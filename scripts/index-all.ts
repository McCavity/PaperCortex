/**
 * Batch indexing script — embeds all Paperless-ngx documents into the vector store.
 *
 * Usage: npx tsx scripts/index-all.ts
 *
 * Environment variables are read from .env (same as MCP server).
 */

import { createPaperlessClient } from "../src/paperless/client.js";
import { createOllamaClient } from "../src/embeddings/ollama.js";
import { createVectorStore } from "../src/embeddings/store.js";
import Database from "better-sqlite3";

// ---------------------------------------------------------------------------
// Config from environment
// ---------------------------------------------------------------------------

const PAPERLESS_URL   = process.env["PAPERLESS_URL"]   ?? "http://localhost:8000";
const PAPERLESS_TOKEN = process.env["PAPERLESS_TOKEN"] ?? "";
const OLLAMA_URL      = process.env["OLLAMA_URL"]      ?? "http://localhost:11434";
const OLLAMA_MODEL    = process.env["OLLAMA_MODEL"]    ?? "qwen2.5:14b";
const EMBEDDING_MODEL = process.env["OLLAMA_EMBEDDING_MODEL"] ?? "nomic-embed-text";
const VECTOR_DB_PATH  = process.env["VECTOR_DB_PATH"]  ?? "./data/vectors.db";
const BATCH_SIZE      = 3;  // Dokumente parallel embedden (konservativ)
const PAGE_SIZE       = 50; // Paperless-Seite
const BATCH_DELAY_MS  = 200; // Pause zwischen Batches
const MAX_RETRIES     = 3;  // Retry bei Connection-Fehlern

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const paperless = createPaperlessClient({ baseUrl: PAPERLESS_URL, token: PAPERLESS_TOKEN });
  const ollama    = createOllamaClient({ baseUrl: OLLAMA_URL, model: OLLAMA_MODEL, embeddingModel: EMBEDDING_MODEL });
  const store     = createVectorStore({ dbPath: VECTOR_DB_PATH });

  // Health check
  const health = await ollama.healthCheck();
  if (!health.ok) {
    console.error("❌ Ollama nicht erreichbar:", OLLAMA_URL);
    process.exit(1);
  }
  console.log(`✅ Ollama OK (${EMBEDDING_MODEL})`);

  // Tag-ID → Name Mapping aufbauen (alle Seiten)
  const tagMap = new Map<number, string>();
  let tagUrl: string | null = "/tags/?page_size=100";
  while (tagUrl) {
    const tagsResult = await paperless.getTags();
    for (const tag of tagsResult.results) {
      tagMap.set(tag.id, tag.name);
    }
    tagUrl = tagsResult.next ?? null;
    break; // getTags() hat keine page-Parameter — direkte API-Abfrage stattdessen
  }
  // Alle Tags direkt per API holen (paginiert)
  tagMap.clear();
  let tagPage = 1;
  while (true) {
    const res = await fetch(
      `${PAPERLESS_URL}/api/tags/?page_size=100&page=${tagPage}`,
      { headers: { Authorization: `Token ${PAPERLESS_TOKEN}`, Accept: "application/json; version=3" } }
    );
    const data = await res.json() as { count: number; next: string | null; results: Array<{id: number; name: string}> };
    for (const tag of data.results) tagMap.set(tag.id, tag.name);
    if (!data.next) break;
    tagPage++;
  }
  console.log(`🏷️  Tags geladen: ${tagMap.size}`);

  // Gesamtzahl Dokumente ermitteln
  const first = await paperless.getDocuments({ page: 1, page_size: 1 });
  const total = first.count;
  const alreadyIndexed = store.count();
  console.log(`📄 Paperless: ${total} Dokumente | Vector Store: ${alreadyIndexed} bereits indexiert\n`);

  let processed = 0;
  let skipped   = 0;
  let errors    = 0;
  // ---------------------------------------------------------------------
  // Skip-Logik 2026-08-24: Existenz ist das FALSCHE Kriterium.
  //
  // Vorher entschied `store.has(doc.id)` — wer einmal drin war, wurde nie
  // wieder angefasst. Folge: Titel, Tags und Inhalt eines geaenderten
  // Dokuments blieben im Index fuer immer auf dem Stand der Erstindizierung.
  // Gemessen am 24.08.2026: fuenf Eintraege trugen noch den Scanner-Dateinamen,
  // obwohl die Dokumente laengst umbenannt waren.
  //
  // Jetzt entscheidet der AENDERUNGSSTAND. `updated_at` ist die Indizierzeit
  // (der Store setzt sie auf datetime('now') in UTC, ohne Zonensuffix);
  // Paperless liefert `modified` als ISO-8601 mit Zone. Beide werden auf
  // Millisekunden normalisiert, sonst vergleicht man Zeichenketten.
  //
  // Bewusst NICHT in src/ geaendert: das hier ist ein eigenes Skript, der
  // Rest des Repos ist Fremdcode (Klon von renefichtmueller/PaperCortex).
  // ---------------------------------------------------------------------
  const indexiertAm = new Map<number, number>();
  {
    const roDb = new Database(VECTOR_DB_PATH, { readonly: true });
    try {
      for (const r of roDb.prepare(
        "SELECT document_id, updated_at FROM embeddings",
      ).all() as { document_id: number; updated_at: string }[]) {
        // "2026-08-24 15:34:12" ist UTC ohne Suffix -> explizit als UTC lesen
        const t = Date.parse(r.updated_at.replace(" ", "T") + "Z");
        if (!Number.isNaN(t)) indexiertAm.set(r.document_id, t);
      }
    } finally {
      roDb.close();
    }
    console.log(`🕒 Indizierzeitpunkte gelesen: ${indexiertAm.size}`);
  }

  /** true = das Dokument ist unveraendert seit der letzten Indizierung. */
  function unveraendert(doc: { id: number; modified?: string }): boolean {
    const seit = indexiertAm.get(doc.id);
    if (seit === undefined) return false;          // gar nicht im Store
    if (!doc.modified) return true;                // ohne Zeitstempel: wie bisher
    const geaendert = Date.parse(doc.modified);
    return Number.isNaN(geaendert) ? true : geaendert <= seit;
  }

  let aktualisiert = 0;
  let page      = 1;

  // Aufraeumstufe 2026-09-21: jede von Paperless gelieferte ID mitschreiben.
  // Ohne diese Menge kann der Lauf geloeschte Dokumente nicht erkennen — er
  // iteriert ja nur ueber das, was Paperless noch hat.
  const gesehen = new Set<number>();
  let durchlaufVollstaendig = false;

  const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

  async function fetchDocumentWithRetry(id: number, retries = MAX_RETRIES): Promise<typeof paperless extends { getDocument(id: number): Promise<infer T> } ? T : never> {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        return await paperless.getDocument(id) as any;
      } catch (err) {
        if (attempt === retries) throw err;
        await sleep(500 * attempt);
      }
    }
    throw new Error("unreachable");
  }

  while (true) {
    let result;
    try {
      // ordering ist PFLICHT, nicht Kosmetik: ohne stabile Sortierung ist jede Seite
      // eine eigene SQL-Abfrage ohne ORDER BY — Zeilen erscheinen doppelt und andere
      // fallen ganz durch. Gemessen 24.08.2026 bei page_size=50 ueber 1526 Dokumente:
      // ohne ordering 1519 eindeutig (7 Doppelte, 5 Dokumente NIE geliefert),
      // mit ordering=id 1526 eindeutig, 0 Doppelte, ueber drei Laeufe deterministisch.
      // Der Lauf meldete dabei "Fehler: 0" — der Verlust war vollstaendig stumm.
      result = await paperless.getDocuments({ page, page_size: PAGE_SIZE, ordering: "id" });
    } catch (err) {
      console.error(`\n  ⚠️  Fehler beim Laden von Seite ${page}: ${(err as Error).message} — retry in 2s`);
      await sleep(2000);
      continue;
    }
    for (const d of result.results) gesehen.add(d.id);

    if (result.results.length === 0) { durchlaufVollstaendig = true; break; }

    // Batch-Verarbeitung
    for (let i = 0; i < result.results.length; i += BATCH_SIZE) {
      const batch = result.results.slice(i, i + BATCH_SIZE);

      await Promise.all(batch.map(async (doc) => {
        // Überspringen nur, wenn seit der Indizierung NICHTS passiert ist
        if (unveraendert(doc)) {
          skipped++;
          return;
        }
        if (indexiertAm.has(doc.id)) aktualisiert++;

        const content = (doc.content ?? "").trim();
        if (!content) {
          skipped++;
          return;
        }

        try {
          // Tag-IDs zu Namen auflösen
          const tagNames: string[] = Array.isArray(doc.tags)
            ? doc.tags.map((t: unknown) => {
                const id = typeof t === "number" ? t : Number(t);
                return tagMap.get(id) ?? String(t);
              })
            : [];

          // Embedding generieren
          const text = `${doc.title}\n\n${content.slice(0, 2000)}`;
          const embedding = await ollama.embed(text);

          store.upsert({
            documentId: doc.id,
            vector:     embedding.vector,
            content:    content.slice(0, 1000),
            title:      doc.title,
            tags:       tagNames,
            createdAt:  doc.created ?? new Date().toISOString(),
          });

          processed++;
        } catch (err) {
          errors++;
          console.error(`\n  ⚠️  Dok #${doc.id} "${doc.title}": ${(err as Error).message}`);
        }
      }));

      const done = processed + skipped + errors;
      const pct  = Math.round((done / total) * 100);
      process.stdout.write(`\r  Fortschritt: ${done}/${total} (${pct}%) — indexiert: ${processed}, übersprungen: ${skipped}, Fehler: ${errors}   `);

      await sleep(BATCH_DELAY_MS);
    }

    if (!result.next) { durchlaufVollstaendig = true; break; }
    page++;
  }

  // ---------------------------------------------------------------------
  // Aufraeumstufe 2026-09-21: der Lauf fuegt hinzu und frischt auf — ein in
  // Paperless GELOESCHTES Dokument sah er nie, denn es steht nicht mehr in der
  // Liste, ueber die iteriert wird. Die Zeile blieb im Index liegen, samt
  // Titel, Inhalt und Vektor: die semantische Suche lieferte Treffer auf
  // Dokumente, die es nicht mehr gibt. Gemessen 21.09.2026 an Dokument 8.
  //
  // Drei Sperren, weil ein falsch erkannter "geloescht"-Zustand loescht:
  //   1. Nur nach einem VOLLSTAENDIGEN Durchlauf. Bricht die Paginierung ab,
  //      ist `gesehen` unvollstaendig — dann sieht jedes fehlende Dokument wie
  //      ein geloeschtes aus. Das Flag wird ausschliesslich an den beiden
  //      regulaeren Schleifenenden gesetzt.
  //   2. Deckel bei 5 % des Bestands (mindestens 10). Mehr Kandidaten heissen
  //      nicht "viel geloescht", sondern "die Erkennung ist kaputt".
  //   3. `--keine-aufraeumung` schaltet die Stufe ganz ab.
  //
  // Verglichen wird gegen `indexiertAm` — den Indexstand VOR dem Lauf. In
  // diesem Lauf neu geschriebene Zeilen stehen ohnehin in `gesehen`.
  // ---------------------------------------------------------------------
  let entfernt = 0;
  let aufraeumHinweis = "";

  if (process.argv.includes("--keine-aufraeumung")) {
    aufraeumHinweis = "abgeschaltet (--keine-aufraeumung)";
  } else if (!durchlaufVollstaendig) {
    aufraeumHinweis = "uebersprungen — Durchlauf war unvollstaendig";
  } else {
    const karteileichen = [...indexiertAm.keys()].filter((id) => !gesehen.has(id));
    const deckel = Math.max(10, Math.floor(indexiertAm.size * 0.05));
    if (karteileichen.length > deckel) {
      aufraeumHinweis =
        `VERWEIGERT — ${karteileichen.length} Kandidaten ueber dem Deckel (${deckel}). ` +
        `Das ist vermutlich ein Erkennungsfehler, kein Loeschvorgang. ` +
        `Erste IDs: ${karteileichen.slice(0, 10).join(", ")}`;
    } else {
      for (const id of karteileichen) {
        store.remove(id);
        entfernt++;
      }
      if (entfernt > 0) aufraeumHinweis = `IDs: ${karteileichen.join(", ")}`;
    }
  }

  console.log(`\n\n✅ Fertig!`);
  console.log(`   Neu indexiert: ${processed}`);
  console.log(`   davon Auffrischungen: ${aktualisiert}`);
  console.log(`   Übersprungen:  ${skipped}`);
  console.log(`   Fehler:        ${errors}`);
  console.log(`   Entfernt:      ${entfernt}${aufraeumHinweis ? ` — ${aufraeumHinweis}` : ""}`);
  console.log(`   Vector Store:  ${store.count()} Dokumente gesamt`);

  store.close();
}

main().catch((err) => {
  console.error("Fehler:", err);
  process.exit(1);
});
