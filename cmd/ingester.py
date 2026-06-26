"""
Data Ingester

Reads a Wikipedia XML dump, cleans wiki markup, and writes
documents as JSONL (one JSON object per line).

Mirrors: distributed-search/cmd/ingester/main.go

Usage:
    python cmd/ingester.py --input data.xml --output docs.jsonl --max-docs 50000
"""

import argparse
import json
import logging
import re
import sys
import time
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add project root to sys.path so we can import internal.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from internal.model import Doc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
MIN_BODY_LEN = 50
MAX_BODY_LEN = 2000
DEFAULT_MAX_DOCS = 50000
DEFAULT_WORKERS = 8
BATCH_SIZE = 1000


# ── Text Cleaning ───────────────────────────────────────────────────────────
# Pre-compiled regex patterns for performance
_WIKI_MARKUP = re.compile(r"\'{2,3}|\[\[|\]\]|\{\{|\}\}|\[|\]")
_MULTI_SPACE = re.compile(r"\s{2,}")


def clean_wiki_text(raw: str) -> str:
    """Strip wiki markup, collapse whitespace, truncate to MAX_BODY_LEN chars."""
    text = _WIKI_MARKUP.sub("", raw)
    text = text.replace("\n", " ").replace("\t", " ")
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text[:MAX_BODY_LEN]


def shard_for_doc(doc_id: str, num_shards: int) -> int:
    h = hashlib.md5(doc_id.encode()).digest()
    return int.from_bytes(h[:8], byteorder="big") % num_shards


def process_page(args: tuple) -> Doc | None:
    """Worker function: clean a raw (title, text) tuple into a Doc.
    Returns None if body is too short after cleaning.
    """
    seq_id, title, raw_text = args
    body = clean_wiki_text(raw_text)
    if len(body) < MIN_BODY_LEN:
        return None
    return Doc(
        id=f"wiki_{seq_id}",
        title=title,
        body=body,
    )


# ── XML Streaming Parser ───────────────────────────────────────────────────

def iter_wiki_pages(xml_path: str, max_docs: int = 0):
    """Yield (seq_id, title, text) tuples from a Wikipedia XML dump.
    
    Supports both plain .xml and compressed .xml.bz2 files.
    Uses lxml.etree.iterparse to stream <page> elements without loading
    the entire file into memory.
    """
    import bz2
    from lxml import etree

    ns = None
    pages_read = 0

    if xml_path.endswith(".bz2"):
        log.info("Detected bz2 compression, decompressing on the fly...")
        source = bz2.open(xml_path, "rb")
    else:
        source = xml_path

    context = etree.iterparse(source, events=("end",), tag=None)

    for event, elem in context:
        if ns is None and elem.tag.startswith("{"):
            ns = elem.tag.split("}")[0] + "}"

        tag_local = elem.tag.replace(ns, "") if ns else elem.tag
        if tag_local != "page":
            continue

        title_tag = f"{ns}title" if ns else "title"
        title_elem = elem.find(title_tag)
        if title_elem is None or not title_elem.text:
            elem.clear()
            continue

        rev_tag = f"{ns}revision" if ns else "revision"
        text_tag = f"{ns}text" if ns else "text"
        rev_elem = elem.find(rev_tag)
        text_elem = rev_elem.find(text_tag) if rev_elem is not None else None

        if text_elem is None or not text_elem.text:
            elem.clear()
            continue

        pages_read += 1
        yield (pages_read, title_elem.text, text_elem.text)

        # Free memory for this element
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if pages_read % 10000 == 0:
            log.info("Read %d pages from XML...", pages_read)

        if max_docs > 0 and pages_read >= max_docs:
            log.info("Hit max-docs cap (%d), stopping reader.", max_docs)
            break

    log.info("Reader finished. Total pages pulled from XML: %d", pages_read)


# ── Main ────────────────────────────────────────────────────────────────────

def _process_batch(pool, batch, out_files, num_shards):
    """Process a batch of pages using the shared pool. Returns (written, skipped)."""
    written = 0
    skipped = 0
    futures = [pool.submit(process_page, p) for p in batch]
    for future in as_completed(futures):
        doc = future.result()
        if doc is None:
            skipped += 1
            continue
        shard_id = shard_for_doc(doc.id, num_shards) if num_shards > 1 else 0
        out_files[shard_id].write(json.dumps(doc.to_dict()) + "\n")
        written += 1
    return written, skipped


def main():
    parser = argparse.ArgumentParser(description="Wikipedia XML → JSONL ingester")
    parser.add_argument("--input", required=True, help="Path to Wikipedia XML dump")
    parser.add_argument("--output", default="docs.jsonl", help="Output JSONL file path")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                        help="Max docs to process (0 = no limit)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Number of parallel text-cleaning workers")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Number of shards to split output into")
    args = parser.parse_args()

    log.info(
        "Starting ingester | input=%s output=%s max-docs=%d workers=%d shards=%d",
        args.input, args.output, args.max_docs, args.workers, args.num_shards
    )

    start = time.time()
    total_written = 0
    skipped = 0

    # Open all shard files
    out_files = {}
    if args.num_shards == 1:
        out_files[0] = open(args.output, "w", encoding="utf-8")
    else:
        for i in range(args.num_shards):
            out_files[i] = open(f"shard-{i}.jsonl", "w", encoding="utf-8")

    try:
        # FIX (PERF-4): Create ONE pool, reuse across all batches
        # Old code created a new ProcessPoolExecutor per batch (50 pool creations for 50K docs)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            batch = []

            for page_tuple in iter_wiki_pages(args.input, args.max_docs):
                batch.append(page_tuple)

                if len(batch) >= BATCH_SIZE:
                    w, s = _process_batch(pool, batch, out_files, args.num_shards)
                    total_written += w
                    skipped += s
                    batch = []

                    if total_written % 5000 == 0 and total_written > 0:
                        log.info("Wrote %d docs so far...", total_written)

            # Flush remaining batch
            if batch:
                w, s = _process_batch(pool, batch, out_files, args.num_shards)
                total_written += w
                skipped += s
    finally:
        for f in out_files.values():
            f.close()

    elapsed = time.time() - start
    rate = total_written / elapsed if elapsed > 0 else 0
    log.info(
        "Done! %d docs written, %d skipped in %.1fs (%.0f docs/sec)",
        total_written, skipped, elapsed, rate,
    )


if __name__ == "__main__":
    main()
