"""
Indexer CLI — Phase 1

Reads a JSONL file and builds a Tantivy search index on disk.

Mirrors: distributed-search/cmd/indexer/main.go

Usage:
    python cmd/indexer.py --input docs.jsonl --index search.idx --batch-size 1000

Shard mode (Phase 2):
    python cmd/indexer.py --input shard-0.jsonl --index search.idx --shard-id 0
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from internal.index import SearchIndex
from internal.embed import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="JSONL → Tantivy indexer")
    parser.add_argument("--input", default="docs.jsonl", help="Path to JSONL docs")
    parser.add_argument("--index", default="search.idx", help="Base index directory path")
    parser.add_argument("--shard-id", type=int, default=-1,
                        help="Shard ID (-1 = unsharded single-node mode)")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for periodic commits")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Max docs to index (0 = all)")
    parser.add_argument("--ollama-host", default="",
                        help="Ollama server host (if set, generates embeddings during indexing)")
    args = parser.parse_args()

    # Determine final index path (mirrors Go: fmt.Sprintf("%s-%d", indexBase, shardID))
    index_path = args.index
    if args.shard_id >= 0:
        index_path = f"{args.index}-{args.shard_id}"
        log.info("Shard mode: indexing shard-%d → %s", args.shard_id, index_path)
    else:
        log.info("Unsharded mode → %s", index_path)

    log.info(
        "Starting indexer | input=%s index=%s batch=%d max=%d",
        args.input, index_path, args.batch_size, args.max_docs,
    )

    start = time.time()

    # Create Ollama client if host is provided
    embed_client = None
    if args.ollama_host:
        log.info("Vector mode: generating embeddings via Ollama at %s", args.ollama_host)
        embed_client = OllamaClient(base_url=f"http://{args.ollama_host}:11434")

    # Create or open the Tantivy index
    idx = SearchIndex(index_path, embed_client=embed_client)

    # Index documents from JSONL
    count = idx.index_jsonl(args.input, batch_size=args.batch_size, max_docs=args.max_docs)

    idx.close()

    elapsed = time.time() - start
    log.info("Indexer complete in %.1fs! (%d docs)", elapsed, count)


if __name__ == "__main__":
    main()
