"""
Indexer CLI

Reads a JSONL file and builds a Tantivy search index on disk.

Mirrors: distributed-search/cmd/indexer/main.go

Usage:
    python cmd/indexer.py --input docs.jsonl --index search.idx --batch-size 1000

Shard mode:
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
from internal.embed import create_embed_client

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
    parser.add_argument("--embed-provider", default="",
                        help="Embedding provider: 'local' (ONNX) or 'ollama' (empty = no embeddings)")
    parser.add_argument("--ollama-host", default="localhost",
                        help="Ollama server host (only used if --embed-provider=ollama)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Number of shards to process concurrently (loads model once)")
    parser.add_argument("--input-pattern", default="",
                        help="Input file pattern with {} for shard ID (e.g. shard-{}.jsonl)")
    args = parser.parse_args()

    start = time.time()

    # Create embedding client once
    embed_client = None
    if args.embed_provider:
        log.info("Embedding mode: %s", args.embed_provider)
        embed_client = create_embed_client(
            provider=args.embed_provider,
            ollama_url=f"http://{args.ollama_host}:11434",
        )

    if args.num_shards > 1 and args.input_pattern:
        import concurrent.futures
        
        def process_shard(shard_id):
            shard_input = args.input_pattern.format(shard_id) if "{}" in args.input_pattern else args.input_pattern
            if not Path(shard_input).is_file():
                log.warning("Shard file %s not found, skipping.", shard_input)
                return 0
            
            index_path = f"{args.index}-{shard_id}"
            log.info("Indexing shard-%d → %s", shard_id, index_path)
            
            idx = SearchIndex(index_path, embed_client=embed_client)
            count = idx.index_jsonl(shard_input, batch_size=args.batch_size, max_docs=args.max_docs)
            idx.close()
            return count

        total_docs = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_shards) as executor:
            futures = {executor.submit(process_shard, i): i for i in range(args.num_shards)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    total_docs += future.result()
                except Exception as e:
                    log.error("Shard indexing failed: %s", e)
                
        elapsed = time.time() - start
        log.info("Indexer complete in %.1fs! (%d docs total across %d shards)", elapsed, total_docs, args.num_shards)
        return

    # Fallback to single shard logic
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

    idx = SearchIndex(index_path, embed_client=embed_client)
    count = idx.index_jsonl(args.input, batch_size=args.batch_size, max_docs=args.max_docs)
    idx.close()

    elapsed = time.time() - start
    log.info("Indexer complete in %.1fs! (%d docs)", elapsed, count)


if __name__ == "__main__":
    main()

