#!/bin/bash
# rebuild-index.sh — Full index rebuild pipeline for K8s CronJob.
#
# Fixes Finding 1: The DEPLOYMENT_GUIDE's CronJob ran bare `indexer.py`
# which only creates a single unsharded index. This script orchestrates
# the complete pipeline:
#   1. ingester.py: XML → 8 sharded JSONL files
#   2. indexer.py:  JSONL → 8 Tantivy index partitions (with embeddings)
#
# Usage: ./scripts/rebuild-index.sh <input_xml_or_jsonl> <index_base> <num_shards>
#
# The coordinator's refresh_partitions_task() will pick up the new
# index data within 30 seconds — no restart needed.

set -euo pipefail

INPUT_FILE="${1:-/data/wikipedia.xml}"
INDEX_BASE="${2:-/app/search.idx}"
NUM_SHARDS="${3:-8}"
EMBED_PROVIDER="${EMBED_PROVIDER:-local}"
MAX_DOCS="${MAX_DOCS:-0}"
BATCH_SIZE="${BATCH_SIZE:-2000}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "=== Index Rebuild Pipeline ==="
log "Input:      $INPUT_FILE"
log "Index base: $INDEX_BASE"
log "Shards:     $NUM_SHARDS"
log "Embed:      $EMBED_PROVIDER"
log "Max docs:   $MAX_DOCS (0 = unlimited)"

# ── Step 1: Check if input is XML (needs ingestion) or JSONL (skip to indexing) ──
if [[ "$INPUT_FILE" == *.xml ]] || [[ "$INPUT_FILE" == *.xml.bz2 ]]; then
    log "Step 1/2: Ingesting XML → sharded JSONL files..."
    
    # Change to the tmp dir before running ingester, so shard files land there:
    cd /tmp/shards
    
    python /app/cmd/ingester.py \
        --input "$INPUT_FILE" \
        --output /tmp/shards/docs.jsonl \
        --max-docs "$MAX_DOCS" \
        --workers 4 \
        --num-shards "$NUM_SHARDS"
    
    log "Ingestion complete. Shard files created."
    JSONL_PATTERN="shard"
else
    log "Step 1/2: Input is JSONL, skipping ingestion."
    JSONL_PATTERN="direct"
fi

# ── Step 2: Index each shard ──
log "Step 2/2: Indexing $NUM_SHARDS shards..."

if [ "$JSONL_PATTERN" = "shard" ]; then
    PATTERN="shard-{}.jsonl"
else
    PATTERN="$INPUT_FILE"
fi

log "  Running concurrent indexing across $NUM_SHARDS shards with pattern: $PATTERN"

python /app/cmd/indexer.py \
    --input-pattern "$PATTERN" \
    --index "$INDEX_BASE" \
    --num-shards "$NUM_SHARDS" \
    --batch-size "$BATCH_SIZE" \
    --embed-provider "$EMBED_PROVIDER"

# ── Cleanup temporary JSONL files ──
if [ "$JSONL_PATTERN" = "shard" ]; then
    log "Cleaning up temporary shard JSONL files..."
    cd /app && rm -rf /tmp/shards/shard-*.jsonl
fi

log "=== Index Rebuild Complete ==="
