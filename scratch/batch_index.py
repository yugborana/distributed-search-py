import subprocess
import os
import time

SHARDS = 8
MAX_DOCS = 0 # Index EVERYTHING
OLLAMA_HOST = "localhost"
BATCH_SIZE = 1000

def run_indexing():
    print(f"--- Starting Parallel Indexing for {SHARDS} shards ---")
    processes = []
    for i in range(SHARDS):
        input_file = f"../shard-{i}.jsonl"
        index_path = f"search.idx-{i}"
        
        # Ensure index dir exists and is clean for the new schema
        if os.path.exists(index_path):
            import shutil
            shutil.rmtree(index_path)
        os.makedirs(index_path)

        cmd = [
            "python", "cmd/indexer.py",
            "--input", input_file,
            "--index", index_path,
            "--shard-id", "-1",
            "--max-docs", str(MAX_DOCS),
            "--batch-size", str(BATCH_SIZE),
            "--ollama-host", OLLAMA_HOST
        ]
        print(f"Starting Shard {i}...")
        p = subprocess.Popen(cmd)
        processes.append(p)

    # Wait for all
    for i, p in enumerate(processes):
        p.wait()
        print(f"Shard {i} Indexing Complete.")

if __name__ == "__main__":
    start = time.time()
    run_indexing()
    print(f"Done in {time.time() - start:.1f}s")
