"""
Phase 0 — Baseline Benchmarking

Measures the system's current performance BEFORE any optimization.
Every future phase will be compared against these numbers.

Metrics captured:
  1. Throughput (QPS) — requests completed per second
  2. Latency — p50, p95, p99 per request
  3. Storage — total disk space per shard index
  4. Memory — RSS of each shard container

Usage:
    python benchmarks/baseline_runner.py --coordinator http://localhost:8090
    python benchmarks/baseline_runner.py --coordinator http://localhost:8090 --concurrency 50 100 200
    python benchmarks/baseline_runner.py --coordinator http://localhost:8090 --endpoint hybrid --duration 30
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    import aiohttp
except ImportError:
    print("aiohttp is required: pip install aiohttp")
    sys.exit(1)

try:
    import yaml
except ImportError:
    yaml = None

# ─── Configuration ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default test queries — a mix of common and rare terms
DEFAULT_QUERIES = [
    "climate", "election", "protest", "distributed", "algorithm",
    "neural networks", "government policy", "economic crisis",
    "space exploration", "artificial intelligence", "health care reform",
    "pandemic response", "war conflict", "technology innovation",
    "education reform", "environmental policy", "sports championship",
    "music festival", "film awards", "scientific discovery",
    "earthquake disaster", "flood damage", "wildfire", "hurricane",
    "stock market", "cryptocurrency", "unemployment", "inflation",
    "immigration reform", "trade agreement", "nuclear energy",
    "renewable energy", "carbon emissions", "deforestation",
    "ocean pollution", "biodiversity", "vaccine development",
    "mental health", "obesity epidemic", "water scarcity",
    "urban planning", "transportation", "housing crisis",
    "cybersecurity", "data privacy", "social media regulation",
    "autonomous vehicles", "quantum computing", "gene editing",
    "mars mission", "satellite launch",
]


@dataclass
class LatencyResult:
    """Result from a single benchmark run at one concurrency level."""
    concurrency: int
    total_requests: int
    successful: int
    failed: int
    duration_seconds: float
    qps: float
    latency_min_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    latency_mean_ms: float
    error_types: dict = field(default_factory=dict)


@dataclass
class StorageResult:
    """Disk usage per shard."""
    shard: str
    index_path: str
    size_bytes: int
    size_mb: float


@dataclass
class MemoryResult:
    """Memory usage per shard container."""
    container: str
    rss_mb: float
    limit_mb: float


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    endpoint: str
    system_config: dict
    latency_results: List[LatencyResult]
    storage_results: List[StorageResult]
    memory_results: List[MemoryResult]
    cluster_stats: dict


# ─── Query Sampler ──────────────────────────────────────────────────────────

def build_query_set(coordinator_url: str, sample_size: int = 1000) -> List[str]:
    """
    Build a test query set. Uses default queries and repeats them
    to reach the desired sample size (simulating realistic hot/cold distribution).
    """
    queries = DEFAULT_QUERIES.copy()

    # Weight distribution: some queries are "hot" (repeated more)
    weighted = []
    hot_terms = queries[:10]   # top 10 are "hot" — 5x weight
    cold_terms = queries[10:]  # rest are "cold"

    for q in hot_terms:
        weighted.extend([q] * 5)
    for q in cold_terms:
        weighted.extend([q] * 1)

    # Fill to sample_size
    result = []
    while len(result) < sample_size:
        result.extend(weighted)
    random.shuffle(result)
    return result[:sample_size]


# ─── Latency Benchmark ─────────────────────────────────────────────────────

async def _run_single_request(
    session: aiohttp.ClientSession,
    url: str,
    latencies: list,
    errors: dict,
    semaphore: asyncio.Semaphore,
):
    """Execute one HTTP request and record its latency."""
    async with semaphore:
        start = time.perf_counter()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                await resp.read()
                elapsed = (time.perf_counter() - start) * 1000  # ms
                if resp.status == 200:
                    latencies.append(elapsed)
                else:
                    err_key = f"HTTP_{resp.status}"
                    errors[err_key] = errors.get(err_key, 0) + 1
        except asyncio.TimeoutError:
            errors["timeout"] = errors.get("timeout", 0) + 1
        except aiohttp.ClientError as e:
            err_key = type(e).__name__
            errors[err_key] = errors.get(err_key, 0) + 1
        except Exception as e:
            err_key = type(e).__name__
            errors[err_key] = errors.get(err_key, 0) + 1


async def run_latency_benchmark(
    coordinator_url: str,
    endpoint: str,
    queries: List[str],
    concurrency: int,
    duration_seconds: int,
) -> LatencyResult:
    """
    Run a sustained load test at a given concurrency level.

    Fires requests continuously for `duration_seconds`, using a semaphore
    to cap concurrency. Records per-request latency with time.perf_counter().
    """
    latencies: List[float] = []
    errors: dict = {}
    semaphore = asyncio.Semaphore(concurrency)

    base_path = "/hybrid" if endpoint == "hybrid" else "/search"

    connector = aiohttp.TCPConnector(limit=concurrency + 10, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        start_time = time.perf_counter()
        tasks = []
        query_idx = 0

        # Keep launching requests until duration expires
        while (time.perf_counter() - start_time) < duration_seconds:
            q = queries[query_idx % len(queries)]
            query_idx += 1
            url = f"{coordinator_url}{base_path}?q={q}&limit=5"
            task = asyncio.create_task(
                _run_single_request(session, url, latencies, errors, semaphore)
            )
            tasks.append(task)

            # Small yield to let the event loop breathe
            if query_idx % 100 == 0:
                await asyncio.sleep(0)

        # Wait for all in-flight requests
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.perf_counter() - start_time

    total_requests = len(latencies) + sum(errors.values())
    successful = len(latencies)
    failed = sum(errors.values())

    if latencies:
        latencies.sort()
        result = LatencyResult(
            concurrency=concurrency,
            total_requests=total_requests,
            successful=successful,
            failed=failed,
            duration_seconds=round(total_time, 2),
            qps=round(successful / total_time, 2),
            latency_min_ms=round(latencies[0], 2),
            latency_p50_ms=round(latencies[len(latencies) // 2], 2),
            latency_p95_ms=round(latencies[int(len(latencies) * 0.95)], 2),
            latency_p99_ms=round(latencies[int(len(latencies) * 0.99)], 2),
            latency_max_ms=round(latencies[-1], 2),
            latency_mean_ms=round(statistics.mean(latencies), 2),
            error_types=errors,
        )
    else:
        result = LatencyResult(
            concurrency=concurrency,
            total_requests=total_requests,
            successful=0,
            failed=failed,
            duration_seconds=round(total_time, 2),
            qps=0.0,
            latency_min_ms=0, latency_p50_ms=0, latency_p95_ms=0,
            latency_p99_ms=0, latency_max_ms=0, latency_mean_ms=0,
            error_types=errors,
        )

    return result


# ─── Storage Benchmark ──────────────────────────────────────────────────────

def measure_storage() -> List[StorageResult]:
    """Measure disk space used by each shard's index files."""
    results = []
    for i in range(8):
        idx_path = PROJECT_ROOT / f"search.idx-{i}"
        if idx_path.exists():
            total = sum(f.stat().st_size for f in idx_path.rglob("*") if f.is_file())
            results.append(StorageResult(
                shard=f"shard-{i}",
                index_path=str(idx_path),
                size_bytes=total,
                size_mb=round(total / (1024 * 1024), 2),
            ))
    return results


# ─── Memory Benchmark ───────────────────────────────────────────────────────

def measure_memory() -> List[MemoryResult]:
    """Measure RSS memory of each shard Docker container."""
    results = []
    for i in range(8):
        container = f"shard-{i}"
        try:
            output = subprocess.check_output(
                ["docker", "stats", container, "--no-stream", "--format",
                 "{{.MemUsage}}"],
                text=True, timeout=5
            ).strip()
            # Format: "45.2MiB / 1.946GiB"
            parts = output.split("/")
            rss_str = parts[0].strip()
            limit_str = parts[1].strip() if len(parts) > 1 else "0MiB"

            rss_mb = _parse_mem(rss_str)
            limit_mb = _parse_mem(limit_str)

            results.append(MemoryResult(
                container=container,
                rss_mb=round(rss_mb, 2),
                limit_mb=round(limit_mb, 2),
            ))
        except Exception as e:
            results.append(MemoryResult(container=container, rss_mb=0, limit_mb=0))

    return results


def _parse_mem(s: str) -> float:
    """Parse Docker memory string like '45.2MiB' or '1.946GiB' to MB."""
    s = s.strip()
    if s.endswith("GiB"):
        return float(s[:-3]) * 1024
    elif s.endswith("MiB"):
        return float(s[:-3])
    elif s.endswith("KiB"):
        return float(s[:-3]) / 1024
    elif s.endswith("B"):
        return float(s[:-1]) / (1024 * 1024)
    return 0.0


# ─── Cluster Stats ──────────────────────────────────────────────────────────

async def fetch_cluster_stats(coordinator_url: str) -> dict:
    """Fetch /cluster_stats from the coordinator."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{coordinator_url}/cluster_stats", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception as e:
        print(f"  ⚠ Could not fetch cluster stats: {e}")
    return {}


# ─── Report Generator ───────────────────────────────────────────────────────

def print_report(report: BenchmarkReport):
    """Pretty-print the benchmark report to stdout."""
    print("\n" + "=" * 72)
    print(f"  BASELINE BENCHMARK REPORT")
    print(f"  Timestamp: {report.timestamp}")
    print(f"  Endpoint:  /{report.endpoint}")
    print("=" * 72)

    # Cluster Stats
    cs = report.cluster_stats
    if cs:
        print(f"\n  📊 Cluster: {cs.get('total_shards', '?')} shards, "
              f"{cs.get('total_documents', '?')} documents, "
              f"{cs.get('hot_terms_count', '?')} hot terms")

    # Latency Table
    print(f"\n  {'Conc':>5} │ {'QPS':>8} │ {'p50':>8} │ {'p95':>8} │ {'p99':>8} │ {'Max':>8} │ {'OK':>6} │ {'Err':>5}")
    print(f"  {'─'*5}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*5}")
    for r in report.latency_results:
        print(f"  {r.concurrency:>5} │ {r.qps:>8.1f} │ {r.latency_p50_ms:>7.1f}ms │ "
              f"{r.latency_p95_ms:>7.1f}ms │ {r.latency_p99_ms:>7.1f}ms │ "
              f"{r.latency_max_ms:>7.1f}ms │ {r.successful:>6} │ {r.failed:>5}")
        if r.error_types:
            for etype, count in r.error_types.items():
                print(f"        └─ {etype}: {count}")

    # Storage Table
    if report.storage_results:
        print(f"\n  💾 Storage:")
        total_mb = sum(s.size_mb for s in report.storage_results)
        for s in report.storage_results:
            bar = "█" * max(1, int(s.size_mb / max(total_mb, 1) * 40))
            print(f"    {s.shard:>8}: {s.size_mb:>8.1f} MB  {bar}")
        print(f"    {'TOTAL':>8}: {total_mb:>8.1f} MB")

    # Memory Table
    if report.memory_results and any(m.rss_mb > 0 for m in report.memory_results):
        print(f"\n  🧠 Memory (RSS):")
        for m in report.memory_results:
            if m.rss_mb > 0:
                print(f"    {m.container:>8}: {m.rss_mb:>8.1f} MB / {m.limit_mb:.0f} MB")

    print("\n" + "=" * 72)


def save_report(report: BenchmarkReport, output_path: str):
    """Save the report as JSON for future comparison."""
    data = {
        "timestamp": report.timestamp,
        "endpoint": report.endpoint,
        "system_config": report.system_config,
        "cluster_stats": report.cluster_stats,
        "latency": [asdict(r) for r in report.latency_results],
        "storage": [asdict(s) for s in report.storage_results],
        "memory": [asdict(m) for m in report.memory_results],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  📁 Report saved to: {output_path}")


# ─── Main ───────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Phase 0 — Baseline Benchmark Runner")
    parser.add_argument("--coordinator", default="http://localhost:8090",
                        help="Coordinator URL")
    parser.add_argument("--endpoint", default="search", choices=["search", "hybrid"],
                        help="Which endpoint to benchmark")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[50, 100, 200],
                        help="Concurrency levels to test (e.g. 50 100 200)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration per concurrency level in seconds")
    parser.add_argument("--queries", type=int, default=1000,
                        help="Number of queries in the sample set")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path (default: benchmarks/baseline_TIMESTAMP.json)")
    args = parser.parse_args()

    print(f"\n  🚀 Phase 0 Baseline Benchmark")
    print(f"  ─────────────────────────────")
    print(f"  Coordinator: {args.coordinator}")
    print(f"  Endpoint:    /{args.endpoint}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Duration:    {args.duration}s per level")
    print(f"  Query pool:  {args.queries} queries")

    # Load system config if available
    config = {}
    config_path = PROJECT_ROOT / "system_config.yaml"
    if config_path.exists() and yaml:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"  Config:      {config_path}")

    # Build query set
    print(f"\n  📋 Building query set...")
    queries = build_query_set(args.coordinator, args.queries)
    print(f"     {len(queries)} queries ready (top 5: {queries[:5]})")

    # Smoke test
    print(f"\n  🔍 Smoke test...")
    try:
        async with aiohttp.ClientSession() as session:
            path = "/hybrid" if args.endpoint == "hybrid" else "/search"
            async with session.get(
                f"{args.coordinator}{path}?q=test&limit=1",
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    print(f"     ✅ {args.coordinator}{path} is responding")
                else:
                    print(f"     ❌ Got HTTP {resp.status}. Aborting.")
                    return
    except Exception as e:
        print(f"     ❌ Cannot reach coordinator: {e}")
        return

    # Fetch cluster stats
    print(f"\n  📊 Fetching cluster stats...")
    cluster_stats = await fetch_cluster_stats(args.coordinator)
    if cluster_stats:
        print(f"     {cluster_stats.get('total_shards', '?')} shards, "
              f"{cluster_stats.get('total_documents', '?')} documents")

    # Measure storage
    print(f"\n  💾 Measuring storage...")
    storage_results = measure_storage()
    total_storage = sum(s.size_mb for s in storage_results)
    print(f"     {len(storage_results)} shard indexes, {total_storage:.1f} MB total")

    # Measure memory
    print(f"\n  🧠 Measuring container memory...")
    memory_results = measure_memory()
    total_mem = sum(m.rss_mb for m in memory_results)
    print(f"     {len([m for m in memory_results if m.rss_mb > 0])} containers reporting, "
          f"{total_mem:.1f} MB total RSS")

    # Run latency benchmarks
    latency_results = []
    for conc in args.concurrency:
        print(f"\n  ⚡ Running {conc} concurrent connections for {args.duration}s...")
        result = await run_latency_benchmark(
            args.coordinator, args.endpoint, queries, conc, args.duration
        )
        latency_results.append(result)
        print(f"     QPS: {result.qps:.1f} | p50: {result.latency_p50_ms:.1f}ms | "
              f"p99: {result.latency_p99_ms:.1f}ms | "
              f"OK: {result.successful} | Err: {result.failed}")

    # Build report
    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        endpoint=args.endpoint,
        system_config=config,
        latency_results=latency_results,
        storage_results=storage_results,
        memory_results=memory_results,
        cluster_stats=cluster_stats,
    )

    print_report(report)

    # Save
    output = args.output
    if not output:
        bench_dir = PROJECT_ROOT / "benchmarks"
        bench_dir.mkdir(exist_ok=True)
        output = str(bench_dir / f"baseline_{time.strftime('%Y%m%d_%H%M%S')}.json")
    save_report(report, output)


if __name__ == "__main__":
    asyncio.run(main())
