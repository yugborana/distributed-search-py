"""
Profiling Benchmark — measures both /search and /hybrid with per-stage breakdown.

Usage:
    python benchmarks/profile_runner.py --coordinator http://localhost:8090
    python benchmarks/profile_runner.py --coordinator http://localhost:8090 --concurrency 50 100
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import random
from dataclasses import dataclass, field
from typing import List

try:
    import aiohttp
except ImportError:
    print("aiohttp is required: pip install aiohttp")
    sys.exit(1)


QUERIES = [
    "climate", "election", "algorithm", "neural networks",
    "government policy", "space exploration", "artificial intelligence",
    "pandemic response", "technology innovation", "quantum computing",
    "earthquake disaster", "stock market", "renewable energy",
    "cybersecurity", "autonomous vehicles", "vaccine development",
    "mars mission", "ocean pollution", "gene editing", "housing crisis",
]


@dataclass
class StageTimings:
    embed_ms: List[float] = field(default_factory=list)
    routing_ms: List[float] = field(default_factory=list)
    fanout_ms: List[float] = field(default_factory=list)
    fusion_ms: List[float] = field(default_factory=list)
    total_ms: List[float] = field(default_factory=list)


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = int(len(data_sorted) * p / 100)
    return data_sorted[min(idx, len(data_sorted) - 1)]


async def benchmark_endpoint(
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    queries: List[str],
    concurrency: int,
    duration: int,
) -> dict:
    """Run a sustained benchmark against one endpoint."""
    latencies = []
    errors = 0
    stage_timings = StageTimings()
    async def worker():
        nonlocal errors
        while (time.perf_counter() - start_time) < duration:
            q = queries[random.randint(0, len(queries)-1)]
            start = time.perf_counter()
            try:
                url = f"{base_url}/{endpoint}?q={q}&limit=10"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    data = await resp.json()
                    elapsed = (time.perf_counter() - start) * 1000
                    if resp.status == 200:
                        latencies.append(elapsed)
                        timing = data.get("timing")
                        if timing:
                            stage_timings.embed_ms.append(timing.get("embed_ms", 0))
                            stage_timings.routing_ms.append(timing.get("routing_ms", 0))
                            stage_timings.fanout_ms.append(timing.get("fanout_ms", 0))
                            stage_timings.fusion_ms.append(timing.get("fusion_ms", 0))
                            stage_timings.total_ms.append(timing.get("total_ms", 0))
                    else:
                        errors += 1
            except Exception:
                errors += 1

    start_time = time.perf_counter()
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers, return_exceptions=True)

    wall_time = time.perf_counter() - start_time
    ok = len(latencies)
    qps = ok / wall_time if wall_time > 0 else 0

    result = {
        "endpoint": endpoint,
        "concurrency": concurrency,
        "duration_s": round(wall_time, 1),
        "total_requests": ok + errors,
        "successful": ok,
        "failed": errors,
        "qps": round(qps, 1),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "latency_p99_ms": round(percentile(latencies, 99), 1),
        "latency_mean_ms": round(statistics.mean(latencies), 1) if latencies else 0,
    }

    # Add stage breakdown for hybrid
    if stage_timings.total_ms:
        result["stage_breakdown"] = {
            "embed_p50": round(percentile(stage_timings.embed_ms, 50), 2),
            "embed_p95": round(percentile(stage_timings.embed_ms, 95), 2),
            "routing_p50": round(percentile(stage_timings.routing_ms, 50), 2),
            "fanout_p50": round(percentile(stage_timings.fanout_ms, 50), 2),
            "fanout_p95": round(percentile(stage_timings.fanout_ms, 95), 2),
            "fusion_p50": round(percentile(stage_timings.fusion_ms, 50), 2),
        }

    return result


def print_results(results: List[dict]):
    print("\n" + "=" * 80)
    print("  PROFILING BENCHMARK RESULTS")
    print("=" * 80)

    for r in results:
        endpoint = r["endpoint"].upper()
        print(f"\n  /{endpoint} @ {r['concurrency']} concurrent")
        print(f"  {'─' * 50}")
        print(f"    QPS:      {r['qps']:>8.1f}")
        print(f"    p50:      {r['latency_p50_ms']:>8.1f} ms")
        print(f"    p95:      {r['latency_p95_ms']:>8.1f} ms")
        print(f"    p99:      {r['latency_p99_ms']:>8.1f} ms")
        print(f"    mean:     {r['latency_mean_ms']:>8.1f} ms")
        print(f"    OK/Err:   {r['successful']}/{r['failed']}")

        sb = r.get("stage_breakdown")
        if sb:
            print(f"\n    Stage Breakdown (p50 / p95):")
            print(f"      Embedding:  {sb['embed_p50']:>6.1f}ms / {sb['embed_p95']:>6.1f}ms")
            print(f"      Routing:    {sb['routing_p50']:>6.1f}ms")
            print(f"      Fan-out:    {sb['fanout_p50']:>6.1f}ms / {sb['fanout_p95']:>6.1f}ms")
            print(f"      Fusion:     {sb['fusion_p50']:>6.1f}ms")

    # Comparison table
    search_results = [r for r in results if r["endpoint"] == "search"]
    hybrid_results = [r for r in results if r["endpoint"] == "hybrid"]

    if search_results and hybrid_results:
        print(f"\n  {'─' * 60}")
        print(f"  COMPARISON: /search vs /hybrid")
        print(f"  {'Conc':>5} │ {'Search QPS':>12} │ {'Hybrid QPS':>12} │ {'Ratio':>8}")
        print(f"  {'─'*5}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*8}")
        for s, h in zip(search_results, hybrid_results):
            ratio = s["qps"] / h["qps"] if h["qps"] > 0 else float("inf")
            print(f"  {s['concurrency']:>5} │ {s['qps']:>12.1f} │ {h['qps']:>12.1f} │ {ratio:>7.1f}x")

    print("\n" + "=" * 80)


async def main():
    parser = argparse.ArgumentParser(description="Profiling Benchmark")
    parser.add_argument("--coordinator", default="http://localhost:8090")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--duration", type=int, default=30, help="Seconds per test")
    args = parser.parse_args()

    print(f"\n  🔬 Profiling Benchmark")
    print(f"  Coordinator: {args.coordinator}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Duration:    {args.duration}s per test")

    # Smoke test
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.coordinator}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"  ❌ Health check failed: HTTP {resp.status}")
                    return
                print(f"  ✅ Coordinator is healthy")
    except Exception as e:
        print(f"  ❌ Cannot reach coordinator: {e}")
        return

    results = []
    connector = aiohttp.TCPConnector(limit=300, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        for conc in args.concurrency:
            # Benchmark /search (BM25 only)
            print(f"\n  ⚡ /search @ {conc} concurrent for {args.duration}s...")
            r = await benchmark_endpoint(session, args.coordinator, "search", QUERIES, conc, args.duration)
            results.append(r)
            print(f"     QPS={r['qps']:.1f} p50={r['latency_p50_ms']:.1f}ms p99={r['latency_p99_ms']:.1f}ms")

            # Benchmark /hybrid (BM25 + semantic)
            print(f"\n  ⚡ /hybrid @ {conc} concurrent for {args.duration}s...")
            r = await benchmark_endpoint(session, args.coordinator, "hybrid", QUERIES, conc, args.duration)
            results.append(r)
            print(f"     QPS={r['qps']:.1f} p50={r['latency_p50_ms']:.1f}ms p99={r['latency_p99_ms']:.1f}ms")

    print_results(results)

    # Save results
    outpath = f"benchmarks/profile_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  📁 Saved to {outpath}")


if __name__ == "__main__":
    asyncio.run(main())
