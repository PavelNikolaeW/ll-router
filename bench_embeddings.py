"""Quick latency / throughput probe for the ll-router /v1/embeddings endpoint.

Usage:
    python3 bench_embeddings.py [--url http://127.0.0.1:8000] [--model bge-m3]
                                [--single-runs 20] [--batch-size 50] [--batch-runs 10]

Output is two-column markdown plus a JSON dump on stderr so it can be appended
into docs/EMBEDDINGS.md verbatim.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request


SAMPLE_SHORT = "The agent persisted the fact to long-term memory."
SAMPLE_BATCH_CORPUS = [
    f"chunk {i}: an arbitrary sentence used to populate a multi-turn benchmark context."
    for i in range(64)
]


def _post(url: str, payload: dict, timeout: float = 30.0) -> tuple[float, dict]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return (time.perf_counter() - t0) * 1000.0, data


def _percentiles(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    return {
        "min": round(s[0], 1),
        "p50": round(statistics.median(s), 1),
        "p95": round(s[int(0.95 * (len(s) - 1))], 1),
        "max": round(s[-1], 1),
        "mean": round(statistics.mean(s), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="bge-m3")
    ap.add_argument("--single-runs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--batch-runs", type=int, default=10)
    args = ap.parse_args()

    endpoint = f"{args.url.rstrip('/')}/v1/embeddings"

    # Warmup (model load on first request, can take a few hundred ms).
    print(f"warmup: {endpoint} model={args.model}", file=sys.stderr)
    _post(endpoint, {"model": args.model, "input": SAMPLE_SHORT})

    print(f"single-text × {args.single_runs}…", file=sys.stderr)
    single_ms: list[float] = []
    single_dim = 0
    for _ in range(args.single_runs):
        ms, data = _post(endpoint, {"model": args.model, "input": SAMPLE_SHORT})
        single_ms.append(ms)
        single_dim = len(data["data"][0]["embedding"])

    print(f"batch={args.batch_size} × {args.batch_runs}…", file=sys.stderr)
    batch_texts = SAMPLE_BATCH_CORPUS[: args.batch_size]
    batch_ms: list[float] = []
    for _ in range(args.batch_runs):
        ms, _ = _post(endpoint, {"model": args.model, "input": batch_texts})
        batch_ms.append(ms)

    single_stats = _percentiles(single_ms)
    batch_stats = _percentiles(batch_ms)

    print()
    print(f"BGE-M3 latency on {args.url} (model={args.model}, dim={single_dim}):")
    print()
    print(f"| metric            |   min |   p50 |   p95 |   max |  mean |")
    print(f"|-------------------|------:|------:|------:|------:|------:|")
    print(f"| single-text (ms)  | {single_stats['min']:>5} | {single_stats['p50']:>5} | {single_stats['p95']:>5} | {single_stats['max']:>5} | {single_stats['mean']:>5} |")
    print(f"| batch={args.batch_size:<3} (ms)     | {batch_stats['min']:>5} | {batch_stats['p50']:>5} | {batch_stats['p95']:>5} | {batch_stats['max']:>5} | {batch_stats['mean']:>5} |")
    print()
    print(f"throughput (req/s): single ≈ {1000 / single_stats['p50']:.1f}, "
          f"batch-items/s ≈ {1000 * args.batch_size / batch_stats['p50']:.0f}")

    json.dump({
        "url": args.url,
        "model": args.model,
        "dim": single_dim,
        "single_ms": single_stats,
        "batch_ms": batch_stats,
        "batch_size": args.batch_size,
    }, sys.stderr, indent=2)
    sys.stderr.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
