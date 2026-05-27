"""Load test for ll-router: simulates parallel agent requests."""

import asyncio
import time
import httpx
import sys

ROUTER_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen-27b"
MAX_TOKENS = 100

PROMPTS = [
    "What is 2+2? Answer briefly.",
    "Name 3 planets. Be concise.",
    "Write a haiku about code.",
    "What color is the sky? One word.",
    "Explain gravity in one sentence.",
    "Count from 1 to 10.",
    "What is Python? Brief answer.",
    "Say hello in Japanese.",
    "What is the capital of France?",
    "Name a prime number above 50.",
]


async def send_request(
    client: httpx.AsyncClient,
    prompt: str,
    request_id: int,
) -> dict:
    start = time.perf_counter()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
    }
    try:
        resp = await client.post(ROUTER_URL, json=body)
        elapsed = time.perf_counter() - start
        data = resp.json()

        timings = data.get("timings", {})
        tokens = timings.get("predicted_n", 0)
        tps = timings.get("predicted_per_second", 0)

        return {
            "id": request_id,
            "status": resp.status_code,
            "tokens": tokens,
            "tps": tps,
            "wall_time": elapsed,
            "prompt": prompt[:40],
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "id": request_id,
            "status": "error",
            "error": str(e),
            "wall_time": elapsed,
            "prompt": prompt[:40],
        }


async def run_bench(concurrency: int):
    prompts = PROMPTS[:concurrency]

    print(f"\n{'='*60}")
    print(f"  Load test: {concurrency} parallel requests")
    print(f"  Model: {MODEL} | Max tokens: {MAX_TOKENS}")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient(timeout=300) as client:
        wall_start = time.perf_counter()
        tasks = [
            send_request(client, p, i)
            for i, p in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
        total_wall = time.perf_counter() - wall_start

    total_tokens = 0
    for r in sorted(results, key=lambda x: x["id"]):
        status = r["status"]
        if status == 200:
            print(
                f"  req {r['id']:2d} | {r['tokens']:3d} tok | "
                f"{r['tps']:6.1f} t/s | {r['wall_time']:5.1f}s | {r['prompt']}"
            )
            total_tokens += r["tokens"]
        else:
            print(f"  req {r['id']:2d} | ERROR: {r.get('error', status)}")

    print(f"\n{'─'*60}")
    print(f"  Total tokens:     {total_tokens}")
    print(f"  Total wall time:  {total_wall:.1f}s")
    print(f"  Throughput:       {total_tokens / total_wall:.1f} tok/s (aggregate)")
    print(f"  Avg per request:  {total_wall / concurrency:.1f}s")
    print()


async def main():
    levels = [1, 2, 4]
    if len(sys.argv) > 1:
        levels = [int(x) for x in sys.argv[1:]]

    for c in levels:
        await run_bench(c)


if __name__ == "__main__":
    asyncio.run(main())
