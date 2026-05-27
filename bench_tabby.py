"""Benchmark TabbyAPI (ExLlamaV2/V3) vs llama.cpp results."""

import asyncio
import time
import httpx
import sys

URL = "http://localhost:8010/v1/chat/completions"
MODEL = "Qwen3.5-27B-exl3-4bpw"
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
    "Name a prime number above 50.",
    "What year did WW2 end?",
]

FILLER = "The quick brown fox jumps over the lazy dog. "


async def send_request(client, prompt, rid):
    start = time.perf_counter()
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS}
    resp = await client.post(URL, json=body)
    elapsed = time.perf_counter() - start
    data = resp.json()
    usage = data.get("usage", {})
    return {
        "id": rid,
        "status": resp.status_code,
        "tokens": usage.get("completion_tokens", 0),
        "tps": usage.get("completion_tokens_per_sec", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "wall_time": elapsed,
    }


async def run_bench(concurrency, n_words=0):
    if n_words > 0:
        filler = (FILLER * (n_words // 9 + 1))[:n_words * 6]
        prompts = [f"Summarize: {filler}"] * concurrency
        label = f"{concurrency} parallel, ~{n_words}w prompts"
    else:
        prompts = PROMPTS[:concurrency]
        label = f"{concurrency} parallel, short prompts"

    print(f"\n  {label}")
    print(f"  {'-'*50}")

    async with httpx.AsyncClient(timeout=300) as client:
        wall_start = time.perf_counter()
        tasks = [send_request(client, p, i) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        total_wall = time.perf_counter() - wall_start

    total_tokens = 0
    for r in sorted(results, key=lambda x: x["id"]):
        if r["status"] == 200:
            print(f"    req {r['id']:2d} | {r['tokens']:3d} tok | {r['tps']:6.1f} t/s | {r['wall_time']:5.1f}s | pt={r['prompt_tokens']}")
            total_tokens += r["tokens"]
        else:
            print(f"    req {r['id']:2d} | ERROR {r['status']}")

    agg = total_tokens / total_wall if total_wall > 0 else 0
    print(f"  Total: {total_tokens} tok, {total_wall:.1f}s, {agg:.1f} agg t/s")


async def main():
    print(f"{'='*60}")
    print(f"  TabbyAPI (ExLlamaV3) benchmark: {MODEL}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"{'='*60}")

    # Short prompts
    await run_bench(1)
    await run_bench(4)
    await run_bench(8)

    # Long prompts
    await run_bench(1, n_words=2000)
    await run_bench(8, n_words=2000)


if __name__ == "__main__":
    asyncio.run(main())
