"""Measure how inference speed depends on input context length."""

import asyncio
import time
import httpx

DIRECT_URL = "http://localhost:8001/v1/chat/completions"
MODEL = "Qwen3.5-27B-Q4_K_M.gguf"
MAX_TOKENS = 100
FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog. "


def make_prompt(n_words: int) -> str:
    filler = (FILLER_SENTENCE * (n_words // 9 + 1))[:n_words * 6]
    return f"Summarize the following text in exactly one sentence: {filler}"


async def bench_single(client: httpx.AsyncClient, n_words: int) -> dict:
    prompt = make_prompt(n_words)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
    }
    start = time.perf_counter()
    resp = await client.post(DIRECT_URL, json=body)
    wall = time.perf_counter() - start
    data = resp.json()

    timings = data.get("timings", {})
    usage = data.get("usage", {})
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "generated": timings.get("predicted_n", 0),
        "gen_tps": timings.get("predicted_per_second", 0),
        "prompt_tps": timings.get("prompt_per_second", 0),
        "wall": wall,
    }


async def bench_parallel(client: httpx.AsyncClient, n_words: int, n_parallel: int) -> list[dict]:
    tasks = [bench_single(client, n_words) for _ in range(n_parallel)]
    return await asyncio.gather(*tasks)


async def main():
    word_counts = [50, 200, 500, 2000, 5000, 10000, 20000]

    async with httpx.AsyncClient(timeout=600) as client:
        # Single request - speed vs context length
        print(f"{'='*70}")
        print(f"  Single request: generation speed vs input context length")
        print(f"  Max tokens: {MAX_TOKENS}")
        print(f"{'='*70}")
        print(f"  {'Prompt tok':>10} | {'Gen t/s':>8} | {'Prompt t/s':>10} | {'Wall':>6}")
        print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")

        for nw in word_counts:
            try:
                r = await bench_single(client, nw)
                print(
                    f"  {r['prompt_tokens']:>10} | "
                    f"{r['gen_tps']:>8.1f} | "
                    f"{r['prompt_tps']:>10.1f} | "
                    f"{r['wall']:>5.1f}s"
                )
            except Exception as e:
                print(f"  ~{nw} words: ERROR {e}")

        # 8 parallel with medium context
        print(f"\n{'='*70}")
        print(f"  8 parallel requests: speed vs input context length")
        print(f"{'='*70}")
        print(f"  {'Prompt tok':>10} | {'Avg gen t/s':>11} | {'Agg t/s':>8} | {'Wall':>6}")
        print(f"  {'-'*10}-+-{'-'*11}-+-{'-'*8}-+-{'-'*6}")

        for nw in [50, 500, 2000, 5000]:
            try:
                start = time.perf_counter()
                results = await bench_parallel(client, nw, 8)
                wall = time.perf_counter() - start
                avg_gen = sum(r["gen_tps"] for r in results) / len(results)
                total_tok = sum(r["generated"] for r in results)
                pt = results[0]["prompt_tokens"]
                print(
                    f"  {pt:>10} | "
                    f"{avg_gen:>11.1f} | "
                    f"{total_tok / wall:>8.1f} | "
                    f"{wall:>5.1f}s"
                )
            except Exception as e:
                print(f"  ~{nw} words: ERROR {e}")


if __name__ == "__main__":
    asyncio.run(main())
