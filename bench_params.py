"""Benchmark llama-server with different parameters."""

import asyncio
import subprocess
import time
import httpx
import signal
import sys

LLAMA_SERVER = "/home/pavel/llama.cpp/build/bin/llama-server"
MODEL = "/home/pavel/Models/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf"
PORT = 8001
URL = f"http://localhost:{PORT}/v1/chat/completions"
MODEL_NAME = "Qwen3.5-27B-Q4_K_M.gguf"

PROMPTS_SHORT = [
    "What is 2+2? Answer briefly.",
    "Name 3 planets. Be concise.",
    "Write a haiku about code.",
    "What color is the sky? One word.",
    "Explain gravity in one sentence.",
    "Count from 1 to 10.",
    "What is Python? Brief answer.",
    "Say hello in Japanese.",
]

FILLER = "The quick brown fox jumps over the lazy dog. "


CONFIGS = [
    {
        "name": "baseline (current)",
        "args": ["--ctx-size", "98304", "--parallel", "8", "--cont-batching"],
    },
    {
        "name": "flash-attn ON",
        "args": ["--ctx-size", "98304", "--parallel", "8", "--cont-batching", "--flash-attn", "on"],
    },
    {
        "name": "flash-attn + KV q8_0",
        "args": ["--ctx-size", "98304", "--parallel", "8", "--cont-batching", "--flash-attn", "on",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
    },
    {
        "name": "flash-attn + KV q4_0",
        "args": ["--ctx-size", "98304", "--parallel", "8", "--cont-batching", "--flash-attn", "on",
                 "--cache-type-k", "q4_0", "--cache-type-v", "q4_0"],
    },
    {
        "name": "FA + KV q4_0 + ctx 196608 (24k/slot)",
        "args": ["--ctx-size", "196608", "--parallel", "8", "--cont-batching", "--flash-attn", "on",
                 "--cache-type-k", "q4_0", "--cache-type-v", "q4_0"],
    },
    {
        "name": "FA + KV q8_0 + batch 4096",
        "args": ["--ctx-size", "98304", "--parallel", "8", "--cont-batching", "--flash-attn", "on",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                 "--batch-size", "4096", "--ubatch-size", "1024"],
    },
    {
        "name": "FA + KV q8_0 + mlock",
        "args": ["--ctx-size", "98304", "--parallel", "8", "--cont-batching", "--flash-attn", "on",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--mlock"],
    },
]


async def send_request(client, prompt, model_name):
    body = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 100}
    resp = await client.post(URL, json=body)
    data = resp.json()
    timings = data.get("timings", {})
    return {
        "tokens": timings.get("predicted_n", 0),
        "tps": timings.get("predicted_per_second", 0),
        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
    }


async def bench(parallel, prompts):
    async with httpx.AsyncClient(timeout=300) as client:
        wall_start = time.perf_counter()
        tasks = [send_request(client, p, MODEL_NAME) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall = time.perf_counter() - wall_start

    total_tok = 0
    avg_tps = []
    for r in results:
        if isinstance(r, Exception):
            return {"error": str(r)}
        total_tok += r["tokens"]
        avg_tps.append(r["tps"])

    return {
        "agg_tps": total_tok / wall if wall > 0 else 0,
        "avg_per_req": sum(avg_tps) / len(avg_tps) if avg_tps else 0,
        "wall": wall,
        "tokens": total_tok,
    }


def start_server(extra_args):
    cmd = [LLAMA_SERVER, "--model", MODEL, "--host", "0.0.0.0", "--port", str(PORT),
           "--n-gpu-layers", "99"] + extra_args
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # wait for server to be ready
    for _ in range(60):
        time.sleep(1)
        try:
            r = httpx.get(f"http://localhost:{PORT}/v1/models", timeout=2)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
    return proc


def stop_server(proc):
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=10)
    time.sleep(3)


def get_vram():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip()


async def run_config(config):
    name = config["name"]
    args = config["args"]

    print(f"\n{'─'*60}")
    print(f"  Config: {name}")
    print(f"  Args: {' '.join(args)}")
    print(f"{'─'*60}")

    proc = start_server(args)
    vram = get_vram()
    print(f"  VRAM: {vram}")

    try:
        # 1 parallel, short
        r = await bench(1, PROMPTS_SHORT[:1])
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            return
        print(f"  1×short:  {r['avg_per_req']:5.1f} t/s per req | {r['agg_tps']:5.1f} agg | {r['wall']:.1f}s")

        # 8 parallel, short
        r = await bench(8, PROMPTS_SHORT)
        print(f"  8×short:  {r['avg_per_req']:5.1f} t/s per req | {r['agg_tps']:5.1f} agg | {r['wall']:.1f}s")

        # 8 parallel, ~2k words
        filler = (FILLER * 230)[:2000 * 6]
        long_prompts = [f"Summarize: {filler}"] * 8
        r = await bench(8, long_prompts)
        print(f"  8×long:   {r['avg_per_req']:5.1f} t/s per req | {r['agg_tps']:5.1f} agg | {r['wall']:.1f}s")
    finally:
        stop_server(proc)


async def main():
    print(f"{'='*60}")
    print(f"  llama-server parameter sweep")
    print(f"  Model: Qwen3.5-27B Q4_K_M")
    print(f"{'='*60}")

    for config in CONFIGS:
        await run_config(config)

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
