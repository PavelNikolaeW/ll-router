"""
Benchmark Gemma-4-31B and Gemma-4-26B-A4B on llama.cpp and ik_llama.cpp.

Tests:
  Phase 1 — 31B: GPU speed (1/4/8 parallel, short/long, KV quant)
  Phase 2 — 26B-A4B: GPU full / --cpu-moe / CPU multi-slot
  Phase 3 — llama.cpp vs ik_llama.cpp (winner config from phase 1 & 2)

Usage:
  PYTHONUNBUFFERED=1 python bench_gemma4.py 2>&1 | tee bench_gemma4.log
"""

import asyncio
import subprocess
import time
import httpx
import signal
import sys
import os

# ── Paths ────────────────────────────────────────────────────────────────────
LLAMA  = "/home/pavel/llama.cpp/build/bin/llama-server"
IKLLAMA = "/home/pavel/ik_llama.cpp/build/bin/llama-server"

MODEL_31B   = "/home/pavel/Models/gemma-4-31B-it-GGUF-Q4_K_M/gemma-4-31B-it-Q4_K_M.gguf"
MODEL_26B   = "/home/pavel/Models/gemma-4-26B-A4B-it-GGUF-Q4_K_M/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"

PORT = 8099
URL  = f"http://localhost:{PORT}/v1/chat/completions"

PROMPT_SHORT = "What is 2+2? Answer in one sentence."
PROMPT_CODING = "Write a Python function that sorts a list of integers using quicksort. Return only code."
FILLER = "The quick brown fox jumps over the lazy dog. " * 50  # ~300 words ≈ 400 tok

# Thinking model — needs generous max_tokens
MAX_TOKENS_SHORT = 512
MAX_TOKENS_LONG  = 1024


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_vram():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
        capture_output=True, text=True)
    return r.stdout.strip()


def start_server(binary, model, extra_args, wait=120):
    log_file = "/tmp/llama_bench_server.log"
    cmd = [binary, "--model", model,
           "--host", "0.0.0.0", "--port", str(PORT)] + extra_args
    print(f"  CMD: {' '.join(cmd)}")
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)
    for i in range(wait):
        time.sleep(1)
        # Check if process died
        if proc.poll() is not None:
            with open(log_file) as lf:
                tail = lf.read()[-1000:]
            print(f"  SERVER CRASHED (exit {proc.returncode}):")
            for line in tail.splitlines()[-8:]:
                print(f"    {line}")
            return proc
        try:
            r = httpx.get(f"http://localhost:{PORT}/v1/models", timeout=2)
            if r.status_code == 200:
                print(f"  Server ready in {i+1}s | VRAM: {get_vram()}")
                return proc
        except Exception:
            pass
    print(f"  WARNING: server not ready after {wait}s")
    return proc


def stop_server(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(2)


async def req(client, prompt, max_tokens=MAX_TOKENS_SHORT):
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = await client.post(URL, json=body, timeout=300)
    wall = time.perf_counter() - t0
    data = resp.json()
    timings = data.get("timings", {})
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {
        "tps":     timings.get("predicted_per_second", 0),
        "pp_tps":  timings.get("prompt_eval_per_second", 0),
        "tokens":  timings.get("predicted_n", 0),
        "wall":    wall,
        "content": content[:80].replace("\n", " "),
    }


async def bench_parallel(prompts, max_tokens=MAX_TOKENS_SHORT):
    async with httpx.AsyncClient(timeout=300) as client:
        # warmup
        try:
            await client.post(URL, json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8
            }, timeout=60)
        except Exception:
            pass

        wall_start = time.perf_counter()
        tasks = [req(client, p, max_tokens) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_total = time.perf_counter() - wall_start

    ok = [r for r in results if not isinstance(r, Exception)]
    if not ok:
        errs = [str(r) for r in results if isinstance(r, Exception)]
        return None, f"ALL FAILED: {errs[0][:100]}"

    total_tok = sum(r["tokens"] for r in ok)
    avg_tps   = sum(r["tps"]    for r in ok) / len(ok)
    avg_pp    = sum(r["pp_tps"] for r in ok) / len(ok)
    agg_tps   = total_tok / wall_total if wall_total > 0 else 0

    return {
        "n":       len(ok),
        "agg_tps": agg_tps,
        "avg_tps": avg_tps,
        "avg_pp":  avg_pp,
        "wall":    wall_total,
        "sample":  ok[0]["content"],
    }, None


def hdr(title):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def subhdr(title):
    print(f"\n  {'─'*60}")
    print(f"  {title}")
    print(f"  {'─'*60}")


async def run_bench(binary, model, label, extra_args):
    """Start server, run short×1, short×4, short×8, long×4, return row."""
    proc = start_server(binary, model, extra_args)
    try:
        if proc.poll() is not None:
            print(f"  SKIP: server failed to start")
            return {}
        rows = {}
        for n, prompt, mt, key in [
            (1,  PROMPT_SHORT,  MAX_TOKENS_SHORT, "1×short"),
            (4,  PROMPT_SHORT,  MAX_TOKENS_SHORT, "4×short"),
            (8,  PROMPT_SHORT,  MAX_TOKENS_SHORT, "8×short"),
            (4,  FILLER + " Summarize briefly.", MAX_TOKENS_LONG, "4×long"),
            (8,  FILLER + " Summarize briefly.", MAX_TOKENS_LONG, "8×long"),
        ]:
            prompts = [prompt] * n
            res, err = await bench_parallel(prompts, mt)
            if err:
                rows[key] = f"ERR:{err[:40]}"
                print(f"    {key:10s}: ERROR — {err[:60]}")
            else:
                rows[key] = res
                print(f"    {key:10s}: {res['avg_tps']:5.1f} t/s/req | "
                      f"agg {res['agg_tps']:5.1f} | pp {res['avg_pp']:5.0f} | "
                      f"wall {res['wall']:.1f}s")
        return rows
    finally:
        stop_server(proc)


# ── Phase 1: 31B GPU ─────────────────────────────────────────────────────────
async def phase1_31b():
    hdr("PHASE 1 — Gemma-4-31B  (18 GB dense, 60L, ctx=262k)")

    # 31B model = ~18.5 GB VRAM. Remaining ~5.5 GB for KV cache.
    # FP16 8k×8=65k tokens = ~6.3 GB → OOM. Use parallel=4 for FP16 configs.
    # Q4_0 reduces KV by 4× → can use parallel=8 with reasonable ctx.
    configs = [
        ("llama.cpp  | ngl=99 | ctx=8k  p4 | FP16",
         LLAMA,  ["-ngl", "99", "-c", "8192",  "--parallel", "4", "--cont-batching"]),
        ("llama.cpp  | ngl=99 | ctx=16k p4 | FP16",
         LLAMA,  ["-ngl", "99", "-c", "16384", "--parallel", "4", "--cont-batching"]),
        ("llama.cpp  | ngl=99 | ctx=16k p8 | KV q4_0",
         LLAMA,  ["-ngl", "99", "-c", "16384", "--parallel", "8", "--cont-batching",
                  "-ctk", "q4_0", "-ctv", "q4_0"]),
        ("llama.cpp  | ngl=99 | ctx=32k p8 | KV q4_0",
         LLAMA,  ["-ngl", "99", "-c", "32768", "--parallel", "8", "--cont-batching",
                  "-ctk", "q4_0", "-ctv", "q4_0"]),
        ("ik_llama.cpp | ngl=99 | ctx=16k p4 | FP16",
         IKLLAMA, ["-ngl", "99", "-c", "16384", "--parallel", "4", "--cont-batching"]),
        ("ik_llama.cpp | ngl=99 | ctx=16k p8 | KV q4_0",
         IKLLAMA, ["-ngl", "99", "-c", "16384", "--parallel", "8", "--cont-batching",
                   "-ctk", "q4_0", "-ctv", "q4_0"]),
    ]

    results = []
    for label, binary, extra in configs:
        subhdr(label)
        rows = await run_bench(binary, MODEL_31B, label, extra)
        results.append((label, rows))

    return results


# ── Phase 2: 26B-A4B modes ───────────────────────────────────────────────────
async def phase2_26b():
    hdr("PHASE 2 — Gemma-4-26B-A4B  (16 GB MoE, 30L, 128 experts / 8 active)")

    # 26B-A4B model = ~16 GB VRAM (MoE). Remaining ~8 GB for KV (FP16).
    # 30 layers, GQA (8 kv heads), head_dim~176. ~165 KB/token FP16.
    # 32k×8 = 256k tokens → ~5.3 GB KV → total ~21.3 GB → fits.
    # 64k×8 FP16 → ~10.5 GB KV → total ~26.5 GB → OOM. Use q4_0.
    # --cpu-moe: attention on GPU (<2GB), experts on CPU → VRAM nearly free.
    configs = [
        ("llama.cpp  | ngl=99 | Full GPU | FP16 | ctx=16k p8",
         LLAMA,  ["-ngl", "99", "-c", "16384", "--parallel", "8", "--cont-batching"]),
        ("llama.cpp  | ngl=99 | Full GPU | FP16 | ctx=32k p8",
         LLAMA,  ["-ngl", "99", "-c", "32768", "--parallel", "8", "--cont-batching"]),
        ("llama.cpp  | ngl=99 | Full GPU | KV q4_0 | ctx=64k p8",
         LLAMA,  ["-ngl", "99", "-c", "65536", "--parallel", "8", "--cont-batching",
                  "-ctk", "q4_0", "-ctv", "q4_0"]),
        ("llama.cpp  | --cpu-moe | FP16 | ctx=32k p8",
         LLAMA,  ["-ngl", "99", "--cpu-moe", "-c", "32768", "--parallel", "8",
                  "--cont-batching", "-t", "32"]),
        # CPU multi-slot (user question: throughput vs latency on CPU)
        ("llama.cpp  | CPU ngl=0 | 1 slot | ctx=8k",
         LLAMA,  ["-ngl", "0", "-c", "8192",  "--parallel", "1", "-t", "32"]),
        ("llama.cpp  | CPU ngl=0 | 4 slots | ctx=32k",
         LLAMA,  ["-ngl", "0", "-c", "32768", "--parallel", "4", "--cont-batching",
                  "-t", "32"]),
        ("llama.cpp  | CPU ngl=0 | 8 slots | ctx=64k",
         LLAMA,  ["-ngl", "0", "-c", "65536", "--parallel", "8", "--cont-batching",
                  "-t", "32"]),
        ("ik_llama.cpp | ngl=99 | Full GPU | FP16 | ctx=32k p8",
         IKLLAMA, ["-ngl", "99", "-c", "32768", "--parallel", "8", "--cont-batching"]),
        ("ik_llama.cpp | --cpu-moe | FP16 | ctx=32k p8",
         IKLLAMA, ["-ngl", "99", "--cpu-moe", "-c", "32768", "--parallel", "8",
                   "--cont-batching", "-t", "32"]),
    ]

    results = []
    for label, binary, extra in configs:
        subhdr(label)
        rows = await run_bench(binary, MODEL_26B, label, extra)
        results.append((label, rows))

    return results


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(title, results):
    hdr(f"SUMMARY — {title}")
    col_keys = ["1×short", "4×short", "8×short", "4×long", "8×long"]

    # Header
    print(f"  {'Config':<50} | {'1×sh':>6} | {'4×sh agg':>8} | {'8×sh agg':>8} | {'4×lg agg':>8} | {'8×lg agg':>8}")
    print(f"  {'-'*50}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for label, rows in results:
        def fmt(key):
            r = rows.get(key)
            if r is None:            return "   n/a"
            if isinstance(r, str):   return "   ERR"
            if key.startswith("1×"): return f"{r['avg_tps']:6.1f}"
            return f"{r['agg_tps']:8.1f}"
        cells = [fmt(k) for k in col_keys]
        short_label = label[:50]
        print(f"  {short_label:<50} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]}")

    print(f"\n  Units: 1×short = t/s per request; N×X agg = aggregate t/s (total tokens / wall time)")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 70)
    print("  Gemma-4 Benchmark: 31B (dense) + 26B-A4B (MoE)")
    print("  llama.cpp vs ik_llama.cpp | GPU / cpu-moe / CPU")
    print("=" * 70)
    print(f"  GPU at start: {get_vram()}")
    print(f"  llama.cpp:   {LLAMA}")
    print(f"  ik_llama.cpp:{IKLLAMA}")

    r1 = await phase1_31b()
    print_summary("Gemma-4-31B", r1)

    r2 = await phase2_26b()
    print_summary("Gemma-4-26B-A4B (MoE)", r2)

    print("\n  DONE. Full results above.")


if __name__ == "__main__":
    asyncio.run(main())
