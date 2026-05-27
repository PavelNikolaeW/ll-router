"""
Benchmark Qwen3.5-397B --cpu-moe: max context + speed + needle-in-haystack.
Optimized version — reasonable prompt sizes, unbuffered output.
"""

import subprocess
import time
import json
import random
import urllib.request
import signal
import sys
import concurrent.futures

LLAMA_SERVER = "/home/pavel/llama.cpp/build/bin/llama-server"
MODEL = "/home/pavel/Models/Q6_K/Qwen3.5-397B-A17B-Q6_K-00001-of-00008.gguf"
PORT = 8001
URL = f"http://localhost:{PORT}/v1/chat/completions"
BASE_ARGS = ["-t", "32", "-ngl", "99", "--cpu-moe", "--cont-batching"]


def log(msg, end="\n"):
    print(msg, end=end, flush=True)


def start_server(ctx_size, parallel=1, extra_args=None):
    args = BASE_ARGS + ["-c", str(ctx_size), "--parallel", str(parallel)]
    if extra_args:
        args += extra_args
    cmd = [LLAMA_SERVER, "--model", MODEL, "--host", "0.0.0.0", "--port", str(PORT)] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(180):
        time.sleep(1)
        try:
            r = urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=2)
            if r.status == 200:
                return proc
        except Exception:
            if proc.poll() is not None:
                return None
    return proc


def stop_server(proc):
    if proc is None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    time.sleep(3)


def get_vram():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def get_model_name():
    r = urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=5)
    return json.loads(r.read())["data"][0]["id"]


def chat(model_name, messages, max_tokens=100, temperature=0):
    body = json.dumps({
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=1800)
    return json.loads(r.read())


def generate_filler(n_words):
    topics = [
        "The history of agriculture spans thousands of years. Early humans transitioned from nomadic hunter-gatherer societies to settled agricultural communities around 10,000 BCE. This shift, known as the Neolithic Revolution, occurred independently in several regions worldwide. In the Fertile Crescent, people began cultivating wheat and barley. In East Asia, rice and millet were the primary crops. The Americas saw the domestication of maize, beans, and squash.",
        "Maritime navigation has been essential to human civilization. Ancient Polynesians navigated vast stretches of the Pacific Ocean using stars, currents, and wave patterns. The Chinese developed the magnetic compass during the Han Dynasty. European explorers in the 15th century relied on astrolabes and dead reckoning to cross the Atlantic.",
        "The study of geology reveals Earth's long history. Rocks can be classified into three main types: igneous, sedimentary, and metamorphic. The rock cycle describes the continuous transformation between these types over millions of years.",
        "Music theory encompasses the study of how sounds are organized. The Western musical tradition uses a twelve-tone chromatic scale. Harmony involves the combination of simultaneously sounding notes. Melody refers to a sequence of notes perceived as a single entity.",
        "Botany is the scientific study of plants. Plants are essential to life on Earth, producing oxygen through photosynthesis. The plant kingdom includes mosses, ferns, conifers, and flowering plants with over 300,000 known species.",
        "Architecture reflects the values and capabilities of civilizations. Ancient Egyptian pyramids demonstrate remarkable engineering precision. Greek temples introduced classical orders. Roman architecture advanced with the arch, vault, and dome.",
        "The periodic table organizes chemical elements by atomic number. Dmitri Mendeleev published his first periodic table in 1869, predicting undiscovered elements. Elements in the same group share similar chemical properties.",
        "Meteorology studies atmospheric phenomena and weather patterns. The atmosphere consists of several layers. Weather occurs primarily in the troposphere. High and low pressure systems drive wind patterns.",
    ]
    parts = []
    wc = 0
    while wc < n_words:
        p = random.choice(topics)
        parts.append(p)
        wc += len(p.split())
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════
# Phase 1: Binary search for max context
# ═══════════════════════════════════════════════════════════════
def phase1():
    log(f"\n{'='*70}")
    log(f"  Phase 1: Maximum context (binary search)")
    log(f"{'='*70}\n")

    # FP16 KV — known: 384k=22.8GB OK
    log("  --- FP16 KV cache ---")
    lo, hi, step = 384 * 1024, 480 * 1024, 8192
    last_ok = 384 * 1024
    last_vram = "22796 MiB"
    while lo <= hi:
        mid = ((lo + hi) // 2 // step) * step
        if mid <= last_ok:
            mid = last_ok + step
        if mid > hi:
            break
        log(f"  FP16 {mid//1024}k: ", end="")
        proc = start_server(mid)
        if proc is None or proc.poll() is not None:
            log("FAILED (OOM)")
            stop_server(proc)
            hi = mid - step
        else:
            vram = get_vram()
            log(f"OK, VRAM={vram}")
            last_ok = mid
            last_vram = vram
            lo = mid + step
            stop_server(proc)
    log(f"  >> Max FP16: {last_ok//1024}k tokens, VRAM={last_vram}\n")
    max_fp16 = last_ok

    # Q4 KV — known: 768k=18.5GB OK
    log("  --- Q4 KV cache ---")
    lo, hi, step = 768 * 1024, 1600 * 1024, 32768
    last_ok = 768 * 1024
    last_vram = "18536 MiB"
    while lo <= hi:
        mid = ((lo + hi) // 2 // step) * step
        if mid <= last_ok:
            mid = last_ok + step
        if mid > hi:
            break
        log(f"  Q4 {mid//1024}k: ", end="")
        proc = start_server(mid, extra_args=["-ctk", "q4_0", "-ctv", "q4_0"])
        if proc is None or proc.poll() is not None:
            log("FAILED (OOM)")
            stop_server(proc)
            hi = mid - step
        else:
            vram = get_vram()
            log(f"OK, VRAM={vram}")
            last_ok = mid
            last_vram = vram
            lo = mid + step
            stop_server(proc)
    log(f"  >> Max Q4: {last_ok//1024}k tokens, VRAM={last_vram}\n")
    max_q4 = last_ok

    return max_fp16, max_q4


# ═══════════════════════════════════════════════════════════════
# Phase 2: Speed vs context length
# ═══════════════════════════════════════════════════════════════
def phase2():
    log(f"\n{'='*70}")
    log(f"  Phase 2: Speed vs prompt length")
    log(f"{'='*70}\n")

    # Use 262k context with Q4
    proc = start_server(262144, extra_args=["-ctk", "q4_0", "-ctv", "q4_0"])
    if proc is None:
        log("  FAILED to start server")
        return
    model = get_model_name()
    log(f"  Server: ctx=262k, KV q4, VRAM={get_vram()}\n")

    # Reasonable sizes: up to ~20k words (~26k tokens, ~15 min max)
    sizes = [100, 500, 1000, 2000, 5000, 10000, 20000]

    log(f"  {'Words':>10s} | {'Prompt tok':>10s} | {'Prompt t/s':>10s} | {'Gen t/s':>8s} | {'Wall time':>10s}")
    log(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")

    for nw in sizes:
        filler = generate_filler(nw)
        msgs = [{"role": "user", "content": filler + "\n\nSummarize the above in one sentence."}]
        try:
            t0 = time.monotonic()
            resp = chat(model, msgs, max_tokens=100)
            elapsed = time.monotonic() - t0
            t = resp.get("timings", {})
            log(f"  {nw:>10,d} | {t.get('prompt_n',0):>10,d} | {t.get('prompt_per_second',0):>10.1f} | {t.get('predicted_per_second',0):>8.1f} | {elapsed:>9.1f}s")
        except Exception as e:
            log(f"  {nw:>10,d} | ERROR: {e}")
            break

    stop_server(proc)


# ═══════════════════════════════════════════════════════════════
# Phase 3: Needle in a haystack
# ═══════════════════════════════════════════════════════════════
def phase3():
    log(f"\n{'='*70}")
    log(f"  Phase 3: Needle in a Haystack")
    log(f"{'='*70}\n")

    NEEDLE = "The secret code for project Aurora is 7-4-1-8-3."
    QUESTION = "What is the secret code for project Aurora? Reply with ONLY the code, no explanation."
    EXPECTED = "7-4-1-8-3"

    proc = start_server(262144, extra_args=["-ctk", "q4_0", "-ctv", "q4_0"])
    if proc is None:
        log("  FAILED to start server")
        return
    model = get_model_name()
    log(f"  Needle: \"{NEEDLE}\"")
    log(f"  Expected: \"{EXPECTED}\"\n")

    # Context sizes and needle positions
    ctx_words = [1000, 5000, 10000, 20000]
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    log(f"  {'Context':>10s} | {'Position':>10s} | {'Tokens':>8s} | {'Found':>6s} | {'Answer':>25s} | {'Time':>8s}")
    log(f"  {'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*25}-+-{'-'*8}")

    for cw in ctx_words:
        for pos in positions:
            filler = generate_filler(cw)
            words = filler.split()
            idx = int(len(words) * pos)
            words.insert(idx, f"\n\n{NEEDLE}\n\n")
            text = " ".join(words)

            pos_name = {0.0: "start", 0.25: "25%", 0.5: "middle", 0.75: "75%", 1.0: "end"}[pos]
            msgs = [{"role": "user", "content": text + f"\n\n{QUESTION}"}]

            try:
                t0 = time.monotonic()
                resp = chat(model, msgs, max_tokens=4096, temperature=0)
                elapsed = time.monotonic() - t0

                content = resp["choices"][0]["message"]["content"].strip()
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()

                t = resp.get("timings", {})
                prompt_n = t.get("prompt_n", 0)
                found = "YES" if EXPECTED in content else "NO"
                short = content.replace("\n", " ")[:25]

                log(f"  {cw:>10,d} | {pos_name:>10s} | {prompt_n:>8,d} | {found:>6s} | {short:>25s} | {elapsed:>7.1f}s")
            except Exception as e:
                log(f"  {cw:>10,d} | {pos_name:>10s} | {'?':>8s} | {'ERR':>6s} | {str(e)[:25]:>25s} | {'?':>8s}")

    stop_server(proc)


def main():
    log(f"{'='*70}")
    log(f"  Qwen3.5-397B --cpu-moe: Context, Speed & Needle-in-a-Haystack v2")
    log(f"  Threadripper PRO 3975WX, 503GB RAM, RTX 3090 24GB")
    log(f"{'='*70}")

    max_fp16, max_q4 = phase1()
    phase2()
    phase3()

    log(f"\n{'='*70}")
    log(f"  Done! Max context: FP16={max_fp16//1024}k, Q4={max_q4//1024}k")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
