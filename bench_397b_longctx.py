"""
Benchmark Qwen3.5-397B --cpu-moe: maximum context, speed vs context length, needle-in-a-haystack.

Phase 1: Find max context that fits in 24GB VRAM
Phase 2: Measure prompt processing + generation speed at increasing context lengths
Phase 3: Needle-in-a-haystack at various depths
"""

import subprocess
import time
import json
import random
import urllib.request
import signal
import sys

LLAMA_SERVER = "/home/pavel/llama.cpp/build/bin/llama-server"
MODEL = "/home/pavel/Models/Q6_K/Qwen3.5-397B-A17B-Q6_K-00001-of-00008.gguf"
PORT = 8001
URL = f"http://localhost:{PORT}/v1/chat/completions"

# Fixed args
BASE_ARGS = ["-t", "32", "-ngl", "99", "--cpu-moe", "--cont-batching", "--parallel", "1"]


def start_server(ctx_size, extra_args=None):
    args = BASE_ARGS + ["-c", str(ctx_size)]
    if extra_args:
        args += extra_args
    cmd = [LLAMA_SERVER, "--model", MODEL, "--host", "0.0.0.0", "--port", str(PORT)] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(300):
        time.sleep(1)
        try:
            r = urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=2)
            if r.status == 200:
                return proc
        except Exception:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode()[-300:]
                print(f"    Server exited: {stderr}")
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
    data = json.loads(r.read())
    return data["data"][0]["id"]


def chat(model_name, messages, max_tokens=100, temperature=0):
    body = json.dumps({
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=600)
    return json.loads(r.read())


def generate_filler_text(n_words):
    """Generate boring filler text that's roughly n_words words."""
    topics = [
        "The history of agriculture spans thousands of years. Early humans transitioned from nomadic hunter-gatherer societies to settled agricultural communities around 10,000 BCE. This shift, known as the Neolithic Revolution, occurred independently in several regions worldwide. In the Fertile Crescent, people began cultivating wheat and barley. In East Asia, rice and millet were the primary crops. The Americas saw the domestication of maize, beans, and squash.",
        "Maritime navigation has been essential to human civilization. Ancient Polynesians navigated vast stretches of the Pacific Ocean using stars, currents, and wave patterns. The Chinese developed the magnetic compass during the Han Dynasty. European explorers in the 15th century relied on astrolabes and dead reckoning to cross the Atlantic. Modern GPS systems use satellite signals to determine position with remarkable accuracy.",
        "The study of geology reveals Earth's long history. Rocks can be classified into three main types: igneous, sedimentary, and metamorphic. Igneous rocks form from cooled magma or lava. Sedimentary rocks are created through the accumulation and compression of sediment. Metamorphic rocks result from the transformation of existing rocks under heat and pressure. The rock cycle describes the continuous transformation between these types.",
        "Music theory encompasses the study of how sounds are organized. The Western musical tradition uses a twelve-tone chromatic scale. Harmony involves the combination of simultaneously sounding notes. Melody refers to a sequence of notes perceived as a single entity. Rhythm describes the pattern of sounds and silences in time. These elements combine in countless ways to create the diversity of music we hear.",
        "Botany is the scientific study of plants. Plants are essential to life on Earth, producing oxygen through photosynthesis and forming the base of most food chains. The plant kingdom includes mosses, ferns, conifers, and flowering plants. Flowering plants, or angiosperms, are the most diverse group, with over 300,000 known species. They reproduce through seeds enclosed in fruits.",
        "Architecture reflects the values and capabilities of civilizations. Ancient Egyptian pyramids demonstrate remarkable engineering precision. Greek temples introduced classical orders: Doric, Ionic, and Corinthian. Roman architecture advanced with the arch, vault, and dome. Gothic cathedrals reached unprecedented heights with flying buttresses and pointed arches. Modern architecture embraces steel, glass, and concrete to create innovative structures.",
        "The periodic table organizes chemical elements by atomic number and properties. Dmitri Mendeleev published his first periodic table in 1869, predicting the existence of undiscovered elements. Elements are arranged in periods (rows) and groups (columns). Elements in the same group share similar chemical properties. The table includes metals, nonmetals, and metalloids, each with distinct characteristics.",
        "Meteorology studies atmospheric phenomena and weather patterns. The atmosphere consists of several layers: troposphere, stratosphere, mesosphere, thermosphere, and exosphere. Weather occurs primarily in the troposphere. Air masses of different temperatures and humidity levels interact to create fronts. High and low pressure systems drive wind patterns. Understanding these dynamics allows for weather forecasting.",
    ]
    text_parts = []
    word_count = 0
    while word_count < n_words:
        paragraph = random.choice(topics)
        text_parts.append(paragraph)
        word_count += len(paragraph.split())
    return " ".join(text_parts)


def estimate_tokens(text):
    """Rough estimate: ~0.75 tokens per word for English."""
    return int(len(text.split()) * 1.3)


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Find maximum context size
# ═══════════════════════════════════════════════════════════════════
def phase1_max_context():
    print(f"\n{'='*70}")
    print(f"  Phase 1: Maximum context size (--cpu-moe, 1 slot)")
    print(f"{'='*70}\n")

    # Test without KV quantization
    print("  --- FP16 KV cache ---")
    for ctx in [131072, 196608, 262144, 327680, 393216]:
        print(f"  ctx {ctx//1024}k: ", end="", flush=True)
        proc = start_server(ctx)
        if proc is None or proc.poll() is not None:
            print("FAILED (OOM or error)")
            stop_server(proc)
            break
        vram = get_vram()
        print(f"OK, VRAM={vram}")
        stop_server(proc)

    # Test with KV q4
    print("\n  --- Q4 KV cache ---")
    for ctx in [262144, 393216, 524288, 655360, 786432]:
        print(f"  ctx {ctx//1024}k (q4): ", end="", flush=True)
        proc = start_server(ctx, extra_args=["-ctk", "q4_0", "-ctv", "q4_0"])
        if proc is None or proc.poll() is not None:
            print("FAILED (OOM or error)")
            stop_server(proc)
            break
        vram = get_vram()
        print(f"OK, VRAM={vram}")
        stop_server(proc)

    # Test with KV q4 + multiple slots
    print("\n  --- Q4 KV cache + parallel slots ---")
    for par, ctx in [(2, 262144), (4, 262144), (2, 524288), (4, 524288)]:
        slot_ctx = ctx // par
        print(f"  ctx {ctx//1024}k × {par} slots ({slot_ctx//1024}k/slot, q4): ", end="", flush=True)
        proc = start_server(ctx, extra_args=["-ctk", "q4_0", "-ctv", "q4_0", "--parallel", str(par)])
        if proc is None or proc.poll() is not None:
            print("FAILED (OOM or error)")
            stop_server(proc)
            break
        vram = get_vram()
        print(f"OK, VRAM={vram}")
        stop_server(proc)


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Speed vs context length
# ═══════════════════════════════════════════════════════════════════
def phase2_speed_vs_context():
    print(f"\n{'='*70}")
    print(f"  Phase 2: Speed vs context length (--cpu-moe, ctx 262k q4)")
    print(f"{'='*70}\n")

    # Start with max context
    proc = start_server(262144, extra_args=["-ctk", "q4_0", "-ctv", "q4_0"])
    if proc is None:
        print("  FAILED to start server")
        return

    model_name = get_model_name()
    vram = get_vram()
    print(f"  Server: ctx=262k, KV q4, VRAM={vram}\n")

    # Test with increasing prompt lengths
    prompt_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000, 40000, 80000]

    print(f"  {'Prompt words':>14s} | {'~Tokens':>8s} | {'Prompt t/s':>10s} | {'Gen t/s':>8s} | {'Total time':>10s}")
    print(f"  {'-'*14}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")

    for n_words in prompt_sizes:
        filler = generate_filler_text(n_words)
        actual_words = len(filler.split())
        est_tokens = estimate_tokens(filler)

        messages = [
            {"role": "user", "content": filler + "\n\nSummarize the above text in one sentence."}
        ]

        try:
            t0 = time.monotonic()
            resp = chat(model_name, messages, max_tokens=100)
            elapsed = time.monotonic() - t0

            timings = resp.get("timings", {})
            prompt_tps = timings.get("prompt_per_second", 0)
            gen_tps = timings.get("predicted_per_second", 0)
            prompt_n = timings.get("prompt_n", 0)

            print(f"  {actual_words:>14,d} | {prompt_n:>8,d} | {prompt_tps:>10.1f} | {gen_tps:>8.1f} | {elapsed:>9.1f}s")
        except Exception as e:
            print(f"  {actual_words:>14,d} | {est_tokens:>8,d} | ERROR: {e}")
            break

    stop_server(proc)


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Needle in a haystack
# ═══════════════════════════════════════════════════════════════════
def phase3_needle_in_haystack():
    print(f"\n{'='*70}")
    print(f"  Phase 3: Needle in a Haystack")
    print(f"{'='*70}\n")

    # The needle — a random fact that doesn't fit the filler
    NEEDLE = "The secret code for project Aurora is 7-4-1-8-3."
    QUESTION = "What is the secret code for project Aurora? Answer with just the code, nothing else."
    EXPECTED = "7-4-1-8-3"

    # Start with large context
    proc = start_server(262144, extra_args=["-ctk", "q4_0", "-ctv", "q4_0"])
    if proc is None:
        print("  FAILED to start server")
        return

    model_name = get_model_name()
    print(f"  Needle: \"{NEEDLE}\"")
    print(f"  Question: \"{QUESTION}\"")
    print(f"  Expected: \"{EXPECTED}\"\n")

    # Test at different context sizes and needle positions
    context_sizes = [1000, 5000, 10000, 20000, 50000, 80000]
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]  # 0=start, 1=end

    print(f"  {'Context':>10s} | {'Position':>10s} | {'Prompt tok':>10s} | {'Found':>6s} | {'Answer':>30s} | {'Time':>8s}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*30}-+-{'-'*8}")

    for ctx_words in context_sizes:
        for pos in positions:
            # Generate filler
            filler = generate_filler_text(ctx_words)
            words = filler.split()

            # Insert needle at position
            insert_idx = int(len(words) * pos)
            insert_idx = max(0, min(insert_idx, len(words)))
            words.insert(insert_idx, f"\n\n{NEEDLE}\n\n")
            text_with_needle = " ".join(words)

            pos_label = {0.0: "start", 0.25: "25%", 0.5: "middle", 0.75: "75%", 1.0: "end"}[pos]

            messages = [
                {"role": "user", "content": text_with_needle + f"\n\n{QUESTION}"}
            ]

            try:
                t0 = time.monotonic()
                resp = chat(model_name, messages, max_tokens=200, temperature=0)
                elapsed = time.monotonic() - t0

                content = resp["choices"][0]["message"]["content"].strip()
                # Remove thinking tags if present
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()

                timings = resp.get("timings", {})
                prompt_n = timings.get("prompt_n", 0)

                found = "YES" if EXPECTED in content else "NO"
                answer_short = content[:30].replace("\n", " ")

                print(f"  {ctx_words:>10,d} | {pos_label:>10s} | {prompt_n:>10,d} | {found:>6s} | {answer_short:>30s} | {elapsed:>7.1f}s")
            except Exception as e:
                print(f"  {ctx_words:>10,d} | {pos_label:>10s} | {'?':>10s} | {'ERR':>6s} | {str(e)[:30]:>30s} | {'?':>8s}")

    stop_server(proc)


def main():
    print(f"{'='*70}")
    print(f"  Qwen3.5-397B --cpu-moe: Long Context & Needle-in-a-Haystack")
    print(f"  Hardware: Threadripper PRO 3975WX, 503GB RAM, RTX 3090 24GB")
    print(f"{'='*70}")

    phase1_max_context()
    phase2_speed_vs_context()
    phase3_needle_in_haystack()

    print(f"\n{'='*70}")
    print(f"  All done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
