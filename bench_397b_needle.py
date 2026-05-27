"""Needle-in-a-haystack test for Qwen3.5-397B --cpu-moe.
Max tokens 4096 to accommodate thinking model.
"""

import subprocess
import time
import json
import random
import urllib.request
import signal

LLAMA_SERVER = "/home/pavel/llama.cpp/build/bin/llama-server"
MODEL = "/home/pavel/Models/Q6_K/Qwen3.5-397B-A17B-Q6_K-00001-of-00008.gguf"
PORT = 8001
URL = f"http://localhost:{PORT}/v1/chat/completions"
BASE_ARGS = ["-t", "32", "-ngl", "99", "--cpu-moe", "--cont-batching", "--parallel", "1"]


def log(msg, end="\n"):
    print(msg, end=end, flush=True)


def start_server(ctx_size, extra_args=None):
    args = BASE_ARGS + ["-c", str(ctx_size)]
    if extra_args:
        args += extra_args
    cmd = [LLAMA_SERVER, "--model", MODEL, "--host", "0.0.0.0", "--port", str(PORT)] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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


def chat(model_name, messages, max_tokens=4096, temperature=0):
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
        "The history of agriculture spans thousands of years. Early humans transitioned from nomadic hunter-gatherer societies to settled agricultural communities around 10,000 BCE. This shift, known as the Neolithic Revolution, occurred independently in several regions worldwide. In the Fertile Crescent, people began cultivating wheat and barley. In East Asia, rice and millet were the primary crops.",
        "Maritime navigation has been essential to human civilization. Ancient Polynesians navigated vast stretches of the Pacific Ocean using stars, currents, and wave patterns. The Chinese developed the magnetic compass during the Han Dynasty. European explorers in the 15th century relied on astrolabes and dead reckoning.",
        "The study of geology reveals Earth's long history. Rocks can be classified into three main types: igneous, sedimentary, and metamorphic. The rock cycle describes the continuous transformation between these types over millions of years through various processes.",
        "Music theory encompasses the study of how sounds are organized. The Western musical tradition uses a twelve-tone chromatic scale. Harmony involves the combination of simultaneously sounding notes perceived as a chord.",
        "Botany is the scientific study of plants. Plants are essential to life on Earth, producing oxygen through photosynthesis. The plant kingdom includes mosses, ferns, conifers, and flowering plants with over 300,000 known species.",
        "Architecture reflects the values and capabilities of civilizations. Ancient Egyptian pyramids demonstrate remarkable engineering precision. Greek temples introduced classical orders. Roman architecture advanced with the arch and dome.",
        "The periodic table organizes chemical elements by atomic number. Dmitri Mendeleev published his first periodic table in 1869, successfully predicting the existence and properties of undiscovered elements.",
        "Meteorology studies atmospheric phenomena and weather patterns. The atmosphere consists of several layers. Weather occurs primarily in the troposphere where most of the atmosphere's mass is concentrated.",
    ]
    parts = []
    wc = 0
    while wc < n_words:
        parts.append(random.choice(topics))
        wc += 50  # approx words per topic
    return " ".join(parts)


def main():
    log(f"{'='*75}")
    log(f"  Needle-in-a-Haystack: Qwen3.5-397B --cpu-moe")
    log(f"  max_tokens=4096 (for thinking model)")
    log(f"{'='*75}\n")

    NEEDLE = "The secret code for project Aurora is 7-4-1-8-3."
    QUESTION = "What is the secret code for project Aurora? Reply with ONLY the code."
    EXPECTED = "7-4-1-8-3"

    # Start with large context
    log("  Starting server (262k ctx, KV q4)...", end="")
    proc = start_server(262144, extra_args=["-ctk", "q4_0", "-ctv", "q4_0"])
    if proc is None:
        log(" FAILED")
        return
    model = get_model_name()
    log(f" OK, VRAM={get_vram()}")

    log(f"\n  Needle: \"{NEEDLE}\"")
    log(f"  Expected answer contains: \"{EXPECTED}\"")
    log(f"  max_tokens: 4096\n")

    # First — sanity check without haystack
    log("  --- Sanity check (no haystack) ---")
    msgs = [{"role": "user", "content": f"{NEEDLE}\n\n{QUESTION}"}]
    try:
        resp = chat(model, msgs, max_tokens=4096)
        content = resp["choices"][0]["message"]["content"].strip()
        if "</think>" in content:
            think_part = content.split("</think>")[0]
            answer_part = content.split("</think>")[-1].strip()
            think_tokens = len(think_part.split())
        else:
            answer_part = content
            think_tokens = 0
        found = "YES" if EXPECTED in answer_part else "NO"
        t = resp.get("timings", {})
        log(f"  Answer: '{answer_part[:50]}' | Found: {found} | Think: ~{think_tokens} words | Gen tokens: {t.get('predicted_n', 0)}")
    except Exception as e:
        log(f"  ERROR: {e}")

    # Needle in haystack
    log(f"\n  --- Needle in Haystack ---")
    ctx_words = [1000, 5000, 10000, 20000]
    positions = [0.0, 0.5, 1.0]  # start, middle, end (reduced for speed)

    log(f"  {'Context':>10s} | {'Position':>10s} | {'Tokens':>8s} | {'Found':>6s} | {'Answer':>30s} | {'Gen tok':>8s} | {'Time':>8s}")
    log(f"  {'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*30}-+-{'-'*8}-+-{'-'*8}")

    for cw in ctx_words:
        for pos in positions:
            filler = generate_filler(cw)
            words = filler.split()
            idx = int(len(words) * pos)
            words.insert(idx, f"\n\n{NEEDLE}\n\n")
            text = " ".join(words)

            pos_name = {0.0: "start", 0.5: "middle", 1.0: "end"}[pos]
            msgs = [{"role": "user", "content": text + f"\n\n{QUESTION}"}]

            try:
                t0 = time.monotonic()
                resp = chat(model, msgs, max_tokens=4096, temperature=0)
                elapsed = time.monotonic() - t0

                content = resp["choices"][0]["message"]["content"].strip()
                if "</think>" in content:
                    answer_part = content.split("</think>")[-1].strip()
                else:
                    answer_part = content

                t = resp.get("timings", {})
                prompt_n = t.get("prompt_n", 0)
                gen_n = t.get("predicted_n", 0)
                found = "YES" if EXPECTED in answer_part else "NO"
                short = answer_part.replace("\n", " ")[:30]

                log(f"  {cw:>10,d} | {pos_name:>10s} | {prompt_n:>8,d} | {found:>6s} | {short:>30s} | {gen_n:>8,d} | {elapsed:>7.1f}s")
            except Exception as e:
                log(f"  {cw:>10,d} | {pos_name:>10s} | {'?':>8s} | {'ERR':>6s} | {str(e)[:30]:>30s} | {'?':>8s} | {'?':>8s}")

    stop_server(proc)
    log(f"\n{'='*75}")
    log(f"  Done!")


if __name__ == "__main__":
    main()
