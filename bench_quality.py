"""Compare output quality between FP16 and Q4_0 KV-cache."""

import subprocess
import time
import httpx
import signal

LLAMA_SERVER = "/home/pavel/llama.cpp/build/bin/llama-server"
MODEL = "/home/pavel/Models/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf"
PORT = 8001
URL = f"http://localhost:{PORT}/v1/chat/completions"
MODEL_NAME = "Qwen3.5-27B-Q4_K_M.gguf"

PROMPT = "Write a quicksort algorithm in Python with type hints and docstring. Code only, no explanations. /no_think"

CONFIGS = [
    {
        "name": "FP16 KV-cache",
        "args": ["--ctx-size", "32768", "--parallel", "1", "--cont-batching"],
    },
    {
        "name": "Q8_0 KV-cache",
        "args": ["--ctx-size", "32768", "--parallel", "1", "--cont-batching",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
    },
    {
        "name": "Q4_0 KV-cache",
        "args": ["--ctx-size", "32768", "--parallel", "1", "--cont-batching",
                 "--cache-type-k", "q4_0", "--cache-type-v", "q4_0"],
    },
]


def start_server(extra_args):
    cmd = [LLAMA_SERVER, "--model", MODEL, "--host", "0.0.0.0", "--port", str(PORT),
           "--n-gpu-layers", "99", "--seed", "42"] + extra_args
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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


def generate(prompt, max_tokens=2048):
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant. Respond with code only."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
    }
    r = httpx.post(URL, json=body, timeout=120)
    data = r.json()
    choice = data["choices"][0]["message"]
    content = choice.get("content", "")
    reasoning = choice.get("reasoning_content", "")
    timings = data.get("timings", {})
    return {
        "content": content,
        "reasoning": reasoning,
        "tokens": timings.get("predicted_n", 0),
        "tps": timings.get("predicted_per_second", 0),
    }


def main():
    results = {}
    for config in CONFIGS:
        name = config["name"]
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        proc = start_server(config["args"])
        try:
            r = generate(PROMPT)
            results[name] = r
            print(f"  Tokens: {r['tokens']}, Speed: {r['tps']:.1f} t/s")
            # combine reasoning + content
            full = ""
            if r["reasoning"]:
                full += f"[THINKING]\n{r['reasoning'][:300]}\n...\n\n"
            full += r["content"]
            print(f"\n{full}")
        finally:
            stop_server(proc)

    # Compare outputs
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = results[names[i]]["content"]
            b = results[names[j]]["content"]
            if a == b:
                print(f"  {names[i]} vs {names[j]}: IDENTICAL")
            else:
                # show first difference
                for k, (ca, cb) in enumerate(zip(a, b)):
                    if ca != cb:
                        print(f"  {names[i]} vs {names[j]}: DIFFERENT at char {k}")
                        print(f"    A: ...{a[max(0,k-20):k+30]}...")
                        print(f"    B: ...{b[max(0,k-20):k+30]}...")
                        break
                else:
                    shorter = names[i] if len(a) < len(b) else names[j]
                    print(f"  {names[i]} vs {names[j]}: {shorter} is shorter ({len(a)} vs {len(b)} chars)")


if __name__ == "__main__":
    main()
