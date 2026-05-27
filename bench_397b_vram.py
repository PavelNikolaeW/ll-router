"""Benchmark Qwen3.5-397B --cpu-moe: maximize VRAM usage with context/parallel."""

import subprocess
import time
import httpx
import signal
import concurrent.futures

LLAMA_SERVER = "/home/pavel/llama.cpp/build/bin/llama-server"
MODEL = "/home/pavel/Models/Q6_K/Qwen3.5-397B-A17B-Q6_K-00001-of-00008.gguf"
PORT = 8001
URL = f"http://localhost:{PORT}/v1/chat/completions"

# Базовые аргументы: --cpu-moe -ngl 99 -t 32
BASE_ARGS = ["-t", "32", "-ngl", "99", "--cpu-moe", "--cont-batching"]

CONFIGS = [
    # Baseline
    {"name": "baseline: 8k×1", "args": ["-c", "8192", "--parallel", "1"]},
    # Увеличиваем контекст (1 слот)
    {"name": "ctx 16k×1", "args": ["-c", "16384", "--parallel", "1"]},
    {"name": "ctx 32k×1", "args": ["-c", "32768", "--parallel", "1"]},
    {"name": "ctx 65k×1", "args": ["-c", "65536", "--parallel", "1"]},
    {"name": "ctx 131k×1", "args": ["-c", "131072", "--parallel", "1"]},
    # Параллельность (контекст делится на слоты)
    {"name": "ctx 16k×2 (8k/slot)", "args": ["-c", "16384", "--parallel", "2"]},
    {"name": "ctx 32k×2 (16k/slot)", "args": ["-c", "32768", "--parallel", "2"]},
    {"name": "ctx 32k×4 (8k/slot)", "args": ["-c", "32768", "--parallel", "4"]},
    {"name": "ctx 65k×4 (16k/slot)", "args": ["-c", "65536", "--parallel", "4"]},
    # KV cache quantization — ещё больше контекста
    {"name": "KV q4 + ctx 131k×1", "args": ["-c", "131072", "--parallel", "1", "-ctk", "q4_0", "-ctv", "q4_0"]},
    {"name": "KV q4 + ctx 262k×1", "args": ["-c", "262144", "--parallel", "1", "-ctk", "q4_0", "-ctv", "q4_0"]},
    {"name": "KV q4 + ctx 131k×2", "args": ["-c", "131072", "--parallel", "2", "-ctk", "q4_0", "-ctv", "q4_0"]},
    {"name": "KV q4 + ctx 65k×4 (16k/slot)", "args": ["-c", "65536", "--parallel", "4", "-ctk", "q4_0", "-ctv", "q4_0"]},
]

SHORT_PROMPT = "What is 2+2? Answer briefly."
LONG_PROMPT = "Explain the history of computing from the earliest mechanical calculators to modern quantum computers. Cover key inventions, people, and breakthroughs. " * 15


def start_server(extra_args):
    cmd = [LLAMA_SERVER, "--model", MODEL, "--host", "0.0.0.0", "--port", str(PORT)] + BASE_ARGS + extra_args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for _ in range(300):
        time.sleep(1)
        try:
            r = httpx.get(f"http://localhost:{PORT}/v1/models", timeout=2)
            if r.status_code == 200:
                return proc
        except Exception:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode()[-500:]
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


def bench_single(model_name, prompt, max_tokens=100):
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    r = httpx.post(URL, json=body, timeout=600)
    d = r.json()
    t = d.get("timings", {})
    return {
        "gen_tps": t.get("predicted_per_second", 0),
        "prompt_tps": t.get("prompt_per_second", 0),
    }


def bench_parallel(model_name, prompt, n_parallel, max_tokens=100):
    """Send n_parallel requests simultaneously."""
    def single():
        return bench_single(model_name, prompt, max_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as ex:
        futures = [ex.submit(single) for _ in range(n_parallel)]
        results = [f.result() for f in futures]

    avg_gen = sum(r["gen_tps"] for r in results) / len(results)
    agg_gen = sum(r["gen_tps"] for r in results)
    return avg_gen, agg_gen


def main():
    print(f"{'='*80}")
    print(f"  Qwen3.5-397B --cpu-moe: VRAM utilization sweep")
    print(f"  Goal: fill remaining ~13 GB VRAM with KV-cache")
    print(f"{'='*80}\n")

    model_name = None

    for config in CONFIGS:
        name = config["name"]
        args = config["args"]

        # Determine parallel count from args
        par_idx = args.index("--parallel") if "--parallel" in args else -1
        n_parallel = int(args[par_idx + 1]) if par_idx >= 0 else 1

        print(f"  [{name}] starting...", end="", flush=True)
        proc = start_server(args)
        if proc is None or proc.poll() is not None:
            print(f" FAILED")
            continue

        try:
            if model_name is None:
                r = httpx.get(f"http://localhost:{PORT}/v1/models", timeout=5)
                model_name = r.json()["data"][0]["id"]

            vram = get_vram()

            # Single short request
            r1 = bench_single(model_name, SHORT_PROMPT)

            # Parallel requests if parallel > 1
            if n_parallel > 1:
                avg_gen, agg_gen = bench_parallel(model_name, SHORT_PROMPT, n_parallel)
                print(f"\r  {name:40s} | VRAM: {vram:>10s} | 1×short: {r1['gen_tps']:5.1f} t/s | {n_parallel}×short: {avg_gen:.1f} per / {agg_gen:.1f} agg")
            else:
                print(f"\r  {name:40s} | VRAM: {vram:>10s} | 1×short: {r1['gen_tps']:5.1f} t/s gen | prompt: {r1['prompt_tps']:5.1f} t/s")

        except Exception as e:
            print(f"\r  {name:40s} | ERROR: {e}")
        finally:
            stop_server(proc)

    print(f"\n{'='*80}")
    print(f"  Done!")


if __name__ == "__main__":
    main()
