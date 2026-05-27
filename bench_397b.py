"""Benchmark Qwen3.5-397B on CPU with different parameters."""

import subprocess
import time
import httpx
import signal

LLAMA_SERVER = "/home/pavel/llama.cpp/build/bin/llama-server"
MODEL = "/home/pavel/Models/Q6_K/Qwen3.5-397B-A17B-Q6_K-00001-of-00008.gguf"
PORT = 8001
URL = f"http://localhost:{PORT}/v1/chat/completions"

CONFIGS = [
    {
        "name": "baseline: 32 threads, ngl=0",
        "args": ["-t", "32", "-ngl", "0", "-c", "8192", "--parallel", "1", "--cont-batching"],
    },
    {
        "name": "64 threads (all cores)",
        "args": ["-t", "64", "-ngl", "0", "-c", "8192", "--parallel", "1", "--cont-batching"],
    },
    {
        "name": "32 threads + mlock",
        "args": ["-t", "32", "-ngl", "0", "-c", "8192", "--parallel", "1", "--cont-batching", "--mlock"],
    },
    {
        "name": "32 threads + GPU offload 10 layers",
        "args": ["-t", "32", "-ngl", "10", "-c", "8192", "--parallel", "1", "--cont-batching"],
    },
    {
        "name": "32 threads + GPU offload 20 layers",
        "args": ["-t", "32", "-ngl", "20", "-c", "8192", "--parallel", "1", "--cont-batching"],
    },
    {
        "name": "64 threads + batch 4096",
        "args": ["-t", "64", "-ngl", "0", "-c", "8192", "--parallel", "1", "--cont-batching",
                 "-b", "4096", "-ub", "1024"],
    },
    {
        "name": "32 threads + KV q4_0",
        "args": ["-t", "32", "-ngl", "0", "-c", "8192", "--parallel", "1", "--cont-batching",
                 "-ctk", "q4_0", "-ctv", "q4_0"],
    },
    {
        "name": "64 threads + GPU 10 + mlock",
        "args": ["-t", "64", "-ngl", "10", "-c", "8192", "--parallel", "1", "--cont-batching", "--mlock"],
    },
    {
        "name": "32t + cpu-moe (MoE on CPU, rest on GPU)",
        "args": ["-t", "32", "-ngl", "99", "--cpu-moe", "-c", "8192", "--parallel", "1", "--cont-batching"],
    },
]


def start_server(extra_args):
    cmd = [LLAMA_SERVER, "--model", MODEL, "--host", "0.0.0.0", "--port", str(PORT)] + extra_args
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(300):  # 5 min timeout for 305GB model
        time.sleep(1)
        try:
            r = httpx.get(f"http://localhost:{PORT}/v1/models", timeout=2)
            if r.status_code == 200:
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
    time.sleep(5)


def get_vram():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def bench(model_name):
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
        "max_tokens": 100,
        "temperature": 0,
    }
    r = httpx.post(URL, json=body, timeout=300)
    d = r.json()
    t = d.get("timings", {})
    return {
        "gen_tps": t.get("predicted_per_second", 0),
        "prompt_tps": t.get("prompt_per_second", 0),
        "tokens": t.get("predicted_n", 0),
    }


def main():
    # get model name from first config
    print(f"{'='*65}")
    print(f"  Qwen3.5-397B-A17B Q6_K parameter sweep (CPU)")
    print(f"  Threadripper PRO 3975WX 32c/64t, 503GB RAM, RTX 3090")
    print(f"{'='*65}")

    # first, get model name
    first_proc = start_server(CONFIGS[0]["args"])
    if first_proc is None:
        print("  FAILED to start server")
        return

    try:
        r = httpx.get(f"http://localhost:{PORT}/v1/models", timeout=5)
        model_name = r.json()["data"][0]["id"]
        print(f"  Model: {model_name}\n")
    except Exception:
        model_name = "unknown"

    # run first config
    config = CONFIGS[0]
    vram = get_vram()
    result = bench(model_name)
    print(f"  {config['name']:45s} | {result['gen_tps']:5.1f} t/s gen | {result['prompt_tps']:6.1f} t/s prompt | VRAM: {vram}")

    stop_server(first_proc)

    # run rest
    for config in CONFIGS[1:]:
        print(f"  Loading: {config['name']}...", end="", flush=True)
        proc = start_server(config["args"])
        if proc is None or proc.poll() is not None:
            print(f" FAILED to start")
            continue
        try:
            vram = get_vram()
            result = bench(model_name)
            print(f"\r  {config['name']:45s} | {result['gen_tps']:5.1f} t/s gen | {result['prompt_tps']:6.1f} t/s prompt | VRAM: {vram}")
        except Exception as e:
            print(f"\r  {config['name']:45s} | ERROR: {e}")
        finally:
            stop_server(proc)

    print(f"\n{'='*65}")
    print(f"  Done!")


if __name__ == "__main__":
    main()
