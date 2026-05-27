"""Fine-tune --cpu-moe config for 397B."""
import subprocess, time, httpx, signal

LLAMA = "/home/pavel/llama.cpp/build/bin/llama-server"
MODEL = "/home/pavel/Models/Q6_K/Qwen3.5-397B-A17B-Q6_K-00001-of-00008.gguf"
PORT = 8001; URL = f"http://localhost:{PORT}/v1/chat/completions"

CONFIGS = [
    ("cpu-moe 32t",          ["-t","32","-ngl","99","--cpu-moe","-c","8192","--parallel","1","--cont-batching"]),
    ("cpu-moe 32t + mlock",  ["-t","32","-ngl","99","--cpu-moe","-c","8192","--parallel","1","--cont-batching","--mlock"]),
    ("cpu-moe 24t",          ["-t","24","-ngl","99","--cpu-moe","-c","8192","--parallel","1","--cont-batching"]),
    ("cpu-moe 16t",          ["-t","16","-ngl","99","--cpu-moe","-c","8192","--parallel","1","--cont-batching"]),
    ("cpu-moe 32t ctx 32k",  ["-t","32","-ngl","99","--cpu-moe","-c","32768","--parallel","1","--cont-batching"]),
    ("cpu-moe 32t parallel4",["-t","32","-ngl","99","--cpu-moe","-c","32768","--parallel","4","--cont-batching"]),
    ("cpu-moe 32t KV q4",    ["-t","32","-ngl","99","--cpu-moe","-c","8192","--parallel","1","--cont-batching","-ctk","q4_0","-ctv","q4_0"]),
    ("cpu-moe 32t prio high",["-t","32","-ngl","99","--cpu-moe","-c","8192","--parallel","1","--cont-batching","--prio","2"]),
]

def start(args):
    p = subprocess.Popen([LLAMA,"--model",MODEL,"--host","0.0.0.0","--port",str(PORT)]+args,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(300):
        time.sleep(1)
        try:
            if httpx.get(f"http://localhost:{PORT}/v1/models",timeout=2).status_code==200: return p
        except: 
            if p.poll() is not None: return None
    return p

def stop(p):
    if not p: return
    p.terminate()
    try: p.wait(10)
    except: p.kill()
    time.sleep(5)

def vram():
    return subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader"],
                          capture_output=True,text=True).stdout.strip()

def bench(mn):
    r = httpx.post(URL,json={"model":mn,"messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],
                              "max_tokens":100,"temperature":0},timeout=300)
    t = r.json().get("timings",{})
    return t.get("predicted_per_second",0), t.get("prompt_per_second",0)

print(f"{'Config':35s} | {'Gen':>7s} | {'Prompt':>7s} | VRAM")
print("-"*70)
for name, args in CONFIGS:
    print(f"  Loading {name}...", end="", flush=True)
    p = start(args)
    if not p or p.poll() is not None:
        print(f"\r{'  '+name:35s} | FAILED")
        continue
    try:
        mn = httpx.get(f"http://localhost:{PORT}/v1/models",timeout=5).json()["data"][0]["id"]
        g, pr = bench(mn)
        v = vram()
        print(f"\r{'  '+name:35s} | {g:5.1f} t/s | {pr:5.1f} t/s | {v}")
    except Exception as e:
        print(f"\r{'  '+name:35s} | ERROR: {e}")
    finally:
        stop(p)
