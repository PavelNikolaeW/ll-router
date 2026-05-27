import subprocess, time, httpx, signal

LLAMA = "/home/pavel/llama.cpp/build/bin/llama-server"
MODEL = "/home/pavel/Models/Q6_K/Qwen3.5-397B-A17B-Q6_K-00001-of-00008.gguf"
PORT = 8001

def vram():
    r = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader"],
                       capture_output=True,text=True)
    return int(r.stdout.strip().replace(" MiB",""))

def start(ncpumoe):
    args = ["-t","32","-ngl","99","--n-cpu-moe",str(ncpumoe),"-c","8192","--parallel","1","--cont-batching"]
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

def bench(mn):
    r = httpx.post(f"http://localhost:{PORT}/v1/chat/completions",
                   json={"model":mn,"messages":[{"role":"user","content":"What is 2+2? Brief."}],"max_tokens":100,"temperature":0},timeout=300)
    return r.json().get("timings",{}).get("predicted_per_second",0)

print(f"{'n-cpu-moe':>10} | {'VRAM':>8} | {'t/s':>6}")
print("-"*32)

# 94 layers total. n-cpu-moe=94 means ALL MoE on CPU (same as --cpu-moe)
# Lower = more MoE layers on GPU = more VRAM but potentially faster
for n in [94, 85, 80, 70, 60]:
    print(f"  Loading n-cpu-moe={n}...", end="", flush=True)
    p = start(n)
    if not p or p.poll() is not None:
        print(f"\r{n:>10} | FAILED (OOM?)")
        continue
    try:
        mn = httpx.get(f"http://localhost:{PORT}/v1/models",timeout=5).json()["data"][0]["id"]
        v = vram()
        tps = bench(mn)
        print(f"\r{n:>10} | {v:>6} MB | {tps:>5.1f}")
    except Exception as e:
        print(f"\r{n:>10} | ERROR: {e}")
    finally:
        stop(p)
