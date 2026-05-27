import subprocess, time, httpx, signal

LLAMA = "/home/pavel/llama.cpp/build/bin/llama-server"
MODEL = "/home/pavel/Models/Q6_K/Qwen3.5-397B-A17B-Q6_K-00001-of-00008.gguf"
PORT = 8001; URL = f"http://localhost:{PORT}/v1/chat/completions"

def start(t):
    args = ["-t",str(t),"-ngl","99","--cpu-moe","-c","8192","--parallel","1","--cont-batching"]
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
    r = httpx.post(URL,json={"model":mn,"messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],
                              "max_tokens":100,"temperature":0},timeout=300)
    t = r.json().get("timings",{})
    return t.get("predicted_per_second",0), t.get("prompt_per_second",0)

print(f"{'Threads':>8s} | {'Gen t/s':>8s} | {'Prompt t/s':>10s}")
print("-"*35)
for t in [20, 24, 28, 30, 32, 34, 36, 40]:
    print(f"  Loading {t}t...", end="", flush=True)
    p = start(t)
    if not p or p.poll() is not None:
        print(f"\r{t:>8d} | FAILED")
        continue
    try:
        mn = httpx.get(f"http://localhost:{PORT}/v1/models",timeout=5).json()["data"][0]["id"]
        g, pr = bench(mn)
        print(f"\r{t:>8d} | {g:>8.1f} | {pr:>10.1f}")
    except Exception as e:
        print(f"\r{t:>8d} | ERROR: {e}")
    finally:
        stop(p)
