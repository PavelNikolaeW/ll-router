"""Measure VRAM usage of 27B with different GPU layer counts.
397B --cpu-moe already running on port 8001, using 10.8GB VRAM."""
import subprocess, time, httpx, signal

LLAMA = "/home/pavel/llama.cpp/build/bin/llama-server"
MODEL = "/home/pavel/Models/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf"
PORT = 8010

def vram():
    r = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader"],
                       capture_output=True,text=True)
    return int(r.stdout.strip().replace(" MiB",""))

def start(ngl, ctx, par):
    args = ["-ngl",str(ngl),"-t","16","-c",str(ctx),"--parallel",str(par),"--cont-batching"]
    p = subprocess.Popen([LLAMA,"--model",MODEL,"--host","0.0.0.0","--port",str(PORT)]+args,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
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
                   json={"model":mn,"messages":[{"role":"user","content":"Hi"}],"max_tokens":50,"temperature":0},timeout=120)
    return r.json().get("timings",{}).get("predicted_per_second",0)

base_vram = vram()
print(f"Base VRAM (397B loaded): {base_vram} MiB")
print(f"Available: {24576 - base_vram} MiB")
print()
print(f"{'ngl':>4} | {'ctx':>6} | {'par':>4} | {'VRAM':>6} | {'27B VRAM':>8} | {'t/s':>6}")
print("-"*55)

for ngl, ctx, par in [(5,4096,1),(10,4096,1),(15,4096,1),(20,4096,1),(25,4096,1),(30,4096,1),
                        (15,8192,2),(20,8192,2)]:
    p = start(ngl, ctx, par)
    if not p or p.poll() is not None:
        print(f"{ngl:>4} | {ctx:>6} | {par:>4} | FAILED")
        continue
    try:
        mn = httpx.get(f"http://localhost:{PORT}/v1/models",timeout=5).json()["data"][0]["id"]
        v = vram()
        tps = bench(mn)
        print(f"{ngl:>4} | {ctx:>6} | {par:>4} | {v:>5} | {v-base_vram:>7} | {tps:>5.1f}")
    except Exception as e:
        print(f"{ngl:>4} | {ctx:>6} | {par:>4} | ERROR: {e}")
    finally:
        stop(p)
