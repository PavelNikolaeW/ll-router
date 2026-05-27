"""Find absolute maximum context for 397B --cpu-moe on single RTX 3090.
Run AFTER other servers are stopped (needs full GPU memory).
"""

import subprocess
import time
import urllib.request
import json
import signal

LLAMA_SERVER = "/home/pavel/llama.cpp/build/bin/llama-server"
MODEL = "/home/pavel/Models/Q6_K/Qwen3.5-397B-A17B-Q6_K-00001-of-00008.gguf"
PORT = 8001
BASE_ARGS = ["-t", "32", "-ngl", "99", "--cpu-moe", "--cont-batching", "--parallel", "1"]


def start_server(ctx_size, extra_args=None):
    args = BASE_ARGS + ["-c", str(ctx_size)]
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


def try_config(name, ctx, extra_args=None):
    print(f"  {name}: ", end="", flush=True)
    proc = start_server(ctx, extra_args)
    if proc is None or proc.poll() is not None:
        print("FAILED (OOM)")
        stop_server(proc)
        return False
    vram = get_vram()
    print(f"OK, VRAM={vram}")
    stop_server(proc)
    return True


def binary_search_max(label, lo, hi, step, extra_args=None):
    """Binary search for max ctx that fits."""
    last_ok = lo
    while lo <= hi:
        mid = ((lo + hi) // 2 // step) * step  # align to step
        if mid == last_ok:
            mid += step
        if mid > hi:
            break
        print(f"  [{label}] trying {mid//1024}k...", end="", flush=True)
        proc = start_server(mid, extra_args)
        if proc is None or proc.poll() is not None:
            print(f" FAILED")
            stop_server(proc)
            hi = mid - step
        else:
            vram = get_vram()
            print(f" OK, VRAM={vram}")
            last_ok = mid
            lo = mid + step
            stop_server(proc)
    return last_ok


def main():
    print(f"{'='*70}")
    print(f"  Find maximum context for Qwen3.5-397B --cpu-moe")
    print(f"  RTX 3090 24GB — need full GPU memory free!")
    print(f"{'='*70}\n")

    vram = get_vram()
    print(f"  Current VRAM usage: {vram}")
    vram_mb = int(vram.split()[0])
    if vram_mb > 1000:
        print(f"  WARNING: {vram_mb} MB in use. Kill other GPU processes first!")
        print(f"  Continuing anyway...\n")

    # FP16 KV: known 384k=22.8GB works. Try up to ~440k
    print("  === FP16 KV cache ===")
    max_fp16 = binary_search_max("FP16", 384 * 1024, 460 * 1024, 8192)
    print(f"  >> Max FP16 context: {max_fp16//1024}k tokens\n")

    # Q4 KV: known 768k=18.5GB works. Try up to ~1.5M
    print("  === Q4 KV cache ===")
    max_q4 = binary_search_max("Q4", 768 * 1024, 1536 * 1024, 16384,
                                extra_args=["-ctk", "q4_0", "-ctv", "q4_0"])
    print(f"  >> Max Q4 context: {max_q4//1024}k tokens\n")

    # Multi-slot configs at max
    print("  === Multi-slot with Q4 ===")
    for par in [2, 4]:
        max_ms = binary_search_max(f"Q4 ×{par}", max_q4 // 2, max_q4 * par, 16384,
                                    extra_args=["-ctk", "q4_0", "-ctv", "q4_0",
                                               "--parallel", str(par)])
        slot_ctx = max_ms // par
        print(f"  >> Max Q4 ×{par}: {max_ms//1024}k total ({slot_ctx//1024}k/slot)\n")

    print(f"\n{'='*70}")
    print(f"  Summary:")
    print(f"  Max FP16: {max_fp16//1024}k tokens")
    print(f"  Max Q4:   {max_q4//1024}k tokens")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
