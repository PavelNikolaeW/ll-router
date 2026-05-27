"""Test VRAM usage with long context fills."""

import asyncio
import time
import httpx
import subprocess

DIRECT_URL = "http://localhost:8001/v1/chat/completions"
MODEL = "Qwen3.5-27B-Q4_K_M.gguf"


def get_vram() -> str:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


async def fill_slots(n_slots: int, n_words: int):
    filler = ("The quick brown fox jumps over the lazy dog. " * 100)[:n_words * 5]
    prompt = f"Summarize: {filler}"

    print(f"\n  {n_slots} slots × ~{n_words} words...")
    print(f"  VRAM before: {get_vram()}")

    async with httpx.AsyncClient(timeout=300) as client:
        tasks = []
        for _ in range(n_slots):
            body = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
            }
            tasks.append(asyncio.create_task(client.post(DIRECT_URL, json=body)))

        # check VRAM mid-flight
        await asyncio.sleep(2)
        print(f"  VRAM during: {get_vram()}")

        results = await asyncio.gather(*tasks, return_exceptions=True)

    print(f"  VRAM after:  {get_vram()}")
    for r in results:
        if isinstance(r, Exception):
            print(f"    error: {r}")
        else:
            data = r.json()
            pt = data.get("usage", {}).get("prompt_tokens", "?")
            status = r.status_code
            print(f"    status={status} prompt_tokens={pt}")


async def main():
    print(f"VRAM idle: {get_vram()}")
    await fill_slots(1, 200)
    await fill_slots(4, 200)
    await fill_slots(8, 200)
    await fill_slots(8, 2000)
    print(f"\nVRAM final: {get_vram()}")


if __name__ == "__main__":
    asyncio.run(main())
