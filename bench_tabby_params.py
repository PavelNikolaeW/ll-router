"""Benchmark TabbyAPI with different parameters."""

import asyncio
import subprocess
import time
import httpx
import signal
import json
import yaml

TABBY_DIR = "/home/pavel/tabbyAPI"
CONFIG_PATH = f"{TABBY_DIR}/config.yml"
PORT = 8010
URL = f"http://localhost:{PORT}/v1/chat/completions"
MODEL_DIR = "/home/pavel/Models/exl3"
MODEL_NAME = "Qwen3.5-27B-exl3-4bpw"

PROMPTS_SHORT = [
    "What is 2+2? Answer briefly.",
    "Name 3 planets. Be concise.",
    "Write a haiku about code.",
    "What color is the sky? One word.",
    "Explain gravity in one sentence.",
    "Count from 1 to 10.",
    "What is Python? Brief answer.",
    "Say hello in Japanese.",
]

FILLER = "The quick brown fox jumps over the lazy dog. "

SYSTEM_PROMPT = """You are an AI agent that executes tasks. You MUST follow these rules strictly:
1. Always respond in valid JSON format: {"action": "...", "params": {...}, "reasoning": "..."}
2. Available actions: "search", "read_file", "write_file", "execute", "done"
3. No markdown formatting - raw JSON only
4. Always include reasoning field"""

AGENT_MESSAGES = [
    {"role": "user", "content": "I need you to help me set up a web scraping pipeline"},
    {"role": "assistant", "content": '{"action": "search", "params": {"query": "python web scraping libraries 2024"}, "reasoning": "First researching the best current tools"}'},
    {"role": "user", "content": "Results: beautifulsoup4, scrapy, playwright, httpx. Use httpx + bs4. Create the scraper."},
    {"role": "assistant", "content": '{"action": "write_file", "params": {"path": "/tmp/scraper.py", "content": "import httpx\\nfrom bs4 import BeautifulSoup\\n\\nasync def scrape(url):\\n    async with httpx.AsyncClient() as client:\\n        resp = await client.get(url)\\n        soup = BeautifulSoup(resp.text, \\"html.parser\\")\\n        return {\\"title\\": soup.title.string}"}, "reasoning": "Creating async scraper"}'},
    {"role": "user", "content": "Add error handling, retry with exponential backoff, logging, and save to JSON. Raw JSON only."},
]

CODING_PROMPT = "Write a quicksort algorithm in Python with type hints and docstring. Code only, no explanations."


# TabbyAPI configs to test
CONFIGS = [
    {
        "name": "baseline (FP16 cache, 98k)",
        "model": {
            "cache_mode": "FP16",
            "cache_size": 98304,
            "max_seq_len": 12288,
            "chunk_size": 2048,
        },
    },
    {
        "name": "cache Q8",
        "model": {
            "cache_mode": "Q8",
            "cache_size": 98304,
            "max_seq_len": 12288,
            "chunk_size": 2048,
        },
    },
    {
        "name": "cache Q4",
        "model": {
            "cache_mode": "Q4",
            "cache_size": 98304,
            "max_seq_len": 12288,
            "chunk_size": 2048,
        },
    },
    {
        "name": "Q4 + cache 196k (24k/seq)",
        "model": {
            "cache_mode": "Q4",
            "cache_size": 196608,
            "max_seq_len": 24576,
            "chunk_size": 2048,
        },
    },
    {
        "name": "Q4 + chunk 4096",
        "model": {
            "cache_mode": "Q4",
            "cache_size": 98304,
            "max_seq_len": 12288,
            "chunk_size": 4096,
        },
    },
    {
        "name": "FP16 + batch 16",
        "model": {
            "cache_mode": "FP16",
            "cache_size": 98304,
            "max_seq_len": 12288,
            "chunk_size": 2048,
            "max_batch_size": 16,
        },
    },
]


def write_config(model_overrides):
    config = {
        "network": {
            "host": "0.0.0.0",
            "port": PORT,
            "disable_auth": True,
            "api_servers": ["OAI"],
        },
        "model": {
            "model_dir": MODEL_DIR,
            "model_name": MODEL_NAME,
            "reasoning": True,
            "reasoning_start_token": "<think>",
            "reasoning_end_token": "</think>",
            **model_overrides,
        },
        "sampling": {
            "override_preset": "safe_defaults",
        },
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def start_server():
    proc = subprocess.Popen(
        ["bash", "-c", f"cd {TABBY_DIR} && source venv/bin/activate && python main.py"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(120):
        time.sleep(1)
        try:
            r = httpx.get(f"http://localhost:{PORT}/v1/models", timeout=2)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
    print("  WARNING: Server may not have started properly")
    return proc


def stop_server(proc):
    # kill the whole process group
    import os
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    # also kill any leftover python processes on our port
    subprocess.run(["bash", "-c", f"fuser -k {PORT}/tcp 2>/dev/null"], capture_output=True)
    time.sleep(5)


def get_vram():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip()


async def send_request(client, prompt, is_chat=True, system=None, messages=None):
    if messages:
        msgs = [{"role": "system", "content": system}] + messages if system else messages
    else:
        msgs = [{"role": "user", "content": prompt}]

    body = {
        "model": MODEL_NAME,
        "messages": msgs,
        "max_tokens": 100 if not messages else 2048,
        "temperature": 0,
    }
    start = time.perf_counter()
    resp = await client.post(URL, json=body)
    wall = time.perf_counter() - start
    data = resp.json()
    usage = data.get("usage", {})
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {
        "tokens": usage.get("completion_tokens", 0),
        "tps": usage.get("completion_tokens_per_sec", 0),
        "wall": wall,
        "content": content,
        "status": resp.status_code,
    }


async def bench_speed(label, parallel, prompts):
    async with httpx.AsyncClient(timeout=300) as client:
        wall_start = time.perf_counter()
        tasks = [send_request(client, p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_wall = time.perf_counter() - wall_start

    total_tok = 0
    avg_tps = []
    for r in results:
        if isinstance(r, Exception):
            return f"  {label}: ERROR {r}"
        if r["status"] != 200:
            return f"  {label}: HTTP {r['status']}"
        total_tok += r["tokens"]
        avg_tps.append(r["tps"])

    agg = total_tok / total_wall if total_wall > 0 else 0
    avg = sum(avg_tps) / len(avg_tps) if avg_tps else 0
    return f"  {label}: {avg:5.1f} t/s per req | {agg:5.1f} agg | {total_wall:.1f}s"


async def bench_quality(client):
    """Test coding + agent instruction following."""
    # Coding
    r = await send_request(client, CODING_PROMPT)
    has_code = "def quicksort" in r["content"] or "def quick_sort" in r["content"]
    coding_ok = "OK" if has_code else "FAIL"

    # Agent
    r = await send_request(client, "", system=SYSTEM_PROMPT, messages=AGENT_MESSAGES)
    content = r["content"].strip()
    try:
        data = json.loads(content)
        agent_ok = "OK" if all(k in data for k in ("action", "params", "reasoning")) else "PARTIAL"
    except json.JSONDecodeError:
        agent_ok = "FAIL"

    return coding_ok, agent_ok


async def run_config(config):
    name = config["name"]
    model_opts = config["model"]

    print(f"\n{'─'*60}")
    print(f"  Config: {name}")
    print(f"{'─'*60}")

    write_config(model_opts)
    proc = start_server()
    try:
        vram = get_vram()
        print(f"  VRAM: {vram}")

        # Speed tests
        r = await bench_speed("1×short", 1, PROMPTS_SHORT[:1])
        print(r)

        r = await bench_speed("8×short", 8, PROMPTS_SHORT)
        print(r)

        filler = (FILLER * 230)[:2000 * 6]
        long_prompts = [f"Summarize: {filler}"] * 8
        r = await bench_speed("8×long ", 8, long_prompts)
        print(r)

        # Quality tests
        async with httpx.AsyncClient(timeout=300) as client:
            coding, agent = await bench_quality(client)
        print(f"  Quality: coding={coding}, agent={agent}")

    finally:
        stop_server(proc)


async def main():
    print(f"{'='*60}")
    print(f"  TabbyAPI parameter sweep + quality")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'='*60}")

    for config in CONFIGS:
        await run_config(config)

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
