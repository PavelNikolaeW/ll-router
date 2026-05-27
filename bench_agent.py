"""Test instruction following with agent-like prompts across KV-cache types."""

import subprocess
import time
import httpx
import signal
import json

LLAMA_SERVER = "/home/pavel/llama.cpp/build/bin/llama-server"
MODEL = "/home/pavel/Models/Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf"
PORT = 8001
URL = f"http://localhost:{PORT}/v1/chat/completions"
MODEL_NAME = "Qwen3.5-27B-Q4_K_M.gguf"

SYSTEM_PROMPT = """You are an AI agent that executes tasks. You MUST follow these rules strictly:

1. Always respond in valid JSON format with this exact structure: {"action": "...", "params": {...}, "reasoning": "..."}
2. Available actions: "search", "read_file", "write_file", "execute", "done"
3. Never include markdown formatting or code blocks - raw JSON only
4. If the task is complete, use action "done"
5. Always include reasoning field explaining why you chose this action
6. Params must match the action: search needs "query", read_file needs "path", write_file needs "path" and "content", execute needs "command"
"""

# Multi-turn conversation simulating agent workflow
CONVERSATIONS = [
    {
        "name": "Simple task (short context)",
        "messages": [
            {"role": "user", "content": "Find the Python version installed on this system"},
        ],
    },
    {
        "name": "Multi-step task (medium context)",
        "messages": [
            {"role": "user", "content": "Create a Python script that calculates fibonacci numbers"},
            {"role": "assistant", "content": '{"action": "write_file", "params": {"path": "/tmp/fib.py", "content": "def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    return fibonacci(n-1) + fibonacci(n-2)\\n\\nif __name__ == \\"__main__\\":\\n    for i in range(10):\\n        print(f\\"fib({i}) = {fibonacci(i)}\\")"}, "reasoning": "Writing a fibonacci script with recursive implementation"}'},
            {"role": "user", "content": "Now execute it and verify it works"},
        ],
    },
    {
        "name": "Complex task with lots of context",
        "messages": [
            {"role": "user", "content": "I need you to help me set up a web scraping pipeline"},
            {"role": "assistant", "content": '{"action": "search", "params": {"query": "python web scraping libraries 2024"}, "reasoning": "First researching the best current tools for web scraping"}'},
            {"role": "user", "content": "Search returned: beautifulsoup4, scrapy, playwright, httpx. Let's use httpx + beautifulsoup4. Create the scraper."},
            {"role": "assistant", "content": '{"action": "write_file", "params": {"path": "/tmp/scraper.py", "content": "import httpx\\nfrom bs4 import BeautifulSoup\\n\\nasync def scrape(url: str) -> dict:\\n    async with httpx.AsyncClient() as client:\\n        resp = await client.get(url)\\n        soup = BeautifulSoup(resp.text, \\"html.parser\\")\\n        return {\\"title\\": soup.title.string, \\"links\\": [a[\\"href\\"] for a in soup.find_all(\\"a\\", href=True)]}"}, "reasoning": "Creating async scraper with httpx and beautifulsoup4"}'},
            {"role": "user", "content": "Good. Now add error handling, retry logic with exponential backoff, and logging. Also add a function to save results to a JSON file. Remember: respond in raw JSON only, no markdown."},
        ],
    },
]

CONFIGS = [
    {
        "name": "FP16",
        "args": ["--ctx-size", "32768", "--parallel", "1", "--cont-batching"],
    },
    {
        "name": "Q8_0",
        "args": ["--ctx-size", "32768", "--parallel", "1", "--cont-batching",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
    },
    {
        "name": "Q4_0",
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


def generate(messages):
    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "max_tokens": 2048,
        "temperature": 0,
        "seed": 42,
    }
    r = httpx.post(URL, json=body, timeout=120)
    data = r.json()
    choice = data["choices"][0]["message"]
    content = choice.get("content", "")
    reasoning = choice.get("reasoning_content", "")
    return content, reasoning


def check_json_valid(content):
    """Check if response is valid JSON with required fields."""
    content = content.strip()
    # Remove markdown code blocks if present
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    try:
        data = json.loads(content)
        has_action = "action" in data
        has_params = "params" in data
        has_reasoning = "reasoning" in data
        valid_action = data.get("action") in ("search", "read_file", "write_file", "execute", "done")
        return {
            "valid_json": True,
            "has_action": has_action,
            "has_params": has_params,
            "has_reasoning": has_reasoning,
            "valid_action": valid_action,
            "used_markdown": False,
            "score": sum([has_action, has_params, has_reasoning, valid_action]) / 4,
        }
    except json.JSONDecodeError:
        return {
            "valid_json": False,
            "used_markdown": "```" in content,
            "score": 0,
        }


def main():
    all_results = {}

    for config in CONFIGS:
        name = config["name"]
        all_results[name] = []

        print(f"\n{'='*60}")
        print(f"  KV-cache: {name}")
        print(f"{'='*60}")

        proc = start_server(config["args"])
        try:
            for conv in CONVERSATIONS:
                content, reasoning = generate(conv["messages"])
                check = check_json_valid(content)

                status = "OK" if check["valid_json"] and check["score"] == 1.0 else "PARTIAL" if check["valid_json"] else "FAIL"
                md = " [used markdown!]" if check.get("used_markdown") else ""

                print(f"\n  {conv['name']}: {status}{md} (score: {check['score']:.0%})")
                # show first 200 chars of content
                preview = content.replace("\n", "\\n")[:200]
                print(f"  Output: {preview}")

                all_results[name].append({
                    "conv": conv["name"],
                    "status": status,
                    "score": check["score"],
                    "content": content,
                })
        finally:
            stop_server(proc)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Test':<40} | {'FP16':>6} | {'Q8_0':>6} | {'Q4_0':>6}")
    print(f"  {'-'*40}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for i, conv in enumerate(CONVERSATIONS):
        scores = []
        for name in ["FP16", "Q8_0", "Q4_0"]:
            s = all_results[name][i]["status"]
            scores.append(s)
        print(f"  {conv['name']:<40} | {scores[0]:>6} | {scores[1]:>6} | {scores[2]:>6}")


if __name__ == "__main__":
    main()
