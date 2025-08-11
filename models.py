import os
import time
import json
import requests
from typing import Callable
from tenacity import retry, stop_after_attempt, wait_exponential

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

# Retry decorator
def retry_with_exponential_backoff(retries: int = 3, backoff_in_seconds: int = 2):
    def decorator(func: Callable):
        @retry(stop=stop_after_attempt(retries), wait=wait_exponential(multiplier=backoff_in_seconds))
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Output cleaner
def clean_output(output: str) -> str:
    try:
        if isinstance(output, dict):
            return json.dumps(output, indent=2)
        return output.strip()
    except Exception as e:
        print(f"⚠️ Failed to clean output: {e}")
        return str(output)

# Bullet model wrapper (Together.ai only)
# models.py (replace call_bullet_model)
@retry_with_exponential_backoff(retries=3, backoff_in_seconds=2)
def call_bullet_model(prompt: str, max_tokens: int = 1600) -> str:
    if not TOGETHER_API_KEY:
        raise EnvironmentError("TOGETHER_API_KEY not set.")

    try:
        resp = requests.post(
            "https://api.together.xyz/inference",
            headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
            json={
                "model": TOGETHER_MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,          # allow enough room
                "temperature": 0.4,                # a bit more deterministic
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                # Try to halt before it starts inventing headers or markdown
                "stop": ["\n\n**", "\n**", "###", "\n\n#"],
            },
            timeout=90,
        )

        if os.getenv("DEBUG_RESUME") == "1":
            print("🧪 Raw Together response (first 1k):", resp.text[:1000])

        if resp.status_code != 200:
            print(f"❌ Together API failed: {resp.status_code} – {resp.text}")
            return ""

        result = resp.json()
        output = ""
        if isinstance(result, dict):
            if "choices" in result:
                output = result["choices"][0].get("text", "")
            else:
                output = result.get("output", "")
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)

        cleaned = clean_output(output)
        if os.getenv("DEBUG_RESUME") == "1":
            with open("prompt_logs/together_bullet_output.txt", "w", encoding="utf-8") as f:
                f.write(cleaned)
        return cleaned

    except Exception as e:
        print(f"❌ Exception during Together call: {e}")
        return ""

