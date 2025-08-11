# extract_job_keywords.py

import re
import json
import os
import time
import requests

try:
    # Your existing helper (keep using it if available)
    from resume_tailoring.utils import call_groq
except Exception:
    call_groq = None  # we'll fall back to direct API calls

# --------------------- Debug logging --------------------- #
def _dbg_write(name: str, content: str):
    try:
        os.makedirs("prompt_logs", exist_ok=True)
        with open(os.path.join("prompt_logs", name), "w", encoding="utf-8") as f:
            f.write(content if content is not None else "")
    except Exception:
        pass

# --------------------- Token Sanitizer (drop CSS/JS junk) --------------------- #
SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 +#./&()-_")
BLOCK_TOKENS = {"@media","var(","calc(","rgba(","webkit","moz","svg","inline","fa-","{","}","</",">","script","style","px","em"}

def _is_reasonable_token(tok: str) -> bool:
    if not tok:
        return False
    s = tok.strip()
    if len(s) < 2 or len(s) > 40:
        return False
    low = s.lower()
    if any(t in low for t in BLOCK_TOKENS):
        return False
    if not any(ch.isalnum() for ch in s):
        return False
    if any(c not in SAFE_CHARS for c in s):
        return False
    return True

def _sanitize_list(values):
    seen, keep, drop = set(), [], []
    for v in values or []:
        v = (v or "").strip()
        if not v:
            continue
        lv = v.lower()
        if lv in seen:
            continue
        if _is_reasonable_token(v):
            keep.append(v); seen.add(lv)
        else:
            drop.append(v)
    return keep[:24], drop  # cap for stability

# --------------------- Static Fallback Dictionaries --------------------- #
STATIC_SKILLS = ["Python", "SQL", "Excel", "Tableau", "Power BI", "ETL", "Snowflake", "AWS", "Data Modeling"]
STATIC_VERBS  = ["Analyzed", "Developed", "Built", "Designed", "Implemented", "Led", "Optimized", "Collaborated"]

# --------------------- Direct Groq client (retry/backoff) --------------------- #
def _groq_complete_retry(prompt: str, attempts: int = 3, model_env: str = "GROQ_MODEL") -> str:
    """
    Use resume_tailoring.utils.call_groq if present; otherwise call Groq HTTP API directly.
    Retries on 5xx/429 with exponential backoff. Returns "" on failure.
    """
    # Prefer user's helper if available
    if callable(call_groq):
        for i in range(1, attempts + 1):
            try:
                out = call_groq(prompt)
                return out or ""
            except Exception as e:
                _dbg_write(f"groq_job_keywords_attempt{i}.txt", f"{type(e).__name__}: {e}")
                if i < attempts:
                    time.sleep(1.5 ** i)
        return ""

    # Fallback: direct HTTP
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        _dbg_write("groq_job_keywords_error.txt", "Missing GROQ_API_KEY")
        return ""
    url = "https://api.groq.com/openai/v1/chat/completions"
    model = os.getenv(model_env, "llama-3.1-8b-instant")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    for i in range(1, attempts + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            status = r.status_code
            if status == 429 or status >= 500:
                raise RuntimeError(f"Groq transient status {status}")
            r.raise_for_status()
            j = r.json()
            content = (j.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            if content.strip():
                return content
            raise RuntimeError("Empty Groq content")
        except Exception as e:
            _dbg_write(f"groq_job_keywords_attempt{i}.txt", f"{type(e).__name__}: {e}")
            if i < attempts:
                time.sleep(1.5 ** i)
    return ""

# --------------------- Parsing helpers --------------------- #
def _strip_to_json_block(text: str) -> str:
    """
    Extract a JSON object from raw model text:
    - remove code fences if present
    - if braces exist, slice from first '{' to last '}'
    """
    if not text:
        return ""
    raw = text.strip()
    # code fence path
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            return m.group(1).strip()
    # generic braces path
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end+1]
    return ""

def _titlecase_unique(items):
    return list({(s or "").strip().title() for s in (items or []) if (s or "").strip()})

# --------------------- Extraction Logic --------------------- #
def extract_job_info(job_text: str):
    job_text = (job_text or "").strip()

    # Guard: too-short JD — skip model, provide minimal structure
    if len(job_text) < 300:
        print("⚠️ Job description too short. Skipping model.")
        return {"job_title": "Unknown Title", "skills": [], "verbs": []}

    prompt = (
        "You are a job description parser.\n\n"
        "Given the following job description, extract:\n"
        "1. The most accurate job title (short)\n"
        "2. A list of 8–12 relevant technical skills, soft skills, and domain-specific skills (single words or short phrases)\n"
        "3. A list of 8–12 strong action verbs based on responsibilities\n\n"
        "Respond ONLY with a valid JSON object in the following format. Do NOT include any explanations, markdown, or text outside the JSON.\n\n"
        "{\n"
        "  \"job_title\": \"...\",\n"
        "  \"skills\": [\"...\", \"...\"],\n"
        "  \"verbs\": [\"...\", \"...\"]\n"
        "}\n\n"
        f"Job Description:\n{job_text}"
    )

    raw_response = _groq_complete_retry(prompt).strip()
    _dbg_write("groq_job_keywords_raw.txt", raw_response[:4000])

    json_block = _strip_to_json_block(raw_response)
    if not json_block:
        print("⚠️ No JSON block found in Groq response")
        # proceed to rule-based
        return rule_based_extraction(job_text)

    try:
        parsed = json.loads(json_block)
        skills = _titlecase_unique(parsed.get("skills", []))
        verbs  = _titlecase_unique(parsed.get("verbs", []))

        # sanitize and log drops
        skills_clean, skills_drop = _sanitize_list(skills)
        verbs_clean,  verbs_drop  = _sanitize_list(verbs)
        if skills_drop or verbs_drop:
            _dbg_write("groq_job_keywords_dropped.txt", "SKILLS_DROP:\n" + "\n".join(skills_drop) + "\n\nVERBS_DROP:\n" + "\n".join(verbs_drop))

        return {
            "job_title": (parsed.get("job_title") or "Unknown Title").strip(),
            "skills": skills_clean,
            "verbs": verbs_clean,
        }
    except Exception as e:
        print(f"⚠️ Failed to parse model response as JSON: {e}")
        _dbg_write("groq_job_keywords_parse_error.txt", f"{type(e).__name__}: {e}\n\n{json_block[:4000]}")
        return rule_based_extraction(job_text)

# --------------------- Rule-Based Fallback --------------------- #
def rule_based_extraction(text: str):
    """
    Lightweight regex fallback that extracts a plausible title, skills, and verbs.
    """
    # Title: look for explicit label first, else grab first 'Business Analyst|Data Analyst|...' occurrence
    m = re.search(r"(?i)(?:Job\s*Title|Role)\s*:\s*([^\n\r|]+)", text)
    if m:
        job_title = m.group(1).strip()
    else:
        m2 = re.search(r"(?i)\b(Data\s+Analyst|Business\s+Analyst|Product\s+Analyst|BI\s+Analyst|Analytics\s+Engineer|Data\s+Scientist)\b", text)
        job_title = m2.group(0).strip() if m2 else "Unknown Title"

    # Skills
    skills_pat = re.compile(
        r"\b(SQL|Python|R|Power\s*BI|Excel|Tableau|Looker|Qlik|ETL|ELT|Snowflake|Redshift|BigQuery|Azure|AWS|GCP|Databricks|Airflow|dbt|Data\s+Model(?:ing)?|A/B\s*Testing|Forecasting|Time\s*Series|Regression|Agile|JIRA|Confluence|KPI[s]?)\b",
        re.IGNORECASE
    )
    skills_found = [s.strip().title() for s in skills_pat.findall(text)]

    # Verbs
    verbs_pat = re.compile(
        r"\b(Analyze|Design|Develop|Build|Create|Implement|Optimize|Automate|Lead|Collaborate|Partner|Own|Drive|Deliver|Present|Communicate|Query|Validate|Document|Maintain|Monitor|Troubleshoot|Migrate|Scale|Improve)\w*",
        re.IGNORECASE
    )
    verbs_found = [v.strip().title() for v in verbs_pat.findall(text)]

    # Deduplicate, sanitize, cap
    skills_clean, _ = _sanitize_list(sorted(set(skills_found)) or STATIC_SKILLS)
    verbs_clean, _  = _sanitize_list(sorted(set(verbs_found))  or STATIC_VERBS)

    return {
        "job_title": job_title,
        "skills": skills_clean,
        "verbs": verbs_clean,
    }
