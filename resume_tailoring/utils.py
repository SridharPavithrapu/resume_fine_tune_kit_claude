import os
import json
import pandas as pd
from nltk.stem import PorterStemmer
import re
import requests
from pathlib import Path  # NEW

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from typing import List

def extract_score_from_pdf_text(text: str) -> int:
    """Extract ATS score from Jobscan PDF text."""
    match = re.search(r"ATS\s*Score\s*[:\-]?\s*(\d+)%?", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def extract_keywords_from_text(text: str) -> List[str]:
    """Extract missing keywords section from Jobscan PDF."""
    keywords = []
    pattern = r"Missing Keywords?\s*:(.*?)\n(?:\s{2,}|\n)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        raw_keywords = match.group(1)
        keywords = re.findall(r"[A-Za-z][A-Za-z0-9 &/()+\-]*", raw_keywords)
        keywords = [k.strip() for k in keywords if len(k.strip()) > 1]
    return keywords

def extract_missing_sections_from_text(text: str) -> List[str]:
    """Extract missing sections like EDUCATION, CERTIFICATIONS from Jobscan PDF text."""
    missing_sections = []
    pattern = r"Missing Sections?\s*[:\-]?\s*(.*?)\n"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        raw = match.group(1)
        missing_sections = [s.strip() for s in raw.split(',') if s.strip()]
    return missing_sections


# --------------------- Core LLM Call --------------------- #
def call_groq(prompt):
    headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}
    payload = {
        "model": os.getenv("GROQ_MODEL", "llama3-70b-8192"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Groq API call failed: {e}")
        return ""

# -------------------- Load JD File -------------------- #
def load_job_descriptions(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading job file: {e}")
        return pd.DataFrame()

# -------------------- Normalize Keywords -------------------- #
def normalize_keywords(keywords):
    ps = PorterStemmer()
    return list(set([ps.stem(k.lower().strip()) for k in keywords if k.strip()]))

# -------------------- Save Tailored Resume & JD -------------------- #
def save_tailored_attempts(job_title, jd_text, resumes_with_scores):
    slug = re.sub(r"[^\w]+", "_", job_title.strip().lower())[:40]
    folder = os.path.join(OUTPUT_DIR, slug)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "job_description.txt"), "w") as f:
        f.write(jd_text)
    for i, (score, resume) in enumerate(resumes_with_scores):
        with open(os.path.join(folder, f"attempt_{i+1}_score_{score}.txt"), "w") as f:
            f.write(resume)

# -------------------- Save Prompt + Output JSON -------------------- #
def save_json_log(prompt, output, job_title):
    slug = job_title.replace(" ", "_").lower()[:40]
    folder = os.path.join(OUTPUT_DIR, slug)
    os.makedirs(folder, exist_ok=True)
    log = {"prompt": prompt, "output": output}
    with open(os.path.join(folder, "log.json"), "w") as f:
        json.dump(log, f, indent=2)

def extract_job_title(jd_text, csv_title=None):
    """
    Extracts job title using CSV title (preferred) or job description text.
    """
    # 0. Prefer CSV title if it’s clean
    if csv_title:
        clean_title = csv_title.strip()
        if 5 < len(clean_title) < 120 and not clean_title.lower().startswith("requisition"):
            return clean_title

    # 1. Try labeled lines
    match = re.search(r"(?:Position Title|Job Title|Title)\s*[:\-–]\s*(.+)", jd_text, re.IGNORECASE)
    if match:
        return match.group(1).strip().split("\n")[0]

    # 2. Try bolded lines
    bolded_lines = re.findall(r"\*\*(.+?)\*\*", jd_text)
    for line in bolded_lines:
        if not line.endswith(":") and any(kw in line.lower() for kw in ["analyst", "consultant", "engineer", "manager", "scientist"]):
            return line.strip()

    # 3. Try top lines of JD
    for line in jd_text.strip().splitlines()[:10]:
        line_clean = line.strip()
        if line_clean.endswith(":"):
            continue
        if len(line_clean) < 100 and any(kw in line_clean.lower() for kw in ["analyst", "consultant", "engineer", "manager", "scientist"]):
            return line_clean

    return "Unknown Job Title".title()


# ==================== NEW: Unified Resume Loader (+ CERTS) ==================== #

# Safe imports from generate_with_models to avoid hard dependency if path differs
try:
    from generate_with_models import _load_docx_text, extract_resume_sections, insert_or_replace_section
except Exception:
    _load_docx_text = None
    extract_resume_sections = None
    insert_or_replace_section = None

_EN_DASH = " – "

# Your real certifications (will be injected if missing/empty)
USER_CERTIFICATIONS_BLOCK = (
    "- Google Data Analytics Professional Certification — Google\n"
    "- Excel for Business — Coursera\n"
    "- AI-Driven Analytics Simulation — Tata Consultancy Services"
)

def _normalize_headers_and_spacing(text: str) -> str:
    """Normalize dashes in likely header lines and fix blank-line spacing."""
    # Collapse >2 blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize dash spacing in lines that look like job headers (contain a year or 'Present')
    header_hint = re.compile(r"(19|20)\d{2}|Present", re.IGNORECASE)
    out = []
    for ln in text.splitlines():
        if header_hint.search(ln):
            # add spaces around hyphen/en-dash between non-space chars
            ln = re.sub(r"(\S)\s*[-–]\s*(\S)", r"\1" + _EN_DASH + r"\2", ln)
        out.append(ln.rstrip())
    text = "\n".join(out)

    # Ensure a blank line before EDUCATION (prevents section run-ons)
    text = re.sub(r"\n(?=EDUCATION\b)", "\n\n", text)
    return text

def load_resume_text(path_str: str) -> str:
    """
    Load base resume text from .txt or .docx.
    - Normalizes headers/dashes and spacing.
    - Ensures a CERTIFICATIONS section exists; injects user's certs if missing/empty.
    """
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Resume not found: {path_str}")

    # 1) Read raw text from TXT or DOCX
    if p.suffix.lower() == ".txt":
        text = p.read_text(encoding="utf-8")
    else:
        if _load_docx_text is None:
            raise RuntimeError("DOCX loader not available; either pass a .txt resume or expose _load_docx_text.")
        text = _load_docx_text(str(p))

    # 2) Normalize spacing + headers
    text = _normalize_headers_and_spacing(text)

    # 3) Ensure CERTIFICATIONS exists (inject user's certs if missing)
    if extract_resume_sections and insert_or_replace_section:
        try:
            secs = extract_resume_sections(text)
            cert_body = (secs.get("CERTIFICATIONS") or "").strip()
            if not cert_body:
                text = insert_or_replace_section(text, "CERTIFICATIONS", USER_CERTIFICATIONS_BLOCK)
        except Exception as e:
            # Non-fatal: if extraction fails, append a CERTIFICATIONS block at the end
            print(f"⚠️ CERTIFICATIONS check failed in loader; appending default. Details: {e}")
            if not text.endswith("\n"):
                text += "\n"
            text += "\nCERTIFICATIONS\n" + USER_CERTIFICATIONS_BLOCK + "\n"

    return text
