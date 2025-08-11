import os
import re
from datetime import datetime
from resume_tailoring.guard_clean_resume import (
    clean_full_resume,
    fix_broken_headers,
    patch_job_titles_with_original,
    remove_placeholder_bullets,
    remove_hallucinated_titles
)

# Final cleaning and patching logic
def patch_final_resume(text, base_resume):
    cleaned = clean_full_resume(text, base_resume, verbose=False)   # pass base_resume!
    # DO NOT remap headers by title; causes duplicates when titles repeat
    # cleaned = patch_job_titles_with_original(cleaned, base_resume)
    # DO NOT remove placeholder bullets globally; we already do it locally for WE
    # cleaned = remove_placeholder_bullets(cleaned)
    return cleaned.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


MAX_MODEL_TOKENS = 8192
SAVE_LOGS = True
LOG_DIR = "prompt_logs"
DEBUG = os.getenv("DEBUG_RESUME") == "1"
os.makedirs(LOG_DIR, exist_ok=True)

REQUIRED_SECTIONS = ["SUMMARY", "SKILLS", "WORK EXPERIENCE", "CERTIFICATIONS", "EDUCATION"]


def validate_sections(text):
    required = ["SUMMARY", "SKILLS", "WORK EXPERIENCE", "EDUCATION"]
    missing = [s for s in required if s not in text.upper()]
    return {"valid": len(missing) == 0, "missing_sections": missing}


def check_personal_info_fields(text):
    fields = {
        "name": bool(re.search(r"^[A-Z][a-z]+ [A-Z][a-z]+", text)),
        "email": bool(re.search(r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b", text)),
        "phone": bool(re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", text))
    }
    if DEBUG:
        print(f"👤 Personal info check: {fields}")
    return fields


# Claude-style enhancement: detect placeholder bullets
PLACEHOLDER_PATTERNS = [
    r"^-\s*(lorem ipsum|...|\[.*?\])",
    r"^-\s*insert|add|sample",
    r"^-\s*[A-Z ]{2,} only",
    r"^-\s*-+",
]


def has_placeholder_bullets(text):
    lines = text.splitlines()
    for line in lines:
        if any(re.search(pat, line, re.IGNORECASE) for pat in PLACEHOLDER_PATTERNS):
            if DEBUG:
                print(f"⚠️ Placeholder bullet detected: {line}")
            return True
    return False


# Claude-style fallback patching logic

def patch_resume_with_fallbacks(text, base_text, default_summary, default_experience):
    patched = text
    summary_ok = "SUMMARY" in text and not has_placeholder_bullets(text)
    experience_ok = "WORK EXPERIENCE" in text and "-" in text.split("WORK EXPERIENCE")[-1][:300]

    if not summary_ok:
        print("⚠️ Patching missing/placeholder SUMMARY.")
        patched = re.sub(r"(?s)SUMMARY\n.*?(?=\n[A-Z ]{3,}|$)", f"SUMMARY\n{default_summary}", patched)

    if not experience_ok:
        print("⚠️ Patching incomplete WORK EXPERIENCE.")
        patched = re.sub(r"(?s)WORK EXPERIENCE\n.*?(?=\n[A-Z ]{3,}|$)", f"WORK EXPERIENCE\n{default_experience}", patched)

    return patched


# Optional: quality score (simple heuristic)

def score_resume_quality(text):
    score = 0
    lines = text.splitlines()
    if any(sec in text for sec in REQUIRED_SECTIONS):
        score += 20
    if not has_placeholder_bullets(text):
        score += 40
    if len(lines) > 60:
        score += 40
    return min(score, 100)

def format_prompt(resume_text, jd_text, style_guide, keywords, missing_sections, job_title):
    return (
        f"==== RESUME ====\n{resume_text}\n\n"
        f"==== JOB DESCRIPTION ====\n{jd_text}\n\n"
        f"==== STYLE GUIDE ====\n{style_guide}\n\n"
        f"==== KEYWORDS ====\n{keywords}\n\n"
        f"==== MISSING SECTIONS ====\n{missing_sections}\n\n"
        f"==== JOB TITLE ====\n{job_title}\n"
    )


# Exported functions from this file likely hook into generate_with_models or CLI scripts.
# You can now safely use `patch_resume_with_fallbacks(...)` to enforce final sanity.
# Example usage:
# patched_resume = patch_resume_with_fallbacks(text, base_resume, DEFAULT_SUMMARY, DEFAULT_EXPERIENCE)
