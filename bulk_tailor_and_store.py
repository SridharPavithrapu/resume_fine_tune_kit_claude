import os
import sys
import json
import pandas as pd
from docx import Document
from generate_with_models import tailor_resume_with_models, save_to_docx, parse_sections, remove_hallucinated_titles
from resume_tailoring.guard_clean_resume import clean_full_resume, fix_broken_headers
from resume_tailoring.utils import extract_job_title, load_resume_text
from jobscan.jobscan_driver_debug import get_jobscan_score_and_feedback
from resume_tailoring.inference import format_prompt, validate_sections
from resume_tailoring.guard_clean_resume import patch_job_titles_with_original
from resume_tailoring.inference import patch_final_resume
import time
from playwright.sync_api import sync_playwright

DEBUG = True

os.environ["MallocStackLogging"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.stderr = open(os.devnull, 'w')

jd_file_path = "data/jobspy_jobs.csv"
jd_df = pd.read_csv(jd_file_path) if os.path.exists(jd_file_path) else pd.DataFrame()

def safe_text(text):
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")

def load_docx_text(path):
    doc = Document(path)
    return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())

base_resume = load_resume_text("data/base_resume_test.txt")

with open("prompts/style_resume.txt", "r") as f:
    style_guide = f.read()

os.makedirs("outputs/final_tailored_resumes", exist_ok=True)
os.makedirs("prompt_logs", exist_ok=True)
summary_data = []

print(f"✅ Loaded {len(jd_df)} job listings from {jd_file_path}")

for i, row in jd_df.iterrows():
    print(f"\n📄 [{i+1}/{len(jd_df)}] Processing job: {row.get('title', '')}")
    try:
        title = row["title"]
        jd_text = row["description"]
        job_title = extract_job_title(jd_text, csv_title=title)
        safe_title = title.replace(" ", "_").replace("/", "_")[:50]

        print(f"\n📄 Processing: {title}")
        print(f"📋 Extracted job title: {job_title}")

        best_score = 0
        best_resume = ""
        best_keywords, best_sections = [], []

        for attempt in range(3):
            print(f"\n⏳ Attempt {attempt + 1}")
            try:
                tailored = tailor_resume_with_models(
                    job_title,
                    jd_text,
                    base_resume_path="data/base_resume_test.txt",
                    ats_keywords=best_keywords,
                    ats_sections=best_sections
                )
            except Exception as e:
                print(f"❌ Tailoring failed: {e}")
                continue

            try:
                tailored.encode("utf-8")
            except UnicodeEncodeError as e:
                print("❌ TAILORED TEXT CONTAINS INVALID CHARACTERS — RAW DUMP:")
                print(repr(tailored[:100]))
                raise e

            print("✅ Debug: Tailored resume preview:")
            print(tailored[:100].encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))
            tailored = tailored.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            print(f"✅ Tailor complete. Resume length: {len(tailored)} chars")
            print("🧪 Tailored resume preview (first 500 chars):")
            print(tailored[:500].encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))



            cleaned = patch_final_resume(tailored, base_resume)
            try:
                print("🧾 Final cleaned preview:")
                print(cleaned[:500].encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))
            except Exception as e:
                print(f"❌ Print preview error: {e}")
            cleaned = cleaned.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

            if DEBUG:
                print(f"🧪 Cleaned resume preview (first 500 chars):\n{cleaned[:500]}")

            if "SUMMARY" not in cleaned:
                print("🔧 Injecting fallback SUMMARY...")
                fallback_summary = "SUMMARY\n- Business Analyst with 5+ years of experience in SQL, ETL, Agile, and BI tools, delivering insights and driving project success.\n\n"
                cleaned = fallback_summary + cleaned

            if len(cleaned) > 20000:
                print(f"⚠️ Resume too long ({len(cleaned)} chars), trimming to 20000 chars.")
                cleaned = cleaned[:20000]
            # === Final WE guard (belt & suspenders) ===
            # Use the WE from the tailored text (already guarded in generate_with_models.py)
            # and ensure the cleaned resume still matches it exactly.
            import re

            def _extract_section(text: str, section_name: str) -> str:
                m = re.search(rf"(?ms)^{re.escape(section_name)}\s*\n(.*?)(\n(?=[A-Z][A-Z &/]+\s*$)|\Z)", text)
                return m.group(1) if m else ""

            def _norm_block(s: str) -> str:
                s = (s or "").strip()
                s = re.sub(r"\r\n", "\n", s)
                s = re.sub(r"\n\s*\n+", "\n\n", s)               # collapse blank lines
                s = re.sub(r"^[•\-\u2022]\s*", "- ", s, flags=re.M)  # normalize bullet marks
                s = re.sub(r"\s+", " ", s)
                return s.strip().lower()

            we_tailored = _extract_section(tailored, "WORK EXPERIENCE")
            we_cleaned  = _extract_section(cleaned,  "WORK EXPERIENCE")
            if _norm_block(we_tailored) != _norm_block(we_cleaned):
                if DEBUG: print("♻️ Final guard: WE changed post-clean — restoring rebuilt block.")
                cleaned = _hard_replace_section(cleaned, "WORK EXPERIENCE",
                                                we_tailored if we_tailored.endswith("\n") else we_tailored + "\n")

            # ==========================================


            section_check = validate_sections(cleaned)
            if not section_check["valid"]:
                print(f"❗ Missing Sections: {section_check['missing_sections']}")
                if "CERTIFICATIONS" in section_check['missing_sections']:
                    cleaned += "\nCERTIFICATIONS\n- [Insert Certification Name]"
                if "EDUCATION" in section_check['missing_sections']:
                    cleaned += "\nEDUCATION\n- [Insert Degree, University Name]"

            parsed = parse_sections(cleaned)
            print(f"📦 Sections found in tailored resume: {list(parsed.keys())}")

            if "SUMMARY" in parsed and parsed["SUMMARY"].lower().startswith("here is"):
                parsed["SUMMARY"] = parsed["SUMMARY"].split(":", 1)[-1].strip()
                cleaned = cleaned.replace(parsed["SUMMARY"], parsed["SUMMARY"].strip())

            print("📸 Attempting OCR scan via Jobscan...")

            try:
                score, best_keywords, best_sections = get_jobscan_score_and_feedback(cleaned, jd_text)
            except Exception as e:
                print(f"❌ Jobscan scan failed: {e}")
                score, best_keywords, best_sections = 0, [], []

            print(f"📊 OCR Score Extracted: {score}")
            print(f"🔍 Missing Keywords: {best_keywords}")
            print(f"🔧 Missing Sections: {best_sections}")

            try:
                score_str = str(score).strip().replace('%', '')
                score_val = int(score_str)
            except Exception as e:
                print(f"⚠️ Could not parse score: {e}")
                score_val = 0

            print(f"▶️ ATS score after attempt {attempt + 1}: {score_val}")
            if score_val > best_score:
                best_score = score_val
                best_resume = cleaned

            try:
                with open(f"prompt_logs/{safe_title}_attempt{attempt + 1}.txt", "w", encoding="utf-8", errors="ignore") as logf:
                    logf.write(safe_text(cleaned))

                with open(f"prompt_logs/{safe_title}_attempt{attempt + 1}.log", "w", encoding="utf-8", errors="ignore") as meta:
                    meta.write(f"Job Title: {job_title}\n")
                    meta.write(f"Attempt: {attempt + 1}\n")
                    meta.write(f"ATS Score: {score}\n")
                    meta.write(f"Missing Keywords: {best_keywords}\n")
                    meta.write(f"Missing Sections: {best_sections}\n\n")
            except Exception as e:
                print(f"⚠️ Failed to write log files: {e}")

            if score_val >= 80:
                print("🎯 Target met. Stopping early.")
                break
            elif score_val < 40 and attempt < 2:
                print("🔁 Score too low — likely OCR error. Retrying...")
                continue
            elif not best_keywords and attempt < 2:
                print("🔁 Missing keyword list empty — retrying to regenerate summary + bullets.")
                continue
            elif attempt < 2:
                print(f"➡️ Retrying for better ATS score... ({attempt + 2}/3)")

            time.sleep(2)

        if not best_resume.strip():
            print("❌ No valid resume generated. Skipping save.")
            continue

        docx_path = f"outputs/final_tailored_resumes/{safe_title}_ATS{best_score}.docx"
        save_to_docx(best_resume, docx_path)
        print(f"📁 Resume saved → {docx_path}")

        try:
            with open(f"outputs/final_tailored_resumes/{safe_title}_ATS{best_score}.txt", "w", encoding="utf-8", errors="ignore") as outf:
                outf.write(best_resume)

            final_prompt = format_prompt(base_resume, jd_text, style_guide, best_keywords, best_sections, job_title)
            with open(f"outputs/final_tailored_resumes/{safe_title}_prompt.txt", "w", encoding="utf-8", errors="ignore") as pf:
                pf.write(final_prompt)
        except Exception as e:
            print(f"⚠️ Failed to write final output files: {e}")

        summary_data.append({
            "title": title,
            "score": best_score,
            "file": docx_path
        })

        print(f"✅ Best resume saved: {docx_path}")
        if best_score < 80:
            print(f"⚠️ Did not reach 80% ATS for: {title}")

        time.sleep(5)

    except Exception as e:
        print(f"❌ Failed processing job #{i+1} due to error:\n{e}")
        continue

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("outputs/ats_score_summary.csv", index=False)
print("\n📊 Summary written to: outputs/ats_score_summary.csv")
