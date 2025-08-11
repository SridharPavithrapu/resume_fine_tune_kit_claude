import os
import re
import time
import asyncio
import threading
from typing import List, Tuple
from pathlib import Path
import json

from dotenv import load_dotenv
from PIL import Image, ImageOps
import pytesseract
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as ATimeoutError, expect

load_dotenv()
# Treat any of these as a 'missing' mark in the Resume column
CROSS_CHARS = "xX✗✘✖×❌"  # plain x, capital X, ballot crosses, heavy cross, multiplication sign, emoji cross
CROSS_RE = re.compile(rf"^\s*[{re.escape(CROSS_CHARS)}]\s*$")

EMAIL = os.getenv("JOBSCAN_EMAIL", "").strip()
PASSWORD = os.getenv("JOBSCAN_PASSWORD", "").strip()
HEADLESS = False
OCR_DEBUG = os.getenv("OCR_DEBUG", "0") == "1"

PDF_PATH = "jobscan_full_rendered.pdf"
SCORE_WIDGET_SEL = "#score"

async def _ensure_skills_comparison_tab(page):
    async def click_if_visible(label):
        loc = page.get_by_role("tab", name=re.compile(label, re.I))
        try:
            if await loc.count():
                await loc.first.click()
        except Exception:
            pass
    await click_if_visible("^Skills\\s*Comparison$")
    # do it for both Hard and Soft; page-level tabs are reused
    await page.wait_for_timeout(200)

# -------------------- keyword sanitizer --------------------
SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 +#./&()-_")
BLOCK_TOKENS = {"@media","var(","calc(","rgba(","webkit","moz","svg","inline","fa-","{","}","</",">","script","style"}

def _norm_skill(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _dedupe_ci(seq):
    seen = set()
    out = []
    for s in seq or []:
        k = _norm_skill(s).lower()
        if k and k not in seen:
            out.append(_norm_skill(s))
            seen.add(k)
    return out

def is_reasonable_keyword(kw: str) -> bool:
    if not kw:
        return False
    s = kw.strip()
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

def sanitize_keywords(keywords):
    keep, drop, seen = [], [], set()
    for k in (keywords or []):
        k = (k or "").strip()
        if not k:
            continue
        lk = k.lower()
        if lk in seen:
            continue
        if is_reasonable_keyword(k):
            keep.append(k); seen.add(lk)
        else:
            drop.append(k)
    return keep[:80], drop  # cap for stability


# -------------------- helpers --------------------

def _require_credentials():
    if not EMAIL or not PASSWORD:
        raise RuntimeError("Missing JOBSCAN_EMAIL or JOBSCAN_PASSWORD env vars")

async def _dismiss_overlays(page):
    selectors = [
        "button:has-text('Accept')","button:has-text('Dismiss')","button:has-text('Got it')",
        ".shepherd-modal-overlay-container","button:has-text('Close')","[aria-label='Close']",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if await loc.count() > 0 and await loc.first.is_visible():
                try:
                    await loc.first.click(timeout=2000)
                    await page.wait_for_timeout(250)
                except Exception:
                    pass
        except Exception:
            pass
    try:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(150)
    except Exception:
        pass

async def _wait_for_new_scan(page):
    candidates = [
        "span.title:has-text('New Scan')","button:has-text('New Scan')","text=New Scan",
        "text=Start a new scan","[data-test='new-scan-button']",
    ]
    for _ in range(3):
        for sel in candidates:
            try:
                await page.wait_for_selector(sel, timeout=4000)
                return sel
            except Exception:
                continue
        await page.reload()
    raise ATimeoutError("Could not find 'New Scan' action after retries")

async def _click_unique(page, selector: str, retries: int = 2):
    for attempt in range(retries + 1):
        loc = page.locator(selector)
        if await loc.count() == 0:
            await page.wait_for_timeout(300)
            continue
        target = loc.first
        try:
            await target.wait_for(state="visible", timeout=5000)
            # wait until enabled
            for _ in range(12):
                try:
                    if await target.is_enabled():
                        break
                except Exception:
                    pass
                await page.wait_for_timeout(200)
            await target.click(timeout=5000)
            return True
        except Exception as e:
            print(f"⚠️ Click attempt {attempt+1} failed for {selector}: {e}")
            await _dismiss_overlays(page)
            await page.wait_for_timeout(250)
    return False

async def _stabilize_widget(page, selector: str, shots: int = 12, interval_ms: int = 800) -> None:
    prev = None
    same_count = 0
    for i in range(shots):
        try:
            img = await page.locator(selector).screenshot()
        except Exception:
            await page.wait_for_timeout(interval_ms)
            continue
        if prev is not None and img == prev:
            same_count += 1
            if same_count >= 2:
                print(f"✅ Score gauge stabilized after {i+1} snapshots.")
                return
        else:
            same_count = 0
        prev = img
        await page.wait_for_timeout(interval_ms)

async def _render_pdf_resilient_async(page, pdf_path: str, headless: bool):
    """
    Print from the authenticated page first. If it fails, render in a new headless
    browser that imports the current context's storage_state (keeps auth).
    """
    # Try direct print first (works in most Chromium builds)
    try:
        await page.evaluate("document.body.style.background='white'")
        await page.emulate_media(media="print")
        await page.pdf(path=pdf_path, format="A4", print_background=True)
        return
    except Exception:
        pass

    # Fallback: headless print using current storage_state (AUTHENTICATED)
    try:
        state = await page.context.storage_state()
    except Exception:
        state = None

    from playwright.async_api import async_playwright
    async with async_playwright() as p2:
        b2 = await p2.chromium.launch(headless=True)
        try:
            kwargs = {"storage_state": state} if state else {}
            # ensure geolocation is granted here too, so no popups in the fallback renderer
            kwargs.setdefault("permissions", ["geolocation"])
            kwargs.setdefault("geolocation", {"latitude": 0, "longitude": 0})
            c2 = await b2.new_context(**kwargs)
            try:
                await c2.grant_permissions(["geolocation"], origin="https://app.jobscan.co")
            except Exception:
                pass
            p = await c2.new_page()
            await p.goto(page.url, wait_until="domcontentloaded", timeout=30000)
            try:
                await p.add_style_tag(content="*[role='dialog'], .tooltip, .shepherd-element { display:none !important; visibility:hidden !important; }")
                await p.evaluate("document.body.style.background='white'")
                await p.emulate_media(media="print")
            except Exception:
                pass
            await p.pdf(path=pdf_path, format="A4", print_background=True)
        finally:
            try: await p.close()
            except Exception: pass
            try: await c2.close()
            except Exception: pass
            try: await b2.close()
            except Exception: pass


def _extract_skills_from_pdf(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    text = "\n".join(pg.get_text() for pg in doc)

    def section_skills(section_name: str):
        m = re.search(rf"{section_name}.*?(?=\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
        if not m:
            return []
        block = re.sub(
            r"(Skills Comparison|Highlighted Skills|Add Skill|Copy All|Don't see skills.*?|Page \d+ of \d+|IMPORTANT|Jobscan Report.*?)",
            "", m.group(0), flags=re.DOTALL
        )
        lines = [ln.strip() for ln in block.splitlines()]
        exclusions = {
            "Skill","Resume","Job Description","Update","Add Skill","Copy All",
            "Skills Comparison","Highlighted Skills","Education Match","IMPORTANT",
            "Recruiter tips","from the job description?","HIGH SCORE IMPACT","MEDIUM SCORE IMPACT",
            "Update required education level","your education is noted.","jobs.",
            "frequently in the job description.","Hard skills","Soft skills","Jobscan Report",
            "Resume Tone","Job Level Match","Job Title Match","Date Formatting","Measurable Results",
            "Update scan information","internship or a personal project.","found by our algorithms.",
            "View Measurable Results","and buzzwords were found. Good job!",
            "Improve your job match by including more","Customize this section using your",
        }
        skills = [
            ln for ln in lines
            if re.search(r"[A-Za-z]", ln)
            and len(ln.split()) <= 6
            and len(ln) <= 50
            and ln not in exclusions
            and not ln.endswith(":")
        ]
        return sorted(set(skills))

    return {
        "hard_skills": section_skills("Hard skills"),
        "soft_skills": section_skills("Soft skills"),
    }

def _ocr_percent_from_image(path: str) -> int:
    try:
        img = Image.open(path)
    except Exception:
        return -1
    thresholds = [150, 180, 200]
    scales = [2, 3, 4, 6]
    psms = [6, 7, 8, 11, 13]
    whitelists = [True, False]
    for t in thresholds:
        for s in scales:
            big = img.resize((img.width * s, img.height * s))
            gray = ImageOps.grayscale(big)
            bw = gray.point(lambda x: 0 if x < t else 255, "1")
            for psm in psms:
                for wl in whitelists:
                    config = f"--psm {psm}"
                    if wl:
                        config += " -c tessedit_char_whitelist=0123456789%"
                    text = pytesseract.image_to_string(bw, config=config).strip()
                    m = re.search(r"(\d{1,3})\s*%", text)
                    if m:
                        try:
                            val = int(m.group(1))
                            if 0 <= val <= 100:
                                return val
                        except ValueError:
                            pass
    raw = pytesseract.image_to_string(Image.open(path)).strip()
    m = re.search(r"(\d{1,3})\s*%", raw)
    if m:
        try:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val
        except ValueError:
            pass
    return -1

# -------------------- ENHANCED EXTRACTION FUNCTIONS --------------------

async def _extract_missing_from_dom(page):
    """Enhanced DOM extraction with better selectors and X mark detection."""
    
    js_code = """
    (() => {
        const results = { hard: [], soft: [] };
        
        // Find skills sections by looking for headings
        const sections = document.querySelectorAll('section, div');
        let currentSection = null;
        
        for (const section of sections) {
            const headings = section.querySelectorAll('h1, h2, h3, h4');
            for (const heading of headings) {
                const text = heading.textContent.toLowerCase();
                if (text.includes('hard skills')) {
                    currentSection = 'hard';
                    break;
                } else if (text.includes('soft skills')) {
                    currentSection = 'soft';
                    break;
                }
            }
            
            if (currentSection) {
                // Look for table rows or skill entries
                const rows = section.querySelectorAll('tr, [role="row"], .skill-row');
                
                for (const row of rows) {
                    const cells = row.querySelectorAll('td, th, [role="cell"], .cell');
                    const allText = row.textContent.toLowerCase();
                    
                    // Skip header rows
                    if (allText.includes('skill') && allText.includes('resume') && allText.includes('job description')) {
                        continue;
                    }
                    
                    if (cells.length >= 3) {
                        const skillText = cells[0].textContent.trim();
                        const resumeText = cells[1].textContent.trim();
                        const jdText = cells[2].textContent.trim();
                        
                        // Check for X marks or zero values
                        const hasResumeValue = resumeText && !resumeText.match(/^[xX✗✘✖×❌\\s]*$/) && resumeText !== '0';
                        const hasJdValue = jdText && jdText.match(/\\d+/) && parseInt(jdText.match(/\\d+/)[0]) > 0;
                        
                        if (hasJdValue && !hasResumeValue && skillText.length > 0) {
                            results[currentSection].push({
                                skill: skillText,
                                resume: resumeText,
                                jd: jdText,
                                resumeAria: cells[1].getAttribute('aria-label') || ''
                            });
                        }
                    }
                }
                
                // Also try a different approach - look for spans/divs in a grid pattern
                const skillElements = section.querySelectorAll('[data-test*="skill"], .skill-name, .skill-item');
                for (const skillEl of skillElements) {
                    const skillText = skillEl.textContent.trim();
                    const parent = skillEl.closest('tr, [role="row"], .row');
                    if (parent) {
                        const resumeEl = parent.querySelector('[data-test*="resume"], .resume-count');
                        const jdEl = parent.querySelector('[data-test*="jd"], .jd-count, .job-description-count');
                        
                        if (resumeEl && jdEl) {
                            const resumeValue = resumeEl.textContent.trim();
                            const jdValue = jdEl.textContent.trim();
                            
                            const hasResumeValue = resumeValue && !resumeValue.match(/^[xX✗✘✖×❌\\s]*$/) && resumeValue !== '0';
                            const hasJdValue = jdValue && jdValue.match(/\\d+/) && parseInt(jdValue.match(/\\d+/)[0]) > 0;
                            
                            if (hasJdValue && !hasResumeValue) {
                                results[currentSection].push({
                                    skill: skillText,
                                    resume: resumeValue,
                                    jd: jdValue,
                                    resumeAria: resumeEl.getAttribute('aria-label') || ''
                                });
                            }
                        }
                    }
                }
            }
        }
        
        return results;
    })()
    """
    
    try:
        data = await page.evaluate(js_code)
    except Exception:
        data = {"hard": [], "soft": []}
    
    missing = []
    for section in ["hard", "soft"]:
        for row in data.get(section, []):
            skill = row.get("skill", "").strip()
            if skill and len(skill) > 1:
                missing.append(skill)
    
    # Write debug info
    try:
        Path("jobscan_debug_dom_rows.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
    
    return missing, []

def _parse_skills_section(section_text: str):
    """Parse a skills section to find present/missing skills."""
    present, missing = set(), set()
    
    lines = [line.strip() for line in section_text.split('\n') if line.strip()]
    
    # Remove header/footer noise
    skip_patterns = [
        'skills comparison', 'highlighted skills', 'copy all', 'show more',
        'add skill', 'skill', 'resume', 'job description', 'high score impact',
        'medium score impact', 'hard skills', 'soft skills', "don't see skills"
    ]
    
    cleaned_lines = []
    for line in lines:
        line_lower = line.lower()
        if not any(pattern in line_lower for pattern in skip_patterns):
            cleaned_lines.append(line)
    
    # Parse the skills table format
    i = 0
    while i < len(cleaned_lines):
        line = cleaned_lines[i].strip()
        
        # Skip pure numbers or empty lines
        if not line or line.isdigit():
            i += 1
            continue
            
        # This should be a skill name
        skill = line
        
        # Look ahead for the numbers (resume count, jd count)
        resume_count = None
        jd_count = None
        
        # Check next few lines for numbers
        j = i + 1
        numbers_found = []
        while j < len(cleaned_lines) and j < i + 4:  # Look ahead max 3 lines
            next_line = cleaned_lines[j].strip()
            if next_line.isdigit():
                numbers_found.append(int(next_line))
                j += 1
            else:
                break
        
        # Interpret the numbers based on the pattern we see
        if len(numbers_found) >= 2:
            # Assume first number is resume count, second is JD count
            resume_count = numbers_found[0]
            jd_count = numbers_found[1]
        elif len(numbers_found) == 1:
            # Only JD count found (resume is likely 0/missing)
            resume_count = 0
            jd_count = numbers_found[0]
        
        # Classify as missing or present
        if jd_count and jd_count > 0:
            if resume_count is None or resume_count == 0:
                missing.add(skill)
            else:
                present.add(skill)
        
        # Move to next potential skill (skip the numbers we consumed)
        i = j if numbers_found else i + 1
    
    return present, missing

def _extract_skills_from_text(page_text: str):
    """
    Enhanced extraction that handles the actual Jobscan format better.
    """
    import re
    
    text = page_text or ""
    present, missing = set(), set()
    
    # Find Hard skills section
    hard_match = re.search(r'Hard skills.*?(?=Soft skills|Recruiter tips|$)', text, re.DOTALL | re.IGNORECASE)
    if hard_match:
        hard_section = hard_match.group(0)
        present_h, missing_h = _parse_skills_section(hard_section)
        present.update(present_h)
        missing.update(missing_h)
    
    # Find Soft skills section  
    soft_match = re.search(r'Soft skills.*?(?=Recruiter tips|Formatting|$)', text, re.DOTALL | re.IGNORECASE)
    if soft_match:
        soft_section = soft_match.group(0)
        present_s, missing_s = _parse_skills_section(soft_section)
        present.update(present_s)
        missing.update(missing_s)
    
    return present, missing

def _extract_skills_from_jobscan_format(text: str):
    """
    Specific parser for the exact format seen in Jobscan PDFs.
    This is a fallback that uses pattern matching for known skill formats.
    """
    present, missing = set(), set()
    
    # Look for the skills table patterns in the text
    skills_patterns = [
        # Pattern: skill_name followed by numbers
        r'([a-zA-Z][a-zA-Z\s&./()-]+?)\s+(\d+)\s+(\d+)',
        # Pattern: skill_name on one line, numbers on next lines
        r'([a-zA-Z][a-zA-Z\s&./()-]{2,40}?)\s*\n\s*(\d+)\s*\n\s*(\d+)',
    ]
    
    for pattern in skills_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            skill_name = match.group(1).strip()
            resume_count = int(match.group(2))
            jd_count = int(match.group(3))
            
            # Skip obvious non-skills
            if skill_name.lower() in ['skill', 'resume', 'job description', 'copy all']:
                continue
                
            if len(skill_name) > 1 and jd_count > 0:
                if resume_count == 0:
                    missing.add(skill_name)
                else:
                    present.add(skill_name)
    
    # Also try to extract from the specific format visible in your PDF
    # Look for skills followed by single numbers (likely JD counts)
    skill_lines = []
    lines = text.split('\n')
    
    in_skills_section = False
    for i, line in enumerate(lines):
        line = line.strip()
        if 'Skills Comparison' in line or 'Hard skills' in line or 'Soft skills' in line:
            in_skills_section = True
            continue
        elif 'Recruiter tips' in line or 'Formatting' in line:
            in_skills_section = False
            continue
            
        if in_skills_section and line:
            # Check if this looks like a skill name
            if (re.search(r'[a-zA-Z]', line) and 
                len(line) > 1 and len(line) < 50 and
                not line.isdigit() and
                line.lower() not in ['copy all', 'show more', 'add skill', 'skill', 'resume', 'job description']):
                
                # Look ahead for numbers
                numbers_ahead = []
                for j in range(i+1, min(i+4, len(lines))):
                    next_line = lines[j].strip()
                    if next_line.isdigit():
                        numbers_ahead.append(int(next_line))
                    elif next_line and not next_line.isdigit():
                        break
                
                # If we found numbers, this might be a skill
                if numbers_ahead:
                    if len(numbers_ahead) >= 2:
                        resume_count, jd_count = numbers_ahead[0], numbers_ahead[1]
                    else:
                        resume_count, jd_count = 0, numbers_ahead[0]
                    
                    if jd_count > 0:
                        if resume_count == 0:
                            missing.add(line)
                        else:
                            present.add(line)
    
    return present, missing

def _extract_skills_from_pdf_layout(pdf_path: str):
    """
    Enhanced PDF layout parsing that better handles the Jobscan table structure.
    """
    import fitz, re, json
    from pathlib import Path

    present, missing = set(), set()
    debug = {"hard": [], "soft": []}

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return present, missing

    for page in doc:
        # Get all text blocks with positions
        blocks = page.get_text("dict")["blocks"]
        
        # Extract all text spans with coordinates
        all_spans = []
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        all_spans.append({
                            "text": text,
                            "x": (bbox[0] + bbox[2]) / 2,
                            "y": (bbox[1] + bbox[3]) / 2,
                            "bbox": bbox
                        })
        
        # Sort by Y position to process line by line
        all_spans.sort(key=lambda s: s["y"])
        
        # Group spans by Y coordinate (same line)
        lines = []
        current_line = []
        current_y = None
        
        for span in all_spans:
            if current_y is None or abs(span["y"] - current_y) > 3:  # New line
                if current_line:
                    lines.append(sorted(current_line, key=lambda s: s["x"]))
                current_line = [span]
                current_y = span["y"]
            else:
                current_line.append(span)
        
        if current_line:
            lines.append(sorted(current_line, key=lambda s: s["x"]))
        
        # Find skills tables
        in_hard_skills = False
        in_soft_skills = False
        
        for line_spans in lines:
            line_text = " ".join(span["text"] for span in line_spans).lower()
            
            # Detect section boundaries
            if "hard skills" in line_text:
                in_hard_skills = True
                in_soft_skills = False
                continue
            elif "soft skills" in line_text:
                in_soft_skills = True
                in_hard_skills = False
                continue
            elif any(keyword in line_text for keyword in ["recruiter tips", "formatting", "job level"]):
                in_hard_skills = False
                in_soft_skills = False
                continue
            
            # Process skills lines
            if (in_hard_skills or in_soft_skills) and len(line_spans) >= 2:
                # Look for skill name + numbers pattern
                skill_candidates = []
                numbers = []
                
                for span in line_spans:
                    text = span["text"].strip()
                    if text.isdigit():
                        numbers.append(int(text))
                    elif re.search(r'[a-zA-Z]', text) and not text.lower() in ['copy', 'all', 'skill', 'resume', 'job', 'description']:
                        skill_candidates.append(text)
                
                # Process each potential skill
                for skill in skill_candidates:
                    if len(numbers) >= 2:
                        # Assume format: skill, resume_count, jd_count
                        resume_count = numbers[0] if len(numbers) > 0 else 0
                        jd_count = numbers[1] if len(numbers) > 1 else 0
                        
                        section = "hard" if in_hard_skills else "soft"
                        debug[section].append({
                            "skill": skill,
                            "resume": resume_count,
                            "jd": jd_count
                        })
                        
                        if jd_count > 0 and resume_count == 0:
                            missing.add(skill)
                        elif resume_count > 0:
                            present.add(skill)

    # Write debug info
    try:
        Path("jobscan_pdf_rows.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
    except Exception:
        pass

    return present, missing

def extract_score_from_pdf_text(txt: str) -> str:
    """
    Robustly extract the Match Rate from Jobscan PDF/text.
    Anchor to 'Match Rate' so we don't pick '90% of companies'.
    Returns digits '0'..'100' or '' if not found.
    """
    import re
    if not txt:
        return ""
    t = re.sub(r"[\s\u00A0]+", " ", txt)
    m = re.search(r"match\s*rate\b(.{0,120})", t, flags=re.I)
    window = t[m.start():m.end()] if m else ""
    m2 = re.search(r"(\d{1,3})\s*%", window)
    if m2:
        return m2.group(1)
    m3 = re.search(r"match\s*rate\b\s*(\d{1,3})\b", t, flags=re.I)
    if m3:
        return m3.group(1)
    if m:
        tail = t[m.end():m.end()+120]
        m4 = re.search(r"(\d{1,3})\s*%?", tail)
        if m4:
            return m4.group(1)
    return ""

def extract_missing_sections_from_text(text: str):
    # Conservative: only mark sections missing if Jobscan explicitly says so
    import re
    out = []
    tl = (text or '').lower()
    cues = [
        (r"couldn['']t find an?\s+\"?summary\"?", "SUMMARY"),
        (r"couldn['']t find an?\s+\"?skills\"?", "SKILLS"),
        (r"couldn['']t find an?\s+\"?work experience\"?", "WORK EXPERIENCE"),
        (r"couldn['']t find an?\s+\"?education\"?", "EDUCATION"),
        (r"couldn['']t find an?\s+\"?certifications?\"?", "CERTIFICATIONS"),
    ]
    for rx, sec in cues:
        if re.search(rx, tl, flags=re.I):
            out.append(sec)
    return out


def _parse_jobscan_rendered_pdf(pdf_path: str):
    """Parse score, missing sections, and missing skills from rendered Jobscan PDF."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = "\n".join(pg.get_text("text") for pg in doc)

        def between(text, start, ends):
            i = text.lower().find(start.lower())
            if i == -1: return ""
            j = len(text); lower = text.lower()
            for em in ends:
                k = lower.find(em.lower(), i+len(start))
                if k != -1: j = min(j, k)
            return text[i:j]

        # Score
        m = re.search(r"Match Rate.*?(\d{1,3})\s*%", text, flags=re.I|re.S)
        if not m:
            m = re.search(r"(\d{1,3})\s*%\s*(?:Match|Rate)", text, flags=re.I)
        score = int(m.group(1)) if m else 0

        # Missing sections (heuristic)
        miss_secs = []
        if re.search(r"couldn't find an?\s+\"Education\"", text, flags=re.I): miss_secs.append("EDUCATION")
        if re.search(r"couldn't find an?\s+\"Summary\"", text, flags=re.I): miss_secs.append("SUMMARY")
        if re.search(r"couldn't find an?\s+\"Skills\"", text, flags=re.I): miss_secs.append("SKILLS")
        if re.search(r"couldn't find an?\s+\"Work Experience\"", text, flags=re.I): miss_secs.append("WORK EXPERIENCE")

        # Missing skills
        def parse_skill_block(block_text: str):
            present, missing = set(), set()
            lines = [ln.strip() for ln in block_text.splitlines() if ln.strip()]
            lines = [ln for ln in lines if ln.lower() not in ("copy all","highlighted skills","skills comparison","skill resume job description")]
            row_patterns = [
                re.compile(r"^(?P<skill>.+?)\s+(?P<resume>\d+)\s+(?P<jd>\d+)$"),
                re.compile(r"^(?P<resume>\d+)\s+(?P<jd>\d+)\s+(?P<skill>.+)$"),
            ]
            for ln in lines:
                matched = False
                for rp in row_patterns:
                    m = rp.match(ln)
                    if m:
                        rc, jc = int(m.group("resume")), int(m.group("jd"))
                        sk = m.group("skill").strip().strip(":")
                        if rc > 0:
                            present.add(sk)
                        elif jc > 0:
                            missing.add(sk)
                        matched = True
                        break
                if not matched:
                    pass
            return present, missing

        hard_block = between(text, "Hard skills", ["Soft skills", "Recruiter tips", "Formatting", "Page Setup", "Web Presence"]) or ""
        soft_block = between(text, "Soft skills", ["Recruiter tips", "Formatting", "Page Setup", "Web Presence"]) or ""
        hp, hm = parse_skill_block(hard_block)
        sp, sm = parse_skill_block(soft_block)
        missing_keywords = sorted(hm.union(sm))
        return score, missing_keywords, miss_secs
    except Exception as e:
        print(f"⚠️ PDF fallback parse failed: {e}")
        return 0, [], []

def _extract_missing_sections_text(page_text: str):
    import re
    tl = (page_text or "").lower()
    miss = []
    cues = [
        (r"couldn['']t find an?\s+\"?summary\"?", "SUMMARY"),
        (r"couldn['']t find an?\s+\"?skills\"?", "SKILLS"),
        (r"couldn['']t find an?\s+\"?work experience\"?", "WORK EXPERIENCE"),
        (r"couldn['']t find an?\s+\"?education\"?", "EDUCATION"),
        (r"couldn['']t find an?\s+\"?certifications?\"?", "CERTIFICATIONS"),
    ]
    for rx, sec in cues:
        if re.search(rx, tl, flags=re.I):
            miss.append(sec)
    return miss

async def _read_match_rate_strict(page) -> int | None:
    """
    Read the big match-rate ring (#score) reliably:
    aria-label -> inner_text -> JS -> OCR screenshot of #score.
    """
    import re as _re
    sel = SCORE_WIDGET_SEL

    try:
        aria = await page.locator(sel).get_attribute("aria-label", timeout=1500)
        if aria:
            m = _re.search(r"(\d{1,3})\s*%", aria)
            if m: return int(m.group(1))
    except Exception:
        pass

    try:
        txt = await page.locator(sel).inner_text(timeout=1500)
        m = _re.search(r"(\d{1,3})\s*%", txt or "")
        if m: return int(m.group(1))
    except Exception:
        pass

    try:
        txt = await page.evaluate(f"(()=>{{const el=document.querySelector('{sel}');return el?el.innerText:''}})()")
        m = _re.search(r"(\d{1,3})\s*%", txt or "")
        if m: return int(m.group(1))
    except Exception:
        pass

    try:
        tmp = "jobscan_score_dom.png"
        await page.locator(sel).screenshot(path=tmp)
        v = _ocr_percent_from_image(tmp)
        if isinstance(v, int) and 0 <= v <= 100:
            return v
    except Exception:
        pass
    return None


# -------------------- ASYNC main entry --------------------
score_from_dom = -1
score_from_pdf = -1
missing_keywords_dom: list[str] = []
missing_sections_dom: list[str] = []
pdf_keywords: list[str] = []
raw_keywords: list[str] = []
keywords: list[str] = []
missing_sections: list[str] = []

async def _get_jobscan_score_and_feedback_async(resume_text: str, jd_text: str):
    """
    Live Jobscan run with enhanced skills extraction:
      - Locks score to the big ring (#score) when present
      - Ensures Skills Comparison tab + forces virtualized rows to render
      - Merges missing skills with strict priority: DOM → text snapshot → PDF layout → fallback → PDF text
      - Writes jobscan_result.json and debug text dumps
    """
    _require_credentials()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS, slow_mo=0 if HEADLESS else 50)
        context = await browser.new_context(
            permissions=['geolocation'],
            geolocation={'latitude': 0, 'longitude': 0},
        )
        # Extra belt-and-suspenders
        try:
            await context.grant_permissions(['geolocation'], origin="https://app.jobscan.co")
        except Exception:
            pass

        page = await context.new_page()
        try:
            # --- Login & navigate ---
            print(f"🧭 HEADLESS={HEADLESS}. Navigating to Jobscan login…")
            await page.goto("https://app.jobscan.co/auth/login", timeout=60000)
            await page.wait_for_selector("input[name='email']", timeout=15000)
            await page.wait_for_selector("input[name='password']", timeout=15000)
            await page.fill("input[name='email']", EMAIL)
            await page.fill("input[name='password']", PASSWORD)
            await _click_unique(page, "button:has-text('Sign In')")
            await page.wait_for_load_state("networkidle", timeout=30000)
            await _dismiss_overlays(page)

            print("🔍 Looking for 'New Scan'…")
            new_scan_sel = await _wait_for_new_scan(page)
            await _click_unique(page, new_scan_sel)
            await page.wait_for_timeout(600)

            resume_sel = "textarea[placeholder^='Paste resume']"
            jd_sel = "#jobDescriptionInput, textarea[placeholder^='Paste job description']"
            await page.wait_for_selector(resume_sel, timeout=15000)
            await page.wait_for_selector(jd_sel, timeout=15000)
            await page.fill(resume_sel, resume_text)
            await page.fill(jd_sel, jd_text)

            scan_btn_sel = "button[data-test='scan-button']"
            await page.wait_for_selector(scan_btn_sel, timeout=15000)
            print("⏳ Waiting for scan button to become enabled…")
            scan_btn = page.locator(scan_btn_sel).first
            for _ in range(20):
                try:
                    if await scan_btn.is_enabled():
                        break
                except Exception:
                    pass
                await page.wait_for_timeout(200)

            print("▶️ Clicking Scan…")
            if not await _click_unique(page, scan_btn_sel):
                await _dismiss_overlays(page)
                if not await _click_unique(page, "button:has-text('Scan')"):
                    raise RuntimeError("Scan button could not be clicked")

            # --- Wait for score ring & stabilize ---
            for attempt in range(10):
                try:
                    await page.wait_for_selector(SCORE_WIDGET_SEL, timeout=6000)
                    break
                except Exception:
                    print(f"⏳ Waiting for score… retry {attempt + 1}/10")
                    await _dismiss_overlays(page)
                    await page.wait_for_timeout(400)
            else:
                print("❌ ATS score widget not found; returning zeros.")
                await browser.close()
                return 0, [], []

            print("🧷 Stabilizing score gauge…")
            await _stabilize_widget(page, SCORE_WIDGET_SEL, shots=14, interval_ms=700)

            # --- DOM-first score (lock to ring) ---
            dom_score = await _read_match_rate_strict(page)
            if dom_score is None:
                dom_score = -1
            else:
                print(f"🔎 DOM score: {dom_score}%")

            # Secondary attempts against the ring (won't override if already valid)
            for sel in [SCORE_WIDGET_SEL, f"{SCORE_WIDGET_SEL} [aria-label]", "[data-test='score']", "[data-cy='score']"]:
                try:
                    txt = await page.locator(sel).inner_text()
                    m = re.search(r"(\d{1,3})\s*%", txt or "")
                    if m:
                        val = int(m.group(1))
                        if 0 <= val <= 100:
                            dom_score = val
                            break
                except Exception:
                    continue

            # --- Ensure correct Skills Comparison tab & mount rows ---
            try:
                await _ensure_skills_comparison_tab(page)
            except Exception:
                pass
            await page.wait_for_timeout(200)
            try:
                await page.locator("h2:has-text('Hard skills')").scroll_into_view_if_needed()
                await page.wait_for_timeout(250)
                await page.locator("h2:has-text('Soft skills')").scroll_into_view_if_needed()
                await page.wait_for_timeout(250)
                await page.wait_for_selector(
                    "section:has(h2:has-text('Hard skills')) >> css=tr, "
                    "section:has(h2:has-text('Hard skills')) >> [role='row']",
                    timeout=2000,
                )
            except Exception:
                pass

            # --- DOM missing extraction (enhanced) ---
            try:
                missing_keywords_dom, missing_sections_dom = await _extract_missing_from_dom(page)
                print(f"🔍 DOM extraction found {len(missing_keywords_dom)} missing keywords")
            except Exception as e:
                print(f"⚠️ DOM extraction failed: {e}")
                missing_keywords_dom, missing_sections_dom = [], []

            # --- Text snapshot (enhanced parsing) ---
            page_text = await page.evaluate("document.body.innerText")
            try:
                Path("jobscan_full_rendered.txt").write_text(page_text, encoding="utf-8")
            except Exception:
                pass

            # Enhanced text parsing
            try:
                present_t, missing_t = _extract_skills_from_text(page_text)
                print(f"📄 Text parsing found {len(missing_t)} missing keywords")
            except Exception as e:
                print(f"⚠️ Text parsing failed: {e}")
                present_t, missing_t = set(), set()

            # Fallback: Known skills pattern matching
            try:
                present_f, missing_f = _extract_skills_from_jobscan_format(page_text)
                print(f"🎯 Fallback parsing found {len(missing_f)} missing keywords")
            except Exception as e:
                print(f"⚠️ Fallback parsing failed: {e}")
                present_f, missing_f = set(), set()

            # Also harvest section hints from page text
            miss_sec_text = _extract_missing_sections_text(page_text)
            if miss_sec_text:
                ms = set(missing_sections_dom)
                for s in miss_sec_text:
                    if s not in ms:
                        missing_sections_dom.append(s); ms.add(s)

            # --- PDF processing (enhanced) ---
            try:
                await _dismiss_overlays(page)
            except Exception:
                pass
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            await page.wait_for_timeout(1200)
            await _render_pdf_resilient_async(page, PDF_PATH, HEADLESS)

            # Enhanced PDF parsing
            try:
                doc = fitz.open(PDF_PATH)
                full_text = "\n".join(pg.get_text("text") for pg in doc)
            except Exception:
                full_text = ""

            # PDF: layout-aware parse (enhanced)
            try:
                pdf_present2, pdf_missing2 = _extract_skills_from_pdf_layout(PDF_PATH)
                print(f"📊 PDF layout parsing found {len(pdf_missing2)} missing keywords")
            except Exception as e:
                print(f"⚠️ PDF layout parsing failed: {e}")
                pdf_present2, pdf_missing2 = set(), set()

            # PDF: text-only parse 
            try:
                pdf_present, pdf_missing = _extract_skills_from_text(full_text)
                print(f"📝 PDF text parsing found {len(pdf_missing)} missing keywords")
            except Exception:
                pdf_present, pdf_missing = set(), set()

            # --- Score selection: ring → pdf text → OCR ---
            if dom_score >= 0:
                score_val = dom_score
            else:
                s_from_text = extract_score_from_pdf_text(full_text)
                score_val = int(s_from_text) if s_from_text.isdigit() else -1
                if score_val < 0:
                    # OCR fallback on the ring
                    tmp_path = "jobscan_score_widget.png"
                    try:
                        await page.locator(SCORE_WIDGET_SEL).screenshot(path=tmp_path)
                        score_val = _ocr_percent_from_image(tmp_path)
                        if score_val >= 0 and OCR_DEBUG:
                            print(f"🧪 OCR extracted score: {score_val}%")
                    except Exception as _e:
                        if OCR_DEBUG:
                            print(f"⚠️ OCR fallback failed: {_e}")
                        score_val = 0

            # --- Missing sections: prefer DOM, else PDF text heuristic ---
            missing_sections = missing_sections_dom if missing_sections_dom else extract_missing_sections_from_text(full_text)

            # OCR retry if score looks bogus AND many sections are missing
            if (score_val in {0, -1}) and len(missing_sections) >= 4:
                await page.wait_for_timeout(1500)
                try:
                    tmp_path = "jobscan_score_widget_retry.png"
                    await page.locator(SCORE_WIDGET_SEL).screenshot(path=tmp_path)
                    retry_val = _ocr_percent_from_image(tmp_path)
                    if retry_val > score_val:
                        score_val = retry_val
                        if OCR_DEBUG:
                            print(f"🔁 OCR retry improved score to: {score_val}%")
                except Exception:
                    pass

            # --- Missing keywords selection with priority & debugging ---
            all_sources = {
                "dom": list(missing_keywords_dom),
                "text_snapshot": list(missing_t),
                "pdf_layout": list(pdf_missing2),
                "pdf_text": list(pdf_missing),
                "fallback": list(missing_f)
            }
            
            print(f"🧪 All sources found: {[(k, len(v)) for k, v in all_sources.items()]}")

            # Priority selection
            if missing_keywords_dom:
                raw_keywords = missing_keywords_dom
                source_tag = "dom"
            elif missing_t:
                raw_keywords = list(missing_t)
                source_tag = "text_snapshot"
            elif pdf_missing2:
                raw_keywords = list(pdf_missing2)
                source_tag = "pdf_layout"
            elif missing_f:
                raw_keywords = list(missing_f)
                source_tag = "fallback"
            else:
                raw_keywords = list(pdf_missing)
                source_tag = "pdf_text"

            print(f"🎯 Selected source: {source_tag} with {len(raw_keywords)} raw keywords")
            print(f"🎯 Raw keywords: {raw_keywords}")

            keywords, dropped = sanitize_keywords(raw_keywords)
            keywords = _dedupe_ci(keywords)
            
            print(f"✅ Final keywords after sanitization: {keywords}")

            # --- Write result JSON ---
            try:
                dbg = {
                    "source": source_tag,
                    "score": int(score_val) if str(score_val).isdigit() else 0,
                    "missing_sections": missing_sections,
                    "missing_keywords": keywords,
                    "all_sources": all_sources,  # Add debug info
                }
                print(f"🧮 Final score selected: {int(score_val) if str(score_val).isdigit() else score_val}% (DOM={dom_score})")
                with open("jobscan_result.json","w",encoding="utf-8") as f:
                    json.dump(dbg, f, indent=2)
                print("🧾 Wrote jobscan_result.json")
            except Exception as _e:
                print(f"⚠️ Could not write jobscan_result.json: {_e}")

            # --- Debug dumps (optional) ---
            try:
                with open("jobscan_keywords_raw.txt","w",encoding="utf-8") as f: f.write("\n".join(raw_keywords))
                with open("jobscan_keywords_clean.txt","w",encoding="utf-8") as f: f.write("\n".join(keywords))
                if dropped:
                    with open("jobscan_keywords_dropped.txt","w",encoding="utf-8") as f: f.write("\n".join(dropped))
            except Exception:
                pass

            # --- Clean close & return ---
            score_int = max(0, min(100, int(score_val if isinstance(score_val, int) else 0)))
            try: await page.close()
            except Exception: pass
            try: await context.close()
            except Exception: pass
            try: await browser.close()
            except Exception: pass
            return score_int, keywords, missing_sections

        except Exception as e:
            print(f"❌ Jobscan scan failed: {e}")
            try: await page.close()
            except Exception: pass
            try: await context.close()
            except Exception: pass
            try: await browser.close()
            except Exception: pass
            return 0, [], []


# -------------------- Safe runner for sync/async callers --------------------

def _run_async_safely(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_container = {}
    exc_container = {}

    def _worker():
        try:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result_container["result"] = new_loop.run_until_complete(coro)
            new_loop.close()
        except Exception as e:
            exc_container["exc"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()

    if "exc" in exc_container:
        raise exc_container["exc"]
    return result_container.get("result")

# -------------------- Public entry --------------------

from typing import Tuple, List

def get_jobscan_score_and_feedback(resume_text: str, jd_text: str) -> Tuple[int, List[str], List[str]]:
    """
    Always return (score, missing_keywords, missing_sections).
    Never propagate None to callers.
    """
    try:
        ret = _run_async_safely(_get_jobscan_score_and_feedback_async(resume_text, jd_text))
    except Exception as e:
        print(f"❌ Jobscan scan failed (wrapper): {e}")
        return (0, [], [])

    if not isinstance(ret, tuple) or len(ret) != 3:
        return (0, [], [])
    score, kws, secs = ret
    try:
        score = int(score)
    except Exception:
        score = 0
    if not isinstance(kws, list):
        kws = []
    if not isinstance(secs, list):
        secs = []
    return (score, kws, secs)