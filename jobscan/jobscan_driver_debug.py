import asyncio
from playwright.async_api import async_playwright, expect, TimeoutError
from PIL import Image, ImageOps
import pytesseract
import re
import time
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from typing import Tuple, List, Dict
import json

load_dotenv()
EMAIL = os.getenv("JOBSCAN_EMAIL")
PASSWORD = os.getenv("JOBSCAN_PASSWORD")
HEADLESS = False
OCR_DEBUG = os.getenv("OCR_DEBUG", "0") == "1"

class JobscanError(Exception):
    """Custom exception for Jobscan-related errors"""
    pass

async def dismiss_modal_overlays(page, timeout: int = 3000) -> None:
    """Enhanced modal dismissal with multiple strategies"""
    
    dismissal_strategies = [
        # Common modal close buttons
        {"selector": "button:has-text('Dismiss')", "action": "click"},
        {"selector": "button:has-text('Got it')", "action": "click"},
        {"selector": "button:has-text('Close')", "action": "click"},
        {"selector": "button:has-text('OK')", "action": "click"},
        {"selector": "button:has-text('Accept')", "action": "click"},
        {"selector": "[data-testid='close-button']", "action": "click"},
        {"selector": ".modal-close, .close-button", "action": "click"},
        
        # Overlay containers to hide
        {"selector": ".shepherd-modal-overlay-container", "action": "hide"},
        {"selector": "[class*='modal'][class*='overlay']", "action": "hide"},
        {"selector": "[role='dialog']", "action": "hide"},
    ]
    
    for strategy in dismissal_strategies:
        try:
            elements = page.locator(strategy["selector"])
            count = await elements.count()
            
            if count > 0:
                print(f"🧹 Found {count} modal elements: {strategy['selector']}")
                
                if strategy["action"] == "click":
                    for i in range(count):
                        element = elements.nth(i)
                        if await element.is_visible(timeout=1000):
                            await element.click(timeout=2000)
                            await page.wait_for_timeout(500)
                            
                elif strategy["action"] == "hide":
                    await page.evaluate(f"""
                        document.querySelectorAll('{strategy["selector"]}')
                        .forEach(el => el.style.display = 'none');
                    """)
                    
        except Exception as e:
            # Silently continue - modal dismissal is best effort
            continue
    
    # Final escape key press to dismiss any remaining overlays
    try:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(1000)
    except Exception:
        pass

async def extract_ats_feedback(page) -> Tuple[List[str], List[str]]:
    """Extract ATS feedback with improved parsing"""
    
    try:
        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
        
        # Enhanced keyword extraction
        missing_keywords = []
        keyword_selectors = [
            "ul.keywords-missing li",
            ".missing-keywords li", 
            "[data-testid='missing-keywords'] li",
            ".keyword-missing"
        ]
        
        for selector in keyword_selectors:
            elements = soup.select(selector)
            for element in elements:
                keyword = element.get_text(strip=True)
                if keyword and keyword not in missing_keywords:
                    missing_keywords.append(keyword)
        
        # Enhanced section extraction
        missing_sections = []
        section_selectors = [
            "div.missing-section span.section-name",
            ".missing-sections .section-name",
            "[data-testid='missing-sections'] .section",
            ".section-missing"
        ]
        
        for selector in section_selectors:
            elements = soup.select(selector)
            for element in elements:
                section = element.get_text(strip=True)
                if section and section not in missing_sections:
                    missing_sections.append(section)
        
        # Clean and filter results
        missing_keywords = [kw for kw in missing_keywords if len(kw) > 1 and len(kw) < 50]
        missing_sections = [sec for sec in missing_sections if len(sec) > 2 and len(sec) < 30]
        
        return missing_keywords[:15], missing_sections[:10]  # Limit results
        
    except Exception as e:
        print(f"⚠️ Failed to extract ATS feedback: {e}")
        return [], []

def extract_score_with_ocr(image_path: str) -> str:
    """Enhanced OCR score extraction using the working approach from the original"""
    
    try:
        img = Image.open(image_path)
        thresholds = [150, 180, 200]
        upscales = [2, 3, 5, 8]
        psms = [3, 6, 7, 8, 10, 11, 13]

        # Try the working approach first
        for t in thresholds:
            for s in upscales:
                big = img.resize((img.width * s, img.height * s))
                gray = ImageOps.grayscale(big)
                bw = gray.point(lambda x: 0 if x < t else 255, '1')
                for psm in psms:
                    for wl in [True, False]:
                        config = f"--psm {psm} "
                        if wl:
                            config += "-c tessedit_char_whitelist=0123456789%"
                        text = pytesseract.image_to_string(bw, config=config).strip()
                        match = re.search(r"(\d{1,3})%", text)
                        if match:
                            score = match.group(0)
                            print(f"[OCR SUCCESS] Extracted ATS Score: {score}")
                            return score
                    if match: break
                if match: break
            if match: break

        # Fallback approach from working version
        if not match:
            fallback_text = pytesseract.image_to_string(img).strip()
            fallback_match = re.search(r"(\d{1,3})%", fallback_text)
            if fallback_match:
                score = fallback_match.group(0)
                print(f"[OCR SUCCESS - Fallback] Extracted ATS Score: {score}")
                return score
            else:
                print("❌ OCR failed — returning score=0 for ATS retry loop.")
                return "0%"

    except Exception as e:
        print(f"❌ OCR processing failed: {e}")
        return "0%"

def extract_skills_from_pdf(pdf_path: str) -> Dict[str, List[str]]:
    """Extract skills from PDF using the working approach"""
    
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()

        def extract_section_skills(section_name):
            pattern = re.compile(rf"{section_name}.*?(?=\n\n|\Z)", re.DOTALL | re.IGNORECASE)
            match = pattern.search(text)
            if not match:
                return []

            block = re.sub(
                r"(Skills Comparison|Highlighted Skills|Add Skill|Copy All|Don't see skills.*?|Page \d+ of \d+|IMPORTANT|Jobscan Report.*?)",
                "", match.group(0), flags=re.DOTALL
            )

            raw_lines = [line.strip() for line in block.splitlines()]
            exclusions = {
                "Skill", "Resume", "Job Description", "Update", "Add Skill", "Copy All",
                "Skills Comparison", "Highlighted Skills", "Education Match", "IMPORTANT",
                "Recruiter tips", "from the job description?", "HIGH SCORE IMPACT", "MEDIUM SCORE IMPACT",
                "Update required education level", "your education is noted.", "jobs.",
                "frequently in the job description.", "Hard skills", "Soft skills", "Jobscan Report",
                "Resume Tone", "Job Level Match", "Job Title Match", "Date Formatting", "Measurable Results",
                "Update scan information", "internship or a personal project.",
                "found by our algorithms.", "View Measurable Results", "and buzzwords were found. Good job!",
                "Improve your job match by including more", "Customize this section using your"
            }

            skills = [
                line for line in raw_lines
                if re.search(r"[A-Za-z]", line)
                and len(line.split()) <= 6
                and len(line) <= 50
                and line.lower() not in exclusions
                and not line.endswith(":")
            ]

            return sorted(set(skills))

        return {
            "hard_skills": extract_section_skills("Hard skills"),
            "soft_skills": extract_section_skills("Soft skills"),
        }
        
    except Exception as e:
        print(f"❌ PDF skill extraction failed: {e}")
        return {"hard_skills": [], "soft_skills": []}

async def wait_for_score_stabilization(page, max_wait: int = 20) -> None:
    """Wait for score gauge to stabilize using the working approach"""
    
    print("⏳ Waiting for score gauge to stabilize...")
    
    previous_img = None
    for i in range(max_wait):
        try:
            path = "score_section.png"
            await page.locator("#score").screenshot(path=path)
            img_bytes = open(path, "rb").read()
            if i > 0 and img_bytes == previous_img:
                print(f"✅ Score gauge stabilized after {i} seconds.")
                return
            previous_img = img_bytes
            await page.wait_for_timeout(1000)
        except Exception as e:
            print(f"⚠️ Score stabilization check failed: {e}")
            await page.wait_for_timeout(1000)

async def perform_jobscan_analysis(page, resume_text: str, jd_text: str) -> Tuple[str, List[str], List[str], Dict]:
    """Main Jobscan analysis workflow using the working dashboard approach"""
    
    print("📝 Starting Jobscan analysis...")
    
    # Wait for dashboard
    for attempt in range(3):
        try:
            print(f"🔁 Checking dashboard (attempt {attempt + 1}/3)...")
            await page.wait_for_selector("span.title:has-text('New Scan')", timeout=10000)

            # ✅ INSERT HERE: Dismiss location permission
            try:
                if await page.locator("text=app.jobscan.co wants to").is_visible():
                    await page.locator("text=Never allow").click()
                    print("✅ Dismissed location permission popup")
            except Exception as e:
                print(f"⚠️ Location popup not found or already dismissed: {e}")

            break
        except:
            print(f"⏳ Retry #{attempt + 1}: Reloading dashboard...")
            await page.screenshot(path=f"dashboard_retry_{attempt + 1}.png")
            await page.reload()

    await page.click("span.title:has-text('New Scan')")
    await page.wait_for_timeout(1000)
    
    # Dismiss any initial modals
    await dismiss_modal_overlays(page)
    
    # Fill resume and job description using working version approach
    resume_field = page.locator("textarea[placeholder^='Paste resume']")
    jd_field = page.locator("#jobDescriptionInput")
    await expect(resume_field).to_be_visible(timeout=10000)
    await expect(jd_field).to_be_visible(timeout=10000)

    await resume_field.fill(resume_text)
    await jd_field.fill(jd_text)
    
    # Start scan using exact working version approach
    scan_button = page.locator("button[data-test='scan-button']")

    try:
        print("⏳ Waiting for scan button to become enabled...")
        await expect(scan_button).to_be_enabled(timeout=10000)
        print("✅ Scan button is enabled. Attempting click...")
        await scan_button.click()
        await page.wait_for_timeout(500)
    except Exception as e:
        print(f"⚠️ Scan button click failed on first try: {e}")
        await page.screenshot(path="scan_click_failed.png")
        print("🔁 Retrying scan button click...")
        try:
            await page.keyboard.press("Escape")  # Dismiss any overlay
            await page.wait_for_timeout(1000)
            await scan_button.click()
            await page.wait_for_timeout(500)
        except Exception as e2:
            print(f"❌ Scan click retry also failed: {e2}")
            raise
    
    # 🔧 NEW: Dismiss Jobscan Report Modal (if visible)
    try:
        await page.wait_for_selector("text=Jobscan Report", timeout=3000)
        print("⚠️ Detected Jobscan Report modal. Dismissing...")
        await page.locator("text=Dismiss").click()
        await page.wait_for_timeout(500)
    except Exception as e:
        print(f"⚠️ Failed to dismiss Jobscan Report modal: {e}")

    # 🧹 NEW: General Modal Sweeper
    try:
        modals = page.locator("div[class*='modal'], div[class*='overlay'], .shepherd-modal-overlay-container")
        if await modals.count() > 0 and await modals.first.is_visible():
            print("🧹 Sweeping visible modals with ESC...")
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(1000)
    except Exception as e:
        print(f"⚠️ Modal sweep failed: {e}")
    
    # Wait for results using working approach
    print("⏳ Waiting for scan results...")
    
    for attempt in range(6):
        try:
            await page.wait_for_selector("#score", timeout=10000)
            break
        except:
            print(f"⏳ Waiting for score... retry {attempt + 1}")
            await page.wait_for_timeout(2000)
    else:
        print("❌ ATS score never appeared.")
        return "N/A", [], [], {}

    # Handle overlays from working version
    overlays = [
        page.locator("button:has-text('Dismiss')"),
        page.locator("button:has-text('Got it')"),
        page.locator(".shepherd-modal-overlay-container"),
        page.locator("button:has-text('Accept')"),
    ]
    for overlay in overlays:
        try:
            if await overlay.count() > 0:
                print(f"⚠️ Found overlay: {await overlay.first.inner_text()}")
                await overlay.first.click()
                await page.wait_for_timeout(300)
        except Exception as e:
            print(f"⚠️ Failed to dismiss overlay: {e}")

    # Wait for stabilization and extract score using working approach
    await wait_for_score_stabilization(page)
    
    print("📸 Extracting ATS score...")
    score_image_path = "score_section.png"
    await page.locator("#score").screenshot(path=score_image_path)
    score = extract_score_with_ocr(score_image_path)
    
    # Extract ATS feedback using working version approach
    _, sections = await extract_ats_feedback(page)
    
    # Generate PDF and extract skills using exact working version approach
    await page.wait_for_timeout(1000)
    await page.pdf(path="jobscan_full_rendered.pdf", format="A4", print_background=True)
    missed_skills = extract_skills_from_pdf("jobscan_full_rendered.pdf")
    keywords = missed_skills["hard_skills"] + missed_skills["soft_skills"]

    if OCR_DEBUG:
        print("📋 Raw PDF skill dump:")
        for k, v in missed_skills.items():
            print(f"  - {k}: {v}")
        with open("jobscan_debug_output.log", "w") as f:
            f.write(f"Score: {score}\n")
            f.write(f"Missing Keywords: {keywords}\n")
            f.write(f"Missing Sections: {sections}\n")
            f.write(f"Extracted Skills: {missed_skills}\n")

    if isinstance(score, str) and score.endswith('%'):
        score = score.replace('%', '')
    try:
        score_int = int(score)
    except ValueError:
        score_int = 0

    return score_int, keywords, sections, missed_skills

async def run_jobscan(resume_text: str, jd_text: str) -> Tuple[str, List[str], List[str], Dict]:
    """Enhanced main Jobscan runner with integrated working solutions"""
    
    if not EMAIL or not PASSWORD:
        raise JobscanError("Jobscan credentials not configured")
    
    # Input validation
    if len(resume_text.strip()) < 100:
        raise JobscanError("Resume text too short")
    
    if len(jd_text.strip()) < 50:
        raise JobscanError("Job description too short")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            # Login using working approach
            print("🔐 Logging into Jobscan...")
            await page.goto("https://app.jobscan.co/auth/login", timeout=60000)
            await page.fill("input[name='email']", EMAIL)
            await page.fill("input[name='password']", PASSWORD)
            await page.click("button:has-text('Sign In')")
            await page.wait_for_load_state('networkidle')

            if "Invalid email or password" in await page.content():
                print("❌ Login failed: check credentials.")
                await page.screenshot(path="login_failed.png")
                return "N/A", [], [], {}

            # Location permission handling from working version
            try:
                if await page.locator("text=app.jobscan.co wants to").is_visible():
                    await page.locator("text=Never allow").click()
                    print("✅ Dismissed location permission popup")
            except Exception as e:
                print(f"⚠️ Location popup not found or already dismissed: {e}")
            
            # Perform analysis (page is already on dashboard after login)
            return await perform_jobscan_analysis(page, resume_text, jd_text)
            
        except JobscanError:
            raise  # Re-raise our custom errors
        except Exception as e:
            # Save screenshot for debugging
            await page.screenshot(path="jobscan_error.png")
            print(f"[ERROR] Jobscan flow failed: {e}")
            return "N/A", [], [], {}
        finally:
            await browser.close()

def get_jobscan_score(resume_text: str, jd_text: str) -> Tuple[str, List[str], List[str], Dict]:
    """Synchronous wrapper for Jobscan analysis"""
    return asyncio.run(run_jobscan(resume_text, jd_text))

def get_jobscan_score_and_feedback(resume_text: str, jd_text: str) -> Tuple[str, List[str], List[str]]:
    """Get Jobscan score and feedback (simplified return)"""
    score, keywords, sections, _ = get_jobscan_score(resume_text, jd_text)
    return score, keywords, sections