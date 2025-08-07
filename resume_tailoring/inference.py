import os
import re
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv
load_dotenv()
from .guard_clean_resume import (
    fix_broken_headers, remove_hallucinated_titles, 
    patch_job_titles_with_original, remove_placeholder_bullets,
    clean_full_resume, validate_resume_structure
)

MAX_MODEL_TOKENS = 8192
SAVE_LOGS = True
LOG_DIR = "prompt_logs"
DEBUG = os.getenv("DEBUG_RESUME") == "1"
os.makedirs(LOG_DIR, exist_ok=True)

REQUIRED_SECTIONS = ["SUMMARY", "SKILLS", "WORK EXPERIENCE", "CERTIFICATIONS", "EDUCATION"]

def validate_sections(text: str) -> Dict[str, any]:
    """Enhanced section validation with detailed feedback"""
    present = set()
    missing = []
    section_details = {}
    
    for section in REQUIRED_SECTIONS:
        if section in text.upper():
            present.add(section)
            # Get section details
            section_details[section] = analyze_section_content(text, section)
        else:
            missing.append(section)
    
    # Additional quality checks
    quality_issues = []
    
    if "WORK EXPERIENCE" in present:
        job_count = count_work_experience_jobs(text)
        if job_count < 2:
            quality_issues.append(f"Only {job_count} work experience entries found")
    
    if DEBUG:
        print(f"✅ Present sections: {present}")
        print(f"❌ Missing sections: {missing}")
        if quality_issues:
            print(f"⚠️ Quality issues: {quality_issues}")
    
    return {
        "valid": len(missing) == 0,
        "missing_sections": missing,
        "present_sections": list(present),
        "section_details": section_details,
        "quality_issues": quality_issues
    }

def analyze_section_content(text: str, section_name: str) -> Dict[str, any]:
    """Analyze the content quality of a specific section"""
    section_start = text.upper().find(section_name)
    if section_start == -1:
        return {"exists": False}
    
    # Find section boundaries
    next_section_start = len(text)
    for other_section in REQUIRED_SECTIONS:
        if other_section != section_name:
            other_start = text.upper().find(other_section, section_start + 1)
            if other_start != -1 and other_start < next_section_start:
                next_section_start = other_start
    
    section_content = text[section_start:next_section_start].strip()
    lines = [line.strip() for line in section_content.splitlines() if line.strip()]
    
    analysis = {
        "exists": True,
        "line_count": len(lines),
        "character_count": len(section_content),
        "has_bullets": any(line.startswith(('-', '•', '●')) for line in lines),
    }
    
    if section_name == "WORK EXPERIENCE":
        analysis["job_count"] = count_jobs_in_section(section_content)
        analysis["total_bullets"] = len([line for line in lines if line.startswith(('-', '•', '●'))])
    
    return analysis

def count_work_experience_jobs(text: str) -> int:
    """Count the number of job entries in work experience"""
    # Look for patterns like "Job Title – Company, Date"
    job_pattern = r"^.+?[–-].+?\d{4}"
    lines = text.splitlines()
    
    job_count = 0
    for line in lines:
        line = line.strip()
        if re.match(job_pattern, line) and not line.startswith(('-', '•', '●')):
            job_count += 1
    
    return job_count

def count_jobs_in_section(section_content: str) -> int:
    """Count jobs within a work experience section"""
    job_pattern = r"^.+?[–-].+?\d{4}"
    return len([line for line in section_content.splitlines() 
               if re.match(job_pattern, line.strip()) and not line.strip().startswith(('-', '•', '●'))])

def check_personal_info_fields(text: str) -> Dict[str, any]:
    """Enhanced personal information validation"""
    fields = {
        "name": bool(re.search(r"^[A-Z][a-z]+ [A-Z][a-z]+", text)),
        "email": bool(re.search(r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b", text)),
        "phone": bool(re.search(r"(\+?\d{1,3}[-.\s])?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)),
        "location": bool(re.search(r"[A-Z][a-z]+,\s*[A-Z]{2}", text)),  # City, State format
    }
    
    missing = [k for k in ["name", "email", "phone"] if not fields[k]]
    
    return {
        "valid": len(missing) == 0,
        "missing_fields": missing,
        "found_fields": [k for k, v in fields.items() if v],
        "completeness_score": sum(fields.values()) / len(fields)
    }

def remove_placeholder_bullets_enhanced(text: str) -> str:
    """Enhanced placeholder removal with more patterns"""
    patterns = [
        r"- Additional relevant responsibility \d+\n?",
        r"- \[.*?\].*?\n?",  # Bracketed placeholders
        r"- TBD.*?\n?",      # TBD items
        r"- TODO.*?\n?",     # TODO items
        r"- \.\.\.\n?",      # Ellipsis placeholders
        r"- Add .*?\n?",     # "Add something" placeholders
        r"- Insert .*?\n?",  # "Insert something" placeholders
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    return text

def enforce_min_bullets_per_job(resume_text: str, min_bullets: int = 6) -> str:
    """
    Enhanced bullet enforcement with better job detection and quality fallbacks
    """
    if "WORK EXPERIENCE" not in resume_text.upper():
        return resume_text

    lines = resume_text.splitlines()
    new_lines = []
    in_experience = False
    current_job = []
    bullet_count = 0
    job_header_pattern = r"^.+?[–-].+?\d{4}"

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Detect work experience section start
        if line_stripped.upper() == "WORK EXPERIENCE":
            in_experience = True
            new_lines.append(line)
            continue

        if in_experience:
            # Check for job header
            if (re.match(job_header_pattern, line_stripped) and 
                not line_stripped.startswith(("-", "•", "●"))):
                
                # Process previous job if exists
                if current_job:
                    if bullet_count < min_bullets:
                        # Add quality fallback bullets instead of generic ones
                        fallback_bullets = generate_fallback_bullets(
                            current_job[0] if current_job else "", 
                            min_bullets - bullet_count
                        )
                        current_job.extend(fallback_bullets)
                    new_lines.extend(current_job)
                    current_job = []
                
                # Start new job
                current_job = [line]
                bullet_count = 0
                
            elif line_stripped.startswith(("-", "•", "●")):
                if current_job:  # Only add if we have a job header
                    current_job.append(line)
                    bullet_count += 1
                else:
                    new_lines.append(line)  # Orphaned bullet
                    
            elif line_stripped == "" or line_stripped.upper() in REQUIRED_SECTIONS:
                # End of work experience section
                if current_job:
                    if bullet_count < min_bullets:
                        fallback_bullets = generate_fallback_bullets(
                            current_job[0] if current_job else "",
                            min_bullets - bullet_count
                        )
                        current_job.extend(fallback_bullets)
                    new_lines.extend(current_job)
                    current_job = []
                    
                in_experience = (line_stripped != "" and 
                               line_stripped.upper() not in REQUIRED_SECTIONS)
                new_lines.append(line)
                
            else:
                if current_job:
                    current_job.append(line)
                else:
                    new_lines.append(line)
        else:
            new_lines.append(line)

    # Handle final job if exists
    if current_job:
        if bullet_count < min_bullets:
            fallback_bullets = generate_fallback_bullets(
                current_job[0] if current_job else "",
                min_bullets - bullet_count
            )
            current_job.extend(fallback_bullets)
        new_lines.extend(current_job)

    return "\n".join(new_lines)

def generate_fallback_bullets(job_header: str, count_needed: int) -> List[str]:
    """Generate contextual fallback bullets based on job header"""
    
    # Extract job context from header
    job_context = "business analysis"
    if job_header:
        if any(term in job_header.lower() for term in ["analyst", "analysis"]):
            job_context = "business analysis and data insights"
        elif any(term in job_header.lower() for term in ["data", "bi", "intelligence"]):
            job_context = "data analysis and reporting"
        elif any(term in job_header.lower() for term in ["consultant", "consulting"]):
            job_context = "consulting and process improvement"
    
    # Quality fallback bullets
    fallback_templates = [
        f"- Collaborated with cross-functional teams to deliver {job_context} solutions",
        f"- Supported data-driven decision making through comprehensive analysis and reporting",
        f"- Participated in process improvement initiatives to enhance operational efficiency",
        f"- Documented business requirements and technical specifications for key projects",
        f"- Facilitated meetings and presentations to communicate findings to stakeholders",
        f"- Maintained and updated analytical models and reporting systems",
    ]
    
    return fallback_templates[:count_needed]

def trim_text_to_fit_token_budget(base_resume: str, jd: str, style_guide: str, 
                                 max_prompt_tokens: int = 3500) -> str:
    """
    Smart text trimming that preserves important content
    """
    # Rough token estimation (4 chars per token)
    total_chars = len(base_resume) + len(jd) + len(style_guide)
    max_chars = max_prompt_tokens * 4
    
    if total_chars <= max_chars:
        return jd  # No trimming needed
    
    # Trim job description intelligently
    jd_lines = jd.splitlines()
    
    # Keep important sections
    important_keywords = [
        "responsibilities", "requirements", "qualifications", "skills",
        "experience", "education", "must have", "preferred"
    ]
    
    important_lines = []
    other_lines = []
    
    for line in jd_lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in important_keywords):
            important_lines.extend(jd_lines[max(0, jd_lines.index(line)-2):
                                             min(len(jd_lines), jd_lines.index(line)+5)])
            break
        else:
            other_lines.append(line)
    
    # Combine important content first, then add other content as space allows
    trimmed_jd = "\n".join(important_lines)
    remaining_chars = max_chars - len(base_resume) - len(style_guide) - len(trimmed_jd)
    
    if remaining_chars > 0:
        additional_content = "\n".join(other_lines)[:remaining_chars]
        trimmed_jd = trimmed_jd + "\n" + additional_content
    
    return trimmed_jd

def format_prompt(base_resume: str, jd: str, style_guide: str, 
                 ats_keywords: List[str] = None, ats_sections: List[str] = None, 
                 job_title: str = None) -> str:
    """
    Enhanced prompt formatting with better structure and instructions
    """
    
    # Trim content to fit token budget
    trimmed_jd = trim_text_to_fit_token_budget(base_resume, jd, style_guide)
    
    prompt = f"""{style_guide.strip()}

###

BASE RESUME (for reference only):
{base_resume.strip()}

JOB DESCRIPTION (for reference only):
{trimmed_jd.strip()}"""

    # Add ATS feedback if available
    if ats_keywords or ats_sections:
        prompt += "\n\nATS OPTIMIZATION FEEDBACK:"
        if ats_keywords:
            prompt += f"\nMissing Keywords (incorporate naturally): {', '.join(ats_keywords)}"
        if ats_sections:
            prompt += f"\nMissing/Weak Sections (strengthen these): {', '.join(ats_sections)}"

    # Add specific job title guidance
    if job_title:
        prompt += f"\n\nTARGET ROLE: {job_title}"
        prompt += "\nEnsure the resume clearly positions the candidate for this specific role."

    prompt += """

###

GENERATION INSTRUCTIONS:
1. Generate a complete, tailored resume following the structure above
2. Tailor each bullet point using relevant skills and tools from the job description
3. Include the job title naturally in SUMMARY or matching work experience
4. Ensure all sections are present and properly formatted
5. Start your output immediately with the resume content - no meta-text

Begin resume generation:

SUMMARY
"""
    
    return prompt

def save_prompt_and_output(prompt: str, output: str, job_title: str = "unknown") -> None:
    """Enhanced logging with better organization"""
    if not SAVE_LOGS:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_job_title = re.sub(r'[^\w\s-]', '', job_title).strip()[:30]
    
    log_filename = f"prompt_{safe_job_title}_{timestamp}.txt"
    log_path = os.path.join(LOG_DIR, log_filename)
    
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + " PROMPT " + "=" * 50 + "\n")
            f.write(prompt)
            f.write("\n\n" + "=" * 50 + " OUTPUT " + "=" * 50 + "\n")
            f.write(output)
            f.write("\n\n" + "=" * 50 + " METADATA " + "=" * 50 + "\n")
            f.write(f"Job Title: {job_title}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Prompt Length: {len(prompt)} chars\n")
            f.write(f"Output Length: {len(output)} chars\n")
    except Exception as e:
        print(f"⚠️ Failed to save prompt log: {e}")

def patch_final_resume(tailored_text: str, base_resume_text: str) -> str:
    """
    Enhanced final patching with comprehensive fixes and validation
    """
    print("🔧 Applying final resume patches...")
    
    # Step 1: Remove obvious hallucinations
    cleaned = remove_hallucinated_titles(tailored_text)
    
    # Step 2: Restore original job headers for consistency
    patched = patch_job_titles_with_original(cleaned, base_resume_text)
    
    # Step 3: Ensure minimum bullet count per job
    patched = enforce_min_bullets_per_job(patched, min_bullets=6)
    
    # Step 4: Remove all placeholder content
    patched = remove_placeholder_bullets_enhanced(patched)
    
    # Step 5: Fix formatting issues
    final = fix_broken_headers(patched)
    
    # Step 6: Final quality validation
    validation = validate_resume_structure(final)
    if not validation["valid"]:
        print(f"⚠️ Final validation issues: {validation}")
    
    print("✅ Final resume patching completed")
    return final

def tailor_resume(*args, **kwargs):
    """Placeholder - actual tailoring is handled in generate_with_models.py"""
    raise NotImplementedError(
        "Resume tailoring logic has been moved to generate_with_models.py. "
        "Use tailor_resume_with_models() instead."
    )

def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation)"""
    return len(text) // 4

def validate_final_output_quality(text: str) -> Dict[str, any]:
    """Comprehensive quality validation for final output"""
    
    issues = []
    warnings = []
    metrics = {}
    
    # Basic length checks
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.splitlines())
    
    metrics.update({
        "character_count": char_count,
        "word_count": word_count, 
        "line_count": line_count
    })
    
    if char_count < 1500:
        issues.append("Resume too short (< 1500 characters)")
    elif char_count > 30000:
        warnings.append("Resume very long (> 30000 characters)")
    
    # Section validation
    section_validation = validate_sections(text)
    if not section_validation["valid"]:
        issues.extend([f"Missing section: {s}" for s in section_validation["missing_sections"]])
    
    # Content quality checks
    if "WORK EXPERIENCE" in text:
        job_count = count_work_experience_jobs(text)
        metrics["job_count"] = job_count
        if job_count < 2:
            warnings.append(f"Only {job_count} work experience entries")
    
    # Check for placeholder content
    placeholder_patterns = [
        r"\[.*?\]", r"TBD", r"TODO", r"Additional relevant responsibility",
        r"Insert.*", r"Add.*here", r"\.\.\."
    ]
    
    placeholder_count = 0
    for pattern in placeholder_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        placeholder_count += len(matches)
    
    if placeholder_count > 0:
        warnings.append(f"Contains {placeholder_count} placeholder items")
    
    metrics["placeholder_count"] = placeholder_count
    
    # Personal info validation
    personal_info = check_personal_info_fields(text)
    if not personal_info["valid"]:
        issues.extend([f"Missing personal info: {field}" for field in personal_info["missing_fields"]])
    
    metrics["personal_info_completeness"] = personal_info["completeness_score"]
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "metrics": metrics,
        "overall_score": calculate_quality_score(metrics, issues, warnings)
    }

def calculate_quality_score(metrics: Dict, issues: List, warnings: List) -> float:
    """Calculate overall quality score (0-100)"""
    
    base_score = 100.0
    
    # Deduct for issues
    base_score -= len(issues) * 15  # Major deduction for issues
    base_score -= len(warnings) * 5  # Minor deduction for warnings
    
    # Adjust based on metrics
    if "job_count" in metrics:
        if metrics["job_count"] >= 3:
            base_score += 5
        elif metrics["job_count"] < 2:
            base_score -= 10
    
    if "personal_info_completeness" in metrics:
        completeness_bonus = metrics["personal_info_completeness"] * 10
        base_score += completeness_bonus
    
    if "placeholder_count" in metrics and metrics["placeholder_count"] > 0:
        base_score -= metrics["placeholder_count"] * 2
    
    # Ensure score is between 0 and 100
    return max(0.0, min(100.0, base_score))