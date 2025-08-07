import re
from typing import Tuple, List, Set, Dict

KNOWN_DEGREES = [
    "Master of Science in Computer Science, University of Bridgeport",
    "Bachelor of Technology in Computer Science & Engineering, KMIT"
]

KNOWN_CERTS = [
    "Google Data Analytics (Coursera)",
    "Excel for Business (Coursera)"
]

# Valid section headers
VALID_SECTIONS = ["SUMMARY", "SKILLS", "WORK EXPERIENCE", "CERTIFICATIONS", "EDUCATION"]

# ----------- Utility Functions -----------

def normalize(text: str) -> str:
    """Normalize text for comparison (lowercase, remove punctuation)."""
    return re.sub(r"[^a-z0-9 ]+", "", text.strip().lower())

def fix_broken_headers(text: str) -> str:
    """
    Fixes broken headers like 'WO\nRK EXPERIENCE'.
    Preserves job headers (e.g. "Analyst – Cognizant") by only merging broken full-caps words.
    """
    # Fix only known broken section headers
    broken_headers = [
        ("ED\nUCATION", "EDUCATION"),
        ("WO\nRK EXPERIENCE", "WORK EXPERIENCE"), 
        ("WORK\nEXPERIENCE", "WORK EXPERIENCE"),
        ("CE\nRTIFICATIONS", "CERTIFICATIONS"),
        ("CERTIF\nICATIONS", "CERTIFICATIONS"),
        ("SU\nMMARY", "SUMMARY"),
        ("SK\nILLS", "SKILLS"),
    ]

    for broken, fixed in broken_headers:
        text = text.replace(broken, fixed)

    # Carefully join only all-caps broken headers, not job titles
    def fix_caps(match):
        return match.group(0).replace("\n", "")

    text = re.sub(r"\b([A-Z]{2,})\n([A-Z]{2,})\b", fix_caps, text)
    
    # Additional cleanup for common formatting issues
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)  # Multiple blank lines to double
    text = re.sub(r"(\n[A-Z ]{3,})\n\s*\n", r"\1\n", text)  # Remove extra line after headers
    
    return text

def find_section_bounds(text: str, section_name: str) -> Tuple[int, int]:
    """Find start and end character positions of a section in the resume."""
    text = fix_broken_headers(text)  # ensure headers are normalized first
    pattern = re.compile(rf"(?m)^{re.escape(section_name.upper())}\s*$")
    match = pattern.search(text)
    if not match:
        return -1, -1
    start = match.start()
    after = text[start + len(section_name) + 1:]
    next_section = re.search(r"(?m)^[A-Z ]{3,}$", after)
    end = start + len(section_name) + 1 + next_section.start() if next_section else len(text)
    return start, end

def remove_placeholder_bullets(text: str) -> str:
    """Removes bullets like '- Additional relevant responsibility 1' used as fallback fillers."""
    patterns = [
        r"- Additional relevant responsibility \d+\n?",
        r"- \[.*?\]\n?",  # Remove bracketed placeholders
        r"- TBD.*?\n?",   # Remove TBD items
        r"- TODO.*?\n?",  # Remove TODO items
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    
    return text

def is_valid_job_header(line: str) -> bool:
    """
    Improved logic to detect valid job headers.
    Must contain company separator and date, but not start with bullet.
    """
    line = line.strip()
    if not line or line.startswith(("-", "•", "●")):
        return False
    
    # Must have en-dash or hyphen separator
    if "–" not in line and " - " not in line:
        return False
    
    # Must contain a year (proxy for dates)
    if not re.search(r"\b\d{4}\b", line):
        return False
    
    # Should not be too short or too long
    if len(line) < 10 or len(line) > 200:
        return False
        
    return True

# ----------- WORK EXPERIENCE Functions -----------

def extract_experience_pairs(resume_text: str) -> Set[Tuple[str, str]]:
    """
    Extract pairs of (job title, company) from WORK EXPERIENCE headers.
    Format expected: "Job Title – Company, Year" or "Job Title - Company Year"
    """
    # More flexible pattern to handle different separators
    patterns = [
        r"^(.*?)\s*–\s*(.*?),?\s*.*?\d{4}.*?$",  # En-dash with optional comma
        r"^(.*?)\s+-\s+(.*?),?\s*.*?\d{4}.*?$",   # Regular dash
    ]
    
    lines = resume_text.splitlines()
    exp_pairs = set()
    
    for line in lines:
        line = line.strip()
        if not is_valid_job_header(line):
            continue
            
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                title = match.group(1).strip()
                company = match.group(2).strip()
                
                # Clean up common suffixes from company names
                company = re.sub(r",.*$", "", company).strip()
                
                if title and company and len(title) > 2 and len(company) > 2:
                    exp_pairs.add((normalize(title), normalize(company)))
                break
    
    return exp_pairs

def parse_work_experience_section(section_text: str) -> List[Dict[str, any]]:
    """
    Parse work experience section into structured job blocks.
    Returns list of jobs with headers and bullets.
    """
    jobs = []
    lines = section_text.strip().splitlines()
    current_job = {"header": "", "bullets": []}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if is_valid_job_header(line):
            # Save previous job if complete
            if current_job["header"] and current_job["bullets"]:
                jobs.append(current_job)
            current_job = {"header": line, "bullets": []}
            
        elif line.startswith(("-", "•", "●")):
            if current_job["header"]:
                # Normalize bullet character
                normalized_bullet = line.replace("•", "-").replace("●", "-")
                current_job["bullets"].append(normalized_bullet)
        elif current_job["header"]:
            # Treat as continuation of previous bullet or new bullet
            if current_job["bullets"]:
                # Append to last bullet if it doesn't look like a new bullet
                if not any(verb in line.lower()[:20] for verb in [
                    'developed', 'managed', 'created', 'led', 'analyzed', 'designed'
                ]):
                    current_job["bullets"][-1] += " " + line
                else:
                    # Treat as new bullet
                    current_job["bullets"].append(f"- {line}")
            else:
                # First bullet for this job
                current_job["bullets"].append(f"- {line}")

    # Append last job if valid
    if current_job["header"] and current_job["bullets"]:
        jobs.append(current_job)

    return jobs

def clean_tailored_work_experience(tailored_resume: str, base_resume: str = None, verbose: bool = False) -> str:
    """
    Clean work experience section by removing hallucinated jobs and fixing formatting.
    """
    if not base_resume:
        return tailored_resume

    base_pairs = extract_experience_pairs(base_resume)
    start, end = find_section_bounds(tailored_resume, "WORK EXPERIENCE")
    
    if start == -1:
        if verbose:
            print("❌ No WORK EXPERIENCE section found.")
        return tailored_resume

    before = tailored_resume[:start]
    work_section = tailored_resume[start:end]
    after = tailored_resume[end:]

    lines = work_section.splitlines()
    cleaned_lines = []
    skip_job_block = False
    skip_bullet_count = 0

    for line in lines:
        line_stripped = line.strip()
        
        # Check if this is a job header
        if is_valid_job_header(line_stripped):
            # Extract title and company for validation
            job_valid = False
            for pattern in [r"^(.*?)\s*–\s*(.*?),?\s*.*?\d{4}.*?$", r"^(.*?)\s+-\s+(.*?),?\s*.*?\d{4}.*?$"]:
                match = re.match(pattern, line_stripped)
                if match:
                    title = normalize(match.group(1).strip())
                    company = normalize(match.group(2).strip().split(',')[0])
                    
                    if (title, company) in base_pairs:
                        job_valid = True
                        break
            
            if job_valid:
                skip_job_block = False
                skip_bullet_count = 0
                cleaned_lines.append(line)
                if verbose:
                    print(f"✅ Kept valid job: {line_stripped}")
            else:
                skip_job_block = True
                skip_bullet_count = 0
                if verbose:
                    print(f"🧹 Removed hallucinated job: {line_stripped}")
                    
        elif skip_job_block:
            # Skip bullets for invalid jobs, but with limits
            if line_stripped.startswith(("-", "•", "●")):
                skip_bullet_count += 1
                if skip_bullet_count <= 10:  # Limit how many bullets we skip
                    continue
            # Stop skipping if we've seen too many bullets or hit a section header
            if skip_bullet_count > 10 or line_stripped.isupper():
                skip_job_block = False
            
            if not skip_job_block:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    cleaned_work_section = "\n".join(cleaned_lines).strip()
    return before + "\n" + cleaned_work_section + "\n" + after

def remove_hallucinated_titles(text: str) -> str:
    """
    Remove job blocks where job titles start with hallucination markers.
    """
    lines = text.splitlines()
    cleaned = []
    skip_block = False

    for line in lines:
        line_stripped = line.strip()
        
        # Check for hallucination markers
        if re.match(r"^[❗⚠️🚫].*–.*", line_stripped):
            skip_block = True
            print(f"🧹 Removing hallucinated job: {line_stripped}")
            continue
            
        # Check for other suspicious patterns
        suspicious_patterns = [
            r"^\[.*\].*–.*",  # [Bracketed] titles
            r"^TBD.*–.*",     # TBD titles
            r"^TODO.*–.*",    # TODO titles
        ]
        
        if any(re.match(pattern, line_stripped) for pattern in suspicious_patterns):
            skip_block = True
            print(f"🧹 Removing suspicious job: {line_stripped}")
            continue
            
        if skip_block:
            # Continue skipping bullets and empty lines
            if (line_stripped.startswith(("-", "•", "●")) or 
                not line_stripped or
                line_stripped.startswith(" ")):
                continue
            else:
                # Hit a new section or job, stop skipping
                skip_block = False
                
        if not skip_block:
            cleaned.append(line)

    return "\n".join(cleaned)

def patch_job_titles_with_original(tailored_text: str, original_text: str) -> str:
    """
    Replace altered job headers in tailored text with original ones.
    Uses intelligent matching to preserve company names and dates.
    """
    # Extract original job headers
    original_jobs = []
    for line in original_text.splitlines():
        if is_valid_job_header(line.strip()):
            original_jobs.append(line.strip())
    
    if not original_jobs:
        return tailored_text
    
    lines = tailored_text.splitlines()
    patched_lines = []
    original_job_index = 0
    
    for line in lines:
        if is_valid_job_header(line.strip()) and original_job_index < len(original_jobs):
            # Replace with original header
            patched_lines.append(original_jobs[original_job_index])
            original_job_index += 1
        else:
            patched_lines.append(line)
    
    return "\n".join(patched_lines)

# ----------- CERTIFICATIONS Functions -----------

def extract_certifications(text: str) -> List[str]:
    """Extract certification entries from text"""
    start, end = find_section_bounds(text, "CERTIFICATIONS")
    if start == -1:
        return []
    
    lines = text[start:end].splitlines()[1:]  # Skip header
    certs = []
    
    for line in lines:
        line = line.strip()
        if line and not line.isupper():  # Skip empty lines and section headers
            certs.append(normalize(line))
    
    return certs

def clean_certifications_section(tailored_resume: str, base_resume: str = None, verbose: bool = False) -> str:
    """Clean certifications section to remove hallucinated entries"""
    if not base_resume:
        return tailored_resume

    base_certs = extract_certifications(base_resume)
    start, end = find_section_bounds(tailored_resume, "CERTIFICATIONS")
    
    if start == -1:
        return tailored_resume

    before = tailored_resume[:start]
    cert_section = tailored_resume[start:end]
    after = tailored_resume[end:]

    lines = cert_section.splitlines()
    cleaned = [lines[0]]  # Keep header
    
    for line in lines[1:]:
        line_stripped = line.strip()
        if not line_stripped:
            cleaned.append(line)
            continue
            
        norm_line = normalize(line_stripped)
        
        # Check if this certification matches any from base resume
        is_valid = any(norm_base in norm_line or norm_line in norm_base 
                      for norm_base in base_certs)
        
        if is_valid:
            cleaned.append(line)
        elif verbose:
            print(f"🧹 Removed hallucinated certification: {line_stripped}")

    return before + "\n" + "\n".join(cleaned).strip() + "\n" + after

# ----------- EDUCATION Functions -----------

def extract_education(text: str) -> List[str]:
    """Extract education entries from text"""
    start, end = find_section_bounds(text, "EDUCATION")
    if start == -1:
        return []
    
    lines = text[start:end].splitlines()[1:]  # Skip header
    education = []
    
    for line in lines:
        line = line.strip()
        if line and not line.isupper():  # Skip empty lines and section headers
            education.append(normalize(line))
    
    return education

def clean_education_section(tailored_resume: str, base_resume: str = None, verbose: bool = False) -> str:
    """Clean education section to remove hallucinated entries"""
    if not base_resume:
        return tailored_resume

    base_edu = extract_education(base_resume)
    start, end = find_section_bounds(tailored_resume, "EDUCATION")
    
    if start == -1:
        return tailored_resume

    before = tailored_resume[:start]
    edu_section = tailored_resume[start:end]
    after = tailored_resume[end:]

    lines = edu_section.splitlines()
    cleaned = [lines[0]]  # Keep header
    
    for line in lines[1:]:
        line_stripped = line.strip()
        if not line_stripped:
            cleaned.append(line)
            continue
            
        norm_line = normalize(line_stripped)
        
        # Check if this education matches any from base resume
        is_valid = any(norm_base in norm_line or norm_line in norm_base 
                      for norm_base in base_edu)
        
        if is_valid:
            cleaned.append(line)
        elif verbose:
            print(f"🧹 Removed hallucinated education: {line_stripped}")

    return before + "\n" + "\n".join(cleaned).strip() + "\n" + after

# ----------- Main Cleaning Functions -----------

def clean_full_resume(tailored_resume: str, base_resume: str = None, verbose: bool = False) -> str:
    """
    Complete resume cleaning pipeline with all fixes applied.
    """
    if not base_resume:
        print("⚠️ No base resume provided for comparison cleaning")
        return fix_basic_issues(tailored_resume)

    print("🧹 Starting comprehensive resume cleaning...")
    
    # Step 1: Fix basic formatting issues
    text = fix_broken_headers(tailored_resume)
    text = remove_placeholder_bullets(text)
    
    # Step 2: Remove hallucinated content
    text = remove_hallucinated_titles(text)
    
    # Step 3: Clean work experience (most important)
    text = clean_tailored_work_experience(text, base_resume, verbose)
    
    # Step 4: Restore original headers for consistency
    text = patch_job_titles_with_original(text, base_resume)
    
    # Step 5: Clean certifications and education
    text = clean_certifications_section(text, base_resume, verbose)
    text = clean_education_section(text, base_resume, verbose)
    
    # Step 6: Final formatting cleanup
    text = final_formatting_cleanup(text)
    
    if verbose:
        print("✅ Resume cleaning completed")
    
    return text

def fix_basic_issues(text: str) -> str:
    """Fix basic formatting issues when no base resume is available"""
    text = fix_broken_headers(text)
    text = remove_placeholder_bullets(text)
    text = remove_hallucinated_titles(text)
    return final_formatting_cleanup(text)

def final_formatting_cleanup(text: str) -> str:
    """Final pass to clean up formatting issues"""
    # Remove excessive blank lines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Ensure proper spacing after section headers
    for section in VALID_SECTIONS:
        text = re.sub(f'({section})\\n([^\\n])', f'\\1\\n\\n\\2', text)
    
    # Clean up bullet formatting
    text = re.sub(r'\n\s*-\s*-\s*', '\n- ', text)  # Fix double bullets
    text = re.sub(r'\n\s*-\s*\n\s*([A-Z])', r'\n- \1', text)  # Fix broken bullets
    
    # Remove trailing whitespace
    lines = [line.rstrip() for line in text.splitlines()]
    text = '\n'.join(lines)
    
    # Ensure file ends with single newline
    text = text.strip() + '\n'
    
    return text

def validate_resume_structure(text: str) -> Dict[str, any]:
    """Validate that resume has proper structure"""
    missing_sections = []
    found_sections = []
    
    for section in VALID_SECTIONS:
        if section in text:
            found_sections.append(section)
        else:
            missing_sections.append(section)
    
    # Check for work experience jobs
    work_jobs = 0
    if "WORK EXPERIENCE" in text:
        start, end = find_section_bounds(text, "WORK EXPERIENCE")
        if start != -1:
            work_section = text[start:end]
            work_jobs = len([line for line in work_section.splitlines() 
                           if is_valid_job_header(line.strip())])
    
    return {
        "valid": len(missing_sections) == 0,
        "missing_sections": missing_sections,
        "found_sections": found_sections,
        "work_experience_jobs": work_jobs,
        "total_length": len(text),
        "line_count": len(text.splitlines())
    }