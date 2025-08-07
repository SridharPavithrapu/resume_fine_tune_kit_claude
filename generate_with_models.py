import os
from docx import Document
from models import call_summary_model, call_bullet_model, call_groq_direct
from extract_job_keywords import extract_job_info
from docx.shared import Pt
import re
import json
import time
import requests
from dotenv import load_dotenv
load_dotenv()
from resume_tailoring.guard_clean_resume import (
    fix_broken_headers, remove_hallucinated_titles, 
    remove_placeholder_bullets, parse_work_experience_section,
    clean_full_resume, validate_resume_structure
)

SOFT_SKILLS = [
    "communication", "teamwork", "problem-solving", "adaptability",
    "critical thinking", "time management", "leadership", "attention to detail",
    "collaboration", "analytical thinking", "project management"
]

DEFAULT_HEADER = "Yoshitha Mudulodu    Email: yoshitha4589@gmail.com    Mobile: +1(669)-399-4052\nMilpitas, CA 95035"
DEFAULT_CERTIFICATIONS = "- Google Data Analytics (Coursera)\n- Excel for Business (Coursera)"
DEFAULT_EDUCATION = "Master of Science in Computer Science, University of Bridgeport, Connecticut  August 2021 – May 2023\nBachelor of Technology in Computer Science & Engineering, KMIT, India  August 2015 – May 2019"

def remove_jobscan_artifacts(text: str) -> str:
    """Remove common Jobscan scanning artifacts from text"""
    artifacts = [
        "education match", "updating scan information", "HIGH SCORE IMPACT",
        "maintaining job title match", "updating required education level",
        "ensuring education match", "ensuring HIPAA compliance", "MEDIUM SCORE IMPACT",
        "recruiter tips", "view measurable results", "improve your job match"
    ]
    
    for phrase in artifacts:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    
    return text

def load_docx_text(path: str) -> str:
    """Load text content from DOCX file"""
    try:
        doc = Document(path)
        return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        print(f"❌ Failed to load DOCX file {path}: {e}")
        return ""

def parse_sections(text: str) -> dict:
    """Parse resume text into sections"""
    SECTIONS = ["SUMMARY", "SKILLS", "WORK EXPERIENCE", "CERTIFICATIONS", "EDUCATION"]
    sections = {}
    current_section = None
    current_content = []
    
    for line in text.splitlines():
        line_upper = line.strip().upper().replace("\t", "")
        
        if line_upper in SECTIONS:
            # Save previous section
            if current_section and current_content:
                sections[current_section] = "\n".join(current_content).strip()
            
            # Start new section
            current_section = line_upper
            current_content = []
        elif current_section:
            current_content.append(line.rstrip())
    
    # Save last section
    if current_section and current_content:
        sections[current_section] = "\n".join(current_content).strip()
    
    return sections

def create_keyword_groups(all_keywords: list) -> dict:
    """Group keywords into categories for better prompt organization"""
    technical_keywords = []
    soft_skill_keywords = []
    domain_keywords = []
    
    technical_terms = ['sql', 'python', 'excel', 'tableau', 'power bi', 'etl', 'aws', 'azure', 'r', 'sas', 'hadoop', 'spark']
    soft_skill_terms = ['communication', 'leadership', 'teamwork', 'collaboration', 'management', 'analytical']
    
    for keyword in all_keywords:
        keyword_lower = keyword.lower()
        if any(tech in keyword_lower for tech in technical_terms):
            technical_keywords.append(keyword)
        elif any(soft in keyword_lower for soft in soft_skill_terms):
            soft_skill_keywords.append(keyword)
        else:
            domain_keywords.append(keyword)
    
    return {
        "Technical Skills": technical_keywords[:6],
        "Soft Skills": soft_skill_keywords[:4], 
        "Domain Terms": domain_keywords[:6]
    }

def generate_tailored_summary(original_summary: str, job_title: str, keyword_groups: dict) -> str:
    """Generate tailored summary using improved prompting"""
    
    # Create focused keyword list for summary
    summary_keywords = []
    for group, keywords in keyword_groups.items():
        summary_keywords.extend(keywords[:3])  # Top 3 from each group
    
    summary_prompt = f"""Rewrite this resume summary for a {job_title} position.

REQUIREMENTS:
- Keep it 3-4 sentences, paragraph format (not bullets)
- Naturally incorporate these relevant keywords: {', '.join(summary_keywords[:8])}
- Maintain professional tone and quantify experience where possible
- Start directly with the summary content, no explanations

Original Summary:
{original_summary}"""

    print("🧠 Generating tailored summary...")
    
    # Try multiple approaches for better results
    for attempt in range(2):
        raw_summary = call_summary_model(summary_prompt, max_new_tokens=300)
        
        if not raw_summary:
            print(f"⚠️ Summary generation failed (attempt {attempt + 1})")
            continue
            
        # Clean the summary
        cleaned_summary = clean_summary_output(raw_summary)
        
        if cleaned_summary and len(cleaned_summary.split()) > 20:
            print("✅ Summary generated successfully")
            return cleaned_summary
        
        print(f"⚠️ Summary too short or empty (attempt {attempt + 1})")
    
    # Fallback summary
    fallback_summary = f"Experienced {job_title} with expertise in {', '.join(summary_keywords[:4])}. Proven track record of delivering data-driven insights and supporting business objectives through analytical problem-solving and effective communication."
    print("⚠️ Using fallback summary")
    return fallback_summary

def clean_summary_output(text: str) -> str:
    """Clean summary output from model"""
    # Remove common prefixes
    prefixes_to_remove = [
        r"^(Here is|Here's)\s+(a\s+)?(rewritten\s+)?summary:?\s*",
        r"^Summary:?\s*",
        r"^Rewritten:?\s*",
        r"^Based on.*?:\s*",
    ]
    
    for prefix in prefixes_to_remove:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)
    
    # Remove artifacts
    text = remove_jobscan_artifacts(text)
    
    # Clean up formatting
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    # Ensure it doesn't have bullet points
    if text.startswith('-') or '•' in text:
        # Convert bullets to paragraph
        lines = [line.strip('- •').strip() for line in text.split('\n') if line.strip()]
        text = ' '.join(lines)
    
    return text

def generate_work_experience_bullets(jobs: list, job_title: str, keywords: list, job_description: str) -> dict:
    """Generate improved work experience bullets for each job"""
    
    if not jobs:
        print("❌ No jobs found to rewrite")
        return {}
    
    # Create visualization tools list from keywords
    viz_tools = ["Power BI", "Tableau", "Looker", "Qlik", "Excel", "Spotfire", "Python", "R"]
    required_viz_tools = [tool for tool in viz_tools if any(tool.lower() in kw.lower() for kw in keywords)]
    
    print(f"🔍 Found {len(jobs)} jobs to rewrite")
    print(f"🎯 Required visualization tools: {required_viz_tools}")
    
    rewritten_jobs = {}
    
    for i, job in enumerate(jobs):
        print(f"\n📋 Rewriting job {i+1}: {job['header'][:50]}...")
        
        # Distribute keywords across jobs to avoid repetition
        job_keywords = keywords[i*3:(i+1)*3 + 6]  # Different slice per job
        
        # Include visualization tools strategically
        if required_viz_tools and i < len(required_viz_tools):
            job_keywords.extend(required_viz_tools[i:i+2])
        
        # Create bullet generation prompt
        bullet_prompt = f"""Rewrite these bullet points for the role of {job_title}:

JOB CONTEXT: {job['header']}

ORIGINAL BULLETS:
{chr(10).join(job['bullets'])}

REQUIREMENTS:
- Generate exactly 6 bullet points starting with "-"
- Include quantified achievements (percentages, numbers, time savings)
- Use action verbs: Developed, Analyzed, Managed, Created, Optimized, Led
- Naturally incorporate these keywords if relevant: {', '.join(job_keywords[:8])}
- Each bullet should be 1-2 lines maximum
- Make bullets specific to business analysis and data work

Generate 6 bullets now:"""

        # Generate bullets with retry logic
        for attempt in range(2):
            bullets_text = call_bullet_model(bullet_prompt, max_tokens=800)
            
            if bullets_text:
                cleaned_bullets = clean_bullet_output(bullets_text)
                bullet_lines = [line.strip() for line in cleaned_bullets.split('\n') 
                               if line.strip() and line.strip().startswith('-')]
                
                if len(bullet_lines) >= 4:  # At least 4 good bullets
                    # Ensure we have exactly 6 bullets
                    while len(bullet_lines) < 6:
                        bullet_lines.append(f"- Collaborated with cross-functional teams to deliver business solutions")
                    
                    rewritten_jobs[job['header']] = bullet_lines[:6]
                    print(f"✅ Generated {len(bullet_lines[:6])} bullets for job {i+1}")
                    break
                else:
                    print(f"⚠️ Only got {len(bullet_lines)} bullets, retrying...")
            else:
                print(f"⚠️ No bullets generated for job {i+1}, attempt {attempt + 1}")
        
        if job['header'] not in rewritten_jobs:
            print(f"❌ Failed to generate bullets for job {i+1}, using originals")
            rewritten_jobs[job['header']] = job['bullets'][:6]
    
    return rewritten_jobs

def clean_bullet_output(text: str) -> str:
    """Clean bullet point output from model"""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip meta-commentary
        skip_phrases = [
            'here are', 'bullet points', 'rewritten', 'bullets', 'experience',
            'responsibilities include', 'key achievements', 'accomplishments'
        ]
        if any(phrase in line.lower() for phrase in skip_phrases):
            continue
        
        # Ensure proper bullet format
        if not line.startswith('-'):
            if len(line) > 15 and any(verb in line.lower()[:30] for verb in [
                'developed', 'analyzed', 'managed', 'created', 'led', 'optimized', 'implemented'
            ]):
                line = f"- {line}"
            else:
                continue
        
        # Clean the line
        line = line.replace('•', '-').replace('*', '-')
        
        # Ensure proper capitalization
        if line.startswith('- ') and len(line) > 2:
            content = line[2:].strip()
            if content and content[0].islower():
                line = f"- {content[0].upper()}{content[1:]}"
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def rebuild_resume_sections(sections: dict, job_title: str, all_keywords: list, rewritten_jobs: dict) -> list:
    """Rebuild resume with all sections in proper order"""
    
    rebuilt_lines = [DEFAULT_HEADER, ""]
    
    # SUMMARY section
    rebuilt_lines.extend(["SUMMARY", ""])
    if "SUMMARY" in sections:
        keyword_groups = create_keyword_groups(all_keywords)
        tailored_summary = generate_tailored_summary(sections["SUMMARY"], job_title, keyword_groups)
        rebuilt_lines.extend([tailored_summary, ""])
    else:
        rebuilt_lines.extend([f"Experienced {job_title} with strong analytical and communication skills.", ""])
    
    # SKILLS section
    rebuilt_lines.extend(["SKILLS", ""])
    if "SKILLS" in sections and sections["SKILLS"].strip():
        rebuilt_lines.extend([sections["SKILLS"], ""])
    else:
        # Generate basic skills if missing
        basic_skills = "• SQL, Excel, Data Analysis, Business Intelligence\n• Communication, Problem Solving, Project Management"
        rebuilt_lines.extend([basic_skills, ""])
    
    # WORK EXPERIENCE section
    rebuilt_lines.extend(["WORK EXPERIENCE", ""])
    if rewritten_jobs:
        for job_header, bullets in rewritten_jobs.items():
            rebuilt_lines.append(job_header)
            rebuilt_lines.extend(bullets)
            rebuilt_lines.append("")  # Spacing between jobs
    elif "WORK EXPERIENCE" in sections:
        # Fallback to original if rewriting failed
        rebuilt_lines.extend([sections["WORK EXPERIENCE"], ""])
    
    # CERTIFICATIONS section
    rebuilt_lines.extend(["CERTIFICATIONS", ""])
    if "CERTIFICATIONS" in sections and sections["CERTIFICATIONS"].strip():
        rebuilt_lines.extend([sections["CERTIFICATIONS"], ""])
    else:
        rebuilt_lines.extend([DEFAULT_CERTIFICATIONS, ""])
    
    # EDUCATION section
    rebuilt_lines.extend(["EDUCATION", ""])
    if "EDUCATION" in sections and sections["EDUCATION"].strip():
        # Check if education looks complete
        edu_text = sections["EDUCATION"]
        if "master" in edu_text.lower() and "bachelor" in edu_text.lower():
            rebuilt_lines.extend([edu_text, ""])
        else:
            # Use default if incomplete
            rebuilt_lines.extend([DEFAULT_EDUCATION, ""])
    else:
        rebuilt_lines.extend([DEFAULT_EDUCATION, ""])
    
    return rebuilt_lines

def tailor_resume_with_models(job_title: str, job_description: str, base_resume_path: str = "data/YoshithaM_Resume_W2.docx", 
                             ats_keywords: list = None, ats_sections: list = None) -> str:
    """
    Main function to tailor resume using AI models with improved error handling and output quality.
    """
    
    print(f"\n🎯 Starting resume tailoring for: {job_title}")
    
    # Load base resume
    base_text = load_docx_text(base_resume_path)
    if not base_text:
        print("❌ Failed to load base resume")
        return ""
    
    # Parse sections
    sections = parse_sections(base_text)
    print(f"📦 Extracted sections: {list(sections.keys())}")
    
    # Extract job information
    try:
        job_info = extract_job_info(f"Job Title: {job_title}\n\n{job_description}")
        print("🔍 Job info extracted successfully")
    except Exception as e:
        print(f"⚠️ Job info extraction failed: {e}")
        job_info = {"job_title": job_title, "skills": [], "verbs": []}
    
    # Build comprehensive keyword list
    all_keywords = build_keyword_list(job_info, ats_keywords, job_description)
    print(f"🔑 Total keywords for tailoring: {len(all_keywords)}")
    
    # Parse and rewrite work experience
    rewritten_jobs = {}
    if "WORK EXPERIENCE" in sections:
        try:
            jobs = parse_work_experience_section(sections["WORK EXPERIENCE"])
            if jobs:
                rewritten_jobs = generate_work_experience_bullets(jobs, job_title, all_keywords, job_description)
                print(f"✅ Rewritten {len(rewritten_jobs)} job experiences")
            else:
                print("⚠️ No jobs found in work experience section")
        except Exception as e:
            print(f"❌ Work experience rewriting failed: {e}")
    
    # Rebuild complete resume
    try:
        rebuilt_lines = rebuild_resume_sections(sections, job_title, all_keywords, rewritten_jobs)
        final_text = "\n".join(rebuilt_lines)
        
        # Apply cleaning and validation
        final_text = clean_full_resume(final_text, base_text, verbose=True)
        
        # Validate final output
        validation = validate_resume_structure(final_text)
        if not validation["valid"]:
            print(f"⚠️ Resume validation failed: missing {validation['missing_sections']}")
            # Add missing sections
            final_text = add_missing_sections(final_text, validation["missing_sections"])
        
        print(f"✅ Resume tailoring completed. Final length: {len(final_text)} characters")
        return final_text
        
    except Exception as e:
        print(f"❌ Resume rebuilding failed: {e}")
        return ""

def build_keyword_list(job_info: dict, ats_keywords: list, job_description: str) -> list:
    """Build comprehensive keyword list from multiple sources"""
    
    all_keywords = []
    seen = set()
    
    # Add ATS keywords first (highest priority)
    if ats_keywords:
        for kw in ats_keywords:
            kw = kw.strip()
            if kw and kw.lower() not in seen:
                all_keywords.append(kw)
                seen.add(kw.lower())
    
    # Add job info keywords
    for source_list in [job_info.get("skills", []), job_info.get("verbs", [])]:
        for kw in source_list:
            kw = str(kw).strip()
            if kw and kw.lower() not in seen:
                all_keywords.append(kw)
                seen.add(kw.lower())
    
    # Add relevant soft skills if mentioned in JD
    for skill in SOFT_SKILLS:
        if skill.lower() in job_description.lower() and skill not in seen:
            all_keywords.append(skill.title())
            seen.add(skill.lower())
    
    return all_keywords[:20]  # Limit total keywords

def add_missing_sections(text: str, missing_sections: list) -> str:
    """Add any missing required sections to the resume"""
    
    lines = text.splitlines()
    
    for section in missing_sections:
        if section == "CERTIFICATIONS":
            lines.extend(["", "CERTIFICATIONS", "", DEFAULT_CERTIFICATIONS])
        elif section == "EDUCATION":
            lines.extend(["", "EDUCATION", "", DEFAULT_EDUCATION])
        elif section == "SUMMARY":
            lines.insert(2, "SUMMARY")
            lines.insert(3, "")
            lines.insert(4, "Business Analyst with experience in data analysis and process improvement.")
        elif section == "SKILLS":
            # Find where to insert skills (after summary)
            insert_pos = 6  # Default position
            lines.insert(insert_pos, "")
            lines.insert(insert_pos + 1, "SKILLS")
            lines.insert(insert_pos + 2, "")
            lines.insert(insert_pos + 3, "• Data Analysis, SQL, Excel\n• Communication, Problem Solving")
    
    return "\n".join(lines)

def save_to_docx(text: str, path: str) -> bool:
    """Save resume text to DOCX file with proper formatting"""
    try:
        doc = Document()
        VALID_HEADERS = ["SUMMARY", "SKILLS", "WORK EXPERIENCE", "CERTIFICATIONS", "EDUCATION"]
        
        for line in text.splitlines():
            clean_line = line.strip()
            if not clean_line:
                doc.add_paragraph("")
                continue
            
            # Check if this is a section header
            is_header = False
            for header in VALID_HEADERS:
                if clean_line.upper() == header:
                    para = doc.add_paragraph()
                    run = para.add_run(header)
                    run.bold = True
                    para.paragraph_format.space_after = Pt(6)
                    is_header = True
                    break
            
            if not is_header:
                # Regular content
                clean_line = clean_line.replace("•", "-").replace("·", "-")
                clean_line = clean_line.encode("utf-8", "ignore").decode("utf-8")
                
                para = doc.add_paragraph(clean_line)
                para.paragraph_format.space_after = Pt(3)
        
        doc.save(path)
        print(f"💾 Resume saved to: {path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save DOCX file: {e}")
        return False

def validate_final_output(text: str) -> dict:
    """Validate the final resume output meets quality standards"""
    
    issues = []
    warnings = []
    
    # Check length
    if len(text) < 1000:
        issues.append("Resume too short")
    elif len(text) > 25000:
        warnings.append("Resume very long")
    
    # Check sections
    required_sections = ["SUMMARY", "WORK EXPERIENCE", "EDUCATION"]
    for section in required_sections:
        if section not in text:
            issues.append(f"Missing {section} section")
    
    # Check work experience
    if "WORK EXPERIENCE" in text:
        job_headers = len([line for line in text.splitlines() 
                          if "–" in line and re.search(r'\b\d{4}\b', line)])
        if job_headers < 2:
            warnings.append("Less than 2 job experiences found")
    
    # Check for placeholder content
    placeholders = ["[", "TBD", "TODO", "Additional relevant responsibility"]
    for placeholder in placeholders:
        if placeholder in text:
            warnings.append(f"Contains placeholder content: {placeholder}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "character_count": len(text),
        "line_count": len(text.splitlines())
    }