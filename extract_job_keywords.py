import re
import json
import os
import requests
from typing import Dict, List
import time

def call_groq(prompt: str, max_retries: int = 3) -> str:
    """Improved Groq API call with better error handling"""
    headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}
    
    # Use structured message format for better results
    payload = {
        "model": os.getenv("GROQ_MODEL", "llama3-70b-8192"),
        "messages": [
            {
                "role": "system",
                "content": "You are a job description analyzer. Always respond with valid JSON only, no explanations or additional text."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.2,  # Lower temperature for more consistent JSON
        "max_tokens": 1000
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                json=payload, 
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt
                print(f"⏳ Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            
            # Clean any markdown formatting
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            
            # Validate it's proper JSON
            try:
                json.loads(content)
                return content
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON response (attempt {attempt + 1}): {content[:200]}")
                if attempt < max_retries - 1:
                    continue
                    
        except Exception as e:
            print(f"❌ Groq API call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return ""

def extract_job_info(job_text: str) -> Dict[str, any]:
    """Enhanced job information extraction with better fallbacks"""
    
    # Pre-validation
    if len(job_text.strip()) < 200:
        print("⚠️ Job description too short. Using minimal extraction.")
        return create_minimal_job_info(job_text)

    # Clean job text for better processing
    cleaned_text = clean_job_description(job_text)
    
    # Improved prompt with better structure and examples
    prompt = f"""Extract information from this job description and respond with valid JSON only.

Required JSON format:
{{
    "job_title": "exact job title from description",
    "skills": ["SQL", "Python", "Data Analysis", "Excel", "Tableau", "Communication"],
    "verbs": ["Analyze", "Develop", "Create", "Manage", "Lead", "Design"]
}}

Guidelines:
- job_title: Use the most specific job title mentioned (e.g., "Senior Business Analyst" not just "Analyst")
- skills: Include 6-8 technical and soft skills mentioned in the description
- verbs: Extract 6-8 action words that describe key responsibilities
- Use proper title case for all items
- Focus on skills and verbs that appear multiple times or are emphasized

Job Description:
{cleaned_text[:3000]}"""

    raw_output = call_groq(prompt)

    if os.getenv("DEBUG_RESUME") == "1":
        print("🔎 Raw model output:\n", raw_output[:500])

    # Try to parse the JSON response
    try:
        # Extract JSON from response (handle cases where model adds explanation)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_output, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON structure found")

        parsed = json.loads(json_match.group(0))
        
        # Validate and clean the parsed data
        result = validate_and_clean_extraction(parsed, cleaned_text)
        
        print("✅ LLM successfully parsed job info.")
        print(f"📋 Extracted: {result['job_title']}")
        print(f"🔧 Skills: {len(result['skills'])} items")
        print(f"⚡ Verbs: {len(result['verbs'])} items")
        
        return result
        
    except Exception as e:
        print(f"⚠️ LLM failed to parse JSON: {e}")
        print("🔁 Falling back to rule-based parsing...")
        
        return fallback_rule_based_extraction(cleaned_text)

def clean_job_description(text: str) -> str:
    """Clean and normalize job description text"""
    # Remove excessive whitespace and formatting
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common junk patterns
    junk_patterns = [
        r'Apply now.*?$',
        r'Equal opportunity employer.*?$',
        r'We are an equal.*?$',
        r'EOE.*?$',
        r'Click here to apply.*?$',
        r'To apply.*?$',
        r'Send resume.*?$',
    ]
    
    for pattern in junk_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    return text.strip()

def validate_and_clean_extraction(parsed: Dict, original_text: str) -> Dict[str, any]:
    """Validate and clean extracted job information"""
    
    # Clean job title
    job_title = parsed.get("job_title", "").strip()
    if not job_title or len(job_title) < 3:
        job_title = extract_title_fallback(original_text)
    
    # Remove common prefixes/suffixes that don't add value
    title_cleanups = [
        r'^(Job Title:?|Position:?|Role:?)\s*',
        r'\s*-\s*(Full Time|Part Time|Contract).*$',
        r'\s*\(.*\)$',  # Remove parenthetical info
    ]
    for cleanup in title_cleanups:
        job_title = re.sub(cleanup, '', job_title, flags=re.IGNORECASE)
    
    job_title = job_title.strip()
    
    # Clean and validate skills
    skills = parsed.get("skills", [])
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(',')]
    
    cleaned_skills = []
    skill_keywords = set()
    
    for skill in skills[:12]:  # Limit to 12 initially
        skill = str(skill).strip().title()
        skill_lower = skill.lower()
        
        # Skip invalid skills
        if (len(skill) < 2 or len(skill) > 40 or 
            skill_lower in skill_keywords or
            skill_lower in ['the', 'and', 'with', 'for', 'this', 'that', 'from', 'will']):
            continue
            
        cleaned_skills.append(skill)
        skill_keywords.add(skill_lower)
        
        if len(cleaned_skills) >= 8:  # Final limit
            break
    
    # Clean and validate verbs
    verbs = parsed.get("verbs", [])
    if isinstance(verbs, str):
        verbs = [v.strip() for v in verbs.split(',')]
    
    cleaned_verbs = []
    verb_keywords = set()
    
    for verb in verbs[:12]:  # Limit to 12 initially
        verb = str(verb).strip().title()
        verb_lower = verb.lower()
        
        # Ensure it's actually a verb-like word
        if (len(verb) < 3 or len(verb) > 25 or 
            verb_lower in verb_keywords or
            verb_lower in ['the', 'and', 'with', 'for', 'this', 'that', 'will', 'can', 'may']):
            continue
            
        # Remove common verb suffixes for deduplication
        base_verb = re.sub(r'(ing|ed|es|s)$', '', verb_lower)
        if base_verb not in verb_keywords:
            cleaned_verbs.append(verb)
            verb_keywords.add(verb_lower)
            verb_keywords.add(base_verb)
            
        if len(cleaned_verbs) >= 8:  # Final limit
            break
    
    # Ensure minimum counts with fallbacks
    if len(cleaned_skills) < 4:
        fallback_skills = extract_skills_fallback(original_text)
        for skill in fallback_skills:
            if skill.lower() not in skill_keywords and len(cleaned_skills) < 6:
                cleaned_skills.append(skill)
                skill_keywords.add(skill.lower())
    
    if len(cleaned_verbs) < 4:
        fallback_verbs = extract_verbs_fallback(original_text)
        for verb in fallback_verbs:
            base_verb = re.sub(r'(ing|ed|es|s)$', '', verb.lower())
            if base_verb not in verb_keywords and len(cleaned_verbs) < 6:
                cleaned_verbs.append(verb)
                verb_keywords.add(verb.lower())
                verb_keywords.add(base_verb)
    
    return {
        "job_title": job_title or "Business Analyst",
        "skills": cleaned_skills[:8],
        "verbs": cleaned_verbs[:8]
    }

def extract_title_fallback(text: str) -> str:
    """Fallback method to extract job title"""
    lines = text.split('\n')[:15]  # Check first 15 lines
    
    # Look for common title patterns
    title_patterns = [
        r'(?:Job Title|Position|Role)\s*:?\s*(.+)',
        r'^(.+?(?:Analyst|Engineer|Manager|Specialist|Coordinator|Consultant|Developer|Scientist|Associate|Director).*)$'
    ]
    
    for line in lines:
        line = line.strip()
        if len(line) < 5 or len(line) > 100:
            continue
            
        for pattern in title_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                # Clean the title
                title = re.sub(r'[^\w\s-]', '', title).strip()
                if len(title) > 5:
                    return title
    
    # Fallback to looking for job-related keywords
    job_keywords = ['analyst', 'engineer', 'manager', 'specialist', 'coordinator',
                   'consultant', 'developer', 'scientist', 'associate', 'director']
    
    for line in lines:
        line = line.strip()
        if (5 < len(line) < 100 and 
            any(keyword in line.lower() for keyword in job_keywords)):
            # Clean the title
            title = re.sub(r'[^\w\s-]', '', line).strip()
            if title:
                return title
    
    return "Business Analyst"

def extract_skills_fallback(text: str) -> List[str]:
    """Rule-based fallback for skill extraction"""
    # Comprehensive skills pattern
    technical_skills = [
        'SQL', 'Python', 'R', 'Excel', 'Tableau', 'Power BI', 'PowerBI', 'SAS', 'SPSS', 
        'Hadoop', 'Spark', 'ETL', 'AWS', 'Azure', 'GCP', 'Salesforce', 'SAP',
        'Machine Learning', 'Data Mining', 'Statistics', 'Analytics', 'BI',
        'Business Intelligence', 'Data Visualization', 'Looker', 'Qlik', 'Alteryx'
    ]
    
    soft_skills = [
        'Project Management', 'Agile', 'Scrum', 'Communication', 'Leadership',
        'Problem Solving', 'Critical Thinking', 'Teamwork', 'Collaboration',
        'Presentation', 'Documentation', 'Time Management'
    ]
    
    all_skills = technical_skills + soft_skills
    found_skills = []
    
    for skill in all_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            found_skills.append(skill)
            if len(found_skills) >= 8:
                break
    
    return found_skills

def extract_verbs_fallback(text: str) -> List[str]:
    """Rule-based fallback for verb extraction"""
    common_verbs = [
        'Analyze', 'Develop', 'Create', 'Manage', 'Lead', 'Design', 'Implement', 
        'Build', 'Optimize', 'Collaborate', 'Support', 'Maintain', 'Deliver', 
        'Execute', 'Monitor', 'Research', 'Document', 'Present', 'Coordinate', 
        'Facilitate', 'Transform', 'Improve', 'Drive', 'Enable', 'Establish'
    ]
    
    found_verbs = []
    
    for verb in common_verbs:
        # Look for the verb and its variations
        pattern = r'\b' + re.escape(verb) + r'(?:s|d|ing)?\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_verbs.append(verb)
            if len(found_verbs) >= 8:
                break
    
    return found_verbs

def fallback_rule_based_extraction(job_text: str) -> Dict[str, any]:
    """Complete fallback when LLM fails"""
    print("🛠️ Using rule-based extraction...")
    
    title = extract_title_fallback(job_text)
    skills = extract_skills_fallback(job_text)
    verbs = extract_verbs_fallback(job_text)
    
    # Ensure minimum viable results
    if len(skills) < 4:
        default_skills = ["Data Analysis", "Excel", "Communication", "Problem Solving"]
        skills.extend([s for s in default_skills if s not in skills])
    
    if len(verbs) < 4:
        default_verbs = ["Analyze", "Develop", "Collaborate", "Support"]
        verbs.extend([v for v in default_verbs if v not in verbs])
    
    return {
        "job_title": title,
        "skills": skills[:8],
        "verbs": verbs[:8]
    }

def create_minimal_job_info(job_text: str) -> Dict[str, any]:
    """Create minimal job info for very short descriptions"""
    # Try to extract at least the title from short text
    title = "Business Analyst"
    if len(job_text) > 10:
        potential_title = extract_title_fallback(job_text)
        if potential_title and potential_title != "Business Analyst":
            title = potential_title
    
    return {
        "job_title": title,
        "skills": ["Data Analysis", "Excel", "SQL", "Communication", "Problem Solving", "Teamwork"],
        "verbs": ["Analyze", "Develop", "Support", "Create", "Collaborate", "Improve"]
    }