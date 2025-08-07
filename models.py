import os
import requests
import json
import time
import re
from typing import Optional

# --------------------- Remote Bullet Model (Together.ai) --------------------- #
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 characters per token"""
    return len(text) // 4

def call_summary_model(prompt: str, max_new_tokens: int = 512) -> str:
    """Use Groq for summary generation with better error handling"""
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY not set.")
    
    # Ensure prompt isn't too long
    if estimate_tokens(prompt) > 6000:  # Leave room for response
        print("⚠️ Summary prompt too long, truncating...")
        prompt = prompt[:20000]  # Rough character limit
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system", 
                "content": "You are a professional resume writer. Generate only the rewritten content requested, no explanations or meta-text. Start directly with the summary content."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": max_new_tokens
    }
    
    for attempt in range(3):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions", 
                json=payload, 
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Groq API failed: {response.status_code} – {response.text}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return ""
            
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"].strip()
            
            # Clean up common prompt leakage patterns
            content = clean_model_output(content)
            
            if not content:
                print("⚠️ Empty response from Groq, retrying...")
                if attempt < 2:
                    time.sleep(1)
                    continue
                    
            return content
            
        except Exception as e:
            print(f"❌ Groq API call failed (attempt {attempt + 1}): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return ""
    
    return ""

def call_bullet_model(prompt: str, max_tokens: int = 1024) -> str:
    """Improved Together.ai call with better formatting and error handling"""
    if not TOGETHER_API_KEY:
        raise EnvironmentError("TOGETHER_API_KEY not set.")

    # Add structured formatting instructions to prompt
    structured_prompt = f"""{prompt}

IMPORTANT FORMATTING RULES:
- Generate ONLY bullet points starting with "-"
- Each bullet should be 1-2 lines maximum
- Include quantified achievements where possible
- Do not include any explanatory text, headers, or job titles
- Stop generation after the bullet points
- Each bullet should start with a strong action verb

Generate the bullet points now:"""

    for attempt in range(3):
        try:
            response = requests.post(
                "https://api.together.xyz/inference",
                headers={"Authorization": f"Bearer {TOGETHER_API_KEY}"},
                json={
                    "model": TOGETHER_MODEL,
                    "prompt": structured_prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.6,  # Slightly lower for more consistency
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,  # Reduce repetition
                    "stop": ["###", "---", "\n\n\n", "SUMMARY", "SKILLS", "EDUCATION", "CERTIFICATIONS"],
                },
                timeout=60
            )

            print(f"🧪 Together.ai response status: {response.status_code}")

            if response.status_code != 200:
                print(f"❌ Together.ai API failed: {response.status_code} – {response.text}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return ""

            response_json = response.json()
            choices = response_json.get("choices", [])

            if not choices or "text" not in choices[0]:
                print("❌ Together.ai response missing expected text field.")
                if attempt < 2:
                    time.sleep(1)
                    continue
                return ""

            output = choices[0]["text"].strip()
            
            # Clean and validate output
            output = clean_bullet_output(output)
            
            if not output or len(output.split('\n')) < 3:  # Ensure we got multiple bullets
                print("⚠️ Insufficient bullet points generated, retrying...")
                if attempt < 2:
                    time.sleep(1)
                    continue
                    
            print("🔎 Bullet model response (preview):", output[:300])
            return output

        except Exception as e:
            print(f"❌ Exception during Together.ai call (attempt {attempt + 1}): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return ""

    return ""

def clean_model_output(text: str) -> str:
    """Clean common issues in model outputs"""
    # Remove common prompt leakage patterns
    leakage_patterns = [
        r"^(You are|Rewrite|Original Summary|Rewritten Summary|Here is|Here's)\b.*?[:]\s*",
        r"^(Based on|Given|According to)\b.*?[:]\s*",
        r"^\*\*.*?\*\*:?\s*",  # Bold formatting
        r"^#+\s*.*?[:]\s*",     # Headers
        r"^Summary:?\s*",       # Summary: prefix
        r"^Rewritten:?\s*",     # Rewritten: prefix
    ]
    
    for pattern in leakage_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize line breaks
    text = text.strip()
    
    return text

def clean_bullet_output(text: str) -> str:
    """Clean and format bullet point output"""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip meta-text and headers
        skip_phrases = [
            'here are', 'bullet points', 'responsibilities', 'experience includes',
            'worked on', 'responsible for', 'job duties', 'position involved',
            'key achievements', 'accomplishments', 'duties include'
        ]
        
        if any(skip in line.lower() for skip in skip_phrases):
            continue
            
        # Skip job headers (lines with company names and dates)
        if '–' in line and re.search(r'\b\d{4}\b', line):
            continue
            
        # Ensure bullet format
        if not line.startswith('-') and not line.startswith('•'):
            # Only add bullet if it looks like a responsibility/achievement
            action_verbs = [
                'developed', 'managed', 'created', 'led', 'implemented', 
                'analyzed', 'designed', 'optimized', 'collaborated', 'built',
                'delivered', 'improved', 'executed', 'coordinated', 'facilitated'
            ]
            if (len(line) > 15 and 
                any(verb in line.lower() for verb in action_verbs) and
                not line.lower().startswith(tuple(['the ', 'a ', 'an ']))):
                line = f"- {line}"
            else:
                continue
                
        # Normalize bullet character
        line = line.replace('•', '-').replace('*', '-')
        
        # Ensure proper capitalization
        if line.startswith('- ') and len(line) > 2:
            line = '- ' + line[2:].strip()
            if line[2:3].islower():
                line = line[:2] + line[2:3].upper() + line[3:]
        
        # Skip duplicate or very similar lines
        if not any(similar_line(line, existing) for existing in cleaned_lines):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def similar_line(line1: str, line2: str) -> bool:
    """Check if two lines are too similar"""
    # Simple similarity check based on common words
    words1 = set(line1.lower().split())
    words2 = set(line2.lower().split())
    
    if len(words1) == 0 or len(words2) == 0:
        return False
        
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    similarity = intersection / union if union > 0 else 0
    return similarity > 0.7  # 70% similarity threshold

def validate_model_output(output: str, expected_type: str = "bullets") -> bool:
    """Validate that model output meets basic requirements"""
    if not output or len(output.strip()) < 20:
        return False
        
    if expected_type == "bullets":
        # Should have at least 3 bullet points
        bullet_count = len([line for line in output.split('\n') if line.strip().startswith('-')])
        return bullet_count >= 3
    elif expected_type == "summary":
        # Should be 2-6 sentences, no bullets
        sentences = len([s for s in output.split('.') if s.strip()])
        has_bullets = any(line.strip().startswith('-') for line in output.split('\n'))
        return 2 <= sentences <= 6 and not has_bullets
    
    return True

def call_groq_direct(prompt: str, max_tokens: int = 1024) -> str:
    """Direct Groq call for general purpose use"""
    if not GROQ_API_KEY:
        return ""
        
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions", 
            json=payload, 
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Groq API call failed: {e}")
        return ""