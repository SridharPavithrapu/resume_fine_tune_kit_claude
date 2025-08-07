import os
import json
import pandas as pd
import undetected_chromedriver as uc
from nltk.stem import PorterStemmer
import re
from typing import List, Dict, Optional
import time

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- Driver Management -------------------- #
def launch_driver(chromedriver_path: str = "/opt/homebrew/bin/chromedriver") -> Optional[uc.Chrome]:
    """Launch Chrome driver with enhanced configuration and error handling"""
    
    try:
        options = uc.ChromeOptions()
        
        # Enhanced options for stability
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Headless mode based on environment
        headless_mode = os.getenv("CHROME_HEADLESS", "false").lower() == "true"
        if headless_mode:
            options.add_argument("--headless")
        
        # User agent to avoid detection
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Try to create driver
        driver = uc.Chrome(options=options, driver_executable_path=chromedriver_path)
        
        # Execute script to avoid detection
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        print("✅ Chrome driver launched successfully")
        return driver
        
    except Exception as e:
        print(f"❌ Failed to launch Chrome driver: {e}")
        return None

def close_driver_safely(driver: uc.Chrome) -> None:
    """Safely close Chrome driver"""
    try:
        if driver:
            driver.quit()
            print("✅ Chrome driver closed")
    except Exception as e:
        print(f"⚠️ Error closing driver: {e}")

# -------------------- Data Loading -------------------- #
def load_job_descriptions(file_path: str) -> pd.DataFrame:
    """Enhanced job description loading with validation"""
    try:
        if not os.path.exists(file_path):
            print(f"❌ Job file not found: {file_path}")
            return pd.DataFrame()
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✅ Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print(f"❌ Could not read file with any encoding")
            return pd.DataFrame()
        
        # Validate and clean data
        initial_count = len(df)
        
        # Check for required columns
        required_columns = ['title', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
            # Try to find similar columns
            for missing_col in missing_columns:
                similar_cols = [col for col in df.columns if missing_col.lower() in col.lower()]
                if similar_cols:
                    df[missing_col] = df[similar_cols[0]]
                    print(f"✅ Mapped {similar_cols[0]} to {missing_col}")
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['title', 'description'])
        
        # Clean and validate job descriptions
        df['description'] = df['description'].astype(str)
        df['title'] = df['title'].astype(str)
        
        # Filter out very short descriptions
        df = df[df['description'].str.len() > 50]
        
        # Remove duplicate jobs (by title and partial description)
        df = df.drop_duplicates(subset=['title', 'description'], keep='first')
        
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"⚠️ Filtered from {initial_count} to {final_count} jobs")
        
        print(f"✅ Loaded {final_count} valid job descriptions")
        return df
        
    except Exception as e:
        print(f"❌ Error loading job descriptions: {e}")
        return pd.DataFrame()

# -------------------- Keyword Processing -------------------- #
def normalize_keywords(keywords: List[str]) -> List[str]:
    """Enhanced keyword normalization with better cleaning"""
    if not keywords:
        return []
    
    try:
        ps = PorterStemmer()
        normalized = []
        seen = set()
        
        for keyword in keywords:
            if not keyword or not isinstance(keyword, str):
                continue
            
            # Clean the keyword
            cleaned = keyword.strip().lower()
            cleaned = re.sub(r'[^\w\s-]', '', cleaned)  # Remove special chars except hyphens
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
            
            if len(cleaned) < 2 or len(cleaned) > 50:  # Skip very short/long keywords
                continue
            
            # Stem the keyword
            try:
                stemmed = ps.stem(cleaned)
                if stemmed not in seen and len(stemmed) > 1:
                    normalized.append(keyword.strip())  # Keep original case
                    seen.add(stemmed)
            except Exception:
                # If stemming fails, use original cleaned version
                if cleaned not in seen:
                    normalized.append(keyword.strip())
                    seen.add(cleaned)
        
        return normalized
        
    except Exception as e:
        print(f"⚠️ Error normalizing keywords: {e}")
        return [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()]

def extract_keywords_from_text(text: str, max_keywords: int = 20) -> List[str]:
    """Extract relevant keywords from text using pattern matching"""
    
    if not text or len(text) < 10:
        return []
    
    # Common business analysis and data keywords
    keyword_patterns = [
        # Technical skills
        r'\b(SQL|Python|R|Excel|Tableau|Power BI|PowerBI|SAS|SPSS|Hadoop|Spark)\b',
        r'\b(ETL|AWS|Azure|GCP|Salesforce|SAP|Oracle|MySQL|PostgreSQL)\b',
        r'\b(Machine Learning|Data Mining|Statistics|Analytics|BI)\b',
        r'\b(Business Intelligence|Data Visualization|Reporting|Dashboard)\b',
        
        # Soft skills and methodologies
        r'\b(Agile|Scrum|Project Management|Communication|Leadership)\b',
        r'\b(Problem Solving|Critical Thinking|Collaboration|Teamwork)\b',
        r'\b(Requirements Gathering|Process Improvement|Documentation)\b',
        
        # Business terms
        r'\b(KPI|ROI|Stakeholder|Revenue|Budget|Forecast|Strategy)\b',
        r'\b(Analysis|Research|Optimization|Implementation|Validation)\b'
    ]
    
    found_keywords = []
    
    for pattern in keyword_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found_keywords.extend(matches)
    
    # Remove duplicates while preserving order and original case
    unique_keywords = []
    seen_lower = set()
    
    for keyword in found_keywords:
        keyword_lower = keyword.lower()
        if keyword_lower not in seen_lower:
            unique_keywords.append(keyword)
            seen_lower.add(keyword_lower)
    
    return unique_keywords[:max_keywords]

# -------------------- File Operations -------------------- #
def save_tailored_attempts(job_title: str, jd_text: str, resumes_with_scores: List[tuple]) -> str:
    """Enhanced resume saving with better organization"""
    
    # Create safe directory name
    slug = create_safe_filename(job_title)
    folder = os.path.join(OUTPUT_DIR, slug)
    os.makedirs(folder, exist_ok=True)
    
    try:
        # Save job description
        jd_path = os.path.join(folder, "job_description.txt")
        with open(jd_path, "w", encoding="utf-8") as f:
            f.write(f"Job Title: {job_title}\n\n")
            f.write("="*50 + "\n\n")
            f.write(jd_text)
        
        # Save resume attempts
        for i, (score, resume_text) in enumerate(resumes_with_scores):
            resume_path = os.path.join(folder, f"attempt_{i+1}_score_{score}.txt")
            with open(resume_path, "w", encoding="utf-8") as f:
                f.write(resume_text)
        
        # Create summary file
        summary_path = os.path.join(folder, "summary.json")
        summary = {
            "job_title": job_title,
            "total_attempts": len(resumes_with_scores),
            "scores": [score for score, _ in resumes_with_scores],
            "best_score": max([score for score, _ in resumes_with_scores]) if resumes_with_scores else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 All attempts saved to: {folder}")
        return folder
        
    except Exception as e:
        print(f"❌ Error saving attempts: {e}")
        return ""

def save_json_log(prompt: str, output: str, job_title: str, metadata: Dict = None) -> None:
    """Enhanced JSON logging with metadata"""
    
    slug = create_safe_filename(job_title)
    folder = os.path.join(OUTPUT_DIR, slug)
    os.makedirs(folder, exist_ok=True)
    
    try:
        log_data = {
            "job_title": job_title,
            "prompt": prompt,
            "output": output,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_length": len(prompt),
            "output_length": len(output)
        }
        
        if metadata:
            log_data["metadata"] = metadata
        
        log_path = os.path.join(folder, "processing_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"⚠️ Failed to save JSON log: {e}")

def create_safe_filename(text: str, max_length: int = 50) -> str:
    """Create filesystem-safe filename from text"""
    
    if not text:
        return "unnamed"
    
    # Replace problematic characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    filename = ""
    
    for char in text:
        if char in safe_chars:
            filename += char
        elif char == " ":
            filename += "_"
        # Skip other characters
    
    # Clean up multiple underscores
    filename = re.sub(r'_+', '_', filename)
    filename = filename.strip('_')
    
    # Ensure not empty and limit length
    if not filename:
        filename = "unnamed"
    
    return filename[:max_length]

# -------------------- Job Title Extraction -------------------- #
def extract_job_title(jd_text: str, csv_title: str = None) -> str:
    """
    Enhanced job title extraction with multiple strategies and validation.
    """
    
    # Strategy 1: Use CSV title if it's high quality
    if csv_title:
        cleaned_csv_title = csv_title.strip()
        
        # Quality checks for CSV title
        if (5 <= len(cleaned_csv_title) <= 100 and
            not cleaned_csv_title.lower().startswith(("requisition", "job id", "posting")) and
            not re.search(r'^\d+', cleaned_csv_title) and  # Doesn't start with number
            any(keyword in cleaned_csv_title.lower() for keyword in [
                "analyst", "engineer", "manager", "specialist", "consultant", 
                "coordinator", "developer", "scientist", "associate", "director"
            ])):
            
            print(f"✅ Using CSV title: {cleaned_csv_title}")
            return cleaned_csv_title
    
    print("🔍 Extracting title from job description...")
    
    # Strategy 2: Look for explicit job title labels
    title_patterns = [
        r"(?:Job Title|Position Title|Role|Title)\s*[:\-–]\s*(.+?)(?:\n|$)",
        r"(?:Position|Role)\s*[:\-–]\s*(.+?)(?:\n|$)",
        r"We are (?:looking for|seeking)\s+(?:a|an)\s+(.+?)(?:\s+to|\s+who|\n)",
        r"Join (?:us|our team) as\s+(?:a|an)\s+(.+?)(?:\s+to|\s+and|\n)"
    ]
    
    for pattern in title_patterns:
        matches = re.finditer(pattern, jd_text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            title = match.group(1).strip()
            if validate_job_title(title):
                print(f"✅ Found labeled title: {title}")
                return clean_job_title(title)
    
    # Strategy 3: Look for bold/formatted titles (common in job descriptions)
    formatting_patterns = [
        r"\*\*(.+?)\*\*",  # **Bold**
        r"__(.+?)__",      # __Bold__
        r"<b>(.+?)</b>",   # HTML bold
    ]
    
    for pattern in formatting_patterns:
        matches = re.finditer(pattern, jd_text, re.IGNORECASE)
        for match in matches:
            title = match.group(1).strip()
            if validate_job_title(title) and not title.endswith(":"):
                print(f"✅ Found formatted title: {title}")
                return clean_job_title(title)
    
    # Strategy 4: Analyze first few lines for title-like content
    lines = jd_text.strip().splitlines()[:15]  # Check first 15 lines
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip very short or very long lines
        if len(line) < 5 or len(line) > 150:
            continue
        
        # Skip lines that end with colons (likely headers)
        if line.endswith(":"):
            continue
        
        # Look for job-related keywords
        if validate_job_title(line):
            # Give priority to lines that appear early and contain job keywords
            job_keywords = [
                "analyst", "engineer", "manager", "specialist", "consultant",
                "coordinator", "developer", "scientist", "associate", "director",
                "technician", "administrator", "supervisor", "lead", "senior"
            ]
            
            if any(keyword in line.lower() for keyword in job_keywords):
                print(f"✅ Found title in line {i+1}: {line}")
                return clean_job_title(line)
    
    # Strategy 5: Fallback - use CSV title even if quality is questionable
    if csv_title and len(csv_title.strip()) > 3:
        cleaned = clean_job_title(csv_title)
        print(f"⚠️ Using fallback CSV title: {cleaned}")
        return cleaned
    
    # Strategy 6: Final fallback
    fallback_title = "Business Analyst"
    print(f"⚠️ Using default fallback: {fallback_title}")
    return fallback_title

def validate_job_title(title: str) -> bool:
    """Validate if a string looks like a job title"""
    
    if not title or not isinstance(title, str):
        return False
    
    title = title.strip()
    
    # Length check
    if len(title) < 5 or len(title) > 120:
        return False
    
    # Must contain letters
    if not re.search(r'[A-Za-z]', title):
        return False
    
    # Shouldn't be mostly numbers
    if len(re.findall(r'\d', title)) > len(title) * 0.5:
        return False
    
    # Shouldn't contain too many special characters
    special_chars = len(re.findall(r'[^A-Za-z0-9\s\-&/()]', title))
    if special_chars > len(title) * 0.2:
        return False
    
    # Common job title indicators
    job_indicators = [
        "analyst", "engineer", "manager", "specialist", "consultant", "coordinator",
        "developer", "scientist", "associate", "director", "technician", "administrator",
        "supervisor", "lead", "senior", "junior", "intern", "assistant", "officer"
    ]
    
    # Should contain at least one job indicator
    return any(indicator in title.lower() for indicator in job_indicators)

def clean_job_title(title: str) -> str:
    """Clean and format job title"""
    
    if not title:
        return "Business Analyst"
    
    # Remove common prefixes/suffixes
    prefixes_to_remove = [
        r'^(Job Title|Position|Role|Title)\s*[:\-–]\s*',
        r'^(We are looking for|We are seeking|Join us as)\s+(?:a|an)\s+',
        r'^\d+\.\s*',  # Numbers at start
        r'^\W+',       # Special characters at start
    ]
    
    suffixes_to_remove = [
        r'\s*[:\-–]\s.*',                         # Remove everything after colon or dash
        r'\s*\([^)]*\)',                          # Remove parenthetical info
        r'\s*-\s*(Full Time|Part Time|Contract|Remote|Hybrid).*',  # Employment type
    ]

    cleaned = title.strip()
    
    # Apply prefix removal
    for prefix in prefixes_to_remove:
        cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE).strip()
    
    # Apply suffix removal
    for suffix in suffixes_to_remove:
        cleaned = re.sub(suffix, '', cleaned, flags=re.IGNORECASE).strip()
    
    # Clean whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Title case
    cleaned = cleaned.title()
    
    # Fix common acronyms that shouldn't be title-cased
    acronym_fixes = {
        'Bi ': 'BI ', 'Etl': 'ETL', 'Api': 'API', 'Sql': 'SQL', 
        'Aws': 'AWS', 'Gcp': 'GCP', 'It ': 'IT ', 'Ai ': 'AI ',
        'Ml ': 'ML ', 'Qa ': 'QA ', 'Ui ': 'UI ', 'Ux ': 'UX '
    }
    
    for old, new in acronym_fixes.items():
        cleaned = cleaned.replace(old, new)
    
    # Final validation
    if not cleaned or len(cleaned) < 3:
        return "Business Analyst"
    
    return cleaned

# -------------------- Data Validation -------------------- #
def validate_job_data(df: pd.DataFrame) -> Dict[str, any]:
    """Comprehensive validation of job data"""
    
    validation_results = {
        "total_jobs": len(df),
        "valid_jobs": 0,
        "issues": [],
        "warnings": [],
        "statistics": {}
    }
    
    if df.empty:
        validation_results["issues"].append("No job data provided")
        return validation_results
    
    # Check required columns
    required_columns = ["title", "description"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        validation_results["issues"].append(f"Missing required columns: {missing_columns}")
        return validation_results
    
    # Analyze data quality
    valid_count = 0
    title_lengths = []
    description_lengths = []
    
    for index, row in df.iterrows():
        is_valid = True
        
        # Check title
        title = str(row.get('title', ''))
        if len(title.strip()) < 3:
            validation_results["warnings"].append(f"Row {index}: Title too short")
            is_valid = False
        title_lengths.append(len(title))
        
        # Check description
        description = str(row.get('description', ''))
        if len(description.strip()) < 50:
            validation_results["warnings"].append(f"Row {index}: Description too short")
            is_valid = False
        description_lengths.append(len(description))
        
        if is_valid:
            valid_count += 1
    
    validation_results["valid_jobs"] = valid_count
    validation_results["statistics"] = {
        "average_title_length": sum(title_lengths) / len(title_lengths) if title_lengths else 0,
        "average_description_length": sum(description_lengths) / len(description_lengths) if description_lengths else 0,
        "valid_percentage": (valid_count / len(df)) * 100 if len(df) > 0 else 0
    }
    
    # Summary assessment
    if valid_count == 0:
        validation_results["issues"].append("No valid jobs found")
    elif valid_count / len(df) < 0.5:
        validation_results["warnings"].append(f"Low data quality: only {valid_count}/{len(df)} jobs are valid")
    
    return validation_results

# -------------------- Performance Monitoring -------------------- #
class PerformanceMonitor:
    """Simple performance monitoring for resume processing"""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
        self.metrics = {}
    
    def start(self, operation_name: str = "operation"):
        """Start timing an operation"""
        self.start_time = time.time()
        self.operation_name = operation_name
        print(f"⏱️ Starting {operation_name}...")
    
    def checkpoint(self, name: str):
        """Record a checkpoint"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.checkpoints[name] = elapsed
            print(f"📊 {name}: {elapsed:.2f}s")
    
    def finish(self) -> Dict[str, float]:
        """Finish timing and return metrics"""
        if self.start_time:
            total_time = time.time() - self.start_time
            self.metrics["total_time"] = total_time
            self.metrics["checkpoints"] = self.checkpoints.copy()
            
            print(f"✅ {getattr(self, 'operation_name', 'Operation')} completed in {total_time:.2f}s")
            
            # Reset
            self.start_time = None
            self.checkpoints = {}
            
            return self.metrics
        
        return {}

# -------------------- Configuration Management -------------------- #
def load_config(config_path: str = "config.json") -> Dict[str, any]:
    """Load configuration with defaults"""
    
    default_config = {
        "max_attempts": 3,
        "target_ats_score": 75,
        "min_acceptable_score": 40,
        "timeout_seconds": 120,
        "headless_mode": True,
        "debug_mode": False,
        "save_attempts": True,
        "save_logs": True
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                print(f"✅ Configuration loaded from {config_path}")
        except Exception as e:
            print(f"⚠️ Error loading config, using defaults: {e}")
    
    return default_config

def save_config(config: Dict[str, any], config_path: str = "config.json"):
    """Save configuration to file"""
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Configuration saved to {config_path}")
    except Exception as e:
        print(f"⚠️ Error saving config: {e}")

# -------------------- Utility Functions -------------------- #
def clean_text_for_processing(text: str) -> str:
    """Clean text for better processing"""
    
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common artifacts
    artifacts = [
        r'\[.*?\]',  # Bracketed content
        r'<.*?>',    # HTML tags
        r'&\w+;',    # HTML entities
    ]
    
    for artifact in artifacts:
        text = re.sub(artifact, '', text)
    
    # Clean up formatting
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
    
    return text.strip()

def estimate_processing_time(job_count: int, avg_time_per_job: float = 45) -> str:
    """Estimate total processing time"""
    
    total_seconds = job_count * avg_time_per_job
    
    if total_seconds < 60:
        return f"{total_seconds:.0f} seconds"
    elif total_seconds < 3600:
        minutes = total_seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = total_seconds / 3600
        return f"{hours:.1f} hours"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"