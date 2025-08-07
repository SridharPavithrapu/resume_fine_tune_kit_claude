import os
import sys
import json
import pandas as pd
from docx import Document
from generate_with_models import tailor_resume_with_models, save_to_docx, parse_sections, validate_final_output
from resume_tailoring.guard_clean_resume import clean_full_resume, fix_broken_headers
from resume_tailoring.utils import extract_job_title
from jobscan.jobscan_driver_debug import get_jobscan_score_and_feedback
from resume_tailoring.inference import patch_final_resume, validate_final_output_quality
import time
import traceback
from typing import Dict, List, Tuple

# Configuration
DEBUG = os.getenv("DEBUG_RESUME", "0") == "1"
MAX_ATTEMPTS = 3
TARGET_ATS_SCORE = 75  # Lowered from 80 for more realistic expectations
MIN_ACCEPTABLE_SCORE = 40
JOBSCAN_TIMEOUT = 120  # seconds

# Suppress unnecessary outputs
os.environ["MallocStackLogging"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if not DEBUG:
    sys.stderr = open(os.devnull, 'w')

def load_job_data() -> pd.DataFrame:
    """Load job data with validation"""
    jd_file_path = "data/jobspy_jobs.csv"
    
    try:
        if not os.path.exists(jd_file_path):
            print(f"❌ Job file not found: {jd_file_path}")
            return pd.DataFrame()
        
        jd_df = pd.read_csv(jd_file_path)
        
        # Validate required columns
        required_columns = ["title", "description"]
        missing_columns = [col for col in required_columns if col not in jd_df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        # Filter out rows with missing data
        initial_count = len(jd_df)
        jd_df = jd_df.dropna(subset=required_columns)
        final_count = len(jd_df)
        
        if initial_count != final_count:
            print(f"⚠️ Filtered out {initial_count - final_count} jobs with missing data")
        
        print(f"✅ Loaded {final_count} job listings from {jd_file_path}")
        return jd_df
        
    except Exception as e:
        print(f"❌ Error loading job file: {e}")
        return pd.DataFrame()

def load_base_resume(path: str) -> str:
    """Load base resume with validation"""
    try:
        if not os.path.exists(path):
            print(f"❌ Base resume not found: {path}")
            return ""
        
        doc = Document(path)
        text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        
        if len(text) < 500:
            print(f"⚠️ Base resume seems too short: {len(text)} characters")
        
        print(f"✅ Base resume loaded: {len(text)} characters")
        return text
        
    except Exception as e:
        print(f"❌ Error loading base resume: {e}")
        return ""

def create_safe_filename(title: str, score: int = None) -> str:
    """Create safe filename from job title"""
    # Remove special characters and limit length
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_title = safe_title.replace(' ', '_')[:50]
    
    if score is not None:
        return f"{safe_title}_ATS{score}"
    return safe_title

def setup_output_directories():
    """Create necessary output directories"""
    directories = [
        "outputs/final_tailored_resumes",
        "outputs/debug_logs", 
        "prompt_logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def process_single_job(row_index: int, job_data: pd.Series, base_resume: str, 
                      style_guide: str) -> Dict[str, any]:
    """Process a single job with comprehensive error handling"""
    
    job_title = job_data.get("title", "Unknown Title")
    jd_text = job_data.get("description", "")
    
    print(f"\n{'='*60}")
    print(f"📄 [{row_index + 1}] Processing: {job_title}")
    print(f"{'='*60}")
    
    # Initialize result tracking
    result = {
        "index": row_index,
        "title": job_title,
        "success": False,
        "final_score": 0,
        "attempts": 0,
        "best_resume": "",
        "error": None,
        "processing_time": 0
    }
    
    start_time = time.time()
    
    try:
        # Validate inputs
        if not jd_text or len(jd_text.strip()) < 50:
            raise ValueError("Job description too short or empty")
        
        if not base_resume or len(base_resume.strip()) < 500:
            raise ValueError("Base resume invalid or too short")
        
        # Extract and validate job title
        extracted_job_title = extract_job_title(jd_text, csv_title=job_title)
        print(f"📋 Extracted job title: {extracted_job_title}")
        
        # Initialize tracking variables
        best_score = 0
        best_resume = ""
        best_keywords = []
        best_sections = []
        attempt_results = []
        
        # Attempt resume tailoring with iterative improvement
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"\n⏳ Attempt {attempt}/{MAX_ATTEMPTS}")
            
            attempt_start = time.time()
            attempt_result = {
                "attempt": attempt,
                "success": False,
                "score": 0,
                "resume_length": 0,
                "keywords_found": 0,
                "error": None
            }
            
            try:
                # Generate tailored resume
                print("🤖 Generating tailored resume...")
                tailored_resume = tailor_resume_with_models(
                    job_title=extracted_job_title,
                    job_description=jd_text,
                    base_resume_path="data/YoshithaM_Resume_W2.docx",
                    ats_keywords=best_keywords,
                    ats_sections=best_sections
                )
                
                if not tailored_resume or len(tailored_resume.strip()) < 800:
                    raise ValueError("Generated resume too short or empty")
                
                print(f"✅ Resume generated: {len(tailored_resume)} characters")
                
                # Apply final cleaning and patching
                print("🧹 Applying final cleaning...")
                cleaned_resume = patch_final_resume(tailored_resume, base_resume)
                
                # Validate resume structure
                quality_check = validate_final_output_quality(cleaned_resume)
                if not quality_check["valid"]:
                    print(f"⚠️ Quality issues: {quality_check['issues']}")
                    # Continue anyway but log the issues
                
                # Add fallback sections if missing
                if "SUMMARY" not in cleaned_resume:
                    print("🔧 Adding fallback SUMMARY...")
                    fallback_summary = f"SUMMARY\n\nExperienced {extracted_job_title} with strong analytical and communication skills, delivering data-driven insights and supporting business objectives.\n\n"
                    cleaned_resume = fallback_summary + cleaned_resume
                
                # Limit resume length to prevent issues
                if len(cleaned_resume) > 25000:
                    print(f"⚠️ Resume too long ({len(cleaned_resume)} chars), trimming...")
                    cleaned_resume = cleaned_resume[:25000] + "\n\n[Content truncated for length]"
                
                attempt_result["resume_length"] = len(cleaned_resume)
                attempt_result["success"] = True
                
                # Save attempt for debugging
                if DEBUG:
                    debug_file = f"outputs/debug_logs/{create_safe_filename(job_title)}_attempt{attempt}.txt"
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(cleaned_resume)
                
                print(f"💾 Resume processed successfully ({len(cleaned_resume)} chars)")
                
                # Get ATS score and feedback
                print("📊 Getting ATS score via Jobscan...")
                
                try:
                    score, missing_keywords, missing_sections = get_jobscan_score_and_feedback(
                        cleaned_resume, jd_text
                    )
                    
                    # Parse score
                    if isinstance(score, str):
                        score_str = score.strip().replace('%', '')
                        try:
                            score_int = int(score_str)
                        except (ValueError, TypeError):
                            score_int = 0
                    else:
                        score_int = int(score) if score else 0
                    
                    print(f"📈 ATS Score: {score_int}%")
                    
                    if missing_keywords:
                        print(f"🔍 Missing keywords: {missing_keywords[:5]}...")
                    if missing_sections:
                        print(f"📋 Missing sections: {missing_sections}")
                    
                    attempt_result["score"] = score_int
                    attempt_result["keywords_found"] = len(missing_keywords)
                    
                    # Update best result if this is better
                    if score_int > best_score:
                        best_score = score_int
                        best_resume = cleaned_resume
                        best_keywords = missing_keywords[:10]  # Limit for next attempt
                        best_sections = missing_sections[:5]   # Limit for next attempt
                        
                        print(f"🎯 New best score: {score_int}%")
                    
                    # Check if we should stop early
                    if score_int >= TARGET_ATS_SCORE:
                        print(f"🎉 Target score reached! Stopping early.")
                        break
                    elif score_int < MIN_ACCEPTABLE_SCORE and attempt == 1:
                        print(f"⚠️ Very low score ({score_int}%) - likely OCR error, retrying...")
                    elif not missing_keywords and attempt < MAX_ATTEMPTS:
                        print("⚠️ No missing keywords returned - may need retry")
                    
                except Exception as e:
                    print(f"❌ Jobscan analysis failed: {e}")
                    # Continue with resume even without score
                    if not best_resume or len(cleaned_resume) > len(best_resume):
                        best_resume = cleaned_resume
                        best_score = 0  # Unknown score
                
                attempt_result["processing_time"] = time.time() - attempt_start
                
            except Exception as e:
                print(f"❌ Attempt {attempt} failed: {e}")
                attempt_result["error"] = str(e)
                if DEBUG:
                    traceback.print_exc()
            
            attempt_results.append(attempt_result)
            result["attempts"] = attempt
            
            # Wait between attempts
            if attempt < MAX_ATTEMPTS:
                wait_time = min(5, attempt * 2)
                print(f"⏳ Waiting {wait_time}s before next attempt...")
                time.sleep(wait_time)
        
        # Finalize results
        if best_resume:
            result["success"] = True
            result["best_resume"] = best_resume
            result["final_score"] = best_score
            
            # Save final resume
            safe_filename = create_safe_filename(job_title, best_score)
            
            # Save DOCX
            docx_path = f"outputs/final_tailored_resumes/{safe_filename}.docx"
            if save_to_docx(best_resume, docx_path):
                print(f"💾 Resume saved: {docx_path}")
            
            # Save text version
            txt_path = f"outputs/final_tailored_resumes/{safe_filename}.txt"
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(best_resume)
                print(f"💾 Text version saved: {txt_path}")
            except Exception as e:
                print(f"⚠️ Failed to save text version: {e}")
            
            # Save processing log
            log_data = {
                "job_title": job_title,
                "extracted_title": extracted_job_title,
                "final_score": best_score,
                "attempts": attempt_results,
                "processing_time": time.time() - start_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            log_path = f"outputs/debug_logs/{safe_filename}_log.json"
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"⚠️ Failed to save processing log: {e}")
            
            print(f"✅ Job processing completed - Final score: {best_score}%")
            
            if best_score < TARGET_ATS_SCORE:
                print(f"⚠️ Did not reach target ATS score of {TARGET_ATS_SCORE}%")
        
        else:
            raise ValueError("No valid resume generated after all attempts")
    
    except Exception as e:
        print(f"❌ Job processing failed: {e}")
        result["error"] = str(e)
        if DEBUG:
            traceback.print_exc()
    
    result["processing_time"] = time.time() - start_time
    return result

def generate_summary_report(results: List[Dict]) -> None:
    """Generate comprehensive summary report"""
    
    print(f"\n{'='*60}")
    print("📊 PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    total_jobs = len(results)
    successful_jobs = len([r for r in results if r["success"]])
    failed_jobs = total_jobs - successful_jobs
    
    print(f"📈 Overall Statistics:")
    print(f"   Total jobs processed: {total_jobs}")
    print(f"   Successful: {successful_jobs}")
    print(f"   Failed: {failed_jobs}")
    print(f"   Success rate: {(successful_jobs/total_jobs*100):.1f}%")
    
    if successful_jobs > 0:
        successful_results = [r for r in results if r["success"]]
        scores = [r["final_score"] for r in successful_results if r["final_score"] > 0]
        
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            target_met = len([s for s in scores if s >= TARGET_ATS_SCORE])
            
            print(f"\n📊 ATS Score Statistics:")
            print(f"   Average score: {avg_score:.1f}%")
            print(f"   Highest score: {max_score}%")
            print(f"   Lowest score: {min_score}%")
            print(f"   Target ({TARGET_ATS_SCORE}%+) achieved: {target_met}/{len(scores)} ({target_met/len(scores)*100:.1f}%)")
    
    # Processing time analysis
    processing_times = [r["processing_time"] for r in results if r["processing_time"] > 0]
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        total_time = sum(processing_times)
        print(f"\n⏱️ Performance Statistics:")
        print(f"   Average processing time: {avg_time:.1f}s")
        print(f"   Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Save detailed summary
    summary_data = []
    for result in results:
        summary_data.append({
            "title": result["title"],
            "success": result["success"],
            "final_score": result["final_score"],
            "attempts": result["attempts"],
            "processing_time": round(result["processing_time"], 1),
            "error": result["error"]
        })
    
    try:
        summary_df = pd.DataFrame(summary_data)
        summary_path = "outputs/processing_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n💾 Detailed summary saved: {summary_path}")
    except Exception as e:
        print(f"⚠️ Failed to save summary CSV: {e}")
    
    # Print failed jobs for debugging
    failed_results = [r for r in results if not r["success"]]
    if failed_results:
        print(f"\n❌ Failed Jobs:")
        for result in failed_results:
            print(f"   - {result['title']}: {result['error']}")

def main():
    """Main execution function"""
    print("🚀 Starting Bulk Resume Tailoring Process")
    print(f"Debug mode: {'ON' if DEBUG else 'OFF'}")
    
    # Setup
    setup_output_directories()
    
    # Load data
    jd_df = load_job_data()
    if jd_df.empty:
        print("❌ No job data available. Exiting.")
        return
    
    base_resume = load_base_resume("data/YoshithaM_Resume_W2.docx")
    if not base_resume:
        print("❌ Base resume not available. Exiting.")
        return
    
    # Load style guide
    try:
        with open("prompts/style_resume.txt", "r", encoding="utf-8") as f:
            style_guide = f.read()
    except Exception as e:
        print(f"⚠️ Could not load style guide: {e}")
        style_guide = "Generate a professional resume following standard formatting."
    
    # Process jobs
    results = []
    start_time = time.time()
    
    try:
        for index, row in jd_df.iterrows():
            result = process_single_job(index, row, base_resume, style_guide)
            results.append(result)
            
            # Progress update
            print(f"\n📊 Progress: {len(results)}/{len(jd_df)} jobs processed")
            
            # Brief pause between jobs
            if index < len(jd_df) - 1:
                time.sleep(2)
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user after {len(results)} jobs")
    except Exception as e:
        print(f"\n❌ Unexpected error in main process: {e}")
        if DEBUG:
            traceback.print_exc()
    
    # Generate final report
    total_time = time.time() - start_time
    print(f"\n🏁 Process completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    if results:
        generate_summary_report(results)
    else:
        print("❌ No results to summarize")

if __name__ == "__main__":
    main()