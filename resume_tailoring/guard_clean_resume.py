import re

KNOWN_DEGREES = [
    "Master of Science in Computer Science, University of Bridgeport",
    "Bachelor of Technology in Computer Science & Engineering, KMIT",
]

KNOWN_CERTS = [
    "Google Data Analytics Professional Certification — Google",
    "Excel for Business — Coursera",
    "AI-Driven Analytics Simulation — Tata Consultancy Services",
]

SECTION_HEADERS = ["SUMMARY", "SKILLS", "WORK EXPERIENCE", "CERTIFICATIONS", "EDUCATION"]
SECTION_REGEX = re.compile(rf"(?m)^({'|'.join(map(re.escape, SECTION_HEADERS))})\s*$")


# ----------- Utility -----------

def normalize(text: str) -> str:
    """Normalize text for comparison (lowercase, remove punctuation)."""
    return re.sub(r"[^a-z0-9 ]+", "", text.strip().lower())


def fix_broken_headers(text: str) -> str:
    """
    Fix known broken section headers and rejoin ALL-CAPS headers that were line-broken.
    Ensures we end up with proper 'WORK EXPERIENCE' (not 'WORKEXPERIENCE').
    """
    # Fix specific broken headers
    broken_headers = [
        ("ED\nUCATION", "EDUCATION"),
        ("WO\nRK EXPERIENCE", "WORK EXPERIENCE"),
        ("CE\nRTIFICATIONS", "CERTIFICATIONS"),
        ("SU\nMMARY", "SUMMARY"),
        ("SK\nILLS", "SKILLS"),
    ]
    for broken, fixed in broken_headers:
        text = text.replace(broken, fixed)

    # Generic fix: "WORK\nEXPERIENCE" => "WORK EXPERIENCE"
    def join_all_caps(match: re.Match) -> str:
        return f"{match.group(1)} {match.group(2)}"

    text = re.sub(r"\b([A-Z]{2,})\n([A-Z]{2,})\b", join_all_caps, text)
    return text


def find_section_bounds(text: str, section_name: str):
    """
    Find [start, end) character positions of a section in the resume.
    Start is at the section header line. End is right before the next known section header
    (SUMMARY|SKILLS|WORK EXPERIENCE|CERTIFICATIONS|EDUCATION), or EOF.
    """
    text = fix_broken_headers(text)  # normalize header breaks first
    header_pat = re.compile(rf"(?m)^{re.escape(section_name.upper())}\s*$")
    m = header_pat.search(text)
    if not m:
        return -1, -1

    start = m.start()
    # Search for the next section header AFTER this header line
    after = text[m.end():]
    next_m = SECTION_REGEX.search(after)
    if next_m:
        end = m.end() + next_m.start()
    else:
        end = len(text)
    return start, end


def remove_placeholder_bullets(text: str) -> str:
    """Removes bullets like '- Additional relevant responsibility 1' used as fallback fillers."""
    return re.sub(r"-\s*Additional relevant responsibility \d+\s*\n?", "", text)


# ----------- WORK EXPERIENCE -----------

def extract_experience_pairs(resume_text: str):
    """
    Extract pairs of (job title, company) from WORK EXPERIENCE headers.
    Format expected: 'Job Title – Company, Location  YYYY – YYYY' (year required).
    """
    pattern = r"^(.*?)\s*–\s*(.*?),\s*.*?\b(19|20)\d{2}\b.*?$"
    lines = resume_text.splitlines()
    exp_pairs = set()
    for line in lines:
        m = re.match(pattern, line.strip())
        if m:
            title, company = m.group(1).strip(), m.group(2).strip()
            exp_pairs.add((normalize(title), normalize(company)))
    return exp_pairs


def clean_tailored_work_experience(tailored_resume: str, base_resume: str = None, verbose: bool = False) -> str:
    """
    Conservative cleanup: trust injected headers; remove obvious hallucination markers and placeholder bullets.
    """
    lines = tailored_resume.splitlines()
    cleaned = []
    for line in lines:
        ls = line.strip()
        if any(tag in ls for tag in ["❗", "<hallucinated>", "<placeholder>"]):
            if verbose:
                print(f"🧹 Removed flagged line: {ls}")
            continue
        if re.match(r"-\s*Additional relevant responsibility \d+\s*$", ls):
            if verbose:
                print(f"🧹 Removed placeholder bullet: {ls}")
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def remove_hallucinated_titles(text: str) -> str:
    """
    Remove explicit hallucination title lines (starting with ❗).
    Do NOT aggressively remove entire blocks; only drop the flagged line and any immediate
    bullet spillover (lines starting with '-' or '•') until content changes.
    """
    lines = text.splitlines()
    out = []
    skip_bullets = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("❗"):
            skip_bullets = True
            continue

        if skip_bullets:
            if stripped.startswith("-") or stripped.startswith("•") or not stripped:
                # still in the 'spillover' zone under a hallucinated header line
                continue
            # content changed; stop skipping
            skip_bullets = False

        out.append(line)

    return "\n".join(out)


# ----------- CERTIFICATIONS -----------

def extract_certifications(text: str):
    start, end = find_section_bounds(text, "CERTIFICATIONS")
    if start == -1:
        return []
    lines = text[start:end].splitlines()[1:]  # skip header
    return [normalize(line) for line in lines if line.strip()]


def clean_certifications_section(tailored_resume: str, base_resume: str = None, verbose: bool = False) -> str:
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
    cleaned = [lines[0]]  # header
    for line in lines[1:]:
        norm_line = normalize(line)
        if any(norm_base in norm_line for norm_base in base_certs):
            cleaned.append(line)
        elif verbose:
            print(f"🧹 Removed hallucinated cert: {line.strip()}")

    return before + "\n" + "\n".join(cleaned).strip() + "\n" + after


# ----------- EDUCATION -----------

def extract_education(text: str):
    start, end = find_section_bounds(text, "EDUCATION")
    if start == -1:
        return []
    lines = text[start:end].splitlines()[1:]  # skip header
    return [normalize(line) for line in lines if line.strip()]


def clean_education_section(tailored_resume: str, base_resume: str = None, verbose: bool = False) -> str:
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
    cleaned = [lines[0]]  # header
    for line in lines[1:]:
        norm_line = normalize(line)
        if any(norm_base in norm_line for norm_base in base_edu):
            cleaned.append(line)
        elif verbose:
            print(f"🧹 Removed hallucinated education entry: {line.strip()}")

    return before + "\n" + "\n".join(cleaned).strip() + "\n" + after


# ----------- Final Patch Utility -----------

def clean_full_resume(tailored_resume: str, base_resume: str = None, verbose: bool = False) -> str:
    if not base_resume:
        return tailored_resume

    text = tailored_resume

    # Normalize known section headers to ALL CAPS
    text = re.sub(r"(?m)^\s*Summary\s*$", "SUMMARY", text)
    text = re.sub(r"(?m)^\s*Skills\s*$", "SKILLS", text)
    text = re.sub(r"(?m)^\s*Work Experience\s*$", "WORK EXPERIENCE", text)
    text = re.sub(r"(?m)^\s*Certifications\s*$", "CERTIFICATIONS", text)
    text = re.sub(r"(?m)^\s*Education\s*$", "EDUCATION", text)

    # --- Freeze WORK EXPERIENCE before global cleanups (keeps job headers intact) ---
    m = re.search(r"(?ms)^WORK EXPERIENCE\s*\n(.*?)(\n(?=[A-Z][A-Z &/]+\s*$)|\Z)", text)
    we_body = m.group(1) if m else ""
    text = re.sub(
        r"(?ms)^WORK EXPERIENCE\s*\n.*?(\n(?=[A-Z][A-Z &/]+\s*$)|\Z)",
        "WORK EXPERIENCE\n\n<<WE_PLACEHOLDER>>\\1",
        text,
    )

    # Global cleanups outside WE
    text = fix_broken_headers(text)
    text = remove_hallucinated_titles(text)

    # Restore WE exactly as it was
    text = text.replace("<<WE_PLACEHOLDER>>", we_body if we_body.endswith("\n") else we_body + "\n")

    # WE-only conservative cleaner then hard-stitch the WE it produced
    text_after_we = clean_tailored_work_experience(text, base_resume, verbose)
    m2 = re.search(r"(?ms)^WORK EXPERIENCE\s*\n(.*?)(\n(?=[A-Z][A-Z &/]+\s*$)|\Z)", text_after_we)
    we_final = m2.group(1) if m2 else we_body
    text = re.sub(
        r"(?ms)^WORK EXPERIENCE\s*\n.*?(\n(?=[A-Z][A-Z &/]+\s*$)|\Z)",
        "WORK EXPERIENCE\n\n" + (we_final if we_final.endswith("\n") else we_final + "\n") + r"\1",
        text,
    )

    # --- Case-insensitive upsert for CERTIFICATIONS & EDUCATION ---
    for sec_name in ["CERTIFICATIONS", "EDUCATION"]:
        # remove ALL variants of the section first (case-insensitive)
        text = re.sub(
            rf"(?ms)^\s*{sec_name}\s*\n.*?(\n(?=[A-Z][A-Z &/]+\s*$)|\Z)", "", text, flags=re.I
        )
        # insert original from base once
        b_start, b_end = find_section_bounds(base_resume, sec_name)
        if b_start != -1:
            original_block = base_resume[b_start:b_end].strip()
            text = text.rstrip() + "\n\n" + original_block + "\n"
            if verbose:
                print(f"🔁 Replaced {sec_name} with original (case-insensitive).")

    return text




def patch_job_titles_with_original(tailored_text: str, original_text: str) -> str:
    """
    Safer: only fix truly truncated headers (e.g., 'Data Analyst –').
    Use expected originals IN ORDER to avoid same-title collisions.
    """
    header_rx = re.compile(r"^(?P<h>[^-\n]+?\s*–\s*[^,\n]+?,.*?\b(19|20)\d{2}\b.*)$", re.M)
    originals = [m.group("h").strip() for m in header_rx.finditer(original_text)]

    trunc_rx = re.compile(r"^(?P<title>[^-\n]+?)\s*–\s*$")  # 'Title –' only
    out, idx = [], 0
    for line in tailored_text.splitlines():
        s = line.strip()
        mt = trunc_rx.match(s)
        if mt and idx < len(originals):
            out.append(originals[idx]); idx += 1
        else:
            out.append(line)
    return "\n".join(out)


def remove_placeholders(text: str) -> str:
    lines = text.splitlines()
    cleaned = [line for line in lines if not any(tok in line for tok in ["❗", "TBD", "[PLACEHOLDER]"])]
    return "\n".join(cleaned)
