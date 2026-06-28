"""
llm_engine.py — Local Offline LLM wrapper for Skill Evidence ATS
Uses Qwen2.5-1.5B-Instruct-GGUF via llama-cpp-python (CPU-only).
Falls back to structured heuristics if model is unavailable.
"""
import os
import re
import json
import logging

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("llm-engine")

# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------
is_llm_active = False
_llm = None
_use_ollama = False
_ollama_model = "none"

# Track which filename was actually used so callers can log it
loaded_model_name: str = "none"

_MODEL_OPTIONS = [
    # Preferred: 1.5B, good balance of speed + quality
    ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    # Fallback: 0.5B if HF is slow or disk is tight
    ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
]


def initialize() -> bool:
    """
    Download (once) and load the GGUF model or connect to local Ollama.
    Returns True if the model loaded successfully/Ollama connected, False otherwise.
    """
    global is_llm_active, _llm, loaded_model_name

    # Try Ollama first — fastest, no GGUF needed
    try:
        import urllib.request as _ur
        logger.info("Checking for local Ollama server on http://127.0.0.1:11434...")
        with _ur.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as res:
            data = json.loads(res.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", [])]
            chosen = None
            for candidate in ["qwen2.5-coder:7b", "deepseek-r1:7b"]:
                if candidate in models:
                    chosen = candidate
                    break
            if not chosen and models:
                chosen = models[0]
            if chosen:
                # Store in globals so ask_llm_json can use the Ollama path
                global _use_ollama, _ollama_model
                _use_ollama = True
                _ollama_model = chosen
                is_llm_active = True
                loaded_model_name = f"Ollama ({chosen})"
                logger.info(f"\u2705 Local Ollama server connected using model: {chosen}")
                return True
    except Exception as e:
        logger.info(f"Ollama not found: {e}. Falling back to llama-cpp GGUF.")

    try:
        from llama_cpp import Llama
    except ImportError as e:
        logger.warning(f"llama-cpp-python not installed: {e}")
        return False

    # 1. Check for custom local model path override
    local_path = os.getenv("LOCAL_MODEL_PATH")
    if local_path:
        local_path = local_path.strip('\'"')
        if os.path.exists(local_path):
            try:
                logger.info(f"Attempting to load local model directly from: {local_path}")
                _llm = Llama(
                    model_path=local_path,
                    n_ctx=4096,
                    n_threads=min(os.cpu_count() or 4, 6),  # use up to 6 cores
                    n_batch=512,
                    verbose=False,
                )
                is_llm_active = True
                loaded_model_name = os.path.basename(local_path)
                logger.info(f"✅ Local LLM loaded from file: {loaded_model_name}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load local model from {local_path}: {e}")
        else:
            logger.warning(f"LOCAL_MODEL_PATH specified but file does not exist: {local_path}")

    # 2. Check for Hugging Face Hub downloads (custom repo or defaults)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        logger.warning(f"huggingface-hub not installed; cannot download model: {e}")
        return False

    # Build model options starting with custom configuration if provided
    hf_options = []
    custom_repo = os.getenv("HF_MODEL_REPO_ID")
    custom_file = os.getenv("HF_MODEL_FILENAME")
    if custom_repo and custom_file:
        hf_options.append((custom_repo.strip('\'"'), custom_file.strip('\'"')))
    
    # Add defaults as fallback
    hf_options.extend(_MODEL_OPTIONS)

    cache_dir = os.getenv("HF_CACHE_DIR")
    if cache_dir:
        cache_dir = cache_dir.strip('\'"')
        logger.info(f"Using custom Hugging Face cache directory: {cache_dir}")

    for repo_id, filename in hf_options:
        try:
            logger.info(f"Attempting to load HF model: {filename} from repo {repo_id}")
            download_kwargs = {"repo_id": repo_id, "filename": filename}
            if cache_dir:
                download_kwargs["cache_dir"] = cache_dir

            model_path = hf_hub_download(**download_kwargs)
            _llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=min(os.cpu_count() or 4, 6),  # use up to 6 cores
                n_batch=512,
                verbose=False,
            )
            is_llm_active = True
            loaded_model_name = filename
            logger.info(f"✅ Local LLM loaded: {filename}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load HF model {filename} from {repo_id}: {e}")
            continue

    logger.warning("⚠️  No local GGUF model could be loaded. Using heuristic fallbacks.")
    return False


# ---------------------------------------------------------------------------
# Core query helpers
# ---------------------------------------------------------------------------

def ask_llm_json(system_prompt: str, user_prompt: str, max_tokens: int = 700) -> dict:
    """
    Query the local LLM in JSON mode.
    Returns a parsed dict, or {} on any failure.
    """
    if not is_llm_active:
        return {}

    if _use_ollama:
        try:
            import urllib.request as _ur
            payload = json.dumps({
                "model": _ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_predict": max_tokens},
            }).encode("utf-8")
            req = _ur.Request(
                "http://127.0.0.1:11434/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with _ur.urlopen(req, timeout=60) as res:
                data = json.loads(res.read().decode("utf-8"))
                raw = data.get("message", {}).get("content", "").strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-z]*\n?", "", raw)
                    raw = re.sub(r"\n?```$", "", raw)
                return json.loads(raw)
        except Exception as e:
            logger.error(f"Ollama JSON query failed: {e}")
            return {}

    if _llm is None:
        return {}
    try:
        response = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=0.05,   # near-deterministic for structured extraction
        )
        raw = response["choices"][0]["message"]["content"]
        return json.loads(raw)
    except Exception as e:
        logger.error(f"LLM JSON query failed: {e}")
        return {}


def ask_llm_text(system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
    """
    Generate free-form text from the local LLM.
    Returns an empty string on failure.
    """
    if not is_llm_active:
        return ""

    if _use_ollama:
        try:
            import urllib.request as _ur
            payload = json.dumps({
                "model": _ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.4, "num_predict": max_tokens},
            }).encode("utf-8")
            req = _ur.Request(
                "http://127.0.0.1:11434/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with _ur.urlopen(req, timeout=60) as res:
                data = json.loads(res.read().decode("utf-8"))
                return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"Ollama text query failed: {e}")
            return ""

    if _llm is None:
        return ""
    try:
        response = _llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM text generation failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Domain-specific helpers — JD parsing
# ---------------------------------------------------------------------------

_KNOWN_SKILLS = [
    "Python", "FastAPI", "Flask", "Django", "PostgreSQL", "MySQL", "MongoDB",
    "Redis", "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Kafka", "gRPC",
    "React", "TypeScript", "JavaScript", "Next.js", "Vue.js", "Angular",
    "Java", "Go", "Rust", "C++", "Scala", "Spark", "Airflow", "dbt",
    "PyTorch", "TensorFlow", "scikit-learn", "HuggingFace", "LangChain",
    "LlamaIndex", "Gemini", "GPT-4", "LLaMA", "BERT", "RAG",
    "embeddings", "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus",
    "NDCG", "MRR", "BM25", "vector database", "sentence-transformers",
    "Git", "CI/CD", "Terraform", "Ansible", "SQL", "NoSQL",
    "GraphQL", "REST", "Microservices", "System Design", "Linux",
]


def parse_jd(jd_text: str) -> dict:
    """
    Extract structured profile from a raw JD text.
    Tries local LLM first, falls back to regex heuristics.
    """
    if is_llm_active:
        system = (
            "You are an ATS (Applicant Tracking System) parser. "
            "Extract a structured JSON job profile from the job description. "
            "Output ONLY valid JSON with exactly these keys:\n"
            "- title: string, the job title\n"
            "- requiredSkills: array of SHORT strings. "
            "  RULES: Each item MUST be a specific technology name, programming language, framework, tool, or methodology. "
            "  Examples of CORRECT items: [\"Python\", \"PyTorch\", \"RAG\", \"embeddings\", \"BM25\", \"FAISS\", \"LLMs\", \"fine-tuning\", \"vector search\", \"SQL\"]. "
            "  NEVER write full sentences like 'Experience with embeddings' or 'Deep technical depth'. "
            "  Extract ONLY the technology/tool/skill noun itself.\n"
            "- preferredSkills: array of SHORT strings, same rules as requiredSkills but for nice-to-have items\n"
            "- responsibilities: array of strings (3-5 short sentences describing duties)\n"
            "- seniority: string (e.g. 'Senior', 'Mid-level', 'Junior')\n"
            "- idealProfile: string (1-2 sentences describing the ideal candidate)\n"
            "Do NOT add commentary outside the JSON."
        )
        user = f"Job Description:\n\n{jd_text[:8000]}"
        result = ask_llm_json(system, user, max_tokens=700)
        if result and "title" in result and "requiredSkills" in result:
            # Post-process: strip sentences that slipped through despite the prompt
            def sanitise_skills(skills: list) -> list:
                sentence_words = re.compile(
                    r'\b(experience|knowledge|expertise|depth|understanding|ability|'
                    r'familiarity|proficiency|with|in|of|and|or|the|a|an|to|for|'
                    r'working|strong|good|solid|excellent|proven|demonstrated)\b',
                    re.IGNORECASE
                )
                strip_prefixes = re.compile(
                    r'^(?:experience with|knowledge of|strong|expertise in|'
                    r'proficiency in|familiarity with|understanding of)\s+',
                    re.IGNORECASE
                )
                clean = []
                for s in (skills or []):
                    s = s.strip()
                    # Remove common sentence prefixes
                    s = strip_prefixes.sub("", s).strip()
                    # Skip if still looks like a full sentence (too long or contains verb words)
                    if len(s) > 45:
                        continue
                    # Allow through if it looks like a tech term (not sentence-heavy)
                    word_count = len(s.split())
                    sentence_word_matches = len(sentence_words.findall(s))
                    if word_count > 4 and sentence_word_matches > 0:
                        continue
                    if s:
                        clean.append(s)
                return clean

            result["requiredSkills"] = sanitise_skills(result.get("requiredSkills", []))
            result["preferredSkills"] = sanitise_skills(result.get("preferredSkills", []))
            
            # Guarantee we have something — fall back to heuristic keywords if LLM returned nothing useful
            if not result["requiredSkills"] or not result["preferredSkills"]:
                heuristic = _heuristic_parse_jd(jd_text)
                if not result["requiredSkills"]:
                    result["requiredSkills"] = heuristic["requiredSkills"]
                if not result["preferredSkills"]:
                    result["preferredSkills"] = heuristic["preferredSkills"]
                    
            return result

    # ---- Heuristic fallback ----
    return _heuristic_parse_jd(jd_text)


def _heuristic_parse_jd(jd_text: str) -> dict:
    ltext = jd_text.lower()

    # Title
    title = "Software Engineer"
    for kw, t in [
        ("founding ai", "Founding AI Engineer"),
        ("ai engineer", "AI Engineer"), ("ml engineer", "ML Engineer"),
        ("data scientist", "Data Scientist"), ("data engineer", "Data Engineer"),
        ("backend", "Backend Engineer"), ("frontend", "Frontend Developer"),
        ("fullstack", "Full Stack Developer"), ("devops", "DevOps Engineer"),
        ("platform engineer", "Platform Engineer"),
    ]:
        if kw in ltext:
            title = t
            break

    # Seniority
    seniority = "Mid-level (3-5 years)"
    if re.search(r"\b(senior|lead|principal|staff|5\+?\s*years|7\+?\s*years|8\+?\s*years|9\+?\s*years)\b", ltext):
        seniority = "Senior / Lead (5+ years)"
    elif re.search(r"\b(fresher|entry[- ]level|0[- ]2\s*years|0\s+to\s+2\s*years|intern(ship)?s?)\b", ltext):
        seniority = "Fresher / Entry-level (0-2 years)"

    found = [s for s in _KNOWN_SKILLS if s.lower() in ltext]
    split = max(1, int(len(found) * 0.6))

    return {
        "title": title,
        "requiredSkills": found[:split] or ["Software Engineering", "Problem Solving"],
        "preferredSkills": found[split:] or ["System Design", "Agile Methodologies"],
        "responsibilities": [
            "Design, build, and maintain software systems.",
            "Collaborate with cross-functional teams on feature delivery.",
            "Write clean, well-tested, maintainable code.",
            "Participate in code reviews and architectural discussions.",
        ],
        "seniority": seniority,
        "idealProfile": (
            f"A motivated engineer with hands-on experience in "
            f"{', '.join(found[:3]) if found else 'software development'}. "
            "Strong problem-solving skills and a collaborative mindset."
        ),
    }


# ---------------------------------------------------------------------------
# Domain-specific helpers — Resume parsing
# ---------------------------------------------------------------------------

def parse_resume(resume_text: str) -> dict:
    """
    Extract structured candidate profile from resume text.
    Always uses heuristics — LLM is bypassed for resumes so behaviour is
    identical whether the LLM is connected or not.
    """
    return _heuristic_parse_resume(resume_text)


def _heuristic_parse_resume(text: str) -> dict:
    lines = text.split("\n")
    name = lines[0].strip() if lines else "Candidate"

    found_skills = [s for s in _KNOWN_SKILLS if s.lower() in text.lower()]

    # Extract GitHub URL
    github = None
    gh_match = re.search(r'github\.com/([a-zA-Z0-9_-]+)', text)
    if gh_match:
        github = f"https://github.com/{gh_match.group(1)}"

    linkedin = None
    li_match = re.search(r'linkedin\.com/in/([a-zA-Z0-9_-]+)', text)
    if li_match:
        linkedin = f"https://linkedin.com/in/{li_match.group(1)}"

    return {
        "name": name,
        "skills": found_skills or ["Software Development"],
        "projects": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "achievements": [],
        "socialLinks": {k: v for k, v in {"github": github, "linkedin": linkedin}.items() if v},
    }


# ---------------------------------------------------------------------------
# Domain-specific helpers — Narrative/reasoning generation
# ---------------------------------------------------------------------------

def generate_reasoning(candidate_name: str, rank: int, score: float,
                        stage: str, skills: list[str], jd_title: str) -> str:
    """Generate a 1-2 sentence recruiting summary for a ranked candidate."""
    # [DISABLED FOR SPEED] Generating narratives via LLM for 100+ candidates in batch takes too long.
    # Bypassing this ensures the system relies on the fast heuristic summarizer below.
    """
    if is_llm_active:
        system = "You are a concise technical recruiting assistant. Write 1-2 professional sentences."
        user = (
            f"Candidate '{candidate_name}' (#{rank}, stage: {stage}, score: {score:.1f}) "
            f"is being evaluated for '{jd_title}'. Their top skills are: {', '.join(skills[:6])}. "
            "Write a brief, specific recruiting summary."
        )
        text = ask_llm_text(system, user, max_tokens=120)
        if text:
            return text
    """

    # Heuristic fallback
    # Return empty so the caller (main.py) falls back to the advanced rank.py logic
    return ""


def generate_social_audit_summary(candidate_name: str, repos: list, languages: list) -> str:
    """Generate a social audit justification summary."""
    if is_llm_active and repos:
        system = "You are a technical recruiter reviewing a candidate's GitHub profile. Write 2 sentences."
        repo_names = ", ".join(r.get("name", "") for r in repos[:5])
        lang_list = ", ".join(languages[:5])
        user = (
            f"Candidate: {candidate_name}. "
            f"GitHub repositories: {repo_names}. "
            f"Languages detected: {lang_list}. "
            "Summarise the profile evidence quality in 2 sentences."
        )
        text = ask_llm_text(system, user, max_tokens=100)
        if text:
            return text

    langs = ", ".join(languages[:3]) if languages else "various languages"
    return (
        f"Social audit for {candidate_name} completed. "
        f"GitHub profile shows activity in {langs}, supporting resume credentials."
    )
