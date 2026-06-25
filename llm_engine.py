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
    Download (once) and load the GGUF model.
    Returns True if the model loaded successfully, False otherwise.
    """
    global is_llm_active, _llm, loaded_model_name

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
    if not is_llm_active or _llm is None:
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
    if not is_llm_active or _llm is None:
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
            "You are a senior technical recruiter. Extract a structured JSON job profile "
            "from the job description below. Output ONLY valid JSON with exactly these keys: "
            "title (string), requiredSkills (array of strings), preferredSkills (array of strings), "
            "responsibilities (array of strings, 3-5 items), seniority (string), "
            "idealProfile (string, 1-2 sentences). Do not add any commentary."
        )
        user = f"Job Description:\n\n{jd_text[:3500]}"
        result = ask_llm_json(system, user, max_tokens=650)
        if result and "title" in result and "requiredSkills" in result:
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
    if any(k in ltext for k in ["fresher", "entry-level", "entry level", "0-2 years", "0 to 2 years", "intern"]):
        seniority = "Fresher / Entry-level (0-2 years)"
    elif any(k in ltext for k in ["senior", "lead", "principal", "staff", "5+ years", "7+ years", "8+ years"]):
        seniority = "Senior / Lead (5+ years)"

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
    Tries local LLM first, falls back to heuristics.
    """
    if is_llm_active:
        system = (
            "You are an expert ATS resume parser. Extract a structured JSON profile. "
            "Output ONLY valid JSON with exactly these keys: "
            "name (string), skills (array of strings), "
            "projects (array of objects with keys: title, description, technologies), "
            "experience (array of objects with keys: role, company, duration, description), "
            "education (array of objects with keys: degree, school, year), "
            "certifications (array of strings), achievements (array of strings), "
            "socialLinks (object with optional keys: github, linkedin, portfolio, website). "
            "Extract ALL jobs from the resume, even short ones. Do not invent data."
        )
        user = f"Resume:\n\n{resume_text[:4500]}"
        result = ask_llm_json(system, user, max_tokens=900)
        if result and "name" in result:
            return result

    # ---- Heuristic fallback ----
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

    # Heuristic fallback
    top = ", ".join(skills[:4]) if skills else "general engineering"
    return (
        f"Candidate {candidate_name} is ranked #{rank} with a score of {score:.1f}. "
        f"Profile demonstrates {stage}-level expertise, particularly in {top}."
    )


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
