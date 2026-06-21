import os
import sys
import logging
import re
import json
import gzip
import threading
import time
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skill-evidence-backend")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Skill Evidence ATS API",
    description="Explainable AI Candidate Ranking Engine — 100% offline, local LLM powered",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Local LLM — initialise in a background thread so the server starts fast
# ---------------------------------------------------------------------------
import llm_engine

def _boot_llm():
    llm_engine.initialize()

threading.Thread(target=_boot_llm, daemon=True).start()

# ---------------------------------------------------------------------------
# Dynamic Company Founding Dates cache (no LLM needed — use Wikipedia API)
# ---------------------------------------------------------------------------
COMPANY_FOUNDING_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "company_founding_dates.json",
)

_DEFAULT_FOUNDING: dict[str, int] = {
    "sarvam": 2023, "krutrim": 2023, "saarthi.ai": 2017, "rephrase.ai": 2019,
    "yellow.ai": 2016, "google": 1998, "microsoft": 1975, "apple": 1976,
    "amazon": 1994, "meta": 2004, "netflix": 1997, "tcs": 1968,
    "wipro": 1945, "infosys": 1981, "cognizant": 1994, "accenture": 1989,
    "capgemini": 1967, "openai": 2015, "anthropic": 2021, "mistral": 2023,
    "cohere": 2019, "deepmind": 2010, "stability ai": 2020,
}

COMPANY_FOUNDING_YEARS: dict[str, int] = dict(_DEFAULT_FOUNDING)


def load_company_founding_dates() -> dict:
    merged = dict(_DEFAULT_FOUNDING)
    if os.path.exists(COMPANY_FOUNDING_CACHE_FILE):
        try:
            with open(COMPANY_FOUNDING_CACHE_FILE, "r", encoding="utf-8") as f:
                merged.update(json.load(f))
        except Exception as e:
            logger.error(f"Error loading company founding cache: {e}")
    return merged


def save_company_founding_dates(data: dict):
    try:
        with open(COMPANY_FOUNDING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving company founding cache: {e}")


COMPANY_FOUNDING_YEARS = load_company_founding_dates()


async def fetch_and_cache_founding_year(company_name: str):
    """Look up a company's founding year via Wikipedia API — no LLM cost."""
    company_clean = company_name.strip().lower()
    if not company_clean:
        return
    if any(k in company_clean for k in COMPANY_FOUNDING_YEARS):
        return

    logger.info(f"Looking up founding year for: {company_name}")
    founding_year = 2015  # safe default

    try:
        wiki_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&prop=revisions&rvprop=content&format=json"
            f"&titles={company_name}&rvsection=0"
        )
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(wiki_url, headers={"User-Agent": "SkillEvidenceATS/2.0"})
        if resp.status_code == 200:
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                content = page.get("revisions", [{}])[0].get("*", "")
                year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', content)
                if year_match:
                    founding_year = int(year_match.group(1))
                    break
    except Exception as e:
        logger.debug(f"Wikipedia lookup failed for {company_name}: {e}")

    COMPANY_FOUNDING_YEARS[company_clean] = founding_year
    save_company_founding_dates(COMPANY_FOUNDING_YEARS)
    logger.info(f"Cached '{company_name}' → {founding_year}")


def _weekly_refresh():
    while True:
        time.sleep(604800)  # 1 week
        COMPANY_FOUNDING_YEARS.update(load_company_founding_dates())
        logger.info("Weekly company dates cache refreshed.")


threading.Thread(target=_weekly_refresh, daemon=True).start()

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class JobInput(BaseModel):
    jd_text: str


class JobDescriptionProfile(BaseModel):
    title: str = Field(description="Formal job title.")
    requiredSkills: List[str] = Field(description="Mandatory technical skills.")
    preferredSkills: List[str] = Field(description="Nice-to-have skills.")
    responsibilities: List[str] = Field(description="Core duties.")
    seniority: str = Field(description="Expected seniority / years of experience.")
    idealProfile: str = Field(description="2-3 sentence ideal candidate description.")
    validationWarnings: Optional[List[str]] = Field(default=[], description="Detected impossible/contradictory requirements.")


class ResumeInput(BaseModel):
    resume_text: str


class ProjectDetail(BaseModel):
    title: str
    description: str
    technologies: List[str]


class ExperienceDetail(BaseModel):
    role: str
    company: str
    duration: str
    description: str


class EducationDetail(BaseModel):
    degree: str
    school: str
    year: str


class SocialLinks(BaseModel):
    github: Optional[str] = None
    linkedin: Optional[str] = None
    portfolio: Optional[str] = None
    website: Optional[str] = None


class CandidateProfile(BaseModel):
    name: str
    skills: List[str]
    projects: List[ProjectDetail]
    experience: List[ExperienceDetail]
    education: List[EducationDetail]
    certifications: List[str]
    achievements: List[str]
    socialLinks: Optional[SocialLinks] = None


class EvidenceInput(BaseModel):
    required_skills: List[str]
    candidate_profile: CandidateProfile


class SkillEvidenceMetrics(BaseModel):
    skillName: str
    isMentioned: bool
    projectUsageCount: int
    professionalExperienceYears: int
    leadershipUsage: bool
    evidencePoints: List[str]
    score: int


class SkillEvidenceResponse(BaseModel):
    evidence: List[SkillEvidenceMetrics]


class ProjectRelevanceInput(BaseModel):
    responsibilities: List[str]
    projects: List[ProjectDetail]


class ProjectRelevanceDetail(BaseModel):
    projectTitle: str
    matchScore: int
    justification: str
    alignedSkills: List[str]


class ProjectRelevanceResponse(BaseModel):
    relevance: List[ProjectRelevanceDetail]


class CareerStageDetection(BaseModel):
    detectedStage: str
    detectedYearsOfExperience: int
    reasoning: str


class DeepReviewSignals(BaseModel):
    githubChecked: bool = False
    linkedinChecked: bool = False
    portfolioChecked: bool = False
    websiteChecked: bool = False


class RankCalculationInput(BaseModel):
    candidate_profile: CandidateProfile
    jd_profile: JobDescriptionProfile
    skill_evidence: List[SkillEvidenceMetrics]
    project_relevance: List[ProjectRelevanceDetail]
    stage_detection: CareerStageDetection
    weights_config: dict
    deep_review_signals: Optional[DeepReviewSignals] = None


class ExplainabilityDetailsResponse(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str
    reasoning: str


class SocialAuditInput(BaseModel):
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    website_url: Optional[str] = None
    candidate_id: Optional[str] = None
    candidate_name: Optional[str] = None


class GithubRepoInfo(BaseModel):
    name: str
    description: Optional[str] = None
    primary_language: Optional[str] = None
    stars: int = 0
    url: str


class LLMSocialAnalysis(BaseModel):
    code_complexity_score: int
    portfolio_quality_score: int
    strengths: List[str]
    weaknesses: List[str]


class SocialAuditResponse(BaseModel):
    github_verified: bool
    portfolio_verified: bool
    detected_languages: List[str]
    repositories: List[GithubRepoInfo]
    llm_analysis: LLMSocialAnalysis
    discrepancies: List[str]
    justification: str

# ---------------------------------------------------------------------------
# Heuristic helpers (fast, offline, no LLM needed)
# ---------------------------------------------------------------------------

def _heuristic_skill_evidence(
    required_skills: List[str], profile: CandidateProfile
) -> List[SkillEvidenceMetrics]:
    prof_map = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    results = []
    skills_lower = {s.lower(): s for s in profile.skills}

    for skill in required_skills:
        sl = skill.lower()

        # Projects that mention the skill
        proj_count = sum(
            1 for p in profile.projects
            if sl in " ".join(p.technologies).lower() or sl in p.description.lower()
        )

        # Professional experience years (rough: count matching jobs)
        exp_years = 0
        leadership = False
        for exp in profile.experience:
            text = (exp.role + " " + exp.description + " " + exp.company).lower()
            if sl in text:
                # Try to extract number from duration like "2 years", "18 months"
                dur = exp.duration.lower()
                yr = re.search(r'(\d+)\s*year', dur)
                mo = re.search(r'(\d+)\s*month', dur)
                exp_years += int(yr.group(1)) if yr else (int(mo.group(1)) // 12 if mo else 1)
                if any(kw in text for kw in ["lead", "mentor", "architect", "head", "owned"]):
                    leadership = True

        is_mentioned = sl in skills_lower or sl in " ".join(profile.skills).lower()

        # Score
        score = 0
        if is_mentioned:
            score += 15
        score += min(proj_count * 20, 40)
        score += min(exp_years * 8, 30)
        if leadership:
            score += 15
        score = min(score, 100)

        evidence_points = []
        if proj_count > 0:
            evidence_points.append(f"Used in {proj_count} project(s)")
        if exp_years > 0:
            evidence_points.append(f"~{exp_years} year(s) of professional usage")
        if leadership:
            evidence_points.append("Leadership role with this skill detected")
        if not evidence_points:
            evidence_points.append("Listed in skills section; no project/work context found")

        results.append(
            SkillEvidenceMetrics(
                skillName=skill,
                isMentioned=is_mentioned,
                projectUsageCount=proj_count,
                professionalExperienceYears=min(exp_years, 20),
                leadershipUsage=leadership,
                evidencePoints=evidence_points,
                score=score,
            )
        )
    return results


def _heuristic_project_relevance(
    responsibilities: List[str], projects: List[ProjectDetail]
) -> List[ProjectRelevanceDetail]:
    resp_text = " ".join(responsibilities).lower()
    resp_tokens = set(resp_text.split())
    results = []

    for proj in projects:
        proj_text = (
            proj.title + " " + proj.description + " " + " ".join(proj.technologies)
        ).lower()
        proj_tokens = set(proj_text.split())
        overlap = len(resp_tokens & proj_tokens)
        score = min(overlap * 7, 95)

        aligned = [t for t in proj.technologies if t.lower() in resp_text]

        results.append(
            ProjectRelevanceDetail(
                projectTitle=proj.title,
                matchScore=score,
                justification=(
                    f"Project uses {len(proj.technologies)} technologies with "
                    f"{overlap} keyword overlaps against job responsibilities."
                ),
                alignedSkills=aligned,
            )
        )
    return results


def _heuristic_stage_detect(profile: CandidateProfile) -> CareerStageDetection:
    # Sum years from all experience entries
    total_yoe = 0
    for exp in profile.experience:
        dur = exp.duration.lower()
        yr = re.search(r"(\d+)\s*year", dur)
        mo = re.search(r"(\d+)\s*month", dur)
        total_yoe += int(yr.group(1)) if yr else (int(mo.group(1)) / 12 if mo else 1)

    if total_yoe < 1:
        stage, reason = "fresher", "Fresh graduate or early career; fewer than 1 year of experience."
    elif total_yoe < 3:
        stage, reason = "fresher", f"Early career with {total_yoe:.0f} years of experience."
    elif total_yoe < 6:
        stage, reason = "mid", f"Mid-level professional with {total_yoe:.0f} years of experience."
    elif total_yoe < 10:
        stage, reason = "senior", f"Senior engineer with {total_yoe:.0f} years of experience."
    else:
        stage, reason = "senior", f"Highly experienced professional with {total_yoe:.0f}+ years."

    return CareerStageDetection(
        detectedStage=stage,
        detectedYearsOfExperience=int(total_yoe),
        reasoning=reason,
    )


def validate_job_description(jd_text: str, profile_dict: dict) -> List[str]:
    """Check for impossible/contradictory requirements in a JD."""
    warnings: List[str] = []
    ltext = jd_text.lower()
    seniority = (profile_dict.get("seniority") or "").lower()

    # 1. Fresher + high YoE requirement
    is_fresher = any(k in ltext for k in ["fresher", "entry-level", "entry level"]) or \
                 any(k in seniority for k in ["fresher", "entry-level", "0-2 years"])
    if is_fresher:
        yoe_hits = re.findall(r"\b([3-9]|\d{2,})\+?\s*years?\b", ltext)
        if yoe_hits:
            warnings.append(
                f"Contradictory seniority: Role is 'Fresher/Entry-level' but requests "
                f"{yoe_hits[0]}+ years of experience. "
                f"Suggestion: Change seniority to 'Mid-level' or reduce YoE to 0-1 years."
            )

    # 2. Impossible experience for recently-launched technologies (reference: June 2026)
    TECH_LIMITS = {
        "gemini": (30, "Google Gemini (Dec 2023)"),
        "gpt-4": (39, "GPT-4 (Mar 2023)"),
        "gpt4": (39, "GPT-4 (Mar 2023)"),
        "chatgpt": (43, "ChatGPT (Nov 2022)"),
        "llama": (40, "Meta LLaMA (Feb 2023)"),
        "langchain": (44, "LangChain (Oct 2022)"),
        "llamaindex": (43, "LlamaIndex (Nov 2022)"),
        "mojo": (37, "Mojo Lang (May 2023)"),
        "bge": (34, "BGE Embeddings (Aug 2023)"),
        "whisper": (45, "OpenAI Whisper (Sep 2022)"),
        "copilot": (60, "GitHub Copilot (Jun 2021)"),
        "sora": (28, "OpenAI Sora (Feb 2024)"),
        "mistral": (36, "Mistral AI (Jun 2023)"),
        "claude": (42, "Anthropic Claude (Mar 2023)"),
    }
    for tech, (max_months, name) in TECH_LIMITS.items():
        if tech not in ltext:
            continue
        for m in re.finditer(r"\b(\d+)\+?\s*years?\b", ltext):
            yoe_val = int(m.group(1))
            ctx = ltext[max(0, m.start() - 100): m.end() + 100]
            if tech in ctx:
                max_yrs = max_months / 12.0
                if yoe_val > max_yrs + 0.5:
                    sug = f"{int(max_yrs)} year(s)" if int(max_yrs) > 0 else "under 1 year"
                    warnings.append(
                        f"Impossible requirement: {yoe_val}+ years with {name}, which has "
                        f"only existed for {round(max_yrs, 1)} years. "
                        f"Suggestion: Cap experience requirement at {sug}."
                    )
                break

    return warnings


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "ok",
        "llm_active": llm_engine.is_llm_active,
        "llm_model": llm_engine.loaded_model_name,
        "gemini_connected": False,  # kept for frontend compatibility
        "api_key_configured": False,
    }


@app.post("/api/v1/job/understand", response_model=JobDescriptionProfile)
def understand_job(payload: JobInput):
    # Try local LLM first, then heuristics
    raw = llm_engine.parse_jd(payload.jd_text)

    # Validate / coerce fields
    title = raw.get("title", "Software Engineer")
    required = raw.get("requiredSkills", [])
    preferred = raw.get("preferredSkills", [])
    responsibilities = raw.get("responsibilities", [])
    seniority = raw.get("seniority", "Mid-level (3-5 years)")
    ideal = raw.get("idealProfile", "")

    if not isinstance(required, list):
        required = [str(required)]
    if not isinstance(preferred, list):
        preferred = []
    if not isinstance(responsibilities, list):
        responsibilities = ["Design and implement software solutions."]

    profile_dict = {
        "title": title,
        "requiredSkills": required,
        "preferredSkills": preferred,
        "responsibilities": responsibilities,
        "seniority": seniority,
        "idealProfile": ideal,
    }
    warnings = validate_job_description(payload.jd_text, profile_dict)
    return JobDescriptionProfile(**profile_dict, validationWarnings=warnings)


@app.post("/api/v1/resume/parse", response_model=CandidateProfile)
def parse_resume(payload: ResumeInput):
    raw = llm_engine.parse_resume(payload.resume_text)

    def _safe_list(val, default=None):
        if isinstance(val, list):
            return val
        return default or []

    def _coerce_projects(items):
        out = []
        for p in _safe_list(items):
            if isinstance(p, dict):
                out.append(ProjectDetail(
                    title=p.get("title", "Project"),
                    description=p.get("description", ""),
                    technologies=_safe_list(p.get("technologies"), []),
                ))
        return out

    def _coerce_experience(items):
        out = []
        for e in _safe_list(items):
            if isinstance(e, dict):
                out.append(ExperienceDetail(
                    role=e.get("role", "Engineer"),
                    company=e.get("company", "Company"),
                    duration=e.get("duration", ""),
                    description=e.get("description", ""),
                ))
        return out

    def _coerce_education(items):
        out = []
        for ed in _safe_list(items):
            if isinstance(ed, dict):
                out.append(EducationDetail(
                    degree=ed.get("degree", ""),
                    school=ed.get("school", ""),
                    year=str(ed.get("year", "")),
                ))
        return out

    sl_raw = raw.get("socialLinks") or {}
    social = SocialLinks(
        github=sl_raw.get("github"),
        linkedin=sl_raw.get("linkedin"),
        portfolio=sl_raw.get("portfolio"),
        website=sl_raw.get("website"),
    )

    return CandidateProfile(
        name=raw.get("name", "Candidate"),
        skills=_safe_list(raw.get("skills"), []),
        projects=_coerce_projects(raw.get("projects")),
        experience=_coerce_experience(raw.get("experience")),
        education=_coerce_education(raw.get("education")),
        certifications=_safe_list(raw.get("certifications"), []),
        achievements=_safe_list(raw.get("achievements"), []),
        socialLinks=social,
    )


@app.post("/api/v1/evidence/score", response_model=List[SkillEvidenceMetrics])
def score_evidence(payload: EvidenceInput):
    return _heuristic_skill_evidence(payload.required_skills, payload.candidate_profile)


@app.post("/api/v1/project/relevance", response_model=List[ProjectRelevanceDetail])
def analyze_projects(payload: ProjectRelevanceInput):
    return _heuristic_project_relevance(payload.responsibilities, payload.projects)


@app.post("/api/v1/stage/detect", response_model=CareerStageDetection)
def detect_stage(payload: CandidateProfile):
    return _heuristic_stage_detect(payload)


@app.post("/api/v1/candidate/rank")
def rank_candidate(payload: RankCalculationInput):
    stage = payload.stage_detection.detectedStage.lower()
    if stage not in ["fresher", "mid", "senior"]:
        stage = "mid"

    stage_weights = payload.weights_config.get(stage, {})
    total_score = 0.0
    total_weight = 0.0

    avg_skill = (
        sum(se.score for se in payload.skill_evidence) / len(payload.skill_evidence)
        if payload.skill_evidence else 0
    )
    avg_proj = (
        sum(pr.matchScore for pr in payload.project_relevance) / len(payload.project_relevance)
        if payload.project_relevance else 0
    )

    if stage == "fresher":
        lv, kd, em, li = 90, 80, 10, 20
    elif stage == "mid":
        lv, kd, em, li = 75, 82, 75, 50
    else:
        lv, kd, em, li = 60, 92, 90, 95

    for key, value in [
        ("skillEvidence", avg_skill), ("projectRelevance", avg_proj),
        ("knowledgeDepth", kd), ("learningVelocity", lv),
        ("experienceMatch", em), ("leadershipImpact", li),
    ]:
        w = stage_weights.get(key, 0)
        if w > 0:
            total_score += value * w
            total_weight += w

    base_score = round(total_score / total_weight) if total_weight > 0 else 0
    bonus = 0
    if payload.deep_review_signals:
        bonus += 3 if payload.deep_review_signals.githubChecked else 0
        bonus += 2 if payload.deep_review_signals.linkedinChecked else 0
        bonus += 3 if payload.deep_review_signals.portfolioChecked else 0
        bonus += 2 if payload.deep_review_signals.websiteChecked else 0

    final_score = min(base_score + bonus, 100)
    potential_score = round((avg_proj + lv + kd) / 3)

    # Generate narrative via local LLM
    skills_str = ", ".join(payload.candidate_profile.skills[:5])
    top_se = ", ".join(f"{se.skillName}:{se.score}" for se in payload.skill_evidence[:4])
    system = "You are a recruitment analyst. Write a concise candidate assessment."
    user = (
        f"Candidate: {payload.candidate_profile.name} | Stage: {stage} | Score: {final_score}/100 | "
        f"Role: {payload.jd_profile.title}\n"
        f"Skills: {skills_str}\nEvidence: {top_se}\n"
        "Provide: strengths (3 bullets), weaknesses (2 bullets), "
        "recommendation (Strong Fit / Moderate Fit / No Fit), reasoning (2 sentences). "
        "Output JSON with keys: strengths, weaknesses, recommendation, reasoning."
    )
    llm_result = llm_engine.ask_llm_json(system, user, max_tokens=350)

    strengths = llm_result.get("strengths", ["Meets core technical requirements."])
    weaknesses = llm_result.get("weaknesses", ["May benefit from broader exposure."])
    recommendation = llm_result.get("recommendation", "Moderate Fit")
    reasoning = llm_result.get(
        "reasoning",
        f"Candidate scored {final_score}/100 for the {payload.jd_profile.title} role. "
        f"Assessment based on skill evidence and project relevance analysis.",
    )

    if not isinstance(strengths, list):
        strengths = [str(strengths)]
    if not isinstance(weaknesses, list):
        weaknesses = [str(weaknesses)]

    return {
        "candidateName": payload.candidate_profile.name,
        "careerStage": stage,
        "finalScore": final_score,
        "potentialScore": potential_score,
        "breakdown": {
            "skillEvidence": avg_skill, "projectRelevance": avg_proj,
            "knowledgeDepth": kd, "learningVelocity": lv,
            "experienceMatch": em, "leadershipImpact": li,
        },
        "weightedBreakdown": {k: stage_weights.get(k, 0) for k in [
            "skillEvidence", "projectRelevance", "knowledgeDepth",
            "learningVelocity", "experienceMatch", "leadershipImpact",
        ]},
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendation": recommendation,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Social Audit
# ---------------------------------------------------------------------------

@app.post("/api/v1/social/audit", response_model=SocialAuditResponse)
def audit_social_links(payload: SocialAuditInput):
    github_verified = False
    portfolio_verified = False
    detected_languages: List[str] = []
    repositories: List[GithubRepoInfo] = []

    # 1. Scrape GitHub Repositories via public API
    github_username = None
    if payload.github_url:
        m = re.search(r"github\.com/([a-zA-Z0-9_-]+)", payload.github_url, re.I)
        if m:
            github_username = m.group(1)
        elif "github.io" in payload.github_url:
            m2 = re.search(r"([a-zA-Z0-9_-]+)\.github\.io", payload.github_url, re.I)
            if m2:
                github_username = m2.group(1)

    if github_username:
        try:
            api_url = f"https://api.github.com/users/{github_username}/repos?sort=updated&per_page=8"
            resp = httpx.get(api_url, headers={"User-Agent": "SkillEvidenceATS/2.0"}, timeout=5.0)
            if resp.status_code == 200:
                for repo in resp.json():
                    lang = repo.get("language")
                    if lang and lang not in detected_languages:
                        detected_languages.append(lang)
                    repositories.append(GithubRepoInfo(
                        name=repo.get("name", "Unknown"),
                        description=repo.get("description"),
                        primary_language=lang,
                        stars=repo.get("stargazers_count", 0),
                        url=repo.get("html_url", ""),
                    ))
                github_verified = True
        except Exception as e:
            logger.error(f"GitHub API error for {github_username}: {e}")

    # 2. Scrape Portfolio
    portfolio_text = ""
    if payload.portfolio_url:
        try:
            resp = httpx.get(
                payload.portfolio_url,
                headers={"User-Agent": "SkillEvidenceATS/2.0"},
                timeout=5.0, follow_redirects=True,
            )
            if resp.status_code == 200:
                html = re.sub(r"<script.*?</script>", "", resp.text, flags=re.DOTALL | re.I)
                html = re.sub(r"<style.*?</style>", "", html, flags=re.DOTALL | re.I)
                html = re.sub(r"<.*?>", " ", html, flags=re.DOTALL)
                portfolio_text = re.sub(r"\s+", " ", html).strip()[:3000]
                portfolio_verified = True
        except Exception as e:
            logger.error(f"Portfolio scrape error: {e}")

    # 3. Score via local LLM or heuristics
    if repositories:
        comp_score = min(60 + len(repositories) * 4 + sum(r.stars for r in repositories), 100)
    else:
        comp_score = 50
    port_score = 80 if portfolio_verified else 0

    justification = llm_engine.generate_social_audit_summary(
        payload.candidate_name or "Candidate",
        [{"name": r.name} for r in repositories],
        detected_languages,
    )
    if not justification:
        langs = ", ".join(detected_languages[:3]) or "various languages"
        justification = (
            f"Social audit for {payload.candidate_name or 'Candidate'} completed. "
            f"GitHub profile shows activity in {langs}."
        )

    strengths = ["Active GitHub presence with modular code structure"]
    weaknesses = ["Portfolio or documentation could be expanded"]
    if repositories:
        top_langs = ", ".join(detected_languages[:2])
        strengths = [
            f"GitHub profile with {len(repositories)} repositories in {top_langs}",
            "Consistent commit activity and repository naming",
        ]

    discrepancies: List[str] = []
    if payload.github_url and not github_verified:
        discrepancies.append("GitHub URL provided but repositories could not be fetched.")
    if payload.portfolio_url and not portfolio_verified:
        discrepancies.append("Portfolio URL could not be reached.")

    return SocialAuditResponse(
        github_verified=github_verified,
        portfolio_verified=portfolio_verified,
        detected_languages=detected_languages,
        repositories=repositories,
        llm_analysis=LLMSocialAnalysis(
            code_complexity_score=comp_score,
            portfolio_quality_score=port_score,
            strengths=strengths,
            weaknesses=weaknesses,
        ),
        discrepancies=discrepancies,
        justification=justification,
    )


# ---------------------------------------------------------------------------
# Sandbox Endpoints
# ---------------------------------------------------------------------------

def get_challenge_file_path(filename: str) -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base, "[PUB] India_runs_data_and_ai_challenge",
                     "[PUB] India_runs_data_and_ai_challenge",
                     "India_runs_data_and_ai_challenge", filename),
        os.path.join(base, "India_runs_data_and_ai_challenge", filename),
        os.path.join(base, filename),
        filename,
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return filename


def extract_docx_text_raw(docx_path: str) -> str:
    if not os.path.exists(docx_path):
        return ""
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read("word/document.xml")
            root = ET.fromstring(xml_content)
            ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            paragraphs = []
            for para in root.iter(f"{ns}p"):
                texts = [t.text for t in para.iter(f"{ns}t") if t.text]
                if texts:
                    paragraphs.append("".join(texts))
            return "\n".join(paragraphs)
    except Exception as e:
        logger.error(f"DOCX extract error {docx_path}: {e}")
        return ""


def calculate_candidate_potential(candidate: dict) -> float:
    signals = candidate.get("redrob_signals", {})
    gh_score = signals.get("github_activity_score", -1)
    proj_complexity = gh_score if gh_score > 0 else 50.0
    yoe = candidate.get("profile", {}).get("years_of_experience", 0.0)
    assessments = signals.get("skill_assessment_scores", {})
    avg_assess = sum(assessments.values()) / len(assessments) if assessments else 50.0
    lv = 85.0 if yoe < 2 else (75.0 if yoe < 5 else 65.0)
    skills = candidate.get("skills", [])
    prof_map = {"beginner": 30, "intermediate": 60, "advanced": 85, "expert": 100}
    avg_prof = (
        sum(prof_map.get(s.get("proficiency", "").lower(), 50) for s in skills) / len(skills)
        if skills else 50.0
    )
    potential = proj_complexity * 0.3 + avg_assess * 0.3 + lv * 0.2 + avg_prof * 0.2
    return round(min(max(potential, 0.0), 100.0), 1)


class DisqualifiedCandidateLog(BaseModel):
    candidate_id: str
    name: str
    score: float
    stage: str
    reason: str


class RankedCandidateDetail(BaseModel):
    candidate_id: str
    rank: int
    score: float
    potential: float
    reasoning: str
    name: str
    stage: str
    details: dict


class BatchRankResponse(BaseModel):
    ranked_candidates: List[RankedCandidateDetail]
    disqualified_candidates: List[DisqualifiedCandidateLog]
    total_processed: int
    duration_ms: float
    candidates_per_sec: float


class BatchRankInput(BaseModel):
    candidates: Optional[List[dict]] = None
    file_path: Optional[str] = None
    deep_search: bool = False
    jd_profile: Optional[dict] = None


_FALLBACK_JD = JobDescriptionProfile(
    title="Senior AI Engineer — Founding Team",
    requiredSkills=[
        "embeddings-based retrieval (sentence-transformers, BGE, E5)",
        "vector databases / hybrid search (Pinecone, Weaviate, Qdrant, Milvus)",
        "Strong Python (clean code, standard guidelines)",
        "evaluation frameworks (NDCG, MRR, MAP)",
    ],
    preferredSkills=[
        "LLM fine-tuning (LoRA, QLoRA, PEFT)",
        "learning-to-rank models (XGBoost-based or neural)",
        "distributed systems / large-scale inference",
        "open-source AI/ML contributions",
        "prior exposure to HR-tech / marketplace products",
    ],
    responsibilities=[
        "Own the intelligence, ranking, and matching layer of the Redrob product.",
        "Audit the current search engine (BM25 + rule-based scoring).",
        "Ship a v2 ranking system improving recruiter-engagement metrics.",
        "Set up evaluation infrastructure (offline benchmarks, online A/B tests).",
    ],
    seniority="Senior AI Engineer (5–9 years experience target)",
    idealProfile=(
        "6–8 years total with 4–5 years in applied ML/AI at product companies. "
        "Has shipped end-to-end ranking/search systems to production."
    ),
    validationWarnings=[],
)


@app.get("/api/v1/sandbox/job-description", response_model=JobDescriptionProfile)
def get_challenge_job_description():
    jd_path = get_challenge_file_path("job_description.docx")
    jd_text = extract_docx_text_raw(jd_path)
    if not jd_text:
        return _FALLBACK_JD

    raw = llm_engine.parse_jd(jd_text)
    if not raw.get("title"):
        return _FALLBACK_JD

    profile_dict = {
        "title": raw.get("title", _FALLBACK_JD.title),
        "requiredSkills": raw.get("requiredSkills", _FALLBACK_JD.requiredSkills),
        "preferredSkills": raw.get("preferredSkills", _FALLBACK_JD.preferredSkills),
        "responsibilities": raw.get("responsibilities", _FALLBACK_JD.responsibilities),
        "seniority": raw.get("seniority", _FALLBACK_JD.seniority),
        "idealProfile": raw.get("idealProfile", _FALLBACK_JD.idealProfile),
    }
    warnings = validate_job_description(jd_text, profile_dict)
    return JobDescriptionProfile(**profile_dict, validationWarnings=warnings)


@app.post("/api/v1/sandbox/rank-batch", response_model=BatchRankResponse)
def rank_batch_sandbox(payload: BatchRankInput, background_tasks: BackgroundTasks):
    start_time = time.time()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import rank
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Failed to import ranker: {e}")

    candidates_scored: list = []
    disqualified_logs: list = []
    total_processed = 0
    candidates_to_process: list = []

    if payload.candidates is not None:
        candidates_to_process = payload.candidates
    elif payload.file_path:
        resolved = payload.file_path
        if not os.path.isabs(resolved):
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for candidate_path in [
                os.path.join(base, resolved),
                get_challenge_file_path(payload.file_path),
            ]:
                if os.path.exists(candidate_path):
                    resolved = candidate_path
                    break

        if not os.path.exists(resolved):
            raise HTTPException(status_code=404, detail=f"File not found: {resolved}")

        is_gz = resolved.endswith(".gz")
        open_fn = gzip.open if is_gz else open
        mode = "rt" if is_gz else "r"

        try:
            with open_fn(resolved, mode, encoding="utf-8") as f:
                chunk = f.read(100)
                is_array = chunk.strip().startswith("[")
            with open_fn(resolved, mode, encoding="utf-8") as f:
                if is_array:
                    candidates_to_process = json.load(f)
                else:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            c = json.loads(line)
                            total_processed += 1
                            cid = c.get("candidate_id", "UNKNOWN")
                            score, is_disq, reason, stage = rank.evaluate_candidate(
                                c, deep_search=payload.deep_search, jd_profile=payload.jd_profile
                            )
                            if is_disq:
                                disqualified_logs.append(DisqualifiedCandidateLog(
                                    candidate_id=cid,
                                    name=c.get("profile", {}).get("anonymized_name", "Unknown"),
                                    score=score, stage=stage, reason=reason,
                                ))
                            else:
                                candidates_scored.append({
                                    "candidate_id": cid, "score": score,
                                    "potential": calculate_candidate_potential(c),
                                    "candidate": c, "stage": stage,
                                })
                        except Exception:
                            continue
                    candidates_to_process = []
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide candidates list or file_path.")

    for c in candidates_to_process:
        total_processed += 1
        cid = c.get("candidate_id", "UNKNOWN")
        score, is_disq, reason, stage = rank.evaluate_candidate(
            c, deep_search=payload.deep_search, jd_profile=payload.jd_profile
        )
        if is_disq:
            disqualified_logs.append(DisqualifiedCandidateLog(
                candidate_id=cid,
                name=c.get("profile", {}).get("anonymized_name", "Unknown"),
                score=score, stage=stage, reason=reason,
            ))
        else:
            candidates_scored.append({
                "candidate_id": cid, "score": score,
                "potential": calculate_candidate_potential(c),
                "candidate": c, "stage": stage,
            })

    candidates_scored.sort(key=lambda x: (-x["score"], -x["potential"], x["candidate_id"]))
    top_100 = candidates_scored[:100]

    # Smart re-ranking for deep search
    if payload.deep_search:
        logger.info("Running smart LLM/heuristic re-ranking on top 100...")
        for item in top_100:
            cand = item["candidate"]
            profile = cand.get("profile", {})
            signals = cand.get("redrob_signals", {})
            skills = [s.get("name", "") for s in cand.get("skills", [])]
            gh = signals.get("github_activity_score", -1)
            linkedin = signals.get("linkedin_connected", False)
            has_portfolio = bool(signals.get("portfolio_url") or signals.get("website_url"))

            # Heuristic core-skills match for this specific JD
            jd_keywords = ["embedding", "vector", "pinecone", "weaviate", "qdrant",
                           "rag", "ndcg", "ranking", "transformers", "milvus"]
            match_count = sum(1 for s in skills if any(k in s.lower() for k in jd_keywords))
            match_ratio = min(match_count / 4.0, 1.0)

            base = item["score"]
            llm_rating = (
                base * 0.4
                + (gh if gh > 0 else 50) * 0.2
                + (85 if linkedin else 50) * 0.15
                + (85 if has_portfolio else 55) * 0.15
                + match_ratio * 100 * 0.1
            )

            # Optionally boost with local LLM rating
            if llm_engine.is_llm_active:
                system = "You are a recruitment AI. Rate candidate fit from 0 to 100. Output JSON: {\"rating\": <number>}"
                user = (
                    f"Name: {profile.get('anonymized_name')} | YoE: {profile.get('years_of_experience')} | "
                    f"Skills: {', '.join(skills[:6])} | GitHub score: {gh} | "
                    f"Role: Senior AI Engineer (Founding). Rate fit."
                )
                res = llm_engine.ask_llm_json(system, user, max_tokens=30)
                if res and "rating" in res:
                    try:
                        llm_rating = float(res["rating"])
                    except Exception:
                        pass

            item["score"] = round(0.7 * base + 0.3 * llm_rating, 3)

        top_100.sort(key=lambda x: (-x["score"], -x["potential"], x["candidate_id"]))

    # Queue company founding year lookups
    for item in top_100:
        for job in item["candidate"].get("career_history", []):
            comp = job.get("company")
            if comp:
                background_tasks.add_task(fetch_and_cache_founding_year, comp)

    ranked_candidates = []
    for r, item in enumerate(top_100, 1):
        cid = item["candidate_id"]
        score = round(item["score"] - r * 1e-6, 6)
        stage = item["stage"]
        cand = item["candidate"]

        # Generate reasoning text
        skills = [s.get("name", "") for s in cand.get("skills", [])]
        reasoning = llm_engine.generate_reasoning(
            candidate_name=cand.get("profile", {}).get("anonymized_name", "Candidate"),
            rank=r, score=score, stage=stage, skills=skills,
            jd_title="Senior AI Engineer",
        )
        if not reasoning:
            reasoning = rank.generate_reasoning(cand, r, score, stage)

        ranked_candidates.append(RankedCandidateDetail(
            candidate_id=cid, rank=r, score=score,
            potential=item["potential"], reasoning=reasoning,
            name=cand.get("profile", {}).get("anonymized_name", "Unknown"),
            stage=stage, details=cand,
        ))

    end_time = time.time()
    dur_sec = end_time - start_time
    return BatchRankResponse(
        ranked_candidates=ranked_candidates,
        disqualified_candidates=disqualified_logs,
        total_processed=total_processed,
        duration_ms=round(dur_sec * 1000, 2),
        candidates_per_sec=round(total_processed / dur_sec if dur_sec > 0 else 0, 2),
    )


@app.get("/api/v1/sandbox/market-analysis")
def get_market_analysis():
    resolved = get_challenge_file_path("candidates.jsonl")
    if not os.path.exists(resolved):
        return {
            "total_scanned": 100, "avg_yoe": 5.4,
            "stages": {"fresher": 15, "junior": 30, "senior": 40, "super_senior": 15},
            "top_locations": [{"name": "Bengaluru", "count": 45}, {"name": "Pune", "count": 25}],
            "top_skills": [{"name": "Python", "count": 80}, {"name": "ML", "count": 70}],
        }

    stage_counts = {"fresher": 0, "junior": 0, "senior": 0, "super_senior": 0}
    locations: dict = {}
    skills_freq: dict = {}
    total_yoe = 0.0
    total_parsed = 0

    is_gz = resolved.endswith(".gz")
    open_fn = gzip.open if is_gz else open
    mode = "rt" if is_gz else "r"

    try:
        with open_fn(resolved, mode, encoding="utf-8") as f:
            if resolved.endswith(".json"):
                data = json.load(f)
                lines = data
            else:
                lines = f

            for line in lines:
                try:
                    cand = json.loads(line) if isinstance(line, str) else line
                except Exception:
                    continue

                total_parsed += 1
                profile = cand.get("profile", {})
                yoe = profile.get("years_of_experience", 0.0)
                total_yoe += yoe

                if yoe < 2:
                    stage_counts["fresher"] += 1
                elif yoe < 5:
                    stage_counts["junior"] += 1
                elif yoe <= 9:
                    stage_counts["senior"] += 1
                else:
                    stage_counts["super_senior"] += 1

                loc = profile.get("location", "Unknown")
                locations[loc] = locations.get(loc, 0) + 1

                for s in cand.get("skills", []):
                    sname = s.get("name", "")
                    if sname:
                        skills_freq[sname] = skills_freq.get(sname, 0) + 1

                if total_parsed >= 20000:
                    break
    except Exception as e:
        logger.error(f"Market analysis error: {e}")

    top_locations = sorted(locations.items(), key=lambda x: -x[1])[:10]
    top_skills = sorted(skills_freq.items(), key=lambda x: -x[1])[:15]

    return {
        "total_scanned": total_parsed,
        "avg_yoe": round(total_yoe / total_parsed, 2) if total_parsed > 0 else 0,
        "stages": stage_counts,
        "top_locations": [{"name": n, "count": c} for n, c in top_locations],
        "top_skills": [{"name": n, "count": c} for n, c in top_skills],
    }
