# Skill Evidence ATS — Backend API Server

This is the FastAPI-based backend server for the **Skill Evidence ATS** system. It handles structured candidate resume parsing, job description extraction, and efficient batch ranking execution.

## Features
- **FastAPI Endpoints**: Fast, high-performance, asynchronous REST API layer.
- **Job Description Understanding**: Parses raw JD text into required/preferred skills and target seniority using Gemini 1.5 Flash.
- **Resume Parsing**: Leverages Gemini 1.5 Flash to automatically extract candidate names, skills, career histories, and portfolio links.
- **Batch Evaluation & Stream-based Candidate Discovery**: Ranks massive datasets (100K candidates) using memory-efficient python generator line-readers (<20MB RAM, <13s execution time).
- **Honeypot & Disqualification Systems**: Identifies fraudulent profiles and implements standard hiring criteria exclusions (service companies, CV/Speech-only focus, academic research focus).
- **Deep Search & Auditing**: Simulates live web-scraping verification of candidate social metrics (GitHub activity scores and LinkedIn connectivity) to reward active developers with platform score boosts.


## Getting Started

### Prerequisites
- **Python 3.9+** (Required)

### Setup & Run
1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Configure custom Local GGUF LLM settings in a `.env` file (see the "Adjusting Local GGUF LLM Settings" section below).
4. Run the development server:
   ```bash
   python -m uvicorn main:app --reload --port 8000
   ```

### ⚙️ Adjusting Local GGUF LLM Settings
By default, the backend downloads and caches the Qwen2.5 model files from Hugging Face Hub. You can customize where the GGUF model is loaded or downloaded by copying `backend/.env.example` to `backend/.env` and editing the following variables:

* **Direct Local Path Loading:** Point `LOCAL_MODEL_PATH` to an existing `.gguf` file on your local disk to bypass Hugging Face downloads entirely. This is highly useful if you have pre-downloaded custom models:
  ```env
  LOCAL_MODEL_PATH="C:\path\to\your\model.gguf"
  ```
* **Custom Hugging Face Download:** Specify a custom GGUF model repository and file to download:
  ```env
  HF_MODEL_REPO_ID="Qwen/Qwen2.5-1.5B-Instruct-GGUF"
  HF_MODEL_FILENAME="qwen2.5-1.5b-instruct-q4_k_m.gguf"
  ```
* **Custom Download Location:** Override the local caching directory where downloaded Hugging Face models are stored:
  ```env
  HF_CACHE_DIR="D:\models\huggingface"
  ```

---

## 💻 Command Line Interface & API Execution

### 1. Toggleable Options & Flags
The ranking engine includes several parameters that can be toggled on/off to modify the evaluation context:

| Feature | CLI Flag | API Parameter | Behavior when ON | Behavior when OFF |
| :--- | :--- | :--- | :--- | :--- |
| **Deep Search** | `--deep-search` | `"deep_search": true` | Simulates live web-audit scraping; awards +3.0 boost for active GitHub profiles (>50 score) and +2.0 boost if LinkedIn is connected. | Runs candidate evaluation offline using platform dataset records only. |
| **Custom JD Matching** | `--jd <file_path>` | `"jd_profile": {...}` | Uses custom required and preferred skills extracted from the custom JD to rank candidate suitability. | Default challenge Job Description (Founding AI Engineer) is used. |

### 2. Standalone CLI Ranking Examples (`rank.py`)
To process and rank candidate files directly via the CLI:
- **Default ranking run (Deep Search OFF)**:
  ```bash
  python ../rank.py --candidates "path/to/candidates.jsonl" --out "submission.csv"
  ```
- **Deep Search ON**:
  ```bash
  python ../rank.py --candidates "path/to/candidates.jsonl" --deep-search --out "submission.csv"
  ```
- **Custom JD & Deep Search ON**:
  ```bash
  python ../rank.py --candidates "path/to/candidates.jsonl" --deep-search --jd "path/to/custom_jd.json" --out "submission.csv"
  ```

### 3. API Invocation Examples
**Endpoint**: `POST http://localhost:8000/api/v1/sandbox/rank-batch`

- **Payload format with Deep Search toggled ON and custom JD**:
  ```json
  {
    "file_path": "candidates.jsonl",
    "deep_search": true,
    "jd_profile": {
      "title": "Senior AI Engineer",
      "requiredSkills": ["embeddings-based retrieval systems", "vector databases"],
      "preferredSkills": ["LLM fine-tuning", "learning-to-rank models"]
    }
  }
  ```

- **Payload format with Deep Search toggled OFF (runs fast, no external/audit boosts)**:
  ```json
  {
    "file_path": "candidates.jsonl",
    "deep_search": false
  }
  ```

- **Example request via `curl`**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/sandbox/rank-batch" \
       -H "Content-Type: application/json" \
       -d '{"file_path": "candidates.jsonl", "deep_search": true}'
  ```

---

## 💡 AI Transparency Note
In the interest of professional integrity and engineering transparency, I want to state clearly that **Google Antigravity** (Google DeepMind's advanced agentic coding assistant) was utilized during the design, implementation, and optimization of this project.

This choice was not due to a lack of technical knowledge or programming capability, but rather to maximize efficiency. Translating complex conceptual ideas into a production-ready system within a compressed hackathon timeline is a major constraint. Using AI allowed me to quickly prototype, test, and iterate on my ideas in real-time. System design is fundamentally a process of trial and error, and using Google Antigravity helped streamline this cycle, reduce formatting/boilerplate errors, and deliver a robust solution in the limited time available. I believe in utilizing modern tools to build better software, and I am proud of the hybrid human-AI engineering process used to bring this system to life.

