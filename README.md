# PostCraft

> No matter what you want — we craft it.

AI-powered content studio to create **social media posts**, **images**, and **emails** in one click.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat&logo=flask&logoColor=white)
![CrewAI](https://img.shields.io/badge/CrewAI-1.6-7C5CFF?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## About

PostCraft is a unified AI workspace for creators, marketers, and job seekers. Generate platform-ready posts, studio-quality images, and high-converting emails from a single prompt — with research, filtering, and brand consistency built in.

Live: `http://127.0.0.1:5000`

---

## Features

**📝 Post Studio**
- Generate for LinkedIn, Instagram, Twitter/X, WhatsApp, Facebook, Telegram, YouTube
- Tone control (Professional, Viral, Casual, Informative)
- Optional image attach

**🎨 Image Studio**
- Text-to-image via HuggingFace `fal-ai` (Krea, SD3, FLUX)
- Style presets and model picker

**✉️ Email Studio**
- 11 types including **Job Application** with ATS-friendly subjects
- Maps resume highlights to job description
- Strict JSON output (subjects, preheader, body)

**✨ Prompt Enhancer**
- One-click prompt improvement for posts and images

---

## Tech Stack

- **Backend:** Flask, CrewAI, LiteLLM
- **LLMs:** Gemini 3.6 Flash, Groq (gpt-oss-120b), HuggingFace (LLaMA 3.1)
- **Search & Image:** Serper.dev, Fal-AI
- **Frontend:** HTML, CSS, JavaScript

---

## Architecture & How Agents Work

PostCraft uses **CrewAI** for Post & Email. Image & Enhancer are direct LLM calls (no Crew).

### Overview — How Many Agents?

- **Post Studio:** **4 Agents, 4 Tasks** — full research pipeline
- **Email Studio:** **2 Agents (Groq) / 4 Agents (Gemini/HF)** — Job Apply + Email Writer
- **Image Studio:** **0 Agents** — direct `fal-ai` diffusion
- **Prompt Enhancer:** **0 Agents** — direct Groq `allam-2-7b`

### Flow (Mermaid)

```mermaid
graph TD
    U[User Prompt] --> Q[Query Interpreter]
    Q --> S{LLM?}
    S -- Gemini/HF --> W[Web Search<br/>SerperDevTool<br/>5-8 facts]
    S -- Groq --> WK[Knowledge Summary<br/>3-5 facts<br/>no tool]
    W --> N[News Filter<br/>keep 5-8]
    WK --> N
    N --> P[Post Creator<br/>platform rules]
    N --> E[Email Writer<br/>JSON: subjects/preheader/body/cta]
    P --> OUT[Post Output]
    E --> EOUT[Email JSON → Preview]
    U -.->|Post + Attach checked| V[Visual Prompt<br/>Groq: no text, 30w]
    V --> IMG[fal-ai<br/>krea → SD3 → FLUX<br/>fallback]
    IMG --> POUT[Attached Image]
    U -. Direct .-> IMG2[Image Studio<br/>fal-ai single model]
    U -. Direct .-> ENH[Prompt Enhancer<br/>Groq allam-2-7b<br/>40 words]
```

**Post Studio (4 agents, ~30-45s):**
```
User: "create LinkedIn post about AI news" + Platforms=LinkedIn + Tone=Professional
  ↓
[1] Query Interpreter (Gemini) → {platform:LinkedIn, topic:AI, clean_query:"AI news"}
  ↓
[2] Web Search (Serper/Summary) → 5-8 facts: title+source+summary
  ↓
[3] News Filter → filtered_news (5-8) + removed_items
  ↓
[4] Post Creator → LinkedIn Post 120-200w + hashtags
  ↓ (if attach checked)
[5] Visual Prompt (Groq) → "Futuristic AI, gradient, no text, 4k" → fal-ai fallback 3 models
```

**Email Studio (2 agents Groq / 4 agents Gemini):**
```
Groq (fast, no tool):
  Query Interpreter (extracts topic) → Email Writer (3 ATS subjects + body mapping resume→JD)
  Inputs: job_title, company, hiring_manager, JD, resume_highlights, portfolio, sender_name, tone
  Output: {"subjects":[3], "preheader", "body_text", "body_html", "cta"}

Gemini/HF (research):
  Query Interpreter → Web Search → News Filter → Email Writer (same output)
```

**Image Studio (0 agents):**
```
Prompt → Style → Model (krea/SD3/FLUX) → InferenceClient(provider="fal-ai") → PNG
```

**Prompt Enhancer (0 agents):**
```
Post: "AI news" → Groq allam-2-7b → "Craft punchy LinkedIn post... hook, SEO, hashtags"
Image: "AI workspace" → Groq → "Futuristic AI workspace, neon, 4k, no text"
```

### Fallback Chain

- **Post:** `gemini 3.6-flash` → `groq gpt-oss-120b` → `hf meta-llama` (daily quota 20 vs OTPM 1000)
- **Email:** `groq gpt-oss-120b` → `hf meta-llama` (Gemini removed, no max_tokens)
- **Image (Post only):** `krea-2-Turbo` → `SD3-medium` → `FLUX.1-dev` on `402/credits`

---

## Quick Start

```bash
git clone <your-repo-url>
cd "AI Post Creator"

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install flask crewai crewai-tools huggingface_hub python-dotenv pydantic
```

Create env file:

```bash
cp .env.example .env
# edit .env with your keys
```

Run:

```bash
python app.py
# open http://127.0.0.1:5000
```

---

## Environment Variables

Create `.env` in the project root (see `.env.example`):

```env
GEMINI_API_KEY=your_gemini_key
GOOGLE_API_KEY=your_google_key
SERPER_API_KEY=your_serper_key
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_key
```

| Key | Where to get |
|-----|--------------|
| `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey |
| `SERPER_API_KEY` | https://serper.dev |
| `HF_TOKEN` | https://huggingface.co/settings/tokens |
| `GROQ_API_KEY` | https://console.groq.com/keys |

---

## Usage

1. **Posts:** Enter a topic → choose platform & tone → Generate (add image optionally)
2. **Images:** Enter a prompt → choose style & model → Generate
3. **Emails:** Choose type (e.g., Job Application) → fill job details + your name → Draft

All content can be copied or downloaded as `.txt`.

---

## Project Structure

```
AI Post Creator/
├── app.py              # Flask app
├── main.py             # Streamlit version (legacy)
├── static/
│   ├── css/style.css
│   └── images/
├── templates/
│   └── index.html
├── .env.example
├── README.md
└── LICENSE
```

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/generate_post` | POST | Create posts |
| `/api/generate_image` | POST | Create image |
| `/api/generate_email` | POST | Draft email |
| `/api/enhance_prompt` | POST | Enhance prompt |

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## License

MIT — see [LICENSE](LICENSE).

---

Built with CrewAI, Flask, and PostCraft.
