import os, base64, io, uuid
from dotenv import load_dotenv
# Ensure .env next to this file is loaded (explicit path for Flask cwd issues)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

# Sync Gemini key (global GOOGLE_API_KEY is suspended)
_gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if _gemini_key:
    os.environ["GOOGLE_API_KEY"] = _gemini_key
    os.environ["GEMINI_API_KEY"] = _gemini_key

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or ""
SERPER_KEY = os.getenv("SERPER_API_KEY", "")

from flask import Flask, render_template, request, jsonify, url_for
from huggingface_hub import InferenceClient
from crewai import Agent, Task, LLM, Crew
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
from typing import List

class EmailOutput(BaseModel):
    subjects: List[str] = Field(description="Exactly 3 subject lines")
    preheader: str = Field(description="Short preheader 40-60 chars")
    body_text: str = Field(description="Email body 150-250 words, plain text")
    body_html: str = Field(description="Same body as HTML paragraphs")
    cta: str = Field(description="Call to action line")

class PostOutput(BaseModel):
    posts: dict = Field(description="Platform -> post text mapping")

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- LLM (primary + fallback via LiteLLM) ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
gemini_llm = LLM(model="gemini/gemini-3.6-flash", api_key=_gemini_key, temperature=0.7)
# Email: simple - only openai/gpt-oss-120b as primary (as requested), HF as fallback. No max_tokens limit - prompt controls length.
groq_120b_llm = LLM(model="groq/openai/gpt-oss-120b", api_key=GROQ_API_KEY, temperature=0.6) if GROQ_API_KEY else None
hf_llm = LLM(model="huggingface/meta-llama/Llama-3.1-8B-Instruct", api_key=HF_TOKEN, temperature=0.6) if HF_TOKEN else None
# Keep old vars for enhancer/post compatibility (alias to 120b)
groq_allam_llm = groq_120b_llm
groq_llm = groq_120b_llm
groq_qwen_llm = None
llm = gemini_llm
web_search = SerperDevTool()

def get_llm_with_fallback(primary=gemini_llm):
    """Return LLM with health check order: gemini -> groq -> hf"""
    return primary  # crew kickoff will try primary; caller handles exception fallback

# ---------- Agents (4 original + 1 Email Writer new with Job Apply expertise) ----------
Query_Interpreter_Agent = Agent(role="Query Interpreter", goal="Understand the raw user query and extract structured metadata for content creation. Only interpret, never generate posts.", backstory="You are the first checkpoint. Identify platform(s), topic, content type, and clean query.", verbose=False, llm=llm)
Web_Search_Agent = Agent(role="Web Search Agent", goal="Receive clean query, perform focused web search and return 5-8 factual results (title+source+summary). No opinions.", backstory="You are the fact collector. Fetch reliable recent info only.", tools=[web_search], verbose=False, llm=llm)
News_Filter_Agent = Agent(role="News Filter Agent", goal="Filter raw search results. Keep 5-8 most relevant, remove duplicates/outdated/promotional. Return filtered_news + removed_items.", backstory="You are the gatekeeper. Only credible facts move forward.", verbose=False, llm=llm)
Post_Creator_Agent = Agent(role="Post Creator Agent", goal="Use filtered news to draft platform-specific posts. WhatsApp <=450 chars, LinkedIn 120-200 words, Instagram emoji-rich, Twitter 3-6 thread <=280, Facebook 2-3 para, Telegram concise, YouTube hook+context. Adapt to requested platforms.", backstory="You are the storyteller. Clarity and platform structure first.", verbose=False, llm=llm)
Email_Writer_Agent = Agent(role="Email Writer Agent", goal="You craft ALL email types: Job Application, Cold outreach, Newsletter. For Job Application: write ATS-friendly cover email with 3 distinct subject lines, personalized greeting, 2-3 paragraph pitch linking resume highlights to job description without duplication, strong CTA, professional sign-off using EXACT sender_name provided. Never invent a name.", backstory="You are an elite email copywriter + career coach. You never hallucinate a sender name — use sender_name if provided else [Your Name]. You never duplicate PostCraft description and never include JSON inside body_text.", verbose=False, llm=llm)

# ---------- Tasks (global crews - simple JSON for email) ----------
Query_Interpreter_Task = Task(description="Read raw user request: {query}. Extract platforms, topic, content_type, clean_query. Be concise, return bullet list.", expected_output='Platform: LinkedIn, Topic: AI, Clean query: AI', agent=Query_Interpreter_Agent)
Web_Search_Task = Task(description="Take clean_query and search 5-8 recent facts with source.", expected_output='Bullet list of 5-8 facts with title - source - summary', agent=Web_Search_Agent)
News_Filter_Task = Task(description="Filter raw search: keep 5-8 relevant facts, remove duplicates.", expected_output='Filtered facts as bullet list', agent=News_Filter_Agent)
Post_Creator_Task = Task(description="Turn filtered_news into platform-specific posts for the platforms. Follow length rules.", expected_output='Posts per platform, clearly labeled', agent=Post_Creator_Agent)
Email_Writer_Task = Task(
    description="""Write a {email_type} email for {recipient} about {context}. 
Sender: {sender_name} | Job: {job_title} at {company} | Hiring manager: {hiring_manager} | JD: {job_description} | Resume: {resume_highlights} | Portfolio: {portfolio} | Tone: {tone}.
Rules:
- Subjects: exactly 3 distinct lines, each 6-10 words, must include job_title and company, never just a person's name or raw JD snippet. Example: "Application for Python Developer — Flask Expertise | Infosys"
- Preheader: 1 line 40-60 chars, no "Dear..."
- Body: Dear {hiring_manager} or Dear Hiring Manager, 3 paragraphs max 180 words total, mention PostCraft ONLY once, map 2 resume highlights to JD requirements, no duplication, close with portfolio link and sign-off using EXACT sender_name (if empty use [Your Name]). Never add "Here's the JSON..." or nested JSON inside body_text.
- Return ONLY valid JSON with keys subjects (array 3), preheader, body_text, body_html (same body as HTML <p>), cta. No markdown, no extra keys.""",
    expected_output='{"subjects": ["Application for Python Developer — Flask & CrewAI | Infosys","Python Developer (3 yrs Flask) - Available for Interview - Infosys","Re: Python Developer at Infosys - B.Tech CSE + PostCraft"], "preheader": "Flask REST APIs, CrewAI — 3 yrs Python", "body_text": "Dear Amit Kumar,\\n\\nI am excited to apply...", "body_html": "<p>Dear Amit Kumar,</p><p>...</p>", "cta": "Available for interview this week - portfolio https://..."}',
    agent=Email_Writer_Agent
)

# Crews - Post: 4 tasks (with web search). Email: 1-2 tasks (NO web search) — direct write, no research needed
post_crew = Crew(agents=[Query_Interpreter_Agent, Web_Search_Agent, News_Filter_Agent, Post_Creator_Agent], tasks=[Query_Interpreter_Task, Web_Search_Task, News_Filter_Task, Post_Creator_Task], verbose=False)
email_crew = Crew(agents=[Email_Writer_Agent], tasks=[Email_Writer_Task], verbose=False)

def _build_crews_with_llm(llm_obj):
    """Recreate crews with given LLM. Post: 4 agents with web search. Email: 1 agent only (no web search) — direct write."""
    is_groq = llm_obj and "groq" in str(getattr(llm_obj, "model", "")).lower()
    if is_groq:
        # Email: single Email Writer (no Query) — fastest, no research, avoids OTPM waste
        e = Agent(role="Email Writer Agent", goal="Craft email JSON. Never hallucinate sender name. Never output thinking.", backstory="Email copywriter.", verbose=False, llm=llm_obj)
        et = Task(description="Write a {email_type} email for {recipient} about {context}. Sender: {sender_name} | Job: {job_title} at {company} | Hiring manager: {hiring_manager} | JD: {job_description} | Resume: {resume_highlights} | Portfolio: {portfolio} | Tone: {tone}. CRITICAL: Output ONLY valid JSON with keys subjects (3 distinct, each must include job_title and company), preheader, body_text, body_html, cta. Do NOT write 'I understand...', 'Thought:', 'Email Components:', 'Here's the JSON', or any preamble. No markdown outside JSON, no nested JSON inside body_text.", expected_output='{"subjects":["Application for Python Developer — Flask | Infosys","Python Developer (3 yrs Flask) - Infosys","Re: Python Developer at Infosys"],"preheader":"Flask REST APIs, CrewAI — 3 yrs","body_text":"Dear Amit Kumar,\\n... Best regards,\\nRounak","body_html":"<p>Dear Amit Kumar,</p><p>...</p>","cta":"Available for interview"}', agent=e)
        # Post: lightweight 2 agents for Groq
        q = Agent(role="Query Interpreter", goal="Extract topic and intent from query.", backstory="Checkpoint.", verbose=False, llm=llm_obj)
        p = Agent(role="Post Creator Agent", goal="Draft posts concisely.", backstory="Storyteller.", verbose=False, llm=llm_obj)
        qt = Task(description="Read query: {query}. Extract topic.", expected_output='Topic: ...', agent=q)
        pt = Task(description="Create posts for {query}.", expected_output='Posts labeled', agent=p)
        post = Crew(agents=[q,p], tasks=[qt,pt], verbose=False)
        email = Crew(agents=[e], tasks=[et], verbose=False)
        return post, email
    # Gemini/HF: Post 4 agents, Email still 1 agent (no need for web search in email)
    q = Agent(role="Query Interpreter", goal="Understand raw user query.", backstory="Checkpoint.", verbose=False, llm=llm_obj)
    w = Agent(role="Web Search Agent", goal="Receive clean query, perform focused web search.", backstory="Collector.", tools=[web_search], verbose=False, llm=llm_obj)
    n = Agent(role="News Filter Agent", goal="Filter raw search results.", backstory="Gatekeeper.", verbose=False, llm=llm_obj)
    p = Agent(role="Post Creator Agent", goal="Use filtered news to draft posts.", backstory="Storyteller.", verbose=False, llm=llm_obj)
    e = Agent(role="Email Writer Agent", goal="Craft email JSON with 3 subjects + body. Never output thinking.", backstory="Copywriter.", verbose=False, llm=llm_obj)
    qt = Task(description="Read raw user request: {query}. Extract clean_query.", expected_output='Clean query: ...', agent=q)
    wt = Task(description="Take clean_query and search 5-8 recent facts.", expected_output='Bullet list', agent=w)
    nt = Task(description="Filter raw search: keep most relevant.", expected_output='Filtered facts', agent=n)
    pt = Task(description="Turn filtered_news into posts.", expected_output='Posts', agent=p)
    et = Task(description="Write a {email_type} email for {recipient} about {context}. Sender: {sender_name} | Job: {job_title} at {company} | Hiring manager: {hiring_manager} | JD: {job_description} | Resume: {resume_highlights} | Portfolio: {portfolio} | Tone: {tone}. CRITICAL: Output ONLY valid JSON with keys subjects, preheader, body_text, body_html, cta. Do NOT write 'I understand', 'Thought:', 'Email Components' or nested JSON in body.", expected_output='{"subjects":["S1","S2","S3"],"preheader":"...","body_text":"Dear ...","body_html":"<p>Dear ...</p>","cta":"..."}', agent=e)
    post = Crew(agents=[q,w,n,p], tasks=[qt,wt,nt,pt], verbose=False)
    email = Crew(agents=[e], tasks=[et], verbose=False)
    return post, email

def run_crew_with_fallback(crew, inputs):
    """CrewAI demo: Post uses gemini->groq->hf; Email uses groq/gpt-oss-120b -> HF only (simple as requested)."""
    import time
    is_post = any("Post Creator" in a.role for a in crew.agents)
    is_email = any("Email Writer" in a.role for a in crew.agents)
    last_err = None
    if is_email:
        # Email: simple - gpt-oss-120b primary -> HF fallback (allam/qwen removed per user)
        fallbacks = [groq_120b_llm, hf_llm]
    else:
        # Post: keep Gemini primary
        fallbacks = [gemini_llm, groq_120b_llm, hf_llm]
    fallbacks = [f for f in fallbacks if f is not None]
    for idx, fb in enumerate(fallbacks):
        for attempt in range(2):
            try:
                post_fb, email_fb = _build_crews_with_llm(fb)
                target = post_fb if is_post else email_fb
                result = target.kickoff(inputs=inputs)
                if result is None or str(result).strip() == "":
                    raise ValueError("LLM returned empty response")
                return result
            except Exception as e:
                last_err = e
                try:
                    msg = str(e).encode("ascii", "replace").decode()
                    print(f"[PostCraft fallback {idx} attempt {attempt}] {getattr(fb, 'model', 'unknown')} failed: {msg[:350]}")
                except:
                    print(f"[PostCraft fallback {idx} attempt {attempt}] failed")
                err_low = str(e).lower()
                # Daily quota (Gemini 20/day) -> no wait, immediate fallback
                if "perday" in err_low or "generate requests per day" in err_low:
                    break
                if any(x in err_low for x in ["ratelimit", "429", "otpm", "quota", "resource_exhausted"]):
                    import re
                    m = re.search(r"try again in (\d+)", err_low)
                    wait = int(m.group(1)) + 1 if m else 6
                    wait = min(wait, 20)
                    print(f"  -> OTPM wait {wait}s")
                    time.sleep(wait)
                    if attempt == 0:
                        continue
                elif any(x in err_low for x in ["400", "parsing", "invalid_request", "empty"]):
                    time.sleep(2)
                break
    raise last_err if last_err else RuntimeError("All LLM fallbacks failed. Gemini daily 20 hit, Groq OTPM 1000 hit. Wait 45s or add HF_TOKEN for huggingface fallback.")

# ---------- Image helper (separate) ----------
def generate_image(prompt, model="krea/Krea-2-Turbo"):
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN missing in .env (hf_Kmou...). Add HF_TOKEN to AI Post Creator/.env")
    client = InferenceClient(provider="fal-ai", api_key=HF_TOKEN)
    image = client.text_to_image(prompt, model=model)
    return image  # PIL.Image

def generate_image_with_fallback(prompt):
    """Post media only: try 3 models in order if 402/credits fails. Image media stays single-model (don't touch)."""
    models = ["krea/Krea-2-Turbo", "stabilityai/stable-diffusion-3-medium", "black-forest-labs/FLUX.1-dev"]
    last_err = None
    for m in models:
        try:
            client = InferenceClient(provider="fal-ai", api_key=HF_TOKEN)
            img = client.text_to_image(prompt, model=m)
            return img, m
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "402" in msg or "payment required" in msg or "credits" in msg or "billing" in msg:
                print(f"[Image fallback] {m} failed 402, trying next: {msg[:200]}")
                continue
            print(f"[Image fallback] {m} failed, trying next: {msg[:200]}")
            continue
    raise last_err if last_err else RuntimeError("All 3 image models failed (credits depleted?)")

def create_visual_image_prompt(query):
    """Use Groq small model to create a clean visual prompt from query — no text, no typography."""
    try:
        enhancer = groq_allam_llm or groq_120b_llm or hf_llm or gemini_llm
        prompt = (
            f"You are an expert visual prompt engineer for social media images. "
            f"User wants posts about: '{query}'. "
            f"Create ONE image prompt (under 30 words) describing a photorealistic or illustrated scene that matches the topic. "
            f"Include style, lighting, composition, colors, mood, 4k, high detail. "
            f"CRITICAL: Visual only — no text, no words, no letters, no typography, no captions, no garbled text. "
            f"Return ONLY the prompt, no quotes, no explanation."
        )
        visual = enhancer.call(prompt).strip().strip('"').strip("'")
        if "```" in visual:
            visual = visual.split("```")[-2].strip() if visual.count("```") >=2 else visual.replace("```","").strip()
        # Ensure no text instruction leaked
        if "text" in visual.lower() and "no text" not in visual.lower():
            visual += ", no text, no typography"
        return visual
    except Exception as e:
        print(f"Visual prompt fallback: {e}")
        return f"Abstract {query}, modern gradient, minimal, no text, 4k, high detail"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate_post", methods=["POST"])
def api_generate_post():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    attach = data.get("attach", False)
    platforms = data.get("platforms", "auto")
    tone = data.get("tone", "Professional")
    if not query:
        return jsonify(error="query required"), 400
    # If user selected specific platform via dropdown, enforce it (ignore prompt's extra platforms)
    if platforms and platforms != "auto":
        # Append explicit instruction so crew respects dropdown over raw prompt
        query = f"{query} [IMPORTANT: Generate ONLY for {platforms}. Tone: {tone}. Ignore other platforms mentioned in prompt.]"
    try:
        result = run_crew_with_fallback(post_crew, inputs={"query": query})
        text = str(result).strip()
        # Post-filter: if specific platform selected, keep only that section to avoid LinkedIn+Twitter when only LinkedIn chosen
        if platforms and platforms != "auto" and platforms != "All":
            import re
            # Try to extract that platform's block (e.g., **LinkedIn Post** ... until next **<Platform> Post** or ---)
            pattern = re.compile(rf"(\*\*{re.escape(platforms)}\s+Post\*\*.*?)(?=\n\*\*\w+ Post\*\*|\n---|$)", re.DOTALL | re.IGNORECASE)
            m = pattern.search(text)
            if m:
                text = m.group(1).strip()
        image_url = None
        image_error = None
        image_model_used = None
        visual_prompt_used = None
        if attach:
            # Create clean visual prompt via Groq (understands query, no garbled text)
            visual_prompt = create_visual_image_prompt(query)
            visual_prompt_used = visual_prompt
            img_prompt = visual_prompt
            try:
                img, used_model = generate_image_with_fallback(img_prompt)
                image_model_used = used_model
                fname = f"generated_post_{uuid.uuid4().hex[:8]}.png"
                save_path = os.path.join(app.static_folder, "images", fname)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                img.save(save_path)
                image_url = url_for("static", filename=f"images/{fname}")
                print(f"[Post image] success with {used_model}")
            except Exception as ie:
                msg = str(ie)
                if "402" in msg or "Payment Required" in msg or "credits" in msg.lower():
                    image_error = "All 3 HF image models hit 402 credits depleted. Post generated without image — add credits at huggingface.co/settings/billing or try Image Studio later."
                else:
                    image_error = f"Image attach skipped: {msg[:200]}"
        return jsonify(result=text, image_url=image_url, image_error=image_error, image_model_used=image_model_used, visual_prompt=visual_prompt_used)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/api/generate_image", methods=["POST"])
def api_generate_image():
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    model = data.get("model", "krea/Krea-2-Turbo")
    if not prompt:
        return jsonify(error="prompt required"), 400
    try:
        img = generate_image(prompt, model=model)
        # Save to static for serving
        fname = f"generated_{uuid.uuid4().hex[:8]}.png"
        save_path = os.path.join(app.static_folder, "images", fname)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
        # Also create base64 for immediate preview if needed
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        image_url = url_for("static", filename=f"images/{fname}")
        return jsonify(image_url=image_url, b64=f"data:image/png;base64,{b64}")
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/api/generate_email", methods=["POST"])
def api_generate_email():
    data = request.get_json() or {}
    email_type = data.get("email_type", "Newsletter")
    recipient = data.get("recipient", "General audience")
    context = data.get("context", "").strip() or data.get("query", "")
    tone = data.get("tone", "Professional")
    sender_name = data.get("sender_name", "").strip()
    job_title = data.get("job_title", "")
    company = data.get("company", "")
    hiring_manager = data.get("hiring_manager", "")
    job_description = data.get("job_description", "")
    resume_highlights = data.get("resume_highlights", "")
    portfolio = data.get("portfolio", "")
    if email_type == "Job Application" and job_title:
        context = f"Job: {job_title} at {company}. Hiring manager: {hiring_manager}. Job description: {job_description}. Resume highlights: {resume_highlights}. Portfolio: {portfolio}. Original context: {context}"
    if not context:
        return jsonify(error="context required"), 400
    try:
        inputs = {
            "query": context,
            "email_type": email_type,
            "recipient": recipient or (hiring_manager or company or "Hiring Manager"),
            "context": context,
            "tone": tone,
            "sender_name": sender_name if sender_name else "[Your Name]",
            "job_title": job_title,
            "company": company,
            "hiring_manager": hiring_manager,
            "job_description": job_description,
            "resume_highlights": resume_highlights,
            "portfolio": portfolio,
            "filtered_news": f"portfolio:{portfolio}" if portfolio else "N/A"
        }
        result = run_crew_with_fallback(email_crew, inputs)
        import json, re
        # Helper to clean and parse JSON - handles fences, extra text, and ensures subjects/body are extracted
        def try_parse_json(s):
            s = s.strip()
            # Remove ```json fences if present
            if "```" in s:
                parts = s.split("```")
                for p in parts:
                    p = p.strip()
                    if p.startswith("json"):
                        p = p[4:].strip()
                    if p.startswith("{") and p.endswith("}"):
                        try:
                            return json.loads(p)
                        except:
                            continue
                # fallback to first JSON block
                s = s.replace("```", "")
            # Direct parse
            try:
                return json.loads(s)
            except:
                pass
            # Find JSON object boundaries (first { to last })
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start:end+1])
                except:
                    pass
            return None

        # If result is already dict/Pydantic
        if isinstance(result, dict) and "subjects" in result:
            j = result
            return jsonify(result="", subjects=j.get("subjects",[]), preheader=j.get("preheader",""), body_text=j.get("body_text",""), body_html=j.get("body_html",""), cta=j.get("cta",""))
        if hasattr(result, "model_dump"):
            try:
                j = result.model_dump()
                if "subjects" in j:
                    return jsonify(result="", subjects=j.get("subjects",[]), preheader=j.get("preheader",""), body_text=j.get("body_text",""), body_html=j.get("body_html",""), cta=j.get("cta",""))
            except:
                pass

        text = str(result).strip()
        j = try_parse_json(text)
        if j and isinstance(j, dict) and "subjects" in j:
            # Success - return parsed JSON only, hide raw JSON from UI (result="": no exposure)
            return jsonify(result="", subjects=j.get("subjects",[]), preheader=j.get("preheader",""), body_text=j.get("body_text",""), body_html=j.get("body_html",""), cta=j.get("cta",""))
        # Last resort: parsing failed - return raw as body but don't expose JSON duplication
        # Extract body after Dear if possible to avoid showing Subjects duplication
        body_text = text
        # If text still looks like JSON with subjects, try to extract body_text field via regex as last attempt
        m_body = re.search(r'"body_text"\s*:\s*"(.*?)"\s*,\s*"body_html"', text, re.DOTALL)
        if m_body:
            try:
                body_text = bytes(m_body.group(1), "utf-8").decode("unicode_escape")
            except:
                body_text = m_body.group(1)
        # Clean markdown ** inside body
        return jsonify(result="", subjects=[], preheader="", body_text=body_text, body_html=f"<p>{body_text.replace(chr(10), '<br>')}</p>", cta="")
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/api/enhance_prompt", methods=["POST"])
def api_enhance_prompt():
    """Prompt enhancer via Groq small model - fast, cheap, no CrewAI overhead"""
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    kind = data.get("kind", "post")  # post or image
    if not text:
        return jsonify(error="text required"), 400
    # Use Groq small model for enhancement (allam-2-7b is fastest, 400 tokens)
    enhancer_llm = groq_allam_llm or hf_llm or groq_llm or gemini_llm
    try:
        if kind == "image":
            prompt = f"You are an image prompt engineer. Enhance this image prompt for AI image generation (krea/Krea-2-Turbo / SD3). Add style, lighting, composition, detail, 4k, aspect guidance. Keep under 40 words. Original: {text}. Return ONLY the enhanced prompt, no quotes, no explanation."
        else:
            prompt = f"You are a viral post copywriter. Enhance this social media prompt to be more engaging, SEO-friendly, and platform-optimized. Keep intent, add hook and hashtags suggestion. Keep under 40 words. Original: {text}. Return ONLY the enhanced prompt, no explanation."
        enhanced = enhancer_llm.call(prompt).strip()
        # Clean fences/quotes
        enhanced = enhanced.strip().strip('"').strip("'").strip()
        if "```" in enhanced:
            enhanced = enhanced.split("```")[-2].strip() if enhanced.count("```") >=2 else enhanced.replace("```","").strip()
        return jsonify(enhanced=enhanced)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    # Flask - disable reloader to avoid watchdog crash (cannot schedule new futures after shutdown) while Crew threads run
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
