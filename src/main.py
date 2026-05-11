"""
SHL Assessment Recommender Agent
FastAPI service with conversational AI-powered assessment recommendations.
"""

import json
import os
import re
import time
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data Models ──────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

    @validator("role")
    def role_must_be_valid(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    messages: List[Message]

    @validator("messages")
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


# ─── Catalog Loading ───────────────────────────────────────────────────────────
CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"
VALID_URLS: set = set()
CATALOG: list = []


def load_catalog() -> list:
    global CATALOG, VALID_URLS
    with open(CATALOG_PATH) as f:
        CATALOG = json.load(f)
    VALID_URLS = {item["url"] for item in CATALOG}
    logger.info(f"Loaded {len(CATALOG)} assessments from catalog")
    return CATALOG


# ─── Embedding & Retrieval ────────────────────────────────────────────────────
EMBEDDINGS: Optional[np.ndarray] = None
EMBED_MODEL = None


def get_embed_model():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformers embedding model")
        except Exception as e:
            logger.warning(f"Could not load sentence-transformers: {e}. Using keyword fallback.")
    return EMBED_MODEL


def build_searchable_text(item: dict) -> str:
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keywords", [])),
        " ".join(item.get("job_levels", [])),
        f"test type {item.get('test_type', '')}",
    ]
    return " ".join(parts).lower()


def build_embeddings():
    global EMBEDDINGS
    model = get_embed_model()
    if model is None:
        return
    texts = [build_searchable_text(item) for item in CATALOG]
    EMBEDDINGS = model.encode(texts, convert_to_numpy=True)
    # Normalize for cosine similarity
    norms = np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True)
    EMBEDDINGS = EMBEDDINGS / np.maximum(norms, 1e-9)
    logger.info(f"Built embeddings for {len(CATALOG)} items")


def semantic_search(query: str, top_k: int = 15) -> list:
    """Return top_k catalog items ranked by semantic similarity."""
    model = get_embed_model()
    if model is None or EMBEDDINGS is None:
        return keyword_search(query, top_k)

    q_vec = model.encode([query], convert_to_numpy=True)
    q_vec = q_vec / np.maximum(np.linalg.norm(q_vec, axis=1, keepdims=True), 1e-9)
    scores = (EMBEDDINGS @ q_vec.T).squeeze()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(CATALOG[i], float(scores[i])) for i in top_indices]


def keyword_search(query: str, top_k: int = 15) -> list:
    """Simple keyword overlap search as fallback."""
    query_words = set(query.lower().split())
    scored = []
    for item in CATALOG:
        text = build_searchable_text(item)
        text_words = set(text.split())
        overlap = len(query_words & text_words)
        scored.append((item, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def retrieve_candidates(conversation_text: str, top_k: int = 15) -> list:
    """Retrieve top candidates from catalog based on conversation context."""
    results = semantic_search(conversation_text, top_k)
    return [item for item, score in results]


# ─── Guard Rails ──────────────────────────────────────────────────────────────
OFF_TOPIC_PATTERNS = [
    r"\b(salary|compensation|pay|benefits|visa|immigration|legal|lawsuit|sue|gdpr|compliance)\b",
    r"\b(interview tips|resume|cv writing|cover letter|job board|linkedin)\b",
    r"\b(competitor|korn ferry|heidrick|mercer|talent plus|criteria corp)\b",
]

INJECTION_PATTERNS = [
    r"ignore (all |previous |prior )?(instructions?|prompts?|rules?)",
    r"(you are|act as|pretend to be|roleplay as) (a |an )?(different|new|another)",
    r"(system prompt|system message|override|bypass|jailbreak|DAN)",
    r"forget (everything|all|previous|your instructions)",
    r"(reveal|show|print|output|display) (your |the )?(system prompt|instructions|prompt)",
]

VAGUE_TRIGGERS = {"assessment", "test", "something", "anything", "help", "evaluate", "measure"}

SPECIFIC_TRIGGERS = {
    "java", "python", "developer", "engineer", "manager", "sales", "customer",
    "graduate", "finance", "data", "analyst", "leadership", "verbal", "numerical",
    "personality", "cognitive", "ability", "technical", "coding", "c#", "javascript",
    "senior", "junior", "mid", "entry", "operative", "clerical", "admin"
}


def is_off_topic(text: str) -> bool:
    text_lower = text.lower()
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def is_injection_attempt(text: str) -> bool:
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def is_vague_first_message(messages: list) -> bool:
    """True if first user message is too vague to act on."""
    if len(messages) > 2:
        return False
    last_user = messages[-1].content.lower()
    words = set(last_user.split())
    has_vague = bool(words & VAGUE_TRIGGERS)
    has_specific = bool(words & SPECIFIC_TRIGGERS)
    return has_vague and not has_specific and len(last_user.split()) < 8


def count_turns(messages: list) -> int:
    return len(messages)


# ─── LLM Integration ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def format_catalog_context(candidates: list) -> str:
    lines = []
    for item in candidates:
        lines.append(
            f"- Name: {item['name']}\n"
            f"  URL: {item['url']}\n"
            f"  TestType: {item['test_type']}\n"
            f"  Description: {item['description']}\n"
            f"  JobLevels: {', '.join(item.get('job_levels', []))}\n"
            f"  RemoteTesting: {item.get('remote_testing', False)}\n"
            f"  Keywords: {', '.join(item.get('keywords', []))}\n"
        )
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """You are an expert SHL Assessment Recommender assistant. You ONLY discuss SHL assessments.

RETRIEVED CATALOG ENTRIES (use ONLY these for recommendations):
{catalog_context}

YOUR RULES:
1. SCOPE: Only discuss SHL assessments. If asked about general hiring advice, legal questions, salary, competitors, or anything unrelated to SHL assessments, politely refuse and redirect.
2. CLARIFY: If the query is vague (e.g. "I need an assessment" with no role/context), ask ONE focused clarifying question. Do NOT recommend yet.
3. RECOMMEND: Once you have enough context (role, level, or purpose), recommend 1-10 assessments ONLY from the catalog entries above. NEVER invent names or URLs.
4. REFINE: If the user changes constraints (e.g. "add personality tests", "remove numerical"), update the shortlist accordingly.
5. COMPARE: If asked to compare assessments, use ONLY the catalog data above. Never use prior knowledge.
6. TURN BUDGET: The conversation is capped at 8 turns total. By turn 6, you MUST provide a shortlist even if context is incomplete.
7. END: Set end_of_conversation=true only when the user is satisfied and the task is fully complete.

TEST TYPE CODES:
- A = Ability/Aptitude (cognitive, numerical, verbal, reasoning)
- P = Personality Questionnaire
- K = Knowledge Test (technical/domain)
- S = Simulation/Situational Judgement
- 360 = 360-degree feedback
- C = Competency Framework

RESPONSE FORMAT (respond ONLY with valid JSON, no markdown, no preamble):
{{
  "reply": "Your conversational response here",
  "recommendations": [
    {{"name": "Assessment Name", "url": "https://www.shl.com/...", "test_type": "X"}}
  ],
  "end_of_conversation": false
}}

IMPORTANT:
- recommendations must be [] when still clarifying or refusing
- recommendations must be 1-10 items when committing to a shortlist
- Every URL must come from the catalog entries above — no exceptions
- Do not add commentary outside the JSON object
"""


def call_anthropic(messages: list, catalog_context: str) -> dict:
    import urllib.request
    system = SYSTEM_PROMPT_TEMPLATE.format(catalog_context=catalog_context)
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "system": system,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read())
    return data


def call_gemini(messages: list, catalog_context: str) -> dict:
    import urllib.request
    system = SYSTEM_PROMPT_TEMPLATE.format(catalog_context=catalog_context)
    # Build Gemini contents
    contents = [{"role": "user", "parts": [{"text": system + "\n\nConversation starts now."}]},
                {"role": "model", "parts": [{"text": "Understood. I am ready to help recommend SHL assessments."}]}]
    for m in messages:
        role = "user" if m.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m.content}]})

    payload = json.dumps({
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.2},
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read())
    return data


def call_groq(messages: list, catalog_context: str) -> dict:
    import urllib.request
    system = SYSTEM_PROMPT_TEMPLATE.format(catalog_context=catalog_context)
    msg_list = [{"role": "system", "content": system}]
    msg_list += [{"role": m.role, "content": m.content} for m in messages]
    payload = json.dumps({
        "model": "llama3-70b-8192",
        "messages": msg_list,
        "max_tokens": 1000,
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read())
    return data


def extract_text_from_response(data: dict, provider: str) -> str:
    if provider == "anthropic":
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block["text"]
    elif provider == "gemini":
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
    elif provider == "groq":
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
    return ""


def parse_llm_output(raw: str) -> dict:
    """Robustly parse LLM JSON output."""
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Find JSON object in response
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from LLM output: {raw[:200]}")


def validate_recommendations(recs: list) -> list:
    """Ensure every recommendation URL is from our catalog."""
    validated = []
    for rec in recs:
        url = rec.get("url", "")
        name = rec.get("name", "")
        test_type = rec.get("test_type", "A")
        if url in VALID_URLS:
            validated.append({"name": name, "url": url, "test_type": test_type})
        else:
            # Try to find by name match
            for item in CATALOG:
                if item["name"].lower() == name.lower():
                    validated.append({"name": item["name"], "url": item["url"], "test_type": item["test_type"]})
                    break
            # else: drop it — never return invalid URLs
    return validated[:10]


def call_llm(messages: list, catalog_context: str) -> str:
    """Try available LLM providers in order."""
    errors = []

    if ANTHROPIC_API_KEY:
        try:
            data = call_anthropic(messages, catalog_context)
            text = extract_text_from_response(data, "anthropic")
            if text:
                return text
        except Exception as e:
            errors.append(f"Anthropic: {e}")
            logger.warning(f"Anthropic call failed: {e}")

    if GEMINI_API_KEY:
        try:
            data = call_gemini(messages, catalog_context)
            text = extract_text_from_response(data, "gemini")
            if text:
                return text
        except Exception as e:
            errors.append(f"Gemini: {e}")
            logger.warning(f"Gemini call failed: {e}")

    if GROQ_API_KEY:
        try:
            data = call_groq(messages, catalog_context)
            text = extract_text_from_response(data, "groq")
            if text:
                return text
        except Exception as e:
            errors.append(f"Groq: {e}")
            logger.warning(f"Groq call failed: {e}")

    raise RuntimeError(f"All LLM providers failed: {'; '.join(errors)}")


# ─── Core Agent Logic ──────────────────────────────────────────────────────────
def build_conversation_query(messages: list) -> str:
    """Extract the search query from full conversation history."""
    user_messages = [m.content for m in messages if m.role == "user"]
    return " ".join(user_messages)


def get_fallback_response(reason: str, candidates: list) -> dict:
    """Safe fallback if LLM fails."""
    if reason == "off_topic":
        return {
            "reply": "I can only help with SHL assessment recommendations. Could you describe the role you're hiring for?",
            "recommendations": [],
            "end_of_conversation": False,
        }
    if reason == "injection":
        return {
            "reply": "I'm here to help you find the right SHL assessments. What role are you hiring for?",
            "recommendations": [],
            "end_of_conversation": False,
        }
    if reason == "vague":
        return {
            "reply": "I'd love to help! Could you tell me more about the role you're hiring for? For example, the job title or key responsibilities?",
            "recommendations": [],
            "end_of_conversation": False,
        }
    # Generic fallback with top candidates
    recs = [{"name": c["name"], "url": c["url"], "test_type": c["test_type"]} for c in candidates[:5]]
    return {
        "reply": "Based on your requirements, here are some SHL assessments that may be relevant:",
        "recommendations": recs,
        "end_of_conversation": False,
    }


def agent_respond(request: ChatRequest) -> ChatResponse:
    messages = request.messages
    last_user_content = ""
    for m in reversed(messages):
        if m.role == "user":
            last_user_content = m.content
            break

    # ── Guard: injection attempt ──────────────────────────────────────────────
    full_text = " ".join(m.content for m in messages)
    if is_injection_attempt(full_text) or is_injection_attempt(last_user_content):
        fb = get_fallback_response("injection", [])
        return ChatResponse(**fb)

    # ── Guard: off-topic ──────────────────────────────────────────────────────
    if is_off_topic(last_user_content):
        fb = get_fallback_response("off_topic", [])
        return ChatResponse(**fb)

    # ── Guard: vague first message ────────────────────────────────────────────
    if is_vague_first_message(messages):
        fb = get_fallback_response("vague", [])
        return ChatResponse(**fb)

    # ── Retrieve catalog candidates ───────────────────────────────────────────
    query = build_conversation_query(messages)
    candidates = retrieve_candidates(query, top_k=15)
    catalog_context = format_catalog_context(candidates)

    # ── Call LLM ──────────────────────────────────────────────────────────────
    try:
        raw = call_llm(messages, catalog_context)
        parsed = parse_llm_output(raw)
    except Exception as e:
        logger.error(f"LLM or parse error: {e}")
        fb = get_fallback_response("error", candidates)
        return ChatResponse(**fb)

    # ── Validate & sanitize response ──────────────────────────────────────────
    reply = parsed.get("reply", "I'm here to help you find the right SHL assessments.")
    raw_recs = parsed.get("recommendations", [])
    end_flag = bool(parsed.get("end_of_conversation", False))

    validated_recs = validate_recommendations(raw_recs)

    # ── Turn budget enforcement ───────────────────────────────────────────────
    # If we're at turn 6+ with no recs yet, force a recommendation
    turn_count = count_turns(messages)
    if turn_count >= 6 and not validated_recs:
        forced = [{"name": c["name"], "url": c["url"], "test_type": c["test_type"]} for c in candidates[:5]]
        validated_recs = forced
        reply += " Based on our conversation so far, here are the most relevant assessments I can suggest."

    return ChatResponse(
        reply=reply,
        recommendations=[Recommendation(**r) for r in validated_recs],
        end_of_conversation=end_flag,
    )


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    load_catalog()
    build_embeddings()
    logger.info("SHL Recommender ready")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.time()
    try:
        response = agent_respond(request)
        elapsed = time.time() - start
        logger.info(f"Chat handled in {elapsed:.2f}s | turns={len(request.messages)} | recs={len(response.recommendations)}")
        return response
    except Exception as e:
        logger.error(f"Unexpected error in /chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred."},
    )
