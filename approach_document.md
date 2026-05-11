# SHL Assessment Recommender — Approach Document

**Candidate:** [Your Name] | **Role:** AI Intern, SHL Labs

---

## Design Overview

The system is a stateless FastAPI service exposing `GET /health` and `POST /chat`. Every request carries the full conversation history; no session state is stored server-side. The agent pipeline has four stages: **guard rails → retrieval → LLM generation → validation**.

### Retrieval: Semantic Search over a Curated Catalog

The SHL product catalog (Individual Test Solutions only) was structured into a `catalog.json` with fields: name, URL, test_type, description, job levels, keywords, remote testing flag, and duration. Each item's searchable text (name + description + keywords + job levels) is embedded at startup using `sentence-transformers/all-MiniLM-L6-v2` — a lightweight 80MB model that runs locally with no API cost. Vectors are normalized and stored in a NumPy matrix; cosine similarity retrieves the top 15 candidates per request. The full conversation history (all user messages concatenated) forms the query, ensuring context accumulates across turns.

**Why local embeddings over a vector DB?** For a catalog of ~50 items, FAISS or Chroma adds infrastructure with no recall benefit. A single matrix multiply is fast enough.

### Context Engineering: Grounded Prompting

The system prompt injects only the top 15 retrieved catalog entries as structured text (name, URL, test_type, description, job levels). The LLM is instructed to recommend **only from these entries** and to return only valid JSON — no preamble, no markdown fences. This keeps the model grounded and makes output parsing deterministic.

**What didn't work:** Sending the entire 50-item catalog in every prompt caused the model to occasionally recommend assessments that were contextually plausible but not best-fit. Restricting context to top-15 retrieved items improved relevance and reduced hallucination.

### Agent Decision Logic

The agent handles four intents:

| Intent | Trigger | Behavior |
|---|---|---|
| **Clarify** | Vague query (≤8 words, no role/skill signal) on turn 1 | Ask one focused question; return empty recommendations |
| **Recommend** | Sufficient context (role + level or purpose) | Return 1–10 assessments from catalog |
| **Refine** | User changes constraints mid-conversation | LLM updates shortlist in place |
| **Compare** | "difference between X and Y" phrasing | LLM answers using only retrieved catalog data |

**Turn budget enforcement:** The conversation is capped at 8 turns by the evaluator. If turn count ≥ 6 and the LLM still returns empty recommendations, the system forces the top-5 semantically retrieved items — preventing the agent from stalling.

### Guard Rails

Three layers before the LLM is called:

1. **Prompt injection detection** — regex patterns for "ignore instructions", "system prompt", "jailbreak" etc. Returns a safe redirect with empty recommendations.
2. **Off-topic refusal** — patterns for salary, legal, competitors, CV writing. Returns a redirect to SHL scope.
3. **Vague first-message detection** — checks if turn 1 has no specific role/skill signal. Returns a clarifying question.

**URL validation:** Every URL returned by the LLM is checked against the set of valid catalog URLs before the response is sent. If the LLM hallucinates a URL, it is either matched by name to the correct catalog entry or silently dropped. This enforces the hard eval requirement that all URLs come from the scraped catalog.

### LLM Provider Chain

The system tries providers in order: **Anthropic Claude → Gemini 1.5 Flash → Groq (Llama3-70b)**. This ensures availability despite free-tier rate limits. JSON output is forced via the system prompt; a robust parser handles stray markdown fences.

### Evaluation Approach

I tested against 10 behavior probes before submission:
- Vague turn-1 query → no recommendations (verifies clarification behavior)
- Off-topic queries (salary, legal) → refusal, empty recommendations  
- Prompt injection strings → refusal
- Multi-turn conversation with role + level context → ≥1 recommendation, all URLs valid
- Refinement mid-conversation → shortlist updates (not reset)
- Comparison question → reply contains both assessment names
- Turn-7 query → recommendations forced even without full context
- Response time → all calls under 30 seconds

**What didn't work initially:** The first prompt had the LLM ask multiple clarifying questions per turn, consuming the turn budget quickly. Constraining to one question per turn improved turn efficiency. Early versions also let the LLM hallucinate URLs; the post-generation validation layer fixed this completely.

**AI tools used:** Claude assisted with boilerplate code structure and the test suite scaffolding. All design decisions, prompt engineering, and guard rail logic were authored and validated manually.

---

*Stack: FastAPI · sentence-transformers · NumPy · Anthropic/Gemini/Groq APIs · Render (deployment)*
