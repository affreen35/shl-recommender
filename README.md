# SHL Assessment Recommender

Conversational AI agent that recommends SHL Individual Test Solutions based on hiring needs.

## Architecture

```
POST /chat
    ↓
Guard Rails (injection, off-topic, vague check)
    ↓
Semantic Search over SHL catalog (FAISS + sentence-transformers)
    ↓
LLM (Anthropic → Gemini → Groq fallback chain)
    ↓
JSON validation + URL verification against catalog
    ↓
ChatResponse (reply, recommendations, end_of_conversation)
```

## Project Structure

```
shl-recommender/
├── src/
│   └── main.py          # FastAPI app + agent logic
├── data/
│   └── catalog.json     # SHL Individual Test Solutions catalog
├── test_agent.py        # Full test suite (behavior probes)
├── requirements.txt
├── Dockerfile
├── render.yaml          # Render.com deployment config
└── .env.example         # API key template
```

## Setup (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

Copy `.env.example` to `.env` and fill in at least one LLM key:

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY or GEMINI_API_KEY or GROQ_API_KEY
```

**Getting free API keys:**
- Anthropic: https://console.anthropic.com (best quality)
- Gemini: https://aistudio.google.com (generous free tier)
- Groq: https://console.groq.com (fast, free Llama3-70b)

### 3. Run the server

```bash
# Load env vars and start
export $(cat .env | grep -v ^# | xargs)
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test it

```bash
# Health check
curl http://localhost:8000/health

# Chat example
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a Java developer with 4 years of experience who works with stakeholders"}
    ]
  }'

# Run full test suite
python test_agent.py --url http://localhost:8000
```

## Deployment (Render — Free Tier)

1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml`
5. Add environment variables in the Render dashboard:
   - `ANTHROPIC_API_KEY` (or GEMINI/GROQ)
6. Deploy — your URL will be `https://shl-recommender.onrender.com`
7. **Important:** Add a free uptime monitor at https://uptimerobot.com
   - Monitor URL: `https://your-service.onrender.com/health`
   - Interval: every 5 minutes
   - This prevents cold starts and keeps service within the 30s timeout

## API Reference

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I'm hiring a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, 4 years experience"}
  ]
}
```

**Response:**
```json
{
  "reply": "Based on your requirements, here are 3 assessments...",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

**Test type codes:**
- `A` = Ability/Aptitude (numerical, verbal, cognitive reasoning)
- `P` = Personality Questionnaire
- `K` = Knowledge Test (technical/domain)
- `S` = Simulation / Situational Judgement
- `360` = 360-degree feedback

**Schema rules:**
- `recommendations` is `[]` when agent is clarifying or refusing
- `recommendations` has 1–10 items when agent provides a shortlist
- `end_of_conversation` is `true` only when task is complete
- Every URL comes from the scraped SHL catalog (validated before returning)

## Design Decisions

### Why stateless?
The spec requires it. All conversation history is sent on every call. The agent reconstructs context from the full history each time.

### Why sentence-transformers locally?
No API call needed for embeddings = faster response, no rate limits, zero cost. The 80MB model is downloaded once at build time and cached.

### Why a fallback LLM chain?
Free tiers have rate limits. Anthropic → Gemini → Groq ensures availability. Add your own keys for whichever you prefer.

### Why validate URLs?
The evaluator checks that every returned URL is a real SHL catalog URL. LLMs can hallucinate URLs. We validate every URL against our catalog before returning.

### Turn budget
The evaluator caps at 8 turns. By turn 6, the agent is forced to provide a shortlist even if information is incomplete, rather than risk hitting the cap.

## Evaluation Checklist

- [x] Schema compliance (reply, recommendations, end_of_conversation)
- [x] Items from catalog only (URL validation)
- [x] Turn cap honored (max 8, forced recs at turn 6)
- [x] No recommendation on turn 1 for vague queries
- [x] Off-topic refusal
- [x] Prompt injection refusal
- [x] Refinement updates shortlist
- [x] Comparison uses catalog data
- [x] 30 second timeout respected
- [x] GET /health returns status=ok
