"""
CarID Backend — FastAPI + CarNet + Serper + Groq
─────────────────────────────────────────────────
Endpoints:
  POST /identify      — image → car identification (CarNet + Serper + Groq)
  POST /chat          — conversational chat with memory, compare, clarify, buyer advisor
  POST /chat/stream   — streaming version of /chat (token-by-token SSE)
  POST /compare       — side-by-side structured comparison of 2-4 cars
  GET  /health        — liveness check

Env vars (set in HuggingFace Spaces → Settings → Secrets):
  GROQ_API_KEY   — console.groq.com  (free)
  SERPER_API_KEY — serper.dev        (free, 2500/month)
"""

import os, io, time, json, re, asyncio
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
import requests
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
GROQ_MODEL        = "llama-3.1-8b-instant"           # text model
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # vision model
GROQ_URL          = "https://api.groq.com/openai/v1/chat/completions"
HTML_PATH      = Path(os.getenv("HTML_PATH", "/app/car_id_chat_app.html"))

_http = httpx.Client(timeout=25)

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


# ── Startup ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] CarID ready")
    print(f"  GROQ   : {'set' if GROQ_API_KEY   else 'MISSING'}")
    print(f"  SERPER : {'set' if SERPER_API_KEY else 'MISSING'}")
    yield
    _http.close()
    print("[shutdown] Bye!")


app = FastAPI(title="CarID API", version="5.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class HistoryMessage(BaseModel):
    """One turn in the conversation history sent from the frontend."""
    role: str       # "user" | "assistant"
    content: str

class BuyerProfile(BaseModel):
    """Collected progressively during the car-buying advisor flow."""
    budget:          str = ""   # e.g. "$15-30k"
    priority:        str = ""   # appearance | cost | reliability | performance | comfort
    usage:           str = ""   # family | single | commuting | off-road | mixed
    fuel_pref:       str = ""   # petrol | diesel | electric | hybrid | any
    body_pref:       str = ""   # sedan | SUV | hatchback | coupe | wagon | any
    location:        str = ""   # e.g. "Egypt", "Cairo", "UAE"
    currency:        str = "USD" # e.g. "EGP", "AED", "SAR", "USD"
    questions_asked: int = 0

class ChatRequest(BaseModel):
    # Core question — your frontend sends both "question" and "message"
    question:      str = ""
    message:       str = ""     # alias used by the Next.js frontend
    # Conversation persistence
    conversationId: str = ""
    # Memory — sent by the upgraded frontend on every call
    history:       list[HistoryMessage] = []
    car_context:   dict          = {}   # last identified/discussed car
    # User location — persisted across the whole conversation, not just buyer flow
    user_location: str           = ""   # e.g. "Egypt", "UAE", "Saudi Arabia"
    # Language — detected by frontend from user input
    user_lang:     str           = ""   # "ar" | "en" | ""
    # Buyer advisor state
    buyer_mode:    bool          = False
    buyer_profile: BuyerProfile  = BuyerProfile()

class ChatResponse(BaseModel):
    # Always present
    answer:          str
    reply:           str         = ""   # alias so old frontend field also works
    # Intent classification — used by the frontend to decide how to render
    intent:          str         = "chat"   # chat | clarify | compare | buyer | recommend
    # Option buttons — shown as tappable choices (buyer advisor + clarify)
    options:         list[str]   = []
    # Buyer advisor
    buyer_mode:      bool        = False
    buyer_profile:   BuyerProfile = BuyerProfile()
    # User location — echoed back so frontend can persist it
    user_location:   str         = ""
    # Language — echoed back so frontend knows detected language
    user_lang:       str         = ""
    # Compare — populated when intent == "compare"
    compare_data:    list[dict]  = []
    # Buyer recommendations — populated when intent == "recommend"
    recommendations: list[str]   = []

class CompareRequest(BaseModel):
    cars:    list[str]
    history: list[HistoryMessage] = []
    aspect:  str = ""

# ── Damage detection models ────────────────────────────────────────────────────

class DamageArea(BaseModel):
    component:        str
    severity:         str   # minor | moderate | severe | critical
    description:      str
    repair_cost_min:  int
    repair_cost_max:  int
    is_internal:      bool = False

class InternalRisk(BaseModel):
    component:        str
    likelihood:       str   # likely | possible | unlikely
    reason:           str
    repair_cost_min:  int
    repair_cost_max:  int

class DamageReport(BaseModel):
    safe_to_drive:     bool
    overall_severity:  str            # minor | moderate | severe | critical
    damage_areas:      list[DamageArea]
    internal_risks:    list[InternalRisk]
    total_cost_min:    int
    total_cost_max:    int
    priority_actions:  list[str]
    message:           str = ""


# ══════════════════════════════════════════════════════════════════════════════
# CORE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _groq(messages: list[dict], max_tokens: int = 500) -> str:
    """Send a full multi-turn messages list to Groq (text model) and return the reply."""
    if not GROQ_API_KEY:
        return "Chat unavailable — GROQ_API_KEY not configured."
    try:
        resp = _http.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={"model": GROQ_MODEL, "max_tokens": max_tokens, "messages": messages},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  Groq error: {e}")
        return "I ran into a small issue — please try again in a moment."


def _groq_vision(image_bytes: bytes, system: str, user_text: str, max_tokens: int = 1200) -> str:
    """
    Send an image + text to the Groq vision model (llama-4-scout).
    Falls back gracefully if the vision model is unavailable.
    """
    if not GROQ_API_KEY:
        return ""
    import base64
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text",      "text": user_text},
        ]},
    ]
    try:
        resp = _http.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={"model": GROQ_VISION_MODEL, "max_tokens": max_tokens, "messages": messages},
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"  Groq vision OK ({len(result)} chars)")
        return result
    except Exception as e:
        print(f"  Groq vision error: {e}")
        return ""


def _serper(query: str) -> list[dict]:
    """Web search via Serper. Returns list of {title, snippet, link}."""
    if not SERPER_API_KEY:
        return []
    try:
        resp = _http.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
        )
        resp.raise_for_status()
        return [
            {"title": r.get("title",""), "snippet": r.get("snippet",""), "link": r.get("link","")}
            for r in resp.json().get("organic", [])[:5]
        ]
    except Exception as e:
        print(f"  Serper error: {e}")
        return []


def _web_text(results: list[dict], n: int = 5) -> str:
    if not results:
        return "No web results available."
    return "\n".join(f"- {r['title']}: {r['snippet']}" for r in results[:n])


# ── Image helpers ──────────────────────────────────────────────────────────────

def _prepare_image(raw: bytes) -> bytes:
    """Any format → JPEG, center-crop 16:9, resize 1500×844, compress <1 MB."""
    img = Image.open(io.BytesIO(raw))
    if img.mode in ("RGBA", "P", "LA"):
        canvas = Image.new("RGB", img.size, (128, 128, 128))
        canvas.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        img = canvas
    else:
        img = img.convert("RGB")

    tw, th = 1500, 844
    if img.width / img.height > tw / th:
        nw = int((tw / th) * img.height)
        img = img.crop(((img.width - nw) // 2, 0, (img.width - nw) // 2 + nw, img.height))
    else:
        nh = int(img.width / (tw / th))
        img = img.crop((0, (img.height - nh) // 2, img.width, (img.height - nh) // 2 + nh))
    img = img.resize((tw, th), Image.Resampling.LANCZOS)

    quality = 95
    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", optimize=True, quality=quality)
        data = buf.getvalue()
        if len(data) / 1024 <= 999 or quality <= 15:
            print(f"  Image: {len(data)//1024} KB | q={quality}")
            return data
        quality -= 5


def _call_carnet(raw: bytes) -> dict:
    """POST image to CarNet free endpoint. Returns car dict or {}."""
    try:
        ready = _prepare_image(raw)
        resp  = requests.post(
            "https://carnet.ai/recognize-file",
            headers={
                "X-Requested-With": "XMLHttpRequest",
                "Origin":    "https://carnet.ai",
                "Referer":   "https://carnet.ai/",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
            files={"imageFile": ("car.jpg", ready, "image/jpeg")},
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"  CarNet HTTP {resp.status_code}")
            return {}
        car = resp.json().get("car")
        if not car:
            return {}
        raw_conf = float(car.get("make_prob", 0) or car.get("prob", 0))
        color = car.get("color", "—")
        if isinstance(color, dict):
            color = color.get("name", "—")
        out = {
            "brand":      car.get("make", ""),
            "model":      car.get("model", ""),
            "generation": car.get("generation", ""),
            "color":      color,
            "angle":      car.get("angle", "—"),
            "confidence": round(raw_conf, 1),
        }
        print(f"  CarNet → {out['brand']} {out['model']} ({out['confidence']}%)")
        return out
    except Exception as e:
        print(f"  CarNet error: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# INTENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

# Patterns that signal a comparison request
_COMPARE_RE = re.compile(
    r"\b(compare|vs\.?|versus|difference between|which is better|"
    r"should i (get|buy|choose)|between .+ and|compare (the )?both)\b", re.I
)

# Detect two car names in a sentence even without vs/compare
_TWO_CARS_RE = re.compile(
    r'\b(BMW|Mercedes|Audi|Toyota|Honda|Ford|Alfa Romeo|Volkswagen|Porsche|'
    r'Ferrari|Lamborghini|Hyundai|Kia|Renault|Peugeot|Fiat|Volvo|Mazda|'
    r'Subaru|Nissan|Jeep|Tesla|Lexus|Skoda|Seat|Chevrolet|Dodge|Chrysler)'
    r'[\w\s\-]*\b', re.I
)

# Patterns that signal car-buying intent (English)
_BUY_RE = re.compile(
    r"\b(want to buy|looking to buy|thinking of buying|want a (new )?car|"
    r"help me (find|choose|pick|select)|which car (should|would)|"
    r"recommend.*car|best car for|buy.*car|get a car|need a car|car for me|"
    r"looking for a car|suggest a car|find me a car|i need a car|"
    r"i('m| am) (looking|searching|shopping) for)\b", re.I
)

# Arabic buying intent patterns
_BUY_AR_RE = re.compile(
    r"(عايز|عاوز|أريد|اريد|نفسي|بدي|محتاج|محتاجه|ابي|أبي)"
    r".{0,15}"
    r"(أشتري|اشتري|عربية|سيارة|اشتري|أشتري|سياره|عربيه)",
    re.I
)
# Also catch direct Arabic phrases
_BUY_AR_RE2 = re.compile(
    r"(شراء سيارة|شراء عربية|ابحث عن سيارة|ابحث عن عربية|"
    r"اقتراح سيارة|انصحني بسيارة|انصحني بعربية|نصيحة في سيارة|"
    r"أنسب سيارة|افضل سيارة ل|أفضل سيارة ل)", re.I
)

# Arabic compare intent patterns
_COMPARE_AR_RE = re.compile(
    r"(قارن|مقارنة|الفرق بين|أيهما أفضل|ايهما أحسن|"
    r"أيهما أحسن|ايهما افضل|الفرق|مقارنه)", re.I
)

# Patterns that are too vague without context
_VAGUE_RE = re.compile(
    r"^(tell me|what about|how about|info|more|and|also|explain|"
    r"details?|what do you think|thoughts?|is it good)\b", re.I
)

def _detect_intent(text: str, history: list, car_ctx: dict, buyer_mode: bool) -> str:
    if buyer_mode:                                    return "buyer"
    if _BUY_RE.search(text):                         return "buyer"
    if _BUY_AR_RE.search(text):                      return "buyer"   # Arabic buying intent
    if _BUY_AR_RE2.search(text):                     return "buyer"   # Arabic buying phrases
    if _COMPARE_RE.search(text):                     return "compare"
    if _COMPARE_AR_RE.search(text):                  return "compare" # Arabic compare intent
    # Catch "X and Y" with two known car brands
    car_hits = _TWO_CARS_RE.findall(text)
    if len(set(h.lower() for h in car_hits)) >= 2:   return "compare"
    # Vague with no context → ask for clarification
    if _VAGUE_RE.match(text.strip()) and not car_ctx.get("brand") and not history:
        return "clarify"
    return "chat"

def _extract_car_names(text: str, history: list[HistoryMessage] | None = None) -> list[str]:
    """Split a compare query into individual car names, falling back to history."""
    parts = re.split(r"\bvs\.?\b|\bversus\b|\bcompare\b|\band\b|,", text, flags=re.I)
    cars  = []
    for p in parts:
        p = re.sub(r"\b(the|a|an|compare|between|which is better|or|both|from|price|engine|fuel|all|three|them)\b",
                   "", p, flags=re.I).strip()
        if len(p) > 2:
            cars.append(p)

    # If fewer than 2 found, scan recent history for recommended/discussed cars
    if len(cars) < 2 and history:
        # Strategy 1: look for numbered list items like "1. Tesla Model 3" in last assistant message
        for m in reversed(history[-10:]):
            if m.role != "assistant":
                continue
            # Extract numbered list items — these are usually the recommended cars
            numbered = re.findall(r'\d+\.\s+\*?\*?([A-Z][A-Za-z\s\-]+?)(?:\s*[-–—]|\s*\(|\s*\*\*|$)', m.content)
            seen = {c.strip().lower() for c in cars}
            for n in numbered:
                n = n.strip()
                if len(n) > 3 and n.lower() not in seen:
                    cars.append(n)
                    seen.add(n.lower())
            if len(cars) >= 2:
                break

        # Strategy 2: brand regex scan on last 10 messages
        if len(cars) < 2:
            hist_text = " ".join(m.content for m in history[-10:])
            hist_hits = _TWO_CARS_RE.findall(hist_text)
            seen = {c.strip().lower() for c in cars}
            for h in dict.fromkeys(hist_hits):
                h = h.strip()
                if h.lower() not in seen and len(h) > 2:
                    cars.append(h)
                    seen.add(h.lower())
                if len(cars) >= 4:
                    break

    # Detect "all three/four" — user wants to compare all recommended cars from history
    all_match = re.search(r"\b(all|compare all|all (three|four|of them|of these))\b", text, re.I)
    if all_match and history and len(cars) < 2:
        # grab ALL numbered items from last assistant message
        for m in reversed(history[-10:]):
            if m.role != "assistant":
                continue
            numbered = re.findall(r'\d+\.\s+\*?\*?([A-Z][A-Za-z\s\-]+?)(?:\s*[-–—]|\s*\(|\s*\*\*|$)', m.content)
            if len(numbered) >= 2:
                cars = [n.strip() for n in numbered[:4]]
                break

    return cars[:4]


# ══════════════════════════════════════════════════════════════════════════════
# COMPARE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_car_data(car_name: str, user_location: str = "") -> dict:
    """
    Fetch structured spec/review data for one car.
    When user is in Egypt: runs Egypt-specific searches (Arabic + English),
    injects tax/import knowledge, returns EGP prices.
    """
    is_egypt_user = _is_egypt(user_location)

    if is_egypt_user:
        # 3 targeted searches (Arabic + English combined queries save quota)
        results_specs  = _serper(f"{car_name} specs engine مواصفات محرك")
        results_price  = _serper(f"{car_name} سعر مصر price Egypt 2024 2025")
        results_review = _serper(f"{car_name} pros cons مراجعة عيوب مزايا reliability")
    else:
        results_specs  = _serper(f"{car_name} engine specs horsepower fuel type")
        results_price  = _serper(f"{car_name} price MSRP cost how much")
        results_review = _serper(f"{car_name} pros cons reliability problems owner review")

    def fmt(label: str, results: list[dict]) -> str:
        if not results:
            return f"[{label}]: No results found."
        lines = "\n".join(f"  - {r['title']}: {r['snippet']}" for r in results[:3])
        return f"[{label}]:\n{lines}"

    web_context = "\n\n".join([
        fmt("SPECS & ENGINE", results_specs),
        fmt("PRICE & COST",   results_price),
        fmt("PROS, CONS & RELIABILITY", results_review),
    ])

    # Use condensed market knowledge for data extraction (long version crowds the prompt)
    market_knowledge = EGYPT_TAX_KNOWLEDGE_SHORT if _is_egypt(user_location) else ""

    system = (
        "You are a car data extraction engine. "
        "Your job is to read web search snippets and extract structured car data. "
        "Return ONLY a valid JSON object — no markdown fences, no preamble, no explanation whatsoever. "
        + _currency_note(user_location)
        + (market_knowledge if market_knowledge else "")
        + "\nThe JSON must have EXACTLY these keys:\n"
        "  name        — full car name including brand (string)\n"
        "  price_range — e.g. '$35,000 – $45,000' or '£28,000 – £35,000' (string, use real numbers from snippets)\n"
        "  engine      — e.g. '2.0L Turbo-4, 255hp' or '3.0L Twin-Turbo I6, 503hp' (string)\n"
        "  fuel        — ONLY one of: Petrol, Diesel, Electric, Hybrid, Plug-in Hybrid (string)\n"
        "  reliability — one factual sentence with a source if available (string)\n"
        "  pros        — exactly 3 bullet points as one string separated by • character (string)\n"
        "  cons        — exactly 3 bullet points as one string separated by • character (string)\n"
        "  verdict     — one sentence: who is this car best for? (string)\n\n"
        "STRICT RULES:\n"
        "- For 'fuel', NEVER put a model name or trim level. Only: Petrol, Diesel, Electric, Hybrid, Plug-in Hybrid.\n"
        "- For 'price_range', NEVER put '—' if any price appears anywhere in the snippets — find it.\n"
        "- For 'engine', include displacement AND power if found.\n"
        "- Only use '—' for a field if it is genuinely absent from ALL snippets.\n"
        "- Do NOT truncate. Complete the entire JSON object."
    )

    raw = _groq([
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Car: {car_name}\n\nWeb search results:\n{web_context}"},
    ], max_tokens=600)   # increased from 350 to prevent truncation

    # Clean and parse
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        # Find the JSON object even if there's stray text before/after
        match = re.search(r"\{[\s\S]+\}", clean)
        if not match:
            raise ValueError("No JSON object found in response")
        data = json.loads(match.group(0))
        data["name"] = data.get("name") or car_name

        # Validate fuel field — if it looks like a model name, fix it
        fuel = data.get("fuel", "—")
        valid_fuels = {"petrol", "diesel", "electric", "hybrid", "plug-in hybrid"}
        if fuel.lower() not in valid_fuels and fuel != "—":
            # Try to infer from engine or snippets
            all_text = web_context.lower()
            if "electric" in all_text or "ev" in all_text or "kwh" in all_text:
                data["fuel"] = "Electric"
            elif "hybrid" in all_text:
                data["fuel"] = "Hybrid"
            elif "diesel" in all_text:
                data["fuel"] = "Diesel"
            else:
                data["fuel"] = "Petrol"

        # Validate price_range — if it's "—" but we have price data, try simple extraction
        if data.get("price_range") in ("—", "", None) and results_price:
            price_text = " ".join(r["snippet"] for r in results_price)
            price_match = re.search(r'[\$£€]\s?[\d,]+(?:\s?[-–]\s?[\$£€]?\s?[\d,]+)?', price_text)
            if price_match:
                data["price_range"] = price_match.group(0).strip()

        print(f"  Compare data OK: {car_name} | price={data.get('price_range')} | engine={data.get('engine')}")
        return data

    except Exception as e:
        print(f"  _fetch_car_data parse error for '{car_name}': {e}\n  raw: {raw[:300]}")
        # Last resort: return what we can from raw text
        return {
            "name":        car_name,
            "price_range": _quick_extract_price(results_price),
            "engine":      _quick_extract_engine(results_specs),
            "fuel":        _quick_extract_fuel(results_specs),
            "reliability": _quick_extract_reliability(results_review),
            "pros":        "• See web search results for details",
            "cons":        "• See web search results for details",
            "verdict":     "—",
        }


def _quick_extract_price(results: list[dict]) -> str:
    """Regex fallback to find any price in snippets."""
    for r in results:
        m = re.search(r'[\$£€]\s?[\d,]+(?:\s?[-–]\s?[\$£€]?\s?[\d,]+)?', r.get("snippet",""))
        if m:
            return m.group(0).strip()
    return "—"

def _quick_extract_engine(results: list[dict]) -> str:
    """Regex fallback to find engine info."""
    for r in results:
        m = re.search(r'\d+\.\d+[Ll][^\.,]{0,40}(?:hp|kW|horsepower)', r.get("snippet",""))
        if m:
            return m.group(0).strip()
    return "—"

def _quick_extract_fuel(results: list[dict]) -> str:
    """Keyword fallback for fuel type."""
    text = " ".join(r.get("snippet","") for r in results).lower()
    if "electric" in text or "kwh" in text:  return "Electric"
    if "plug-in hybrid" in text:             return "Plug-in Hybrid"
    if "hybrid" in text:                     return "Hybrid"
    if "diesel" in text:                     return "Diesel"
    return "Petrol"

def _quick_extract_reliability(results: list[dict]) -> str:
    """Pull the first reliability-sounding sentence."""
    for r in results:
        s = r.get("snippet","")
        if any(w in s.lower() for w in ["reliable","reliability","repairpal","problem","issue"]):
            return s[:120] + ("…" if len(s)>120 else "")
    return "—"


def _compare_summary(cars_data: list[dict], aspect: str, history: list[HistoryMessage],
                     user_location: str = "", user_lang: str = "en") -> str:
    """Generate a human-like comparison verdict from structured data."""
    hist_text   = "\n".join(f"{m.role}: {m.content}" for m in history[-6:]) if history else ""
    aspect_note = f" Focus especially on: {aspect}." if aspect else ""
    curr_note   = _currency_note(user_location)
    lang_note   = (
        "\nLANGUAGE RULE: Respond entirely in Arabic. Brand names stay in Latin script.\n"
        if user_lang == "ar" else ""
    )

    return _groq([
        {"role": "system", "content": (
            "You are CarID — a warm, opinionated car expert who talks like a knowledgeable friend. "
            "You just compared some cars. Give a natural, human summary. "
            "Be direct — give a clear winner or recommendation at the end. "
            "Max 3 sentences."
            + aspect_note + curr_note + lang_note
        )},
        {"role": "user", "content": (
            (f"Conversation so far:\n{hist_text}\n\n" if hist_text else "") +
            f"Comparison data:\n{json.dumps(cars_data, indent=2)}\n\nGive a warm, direct summary."
        )},
    ], max_tokens=280)


# ══════════════════════════════════════════════════════════════════════════════
# MARKET-AWARE DATA — EGYPT & ARAB WORLD
# ══════════════════════════════════════════════════════════════════════════════

# Cars officially available and popular in Egypt by budget (EGP ranges as of 2025)
EGYPT_MARKET = {
    "under_500k": [
        "Lada Vesta", "Chery Tiggo 4", "Dongfeng AX7", "JAC J7",
        "Geely Emgrand", "MG 5",
    ],
    "500k_1000k": [
        "Hyundai i10", "Hyundai i20", "Kia Picanto", "Kia Rio",
        "Renault Logan", "Renault Duster", "Toyota Yaris",
        "MG ZS", "Chery Tiggo 7", "Geely Coolray",
    ],
    "1000k_1800k": [
        "Toyota Corolla", "Hyundai Elantra", "Kia Cerato", "Kia K3",
        "Nissan Sentra", "MG HS", "Chery Tiggo 8",
        "Renault Megane", "Skoda Octavia",
    ],
    "1800k_3500k": [
        "Toyota Camry", "Toyota C-HR", "Toyota RAV4",
        "Hyundai Tucson", "Hyundai Santa Fe",
        "Kia Sportage", "Kia Sorento",
        "MG RX8", "Jeep Compass", "Jeep Cherokee",
    ],
    "over_3500k": [
        "BMW 3 Series", "BMW 5 Series", "Mercedes C-Class",
        "Mercedes E-Class", "Audi A4", "Audi Q5",
        "Toyota Land Cruiser", "Lexus ES", "Volvo XC60",
    ],
}

# Brands NOT officially sold in Egypt (avoid recommending)
EGYPT_UNAVAILABLE = {
    "Tesla", "Rivian", "Lucid", "Polestar", "Subaru", "Chrysler",
    "Dodge", "Buick", "Cadillac", "Lincoln", "Pontiac", "Saturn",
    "Acura", "Infiniti", "Genesis", "Alfa Romeo", "Maserati",
}

# Egypt customs & tax knowledge
EGYPT_TAX_KNOWLEDGE = """
EGYPTIAN MARKET FACTS — USE THESE WHEN USER IS IN EGYPT:

Import Tax Brackets (applied on top of car base price):
- Engine under 1600cc → ~40–60% total taxes (most affordable bracket)
- Engine 1600–2000cc → ~80–100% total taxes
- Engine over 2000cc → ~100–135% total taxes (most expensive bracket)
- Electric vehicles → currently high import duties (~40%) + VAT
- Hybrid vehicles → slightly lower taxes than pure ICE equivalents

Price Reality:
- A car priced at $20,000 USD in the US costs approximately 1,800,000–2,200,000 EGP in Egypt
- A car priced at $35,000 USD in the US costs approximately 3,500,000–4,500,000 EGP in Egypt
- NEVER just multiply USD by 50 — always add 80–135% tax depending on engine size
- Egyptian dealer prices are typically 2.5–3.5× the US MSRP in EGP terms

Available Fuel Types in Egypt:
- Petrol (benzine 92 and 95 octane) — most common
- Diesel — available, used in SUVs/trucks
- Electric — limited charging infrastructure, not recommended for most users
- Natural Gas (CNG) — available as conversion but factory CNG cars rare

Popular Brands in Egypt (ranked by availability/service network):
1. Toyota — largest network, best resale value
2. Hyundai — very common, good service
3. Kia — growing fast, good value
4. MG — increasingly popular, good price/specs ratio  
5. Chery — budget-friendly Chinese brand
6. Renault — long history in Egypt
7. Nissan — solid presence
8. Geely — newer but growing
9. Lada — budget segment
10. BMW/Mercedes — luxury segment, very expensive due to taxes

IMPORTANT: Do NOT recommend Tesla, Rivian, Lucid, Subaru, Chrysler, Dodge, or any
car not officially available in Egypt. The user cannot buy or service these cars.
"""

# Per-market knowledge injection
MARKET_KNOWLEDGE = {
    "egypt":      EGYPT_TAX_KNOWLEDGE,
    "cairo":      EGYPT_TAX_KNOWLEDGE,
    "alexandria": EGYPT_TAX_KNOWLEDGE,
}

EGYPT_TAX_KNOWLEDGE_SHORT = (
    "EGYPT MARKET: Prices in EGP. "
    "A $20k USD car costs ~1,800,000–2,200,000 EGP after import taxes. "
    "A $35k USD car costs ~3,500,000–4,500,000 EGP. "
    "NEVER just multiply USD×50 — always account for 80–135% import tax. "
    "Popular brands: Toyota, Hyundai, Kia, MG, Chery, Renault, Nissan, Geely. "
    "Do NOT recommend Tesla, Rivian, Subaru, Dodge, or cars unavailable in Egypt."
)

def _get_market_knowledge(location: str) -> str:
    """Returns market-specific knowledge to inject into Groq prompts."""
    loc = location.lower().strip()
    for key, val in MARKET_KNOWLEDGE.items():
        if key in loc:
            return val
    return ""

def _get_egypt_budget_tier(budget_egp: float) -> str:
    """Map an EGP budget to the right Egyptian market tier."""
    if budget_egp < 500_000:   return "under_500k"
    if budget_egp < 1_000_000: return "500k_1000k"
    if budget_egp < 1_800_000: return "1000k_1800k"
    if budget_egp < 3_500_000: return "1800k_3500k"
    return "over_3500k"

def _egypt_budget_cars(budget_str: str) -> list[str]:
    """Return available Egyptian market cars for a given budget string."""
    # Strip commas and spaces BEFORE regex so "500,000" parses as 500000 not 500
    s = budget_str.lower().replace(",", "").replace(" ", "")
    m = re.search(r"(\d+)", s)
    if not m:
        return EGYPT_MARKET["1000k_1800k"]  # default mid-range
    num = int(m.group(1))
    # Handle k suffix — but only if number looks like a shorthand (< 10000)
    if "k" in s and num < 10000:
        num *= 1000
    # If looks like USD (under 200k), convert to EGP
    if num < 200_000:
        num = num * 50
    tier = _get_egypt_budget_tier(num)
    return EGYPT_MARKET.get(tier, EGYPT_MARKET["1000k_1800k"])

def _is_egypt(location: str) -> bool:
    return any(k in location.lower() for k in ["egypt","cairo","alexandria","egyp"])

# Advisor questions — Egyptian versions (fully Arabic)
ADVISOR_QUESTIONS_EGYPT = [
    {
        "field":   "budget",
        "q":       "ما هي ميزانيتك التقريبية؟",
        "options": ["أقل من 500,000 EGP", "500k – 1M EGP", "1M – 2M EGP", "أكثر من 2M EGP"],
    },
    {
        "field":   "priority",
        "q":       "ما الأهم بالنسبة لك في السيارة؟",
        "options": ["الشكل والمظهر", "توفير الوقود والتكلفة", "الموثوقية وخدمة ما بعد البيع", "الأداء والسرعة"],
    },
    {
        "field":   "usage",
        "q":       "السيارة هتكون لمين أو لإيه؟",
        "options": ["للعيلة كلها", "بس أنا", "التنقل اليومي", "استخدام مختلط"],
    },
    {
        "field":   "fuel_pref",
        "q":       "عندك تفضيل في نوع الوقود؟",
        "options": ["بنزين", "ديزل", "هايبرد", "مش مهم"],
    },
    {
        "field":   "body_pref",
        "q":       "عندك تفضيل في شكل هيكل السيارة؟",
        "options": ["SUV / كروس أوفر", "سيدان", "هاتشباك", "مش مهم"],
    },
]

ADVISOR_QUESTIONS = [
    {
        "field":   "budget",
        "q":       "What's your rough budget?",
        "options": ["Under $15k", "$15k – $30k", "$30k – $50k", "Over $50k"],
    },
    {
        "field":   "priority",
        "q":       "What matters most to you in a car?",
        "options": ["Looks & style", "Low cost", "Reliability", "Performance"],
    },
    {
        "field":   "usage",
        "q":       "Who's the car mainly for?",
        "options": ["The whole family", "Just me", "Daily commuting", "Off-road / adventure"],
    },
    {
        "field":   "fuel_pref",
        "q":       "Any fuel preference?",
        "options": ["Petrol", "Diesel", "Electric", "Hybrid / open to anything"],
    },
    {
        "field":   "body_pref",
        "q":       "Any body style in mind?",
        "options": ["SUV / Crossover", "Sedan", "Hatchback", "Coupe / No preference"],
    },
]

def _profile_complete(p: BuyerProfile) -> bool:
    """Recommend when 4 out of 5 profile fields are filled.
    No longer requires questions_asked >= 4 — that could block users
    who answered multiple fields in one message."""
    filled = sum(1 for f in [p.budget, p.priority, p.usage, p.fuel_pref, p.body_pref] if f.strip())
    return filled >= 4

def _extract_profile(user_text: str, profile: BuyerProfile) -> BuyerProfile:
    """Parse answers — direct lookup for option buttons, Groq fallback for free text.
    This fixes the infinite loop caused by Groq misclassifying button label strings."""

    # Direct lookup tables — exact option button labels → profile field values
    BUDGET_MAP = {
        "under $15k": "Under $15k",
        "$15k – $30k": "$15k – $30k",
        "$30k – $50k": "$30k – $50k",
        "over $50k": "Over $50k",
        "أقل من 500,000 egp": "أقل من 500,000 EGP",
        "500k – 1m egp": "500k – 1M EGP",
        "1m – 2m egp": "1M – 2M EGP",
        "أكثر من 2m egp": "أكثر من 2M EGP",
    }
    PRIORITY_MAP = {
        "looks & style": "appearance", "low cost": "cost",
        "reliability": "reliability", "performance": "performance",
        "الشكل والمظهر": "appearance",
        "توفير الوقود والتكلفة": "cost",
        "الموثوقية وخدمة ما بعد البيع": "reliability",
        "الأداء والسرعة": "performance",
    }
    USAGE_MAP = {
        "the whole family": "family", "just me": "single",
        "daily commuting": "commuting", "off-road / adventure": "off-road",
        "للعيلة كلها": "family", "بس أنا": "single",
        "التنقل اليومي": "commuting", "استخدام مختلط": "mixed",
    }
    FUEL_MAP = {
        "petrol": "petrol", "diesel": "diesel",
        "electric": "electric", "hybrid / open to anything": "hybrid",
        "بنزين": "petrol", "ديزل": "diesel",
        "هايبرد": "hybrid", "مش مهم": "any",
    }
    BODY_MAP = {
        "suv / crossover": "SUV", "sedan": "sedan",
        "hatchback": "hatchback", "coupe / no preference": "any",
        "suv / كروس أوفر": "SUV", "سيدان": "sedan",
        "هاتشباك": "hatchback",
    }

    txt = user_text.strip().lower()

    # Try direct mapping first — no Groq needed, no misclassification
    if not profile.budget    and txt in BUDGET_MAP:
        profile.budget    = BUDGET_MAP[txt];    return profile
    if not profile.priority  and txt in PRIORITY_MAP:
        profile.priority  = PRIORITY_MAP[txt];  return profile
    if not profile.usage     and txt in USAGE_MAP:
        profile.usage     = USAGE_MAP[txt];     return profile
    if not profile.fuel_pref and txt in FUEL_MAP:
        profile.fuel_pref = FUEL_MAP[txt];      return profile
    if not profile.body_pref and txt in BODY_MAP:
        profile.body_pref = BODY_MAP[txt];      return profile

    # Groq fallback — only for free-text answers not matching any option label
    raw = _groq([
        {"role": "system", "content": (
            "Extract car-buying preferences from the user message (Arabic or English). "
            "Return ONLY valid JSON: "
            "budget, priority (appearance|cost|reliability|performance|comfort), "
            "usage (family|single|commuting|off-road|mixed), "
            "fuel_pref (petrol|diesel|electric|hybrid|any), "
            "body_pref (sedan|SUV|hatchback|coupe|any), "
            "location (city/country or empty). "
            "Arabic: الشكل→appearance, التكلفة→cost, الموثوقية→reliability, "
            "الأداء→performance, عيلة→family, بنزين→petrol, ديزل→diesel. "
            "Empty string for anything not mentioned. No markdown."
        )},
        {"role": "user", "content": user_text},
    ], max_tokens=150)
    try:
        updates = json.loads(re.sub(r"```json|```", "", raw).strip())
        for field in ["budget", "priority", "usage", "fuel_pref", "body_pref"]:
            val = updates.get(field, "").strip()
            if val and not getattr(profile, field):
                setattr(profile, field, val)
        loc = updates.get("location", "").strip()
        if loc:
            profile.location = loc
            code, _ = _get_currency(loc)
            profile.currency = code
    except Exception as e:
        print(f"  Profile parse error: {e}")
    return profile

def _next_question(profile: BuyerProfile) -> tuple[str, list[str]] | None:
    """Return (question_text, options) for the next unanswered field.
    Uses Egyptian EGP-based questions when user is in Egypt."""
    questions = ADVISOR_QUESTIONS_EGYPT if _is_egypt(profile.location) else ADVISOR_QUESTIONS
    for q in questions:
        if not getattr(profile, q["field"], "").strip():
            return q["q"], q["options"]
    return None

# Country → currency mapping  (1 USD = X local)
CURRENCY_MAP = {
    "egypt":        ("EGP", 50),
    "cairo":        ("EGP", 50),
    "alexandria":   ("EGP", 50),
    "uae":          ("AED", 3.67),
    "dubai":        ("AED", 3.67),
    "abu dhabi":    ("AED", 3.67),
    "saudi":        ("SAR", 3.75),
    "saudi arabia": ("SAR", 3.75),
    "riyadh":       ("SAR", 3.75),
    "jeddah":       ("SAR", 3.75),
    "jordan":       ("JOD", 0.71),
    "amman":        ("JOD", 0.71),
    "kuwait":       ("KWD", 0.31),
    "qatar":        ("QAR", 3.64),
    "doha":         ("QAR", 3.64),
    "bahrain":      ("BHD", 0.38),
    "oman":         ("OMR", 0.38),
    "morocco":      ("MAD", 10.0),
    "uk":           ("GBP", 0.79),
    "england":      ("GBP", 0.79),
    "europe":       ("EUR", 0.92),
    "germany":      ("EUR", 0.92),
    "france":       ("EUR", 0.92),
    "turkey":       ("TRY", 32.0),
    "india":        ("INR", 83.0),
    "pakistan":     ("PKR", 278.0),
    "nigeria":      ("NGN", 1550.0),
    "kenya":        ("KES", 130.0),
}

def _get_currency(location: str) -> tuple[str, float]:
    """Return (currency_code, usd_rate) for a location string."""
    loc = location.lower().strip()
    for key, val in CURRENCY_MAP.items():
        if key in loc:
            return val
    return ("USD", 1.0)

def _currency_note(location: str) -> str:
    """
    Returns a currency instruction string to inject into ANY Groq system prompt
    when we know the user's location.  Empty string if location unknown.
    """
    if not location:
        return ""
    currency, rate = _get_currency(location)
    if currency == "USD":
        return ""
    return (
        f"\nCURRENCY RULE — VERY IMPORTANT: The user is in {location}. "
        f"Whenever you mention ANY price, cost, or money amount, "
        f"show it in {currency} (not USD, not EUR, not GBP). "
        f"Convert: 1 USD = {rate} {currency}. "
        f"Round to nearest 1,000. "
        f"Example: $20,000 USD = {round(20000*rate/1000)*1000:,} {currency}. "
        f"If you mention a USD price, always follow it with the {currency} equivalent in brackets.\n"
    )


# Arabic Unicode block range: \u0600–\u06FF
_ARABIC_RE = re.compile(r'[\u0600-\u06FF]')

def _is_arabic(text: str) -> bool:
    """True if the message contains Arabic characters."""
    return bool(_ARABIC_RE.search(text))

def _lang_note(text: str) -> str:
    """
    Returns a language instruction to inject into ANY Groq system prompt.
    If the user writes in Arabic → instruct Groq to reply fully in Arabic.
    Otherwise empty string (default English).
    """
    if _is_arabic(text):
        return (
            "\nLANGUAGE RULE — CRITICAL: The user is writing in Arabic. "
            "You MUST respond entirely in Arabic (Modern Standard Arabic or Egyptian dialect is fine). "
            "Do NOT respond in English. Use Arabic script throughout your entire response. "
            "Car names and brand names can remain in Latin script (e.g. BMW, Toyota) but all "
            "explanatory text must be in Arabic.\n"
        )
    return ""

def _format_price(usd_min: int, usd_max: int, currency: str, rate: float) -> str:
    """Convert a USD price range to local currency string."""
    if currency == "USD":
        return f"${usd_min:,}–${usd_max:,}"
    lo  = round(usd_min * rate / 1000) * 1000
    hi  = round(usd_max * rate / 1000) * 1000
    return f"{lo:,}–{hi:,} {currency}"


def _generate_recommendations(profile: BuyerProfile, history: list[HistoryMessage], user_lang: str = "en") -> tuple[str, list[str]]:
    hist_text = "\n".join(f"{m.role}: {m.content}" for m in history[-6:]) if history else ""

    # Resolve local currency
    currency, rate = _get_currency(profile.location)
    location_note  = f"User is in {profile.location}." if profile.location else ""
    currency_note  = (
        f"Show ALL prices in {currency} (not USD). "
        f"Multiply USD price by {rate:.1f} to get {currency}. "
        f"Round to nearest 1,000."
        if currency != "USD"
        else "Show prices in USD."
    )

    # Language instruction
    arabic_note = (
        "\nLANGUAGE RULE — CRITICAL: Respond entirely in Arabic. "
        "Car/brand names stay in Latin script but all text must be Arabic.\n"
        if user_lang == "ar" else ""
    )

    # Parse the budget string into a hard numeric ceiling for the prompt
    budget_str   = profile.budget or "flexible"
    budget_lower = budget_str.lower().replace(",", "")
    is_egypt     = _is_egypt(profile.location)
    budget_ceiling_note = ""

    # Helper: extract first number from string
    def _extract_num(s: str) -> int | None:
        m = re.search(r'(\d[\d]*)', s.replace(",",""))
        if not m: return None
        n = int(m.group(1))
        if "k" in s.lower() and n < 10000: n *= 1000
        return n

    # Detect ceiling based on language and currency
    if any(w in budget_lower for w in ["under","less","below","أقل","أدنى","دون"]):
        num = _extract_num(budget_lower)
        if num:
            # If EGP budget (large number) or Egyptian user with small number (convert)
            if is_egypt:
                egp_ceiling = num if num >= 10_000 else num * 50
                usd_ceiling = egp_ceiling // 50
                budget_ceiling_note = (
                    f"\n🚨 BUDGET CEILING: Max budget is {egp_ceiling:,} EGP (~${usd_ceiling:,} USD). "
                    f"Every recommended car MUST be below {egp_ceiling:,} EGP. Never exceed this."
                )
            else:
                usd_ceiling = num
                budget_ceiling_note = (
                    f"\n🚨 BUDGET CEILING: Max budget is ${usd_ceiling:,} USD. "
                    f"Every recommended car MUST be priced below ${usd_ceiling:,} USD. Never exceed this."
                )

    elif any(c in budget_str for c in ["–","-","—"]) or "egp" in budget_lower:
        # Range like "$15k – $30k" or "1M – 2M EGP" — upper end is ceiling
        nums = re.findall(r'(\d[\d,]*(?:k|m)?)', budget_lower)
        if len(nums) >= 2:
            def parse_n(s: str) -> int:
                s = s.replace(",","")
                n = int(re.sub(r'[km]','',s))
                if "m" in s: n *= 1_000_000
                elif "k" in s: n *= 1_000
                return n
            try:
                ceiling = parse_n(nums[-1])
                if is_egypt:
                    egp_c = ceiling if ceiling >= 10_000 else ceiling * 50
                    budget_ceiling_note = (
                        f"\n🚨 BUDGET CEILING: Max is {egp_c:,} EGP. "
                        f"All recommended cars must be at or below {egp_c:,} EGP."
                    )
                else:
                    budget_ceiling_note = (
                        f"\n🚨 BUDGET CEILING: Max is ${ceiling:,} USD. "
                        f"All recommended cars must be at or below ${ceiling:,} USD."
                    )
            except Exception:
                pass

    profile_summary = (
        f"Budget: {budget_str}\n"
        f"Priority: {profile.priority or '?'}\n"
        f"Usage: {profile.usage or '?'}\n"
        f"Fuel: {profile.fuel_pref or 'any'}\n"
        f"Body style: {profile.body_pref or 'any'}\n"
        f"Location: {profile.location or 'not specified'}"
    )

    # Egyptian market: build a hint list of available cars for this budget
    egypt_cars_hint = ""
    if _is_egypt(profile.location):
        available = _egypt_budget_cars(budget_str)
        egypt_cars_hint = (
            f"\nEGYPT-SPECIFIC: Cars actually available in Egypt at this budget include: "
            f"{', '.join(available)}. "
            f"Prioritize recommending from this list. "
            f"Do NOT recommend: {', '.join(sorted(EGYPT_UNAVAILABLE))}.\n"
        )

    market_knowledge = _get_market_knowledge(profile.location)

    answer = _groq([
        {"role": "system", "content": (
            "You are CarID — a knowledgeable car-buying advisor. "
            f"{location_note} {currency_note}"
            f"{budget_ceiling_note}"
            f"{egypt_cars_hint}"
            f"{market_knowledge}"
            f"{arabic_note}\n\n"
            "Recommend exactly 3 cars that fit the buyer profile. "
            "Budget is the MOST IMPORTANT constraint — never violate it. "
            "Only recommend cars that are actually available and sold in the user's country. "
            "For each car: full name, one sentence why it fits the profile, "
            "and the actual dealer price range in local currency (not converted US MSRP). "
            "Reference the conversation if relevant. "
            "End with: 'Want me to go deeper on any of these, or compare two of them?'"
        )},
        {"role": "user", "content": (
            (f"Conversation:\n{hist_text}\n\n" if hist_text else "") +
            f"Buyer profile:\n{profile_summary}\n\n"
            f"Recommend 3 cars available in {profile.location or 'this market'} "
            f"that strictly fit the budget of {budget_str}."
        )},
    ], max_tokens=650)

    # Extract car names for frontend chips
    names = re.findall(
        r'\b(?:Toyota|Honda|Ford|BMW|Mercedes|Audi|Volkswagen|Hyundai|Kia|Mazda|'
        r'Volvo|Subaru|Nissan|Renault|Peugeot|Fiat|Alfa Romeo|Jeep|Skoda|'
        r'Chevrolet|Tesla|Lexus|Porsche|Mitsubishi|Suzuki|Isuzu|Chery|MG|Dacia|'
        r'Seat|Opel|Vauxhall|Lada|Geely|BYD|Haval)[^\n,]*', answer
    )
    return answer, list(dict.fromkeys(names))[:3]


# ══════════════════════════════════════════════════════════════════════════════
# CLARIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _clarification_question(question: str, history: list[HistoryMessage]) -> tuple[str, list[str]]:
    """Ask a clarifying question and generate 3-4 relevant answer options."""
    hist_text = "\n".join(f"{m.role}: {m.content}" for m in history[-6:]) if history else "None"

    raw = _groq([
        {"role": "system", "content": (
            "You are CarID — a warm car expert. The user asked something vague or ambiguous. "
            "Return ONLY valid JSON with two keys:\n"
            "  question: one short natural clarifying question (max 12 words)\n"
            "  options: array of 3-4 short answer choices (max 5 words each) that cover the "
            "           most likely meanings of what the user asked.\n"
            "No markdown, no explanation — only the JSON object."
        )},
        {"role": "user", "content": (
            f"Conversation:\n{hist_text}\n\n"
            f"User just said: '{question}'"
        )},
    ], max_tokens=150)

    try:
        data = json.loads(re.sub(r"```json|```", "", raw).strip())
        q    = data.get("question", "Could you clarify what you're looking for?")
        opts = data.get("options", [])[:4]
        if not opts:
            opts = ["Reliability", "Price range", "Engine specs", "Pros & cons"]
        return q, opts
    except Exception:
        return "Could you clarify what you're looking for?", ["Reliability", "Price range", "Engine specs", "Pros & cons"]


# ══════════════════════════════════════════════════════════════════════════════
# DAMAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

_DAMAGE_SYSTEM = """You are an expert automotive damage assessor with 20+ years of experience.
Analyse the car damage visible in the image and any description provided.
Return ONLY a valid JSON object — no markdown, no preamble, no explanation.

JSON schema (use exactly these keys):
{
  "safe_to_drive": bool,
  "overall_severity": "minor|moderate|severe|critical",
  "damage_areas": [
    {
      "component": "string (e.g. front bumper, hood, windshield)",
      "severity": "minor|moderate|severe|critical",
      "description": "string — what you see and why this severity",
      "repair_cost_min": int (USD),
      "repair_cost_max": int (USD),
      "is_internal": false
    }
  ],
  "internal_risks": [
    {
      "component": "string (e.g. radiator, airbag sensor, CV axle)",
      "likelihood": "likely|possible|unlikely",
      "reason": "string — mechanical causation explanation",
      "repair_cost_min": int (USD),
      "repair_cost_max": int (USD)
    }
  ],
  "total_cost_min": int,
  "total_cost_max": int,
  "priority_actions": ["string", "string", "string"]
}

SEVERITY SCALE:
- minor: cosmetic only, safe to drive, <$1,000
- moderate: functional impact, drive carefully, $500–$5,000
- severe: safety risk, do NOT drive, $3,000–$15,000
- critical: total loss likely, do NOT drive, $10,000+

INTERNAL RISK RULES — infer hidden damage from visible evidence:
- Front collision → radiator, AC condenser, engine mounts, steering rack, crumple zones, airbag sensors
- Hood buckled / firewall pushed → engine block, transmission, EV battery pack
- Side impact → door intrusion beams, seat belt pretensioners, side curtain airbags, fuel lines
- Rear-end impact → fuel tank, exhaust system, rear suspension geometry
- Undercarriage scrape → oil pan, catalytic converter, transmission oil pan
- Wheel/tyre damage → CV axle, wheel bearing, brake caliper, ABS sensor
- Deployed airbags visible → crash sensors, SRS/ECU module, steering column
- Fluid puddles → identify fluid type (coolant=green/orange, oil=black, brake=clear/yellow, transmission=red)

LIKELIHOOD RATINGS:
- likely (>70%): strong visual evidence of this damage mechanism
- possible (30-70%): impact pattern is consistent but damage is not certain
- unlikely (<30%): theoretically possible but low probability given visible damage

SAFE TO DRIVE RULES — set safe_to_drive=false if ANY of these:
- Any damage_area has severity = severe OR critical
- Any internal_risk has likelihood = likely AND component involves any of:
  brake, steering, fuel, airbag, SRS, suspension, axle, engine mount, subframe, firewall, battery

COST RULES:
- total_cost_min/max = sum of damage_areas costs + sum of "likely" internal_risks costs only
- Use realistic US market repair prices

PRIORITY ACTIONS: top 3 most urgent things to do (get to workshop, don't start engine, etc.)"""

_DAMAGE_NARRATIVE_SYSTEM = """You are CarID — a warm, knowledgeable automotive expert.
You have just analysed a car damage report (JSON). Write a clear, human-friendly summary.

FORMAT RULES:
1. Start with a severity headline using emoji: 🟢 Minor | 🟡 Moderate | 🔴 Severe | ⚫ Critical
2. If safe_to_drive is false, add a bold "⛔ DO NOT DRIVE THIS CAR" warning on its own line
3. List each damage area with its severity emoji and a plain-English explanation
4. Add a "🔧 Suspected Internal Damage" section — list each internal risk with:
   - 🔴 = likely, 🟡 = possible, 🔵 = unlikely
   - Explain WHY it might be damaged in plain language
5. Show the total repair cost range
6. List the top 3 priority actions numbered
7. End with: "⚠️ Internal damage estimates are based on mechanical inference — always get a workshop inspection before buying or repairing."
8. Be honest and direct. Don't sugarcoat severe damage. Max 250 words."""


def _analyse_damage_with_groq(image_bytes: bytes, description: str = "") -> dict:
    """
    Analyse car damage using:
    1. Groq vision model (llama-4-scout) — reads the image directly
    2. Fallback: CarNet identifies the car → Serper finds common issues →
       Groq text model generates a damage report from description alone
    """

    user_text = (
        f"Additional context from user: {description}\n\n"
        if description else ""
    ) + "Analyse ALL visible damage in this image. Return ONLY the JSON object."

    # ── Attempt 1: Groq vision model ──────────────────────────────────────────
    raw = _groq_vision(image_bytes, _DAMAGE_SYSTEM, user_text, max_tokens=1200)

    if raw:
        try:
            clean = re.sub(r"```json|```", "", raw).strip()
            match = re.search(r"\{[\s\S]+\}", clean)
            if match:
                data = json.loads(match.group(0))
                if data.get("damage_areas") is not None:   # valid structure
                    print("  Damage: vision model succeeded")
                    return data
        except Exception as e:
            print(f"  Damage vision parse error: {e} | raw: {raw[:200]}")

    # ── Attempt 2: Fallback — identify car via CarNet + description-based analysis ──
    print("  Damage: vision failed, using CarNet + description fallback")

    # Try to identify the car so we can look up its known weak points
    car_label = "unknown car"
    try:
        carnet = _call_carnet(image_bytes)
        if carnet.get("brand"):
            car_label = f"{carnet['brand']} {carnet.get('model','')}".strip()
            print(f"  Damage fallback: car identified as {car_label}")
    except Exception:
        pass

    # Build a description-based prompt combining user text + car identity
    fallback_user = (
        f"Car: {car_label}\n"
        f"User description of damage: {description if description else 'No description provided — analyse based on car identity and common damage patterns.'}\n\n"
        "Based on the car model and any description provided, generate a realistic damage assessment. "
        "If no specific damage is described, note that a visual inspection is required. "
        "Return ONLY the JSON object."
    )

    raw2 = _groq([
        {"role": "system", "content": _DAMAGE_SYSTEM},
        {"role": "user",   "content": fallback_user},
    ], max_tokens=1200)

    try:
        clean2 = re.sub(r"```json|```", "", raw2).strip()
        match2 = re.search(r"\{[\s\S]+\}", clean2)
        if match2:
            data2 = json.loads(match2.group(0))
            # Mark that this came from fallback so frontend can show a note
            data2["_fallback"] = True
            data2["_car_label"] = car_label
            print("  Damage: fallback text analysis succeeded")
            return data2
    except Exception as e:
        print(f"  Damage fallback parse error: {e}")

    return {}


def _damage_narrative(data: dict) -> str:
    """Second Groq call — turns structured damage JSON into a warm human summary."""
    return _groq([
        {"role": "system", "content": _DAMAGE_NARRATIVE_SYSTEM},
        {"role": "user",   "content": f"Damage report JSON:\n{json.dumps(data, indent=2)}"},
    ], max_tokens=500)


_SAFETY_CRITICAL = re.compile(
    r"\b(brake|steering|fuel|airbag|srs|suspension|axle|engine.mount|"
    r"subframe|firewall|battery)\b", re.I
)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status":         "ok",
        "groq_key_set":   bool(GROQ_API_KEY),
        "serper_key_set": bool(SERPER_API_KEY),
        "pipeline":       "carnet+serper+groq+memory+compare+buyer+damage",
    }


@app.get("/", response_class=HTMLResponse)
async def frontend():
    for p in [HTML_PATH, Path("car_id_chat_app.html")]:
        if p.exists():
            return p.read_text(encoding="utf-8")
    raise HTTPException(404, "Frontend HTML not found.")


# ── /identify ──────────────────────────────────────────────────────────────────

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    """
    Image identification pipeline:
      1. CarNet  → brand + model + generation + color
      2. Serper  → web enrichment (specs, reliability, price)
      3. Groq    → warm, human-like response
    """
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 20 MB)")

    t0     = time.perf_counter()
    carnet = _call_carnet(data)

    if not carnet.get("brand"):
        return {
            "status":        "unknown",
            "prediction":    None,
            "confidence":    0,
            "message":       (
                "Hmm, I couldn't quite make that out. "
                "Try a clearer shot with the full car visible from the front or side — "
                "I'll get it next time! 😄"
            ),
            "details":       {},
            "query_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    brand      = carnet["brand"]
    model      = carnet["model"]
    generation = carnet.get("generation", "")
    color      = carnet.get("color", "")
    confidence = carnet.get("confidence", 0)

    # Serper enrichment
    parts       = [p for p in [brand, model, generation] if p and p not in ("—", "---", "")]
    web_results = _serper(" ".join(parts[:3]) + " specs reliability pros cons review")
    web_text    = _web_text(web_results, 4)

    # Confidence label
    conf_label = (
        "very high confidence" if confidence >= 90 else
        "high confidence"      if confidence >= 75 else
        "medium confidence"    if confidence >= 50 else
        "low confidence — possible match"
    )
    color_note = f" Color detected: {color}." if color and color not in ("—", "") else ""

    # Groq — human-like response
    message = _groq([
        {"role": "system", "content": (
            "You are CarID — a warm, enthusiastic car expert who talks exactly like "
            "an excited knowledgeable friend. A car was just identified from a photo. "
            "React naturally — lead with the car name, add real info from web results, "
            "mention color if available, be honest if confidence is low. "
            "3-5 sentences. End: 'Want to know more about the engine, reliability, or price range?'"
        )},
        {"role": "user", "content": (
            f"Identified: {brand} {model}"
            + (f" ({generation})" if generation else "")
            + color_note
            + f"\nConfidence: {conf_label} ({confidence}%)\n\nWeb info:\n{web_text}"
        )},
    ], max_tokens=350)

    return {
        "status":        "known" if confidence >= 75 else "uncertain",
        "prediction":    f"{brand} {model}",
        "confidence":    confidence,
        "message":       message,
        "details":       {
            "brand":      brand,
            "model":      model,
            "generation": generation,
            "color":      color,
            "angle":      carnet.get("angle", ""),
        },
        "web_source":    web_results[0].get("link", "") if web_results else "",
        "query_time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# ── /compare ───────────────────────────────────────────────────────────────────

@app.post("/compare")
def compare(req: CompareRequest):
    """
    Fetch structured data for each car and return a comparison table + summary.
    Called directly when the frontend wants an explicit compare (optional).
    The /chat endpoint also handles compare intent inline.
    """
    if len(req.cars) < 2:
        raise HTTPException(400, "Need at least 2 cars to compare.")
    if len(req.cars) > 4:
        raise HTTPException(400, "Max 4 cars at a time.")

    cars_data = [_fetch_car_data(c) for c in req.cars]
    summary   = _compare_summary(cars_data, req.aspect, req.history)
    return {"cars": cars_data, "summary": summary}


# ── /damage ────────────────────────────────────────────────────────────────────

@app.post("/damage")
async def damage(
    file:        UploadFile | None = File(default=None),
    description: str               = "",
):
    """
    Damage detection pipeline:
      1. Normalise image with _prepare_image()
      2. Send to Groq vision → structured DamageReport JSON
      3. Force safe_to_drive=False if any safety-critical "likely" internal risk
      4. Second Groq call → warm human-readable narrative
      5. Return full DamageReport
    """
    if not file and not description:
        raise HTTPException(400, "Provide an image file, a description, or both.")

    # Read + normalise image
    image_bytes = b""
    if file:
        raw = await file.read()
        if len(raw) > 20 * 1024 * 1024:
            raise HTTPException(413, "Image too large (max 20 MB)")
        try:
            image_bytes = _prepare_image(raw)
        except Exception as e:
            raise HTTPException(400, f"Could not process image: {e}")

    # Step 1 — structured JSON analysis
    if image_bytes:
        data = _analyse_damage_with_groq(image_bytes, description)
    else:
        # Text-only description — use Groq without vision
        raw_text = _groq([
            {"role": "system", "content": _DAMAGE_SYSTEM},
            {"role": "user",   "content": f"Damage description (no image): {description}"},
        ], max_tokens=1200)
        try:
            clean = re.sub(r"```json|```", "", raw_text).strip()
            m = re.search(r"\{[\s\S]+\}", clean)
            data = json.loads(m.group(0)) if m else {}
        except Exception:
            data = {}

    if not data:
        return DamageReport(
            safe_to_drive=False,
            overall_severity="unknown",
            damage_areas=[],
            internal_risks=[],
            total_cost_min=0,
            total_cost_max=0,
            priority_actions=["Get a professional inspection immediately"],
            message="I couldn't analyse the damage from this input. Please try a clearer photo.",
        )

    # Step 2 — build typed objects
    damage_areas: list[DamageArea] = []
    for a in data.get("damage_areas", []):
        try:
            damage_areas.append(DamageArea(**a))
        except Exception:
            pass

    internal_risks: list[InternalRisk] = []
    for r in data.get("internal_risks", []):
        try:
            internal_risks.append(InternalRisk(**r))
        except Exception:
            pass

    # Step 3 — override safe_to_drive if safety-critical likely internal risk
    safe_to_drive = bool(data.get("safe_to_drive", True))
    for risk in internal_risks:
        if risk.likelihood == "likely" and _SAFETY_CRITICAL.search(risk.component):
            safe_to_drive = False
            break

    # Also force unsafe if any area is severe or critical
    severity_order = {"minor": 0, "moderate": 1, "severe": 2, "critical": 3}
    for area in damage_areas:
        if severity_order.get(area.severity, 0) >= 2:
            safe_to_drive = False
            break

    # Step 4 — recompute overall severity from worst single area
    worst = max(
        (severity_order.get(a.severity, 0) for a in damage_areas),
        default=0
    )
    sev_labels = {0: "minor", 1: "moderate", 2: "severe", 3: "critical"}
    overall_severity = data.get("overall_severity", sev_labels[worst])

    # Step 5 — recompute total cost (external + likely internal only)
    ext_min = sum(a.repair_cost_min for a in damage_areas)
    ext_max = sum(a.repair_cost_max for a in damage_areas)
    int_min = sum(r.repair_cost_min for r in internal_risks if r.likelihood == "likely")
    int_max = sum(r.repair_cost_max for r in internal_risks if r.likelihood == "likely")
    total_min = ext_min + int_min
    total_max = ext_max + int_max

    # Step 6 — human narrative
    is_fallback  = data.get("_fallback", False)
    car_label    = data.get("_car_label", "")
    fallback_note = (
        f"\n\n⚠️ *Note: Direct image analysis was unavailable. "
        f"This report is based on{'  the identified car (' + car_label + ') and' if car_label else ''} "
        f"your description. For accurate damage assessment, please visit a workshop.*"
        if is_fallback else ""
    )

    report_dict = {
        "safe_to_drive":    safe_to_drive,
        "overall_severity": overall_severity,
        "damage_areas":     [a.model_dump() for a in damage_areas],
        "internal_risks":   [r.model_dump() for r in internal_risks],
        "total_cost_min":   total_min,
        "total_cost_max":   total_max,
        "priority_actions": data.get("priority_actions", [])[:3],
    }
    narrative = _damage_narrative(report_dict) + fallback_note

    return DamageReport(
        safe_to_drive=safe_to_drive,
        overall_severity=overall_severity,
        damage_areas=damage_areas,
        internal_risks=internal_risks,
        total_cost_min=total_min,
        total_cost_max=total_max,
        priority_actions=data.get("priority_actions", [])[:3],
        message=narrative,
    )


# ── /chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Full conversational chat with:
      - Complete conversation memory (history injected into every Groq call)
      - Car context (anchors follow-ups like "is it reliable?" to the right car)
      - Compare intent → structured table data returned to frontend
      - Clarify intent → asks one natural question when query is too vague
      - Buyer advisor → progressive Q&A → personalised recommendations
    """
    # Accept both "question" and "message" field names from the frontend
    question = (req.question or req.message).strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")

    history    = req.history or []
    car_ctx    = req.car_context or {}
    buyer_mode = req.buyer_mode
    profile    = req.buyer_profile

    # ── Resolve user location ─────────────────────────────────────────────────
    user_location = req.user_location.strip() or profile.location.strip()
    if not user_location:
        q_lower = question.lower()
        for kw in sorted(CURRENCY_MAP.keys(), key=len, reverse=True):
            if kw in q_lower:
                user_location = kw.title()
                break
    if user_location and not profile.location:
        profile.location = user_location
        code, _ = _get_currency(user_location)
        profile.currency = code

    # ── Detect language ────────────────────────────────────────────────────────
    # Use frontend-provided lang, or auto-detect from message content
    user_lang = req.user_lang or ("ar" if _is_arabic(question) else "en")

    # Also check history for Arabic — user may switch languages mid-conversation
    if user_lang == "en" and history:
        for m in history[-4:]:
            if _is_arabic(m.content):
                user_lang = "ar"
                break

    # Notes injected into every Groq system prompt
    curr_note = _currency_note(user_location)
    lang_note = _lang_note(question) if user_lang == "ar" else ""

    print(f"  [chat] intent-detect | q={question[:60]!r} | "
          f"history={len(history)} | buyer={buyer_mode} | car={car_ctx.get('brand','—')} | "
          f"location={user_location or '—'} | lang={user_lang}")

    intent = _detect_intent(question, history, car_ctx, buyer_mode)
    print(f"  → intent: {intent}")

    # ══ 1. BUYER ADVISOR ════════════════════════════════════════════════════════
    if intent == "buyer" and not buyer_mode:
        profile = BuyerProfile()
        if user_location: profile.location = user_location
        # Use Egyptian questions if user is in Egypt, else use appropriate language
        questions = ADVISOR_QUESTIONS_EGYPT if _is_egypt(user_location) else ADVISOR_QUESTIONS
        first_q   = questions[0]

        # Opening greeting — Arabic or English based on detected language
        if user_lang == "ar":
            opening = (
                "أهلاً! 😊 يسعدني أساعدك تلاقي السيارة المناسبة. "
                "هسألك بعض الأسئلة البسيطة عشان أقدر أنصحك صح.\n\n"
                + first_q["q"]
            )
        else:
            opening = (
                "Oh nice, you're car shopping! I love helping with this. 😊 "
                "Let me ask a few quick questions so I can find your perfect match.\n\n"
                + first_q["q"]
            )

        return ChatResponse(
            answer=opening, reply=opening,
            intent="buyer", buyer_mode=True, buyer_profile=profile,
            options=first_q["options"],
            user_location=user_location, user_lang=user_lang,
        )

    if buyer_mode:
        # Parse their answer into the profile — also detects location
        profile = _extract_profile(question, profile)
        # Sync location back
        if profile.location and not user_location:
            user_location = profile.location
            curr_note = _currency_note(user_location)
            # Switch to Egyptian questions if location just detected
            if _is_egypt(user_location) and user_lang == "en":
                user_lang = "ar"   # Egyptian users likely prefer Arabic

        if _profile_complete(profile):
            answer, rec_names = _generate_recommendations(profile, history, user_lang)
            return ChatResponse(
                answer=answer, reply=answer,
                intent="recommend", buyer_mode=False,
                buyer_profile=profile, recommendations=rec_names,
                user_location=user_location, user_lang=user_lang,
            )

        nxt = _next_question(profile)
        if nxt:
            next_q_text, next_opts = nxt
            return ChatResponse(
                answer=next_q_text, reply=next_q_text,
                intent="buyer", buyer_mode=True, buyer_profile=profile,
                options=next_opts,
                user_location=user_location, user_lang=user_lang,
            )

        # Safety fallback — profile not complete but no more questions to ask
        # Force recommendations with whatever we have
        answer, rec_names = _generate_recommendations(profile, history, user_lang)
        return ChatResponse(
            answer=answer, reply=answer,
            intent="recommend", buyer_mode=False,
            buyer_profile=profile, recommendations=rec_names,
            user_location=user_location, user_lang=user_lang,
        )

    # ══ 2. COMPARE ══════════════════════════════════════════════════════════════
    if intent == "compare":
        car_names = _extract_car_names(question, history)

        if len(car_names) < 2:
            clarify = (
                "Sure, happy to compare! Just to make sure I get it right — "
                "which two (or more) cars exactly? For example: 'BMW E90 vs Alfa Romeo 156'."
            )
            return ChatResponse(
                answer=clarify, reply=clarify, intent="clarify",
                user_location=user_location, user_lang=user_lang,
            )

        # Pass currency info to _fetch_car_data via the aspect string
        cars_data = [_fetch_car_data(c, user_location) for c in car_names]

        aspect_match = re.search(
            r"\b(reliab\w*|price|cost|engine|fuel|comfort|space|family|performance)\b",
            question, re.I
        )
        summary = _compare_summary(cars_data, aspect_match.group(0) if aspect_match else "", history, user_location, user_lang)

        return ChatResponse(
            answer=summary, reply=summary,
            intent="compare", compare_data=cars_data,
            user_location=user_location, user_lang=user_lang,
        )

    # ══ 3. CLARIFY ══════════════════════════════════════════════════════════════
    if intent == "clarify":
        clarify_q, clarify_opts = _clarification_question(question, history)
        return ChatResponse(
            answer=clarify_q, reply=clarify_q,
            intent="clarify", options=clarify_opts,
            user_location=user_location, user_lang=user_lang,
        )

    # ══ 4. NORMAL CONVERSATIONAL CHAT (with full memory) ════════════════════════

    # Build Serper query — include Egyptian market context when relevant
    base_q = (
        f"{car_ctx['brand']} {car_ctx.get('model', '')} {question}".strip()
        if car_ctx.get("brand") else question
    )
    if _is_egypt(user_location):
        # Single combined query (Arabic + English) to save Serper quota
        web_results = _serper(f"{base_q} مصر سعر Egypt price")
    else:
        web_results = _serper(base_q)
    web_text = _web_text(web_results, 5)

    # Car context note injected into system prompt
    car_note = ""
    if car_ctx.get("brand"):
        parts    = [car_ctx.get(k, "") for k in ["brand", "model", "generation"]]
        car_note = (
            f"\nCAR IN FOCUS: {' '.join(p for p in parts if p and p not in ('—', ''))} "
            f"(color: {car_ctx.get('color', '—')}). "
            "Short follow-ups like 'is it reliable?' or 'what about the engine?' "
            "refer to THIS car — answer about it without asking which car.\n"
        )

    system = (
        "You are CarID — a warm, funny, deeply knowledgeable car expert who talks "
        "exactly like a real human friend who loves cars. "
        "You remember EVERYTHING said in this conversation and naturally weave it "
        "into your answers. If something was mentioned earlier, refer back to it naturally. "
        "You use natural phrases: 'honestly', 'the thing is', 'good news', "
        "'to be fair', 'personally I'd…'. "
        "You're direct and give real opinions — not wishy-washy non-answers."
        + car_note
        + curr_note
        + lang_note
        + _get_market_knowledge(user_location) +
        "\nRULES:\n"
        "- Never start with 'Answer:', 'Based on results', or 'According to'.\n"
        "- Always reference earlier conversation when relevant.\n"
        "- Give direct useful answers — specs, prices, owner opinions, reliability.\n"
        "- When user is in Egypt: use real Egyptian dealer prices, not converted US MSRP.\n"
        "- Only recommend cars available in the user's local market.\n"
        "- 3-5 sentences unless user asks for more.\n"
        "- End with a natural, relevant follow-up question.\n"
        "- Never invent facts not in the web results or conversation."
    )

    groq_msgs = [{"role": "system", "content": system}]
    for m in history[-14:]:
        if m.role in ("user", "assistant"):
            groq_msgs.append({"role": m.role, "content": m.content})
    groq_msgs.append({
        "role": "user",
        "content": f"{question}\n\n[Web search results]\n{web_text}",
    })

    answer = _groq(groq_msgs, max_tokens=480)
    return ChatResponse(
        answer=answer, reply=answer, intent="chat",
        buyer_mode=False, buyer_profile=profile,
        user_location=user_location, user_lang=user_lang,
    )


# ── /chat/stream ────────────────────────────────────────────────────────────────

async def _groq_stream(messages: list[dict]):
    """Yield Server-Sent Event lines from Groq streaming API."""
    if not GROQ_API_KEY:
        yield f"data: {json.dumps({'token': 'Chat unavailable — GROQ_API_KEY not set.'})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
        return

    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream(
            "POST", GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "max_tokens": 500, "messages": messages, "stream": True},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return
                try:
                    chunk = json.loads(raw)
                    token = chunk["choices"][0]["delta"].get("content", "")
                    if token:
                        yield f"data: {json.dumps({'token': token})}\n\n"
                except Exception:
                    continue


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming version of /chat for normal conversational responses.
    For buyer/compare/clarify intents, falls back to the regular /chat endpoint
    since those need structured JSON, not a stream.
    """
    question = (req.question or req.message).strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")

    history  = req.history or []
    car_ctx  = req.car_context or {}
    buyer_mode = req.buyer_mode

    # Non-streamable intents — redirect to regular /chat
    intent = _detect_intent(question, history, car_ctx, buyer_mode)
    if intent in ("buyer", "compare", "clarify") or buyer_mode:
        result = chat(req)
        async def _wrap():
            # Send full JSON as a single done event so frontend can handle it normally
            yield f"data: {json.dumps({'done': True, **result.model_dump()})}\n\n"
        return StreamingResponse(_wrap(), media_type="text/event-stream")

    # Normal chat → stream token by token
    serper_q = (
        f"{car_ctx['brand']} {car_ctx.get('model','')} {question}".strip()
        if car_ctx.get("brand") else question
    )
    web_results = _serper(serper_q)
    web_text    = _web_text(web_results, 5)

    car_note = ""
    if car_ctx.get("brand"):
        parts = [car_ctx.get(k,"") for k in ["brand","model","generation"]]
        car_note = (
            f"\nCAR IN FOCUS: {' '.join(p for p in parts if p and p not in ('—',''))} "
            f"(color: {car_ctx.get('color','—')}). "
            "Short follow-ups refer to THIS car.\n"
        )

    system = (
        "You are CarID — a warm, funny, deeply knowledgeable car expert who talks "
        "exactly like a real human friend who loves cars. "
        "You remember EVERYTHING said in this conversation and naturally weave it into your answers. "
        "You use natural phrases: 'honestly', 'the thing is', 'good news', 'to be fair', 'personally I'd…'. "
        "You're direct and give real opinions."
        + car_note +
        "\nRULES:\n"
        "- Never start with 'Answer:', 'Based on results', or 'According to'.\n"
        "- Reference earlier conversation when relevant.\n"
        "- Give direct useful answers — specs, prices, owner opinions, reliability.\n"
        "- 3-5 sentences unless asked for more.\n"
        "- End with a natural follow-up question.\n"
        "- Never invent facts."
    )

    groq_msgs = [{"role": "system", "content": system}]
    for m in history[-14:]:
        if m.role in ("user", "assistant"):
            groq_msgs.append({"role": m.role, "content": m.content})
    groq_msgs.append({"role": "user", "content": f"{question}\n\n[Web search results]\n{web_text}"})

    return StreamingResponse(
        _groq_stream(groq_msgs),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )