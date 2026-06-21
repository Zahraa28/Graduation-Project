"""
fallback_handler.py
────────────────────
IMAGE PATH logic:
 
  CLIP + FAISS → clip_model (brand + model)
  CarNet web   → carnet_model (brand + model)
 
  IF clip_model == carnet_model:
      → merge both outputs
      → Serper search with combined keys
      → highlight pros & cons from DB
      → Groq synthesis → human message
 
  IF clip_model != carnet_model:
      → use CarNet output only (more reliable for unknown cars)
      → Serper search with CarNet keys
      → Groq synthesis → human message
 
TEXT PATH logic (used by /chat in main.py):
  → search metadata by user text
  → if found: Serper search by keys → Groq
  → if not found: Serper direct search → Groq
"""

from ast import MatchOr
import os, json, re, io
import logging
from unicodedata import normalize
from urllib.request import build_opener
import httpx
import requests
from dotenv import load_dotenv
from carnet import call_carnet

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"

CLIP_WEIGHT   = 0.6
CARNET_WEIGHT = 0.4

_http = httpx.Client(timeout=20)


# ─ Groq helper ─

def _groq(system: str, user: str, max_tokens: int = 600) -> str:
    if not GROQ_API_KEY:
        return ""
    resp = _http.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}", 
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _parse_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}

# ── Fusion scoring ──

def _carnet_confidence(carnet: dict) -> float:
    if not carnet:
        return 0.0
    b = carnet.get("brand_prob", 0.0)
    m = carnet.get("model_prob", 0.0)
    g = carnet.get("generation_prob", 0.0)
    return round(0.5 * b + 0.35 * m + 0.15 * g, 4)


def compute_fusion_score(clip_score: float, carnet: dict) -> float:
    if not carnet:
        return round(clip_score, 4)
    return round(CLIP_WEIGHT * clip_score + CARNET_WEIGHT * _carnet_confidence(carnet), 4)


# ──────── Matching ───────────
def _normalise(text: str) -> str:
    """Lowercase, strip spaces and punctuation for comparison."""
    return re.sub(r"[^a-z0-9]", "", text.lower().strip())

def models_match(clip_best: dict, carnet: dict) -> bool:
    """ 
    Return true : 
        if CLIP and Carnet agree on Brand & Model
    By using :
        Normalised comparison to handle minor naming differences
    EX:   "citroen" == "Citroen, "C3 Picasso" == "C3-picasso"
    """
    clip_brand = _normalise(clip_best.get("brand", ""))
    clip_model = _normalise(clip_best.get("model", ""))
    carnet_brand = _normalise(carnet.get("brand", ""))
    carnet_model = _normalise(carnet.get("model", ""))

    # if the brand is missing
    if not clip_brand or not carnet_brand :
        return False

    # Brand comparison
    brand_match: bool= clip_brand == carnet_brand
    # the Exact Model or one contains like "156" vs "Alfa Romeo 156"
    model_match: bool = bool(
        clip_model == carnet_model 
        or (
            clip_model and carnet_model 
            and (
                clip_model in carnet_model or carnet_model in clip_model 
                )
            )
    )

    final_match = brand_match and model_match

    return final_match  

# ── Serper search ──────────────────────────────────────────

def _serper_search(query: str) -> list[dict]:
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
            {
                "title": r.get("title",""), 
                "snippet": r.get("snippet",""), 
                "link": r.get("link","")
            }
            for r in resp.json().get("organic", [])[:5]
        ]
    except Exception as e:
        print(f"  Serper error: {e}")
        return []


def _build_search_query(details: dict, matched: bool) -> str:
    """Build serper query for image path
        if matched : include generation for precision
        if not matched : use carnet brand + model  
    """
    brand = details.get("brand", "")
    model = details.get("model", "")
    generation = details.get("generation", "")

    parts = [p for p in [brand, model] if p and p not in ("—", "---", "")]

    if matched and generation and generation not in ("—", "---"):
        parts.append(generation)
    return " ".join(parts) + " specs reliability pros cons reviews"


# ── Groq synthesis ───
def _synthesise_with_groq(
    db_data: dict,
    web_results: list[dict],
    fusion_score: float,
    matched: bool,
    source_note:str = "",
) -> str:
    """
        Merge DB metadata + web snippets into a warm, human-like message.
        when matched = True, highlight pros & cons from the DB.
    """
    if not GROQ_API_KEY:
        brand = db_data.get("brand", "Unknown")
        model = db_data.get("model", "Unknown")
        return f"I found a match: {brand} {model}. Confidence: {round(fusion_score*100)}% confidence. Want to know more?"

    def v(val):
        return val if val and val not in ("—", "---", "Not_Available", "") else "not available"

    db_summary = (
        f"Brand: {db_data.get('brand','?')}\n"
        f"Model: {db_data.get('model','?')}\n"
        f"Generation: {v(db_data.get('generation',''))}\n"
        f"Year range: {v(db_data.get('year',''))}\n"
        f"Engine/Power: {v(db_data.get('engine',''))}\n"
        f"Body/Fuel: {v(db_data.get('body',''))}\n"
        f"Pros: {v(db_data.get('pros',''))}\n"
        f"Cons: {v(db_data.get('cons',''))}\n"
        f"Color detected: {v(db_data.get('color',''))}"
    )

    web_text = "\n".join(
        f"- {r['title']}: {r['snippet']}" for r in web_results[:4]
    ) if web_results else "No web results available."

    confidence_label = (
        "high confidence — definitely this car"
        if fusion_score >= 0.88
        else "medium confidence — very likely this car"
        if fusion_score >= 0.75
        else "low confidence — possible match, not certain"
    )

    # Extra Info when both CLIP and CarNet agreed

    pros_cons_note = ""
    if matched:
        pros_cons_note = (
            "\n10. IMPORTANT: Both our visual AI and CarNet agreed on this car — "
            "this is a strong match. Make sure to highlight the Pros and Cons "
            "from the database record clearly, even if they are brief."
        )

    system = f"""You are CarID, a friendly and knowledgeable automotive assistant.
Tell the user what car was identified and share the most useful real-world info.
 
RULES:
1. Sound like a real human — warm, helpful, never robotic.
2. NEVER start with "Answer:", "Based on the data", or "According to".
3. Lead with the car name — e.g. "That's an Alfa Romeo 156!" or "Looks like a BMW 3 Series..."
4. Use web results to add real colour: reliability, owner opinions, price range, fun facts.
5. If DB has "not available" but web has it, use the web info.
6. If confidence is low, be honest: "I'm not 100% certain, but this looks like..."
7. Keep it to 3-5 sentences — punchy and useful.
8. End with an invitation: "Want to know more about the engine, reliability, or price?"
9. NEVER make up numbers or specs not in the data.{pros_cons_note}
 
Source note: {source_note}"""
 
    user = (
        f"Confidence: {confidence_label} ({round(fusion_score*100)}%)\n\n"
        f"DATABASE RECORD:\n{db_summary}\n\n"
        f"WEB SEARCH RESULTS:\n{web_text}\n\n"
        "Write a warm, conversational message about this car."
    )


    try:
        return _groq(system, user, max_tokens=300)
    except Exception as e:
        print(f"  Groq synthesis error: {e}")
        brand = db_data.get("brand", "Unknown")
        model = db_data.get("model", "Unknown")
        return (
            f"Looks like a {brand} {model}!" 
            f"({round(fusion_score*100)}% confidence). Want to know more?"
        )


# ── Used by main.py /chat enrichment ─────────────────────────────────────────

def _extract_car_snippets(snippets: list[dict], query: str, carnet: dict) -> dict:
    """
        Extract car data  from serper snippets 
        used by /chat text
    """
    text = "\n".join(f"- {r['title']}: {r['snippet']}" for r in snippets)
    hint = ""
    if carnet.get("brand"):
        hint = f"\nCarNet hint: {carnet['brand']} {carnet.get('model','')} ({carnet.get('brand_prob',0):.0%})"
    try:
        return _parse_json(_groq(
            "Automotive expert. Extract car info from web snippets. Reply ONLY with valid JSON, no other text.",
            f'Search: "{query}"{hint}\nResults:\n{text}\n\n'
            'Reply with ONLY:\n{"brand":"...","model":"...","generation":"...","year":"...",'
            '"engine":"...","body":"...","pros":"...","cons":"...","summary":"..."}',
            max_tokens=400,
        ))
    except Exception:
        return {}


# ── Master dispatcher  "Image" ───
def predict_with_fallback(
    image_bytes: bytes,
    best_score: float,
    best_label: str,
    top_k: list[dict],
    known_threshold: float     = 0.88,
    uncertain_threshold: float = 0.75,
) -> dict:
    """
    IMAGE PATH pipeline:
 
    Step 1 — Run CarNet on the image (brand + model + generation)
    Step 2 — Compare CLIP best match vs CarNet best match
    Step 3 — IF match:
                 merge CLIP DB record + CarNet extras
                 Serper query = brand + model + generation
                 highlight pros & cons
             IF no match:
                 use CarNet output only
                 Serper query = CarNet brand + model
    Step 4 — Groq synthesis → human-like message
    """

    # Step 1 — CarNet
    carnet = call_carnet(image_bytes)
    fusion = compute_fusion_score(best_score, carnet)
    
    clip_best = top_k[0] if top_k else {}
    
    print(f"  CLIP={best_score:.3f} | "
          f"CLIP model={clip_best.get('brand','-')} {carnet.get('model','-')}"
          f"CarNet model={carnet.get('brand','-')} {carnet.get('model','-')}"
          f"({carnet.get('brand_prob',0):.0%}) | fusion={fusion:.3f}")

    # Step 2 — Do CLIP AND CarNet match?
    matched = models_match(clip_best, carnet)
    print(f" Match: {matched}")


    # Step 3.1 — Match & Merge both predctions
    if matched:
        # 1. CLIP DB record (has ptos, cons, year, fuel,...)
        details = dict(clip_best)

        # 2. Add CarNet Extract that DB not have 
        if carnet.get("color") and carnet["color"] not in ("—", ""):
            details["color"] = carnet["color"]
        if carnet.get("angle") and carnet["angle"] not in ("—", ""):
            details["angle"] = carnet["angle"]
        # if the CarNet has a more specific generation
        if (carnet.get("generation") and carnet["generation"] not in ("—", "") 
            and details.get("generation") in ("—", "----", "")):

            details["generation"] = carnet["generation"]

        source_note = (
            f"CLIP and CarNet both identifed:"
            f"{details.get('brand')} {details.get('model')}."
            f"Strong match — pros & cons available from database."
        )

    # Step 3.2 — NO Match — use CarNet only — 
    else:
        if carnet.get("brand"):
            details = {
                "brand":      carnet.get("brand", ""),
                "model":      carnet.get("model", ""),
                "generation": carnet.get("generation", "—"),
                "year": "—", "engine": "—", "body": "—",
                "pros": "—", "cons": "—",
            }

            source_note = (
                f"CLIP suggested {clip_best.get('brand','-')} {clip_best.get('model','-')} "
                f"but CarNet identified {carnet.get('brand')} {carnet.get('model')}. "
                f"Using CarNet result as it is more reliable for brand detection."
            )
        else:
            # CarNet faild too - Fall back to CLIP
            details = dict(clip_best)
            source_note = (
                f"CarNet unavailable. Using CLIP result: "
                f"{clip_best.get('brand','-')} {clip_best.get('model','-')}."
            )

    # Step 4: Serper search 
    search_query = _build_search_query(details, matched)
    print(f"    Serper query: {search_query}")
    web_results = _serper_search(search_query)
    # If no match and web results avaliable, try to extract info
    if not matched and web_results:
        extracted = _extract_car_snippets(web_results, search_query, carnet)
        if extracted.get("brand"):
            for key in ("year", "engine", "body", "pros", "cons"):
                if extracted.get(key) and details.get(key) in ("", "—", "----"):
                    details[key] = extracted[key]

     # Step 5 — Groq synthesis → human-like message
    message = _synthesise_with_groq(
        db_data=details,
        web_results=web_results,
        fusion_score=fusion,
        matched=matched,
        source_note=source_note,
        )

    if fusion >= known_threshold:
        status   = "known"
    elif fusion >= uncertain_threshold:
        status   = "uncertain"
    else:
        status   = "unknown"
    

    prediction = f"{details.get('brand','')} {details.get('model','')}".strip() or None

    return {
        "status":       status,
        "prediction":   prediction,
        "clip_score":   round(best_score, 4),
        "fusion_score": fusion,
        "matched" :     matched,
        "source":       "faiss+carnet-web+serper+groq",
        "from_web":     not matched,
        "message":      message,
        "details":      details,
    }