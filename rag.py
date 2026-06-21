import os, json, re
from pathlib import Path
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


# This looks for the .env file and loads the variables into system
load_dotenv()

METADATA_FILE = Path(
    os.getenv("METADATA_FILE", "metadata_flat.json")
) 
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_MODEL    = "llama-3.1-8b-instant"

# Lightweight sentence-transformer (runs on CPU, ~90 MB)
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

# lazy singleton
_qa_chain = None   

_metadata_records: list[dict] = []

# ── Helpers ──
def _has_value(val: str) -> bool:
    """True if the field has real, not a dash or placeholder. """
    return bool(val) and val.strip() not in {"", "-", "_", "---", "N/A", "n/a"}

def _norm(text: str) -> str:
    """Lowercase, remove punctuation, for fuzzy matching"""
    return re.sub(r"[^a-z0-9\s]", "", text.lower().strip())


# ── DB loader──
def _load_metadata() -> list[dict]:
    """Load and deduplicate metadata_flat.json once, cache it."""
    global _metadata_records
    if _metadata_records:
        return _metadata_records
    try:
        records = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        seen, unique = set(), []
        
        for r in records:
            key = f"{r.get('brand','')}_{r.get('model','')}_{r.get('generation','')}"
            if key not in seen:
                seen.add(key)
                unique.append(r)
        _metadata_records = unique
        print(f"  Metadata: {len(_metadata_records)} unique car records cached")
        return _metadata_records
    except Exception as e :
        print(f"  Metadata load error: {e}")
        return []

# ── Metadata search ───
def search_metadata(user_text: str) -> dict | None:
    """
    Search metadata_flat.json for the best car matching the user's text.
 
    Scoring:
      3 pts → brand AND model both found in query
      2 pts → brand found in query
      1 pt  → any long word matches brand or model
 
    Returns best matching record (score ≥ 2) or None.
    """
    records = _load_metadata()
    if not records:
        return None

    q   = _norm(user_text)
    q_words = set(q.split())

    best, best_score = None, 0

    for record in records:
        brand = _norm(record.get("brand", ""))
        model = _norm(record.get("model", ""))

        brand_match = bool(brand) and brand in q
        model_words = set(model.split())
        model_match = bool(model) and (
            model in q or (len(model_words)>1 and model_words.issubset(q_words))
                       or any(word in q for word in model_words if len(word) > 3)
        )
        
        if brand_match and model_match:
            score = 3
        elif brand_match:
            score = 2
        elif any(word in q for word in brand.split() if len(word) > 3):
            score = 1
        else:
            score = 0

        if score > best_score:
            best_score = score
            best = record

    if best_score >= 2 and best:
        print(f"  Metadata match (score={best_score}): "
              f"{best.get('brand')} {best.get('model')}")
        return best

    print(f"  No metadata match for: {user_text[:60]!r}")
    return None

# ── Document loader for RAG ─── 
def _load_documents() -> list[Document]:
    records = _load_metadata()
    docs = []

    for record in records:
        brand   = record.get("brand", "Unknown")
        model   = record.get("model", "Unknown")
        generation = record.get("generation", "")

        def v(val):
            return val if _has_value(str(val)) else "Not Available"

        content = (
            f"CAR_RECORD\n"
            f"Car: {brand} {model}\n"
            f"Generation: {v(generation)}\n"
            f"Year range: {v(record.get('year', record.get('year_range', '—')))}\n"
            f"Power/Engine: {v(record.get('engine', record.get('power', '—')))}\n"
            f"Price: {v(record.get('price', '—'))}\n"
            f"Body/Fuel: {v(record.get('body', record.get('fuel', '—')))}\n"
            f"Pros: {v(record.get('pros', '—'))}\n"
            f"Cons: {v(record.get('cons', '—'))}"
        )

        docs.append(
            Document(
                page_content=content,
                metadata = {"brand": brand, "model": model, "generation": generation},
            )
        )
    return docs

# ── Prompt ──
CARID_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are CarID, a friendly and knowledgeable automotive assistant. \
You have access to a structured car database. Answer questions naturally and \
conversationally — like a helpful car expert talking to a friend.
 
RULES YOU MUST ALWAYS FOLLOW:
1. NEVER start your reply with "Answer:" — respond naturally, like a person talking.
2. NEVER invent specs, years, or numbers. Only use facts from the database context.
3. MISSING DATA: If a field shows "NOT_AVAILABLE" (e.g. for pros or cons) you can search if you find it, say it \
   naturally: "I have this car in our database, but we don't have specific pros and \
   cons listed for this generation yet."
4. MISINFORMATION DETECTION: If the user's question contains a specific claim \
   (like "500 HP" or "made in 1960") that contradicts the database, correct them \
   gently. For example: "Actually, based on our records, this model produces around \
   150 HP, not 500 — you might be thinking of a different variant."
5. CAR NOT FOUND: If no matching car is in the context, say: "I don't have that \
   specific model in our database yet. Try a slightly different name, or I can tell \
   you about a similar car."
6. LENGTH: Keep answers to 2-4 sentences unless the user explicitly asks for more.
7. TONE: Be warm and engaging. Use phrases like "Based on our records...", \
   "Interestingly...", "Great question!", "From what I can see..." where natural.
 
DATABASE CONTEXT:
{context}
 
USER QUESTION:
{question}
 
YOUR RESPONSE:"""
)


def build_rag():
    """Build and return the QA chain. Call once at startup."""
    global _qa_chain

    if _qa_chain is not None:
        return _qa_chain

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY env var not set")

    print("Building RAG knowledge base …")
    docs = _load_documents()
    print(f"  → {len(docs)} unique car models loaded")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = LangFAISS.from_documents(docs, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        temperature=0,
        model=GROQ_MODEL,
        api_key=SecretStr(GROQ_API_KEY),
    )

    _qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": CARID_PROMPT},
        return_source_documents=False
    ) 

    print("  → RAG ready")
    return _qa_chain

# ── Ask ──
def _clean(text: str) -> str:
    """Strip any leftover 'Answer:' or 'YOUR RESPONSE:' the model might echo."""
    text = re.sub(r"(?i)^(answer|your response)\s*:\s*", "", text.strip())
    return text.strip()


def ask(qa_chain, question: str) -> str:
    """Ask a question and return the answer string."""
    try:
        result = qa_chain.invoke({"query": question})
        return _clean(result.get("result", str(result)))
    except Exception as e:
        return f"I ran into a technical issue trying to answer that: {e}"