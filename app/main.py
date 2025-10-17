#!/usr/bin/env python3
"""
EduNote FastAPI App (robust lazy init + /generate)
- Ensures project root is on sys.path
- Lazily initializes LlamaWrapper with default Alpaca model
- Provides /, /upload, /summarize, /quiz, /generate endpoints
"""

import sys
import os
from pathlib import Path
import tempfile
from typing import Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ensure project root (~/EduNote) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try importing local extractors and Llama wrapper; allow server to run even if missing
try:
    from extractors import extract_text_from_uploaded_file
except Exception:
    extract_text_from_uploaded_file = None

try:
    from edunote_llm.llama_wrapper import LlamaWrapper
except Exception:
    LlamaWrapper = None

app = FastAPI(title="EduNote Backend (lazy init)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default model & binary (edit if your files are elsewhere)
DEFAULT_MODEL = os.path.expanduser("~/EduNote/models/alpaca/claude2-alpaca-7b.Q3_K_M.gguf")
DEFAULT_BIN = os.path.expanduser("~/EduNote/llama.cpp/llama-cli")

_llama = None
def get_llama():
    """Lazy initializer for LlamaWrapper. Returns instance or raises HTTPException."""
    global _llama
    if _llama is None:
        if LlamaWrapper is None:
            raise HTTPException(status_code=500, detail="Llama wrapper module not available.")
        if not os.path.isfile(DEFAULT_MODEL):
            raise HTTPException(status_code=500, detail=f"Model file not found at {DEFAULT_MODEL}")
        if not os.path.isfile(DEFAULT_BIN):
            raise HTTPException(status_code=500, detail=f"llama-cli binary not found at {DEFAULT_BIN}")
        try:
            _llama = LlamaWrapper(model_path=DEFAULT_MODEL, binary_path=DEFAULT_BIN)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize LlamaWrapper: {e}")
    return _llama

# Simple chunker (character-based)
def simple_chunk_text(text: str, max_chars: int = 3500):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + max_chars, L)
        split_index = text.rfind("\n", start, end)
        if split_index <= start:
            split_index = text.rfind(".", start, end)
        if split_index <= start:
            split_index = end
        chunks.append(text[start:split_index].strip())
        start = split_index
    return chunks

# Prompt templates
SUMMARY_PROMPT = """
You are EduNote, a study assistant. Summarize the following study notes into 5 concise bullet points and a 2-sentence summary.

Notes:
\"\"\"
{content}
\"\"\"

Output format:
- Bullet 1
- Bullet 2
...
Summary: <two sentence paragraph>
"""

QUIZ_PROMPT = """
You are EduNote, a study assistant. Create 5 multiple-choice questions from the text below.
Each question should have 4 options labeled A, B, C, D, and mark the correct option with (Answer: X).

Text:
\"\"\"
{content}
\"\"\"
"""

# Pydantic model for /generate endpoint
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 64

@app.get("/")
async def root():
    return {"service": "EduNote", "status": "ok", "model_path": DEFAULT_MODEL, "binary_path": DEFAULT_BIN}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if extract_text_from_uploaded_file is None:
        raise HTTPException(status_code=500, detail="Text extractor not available (extractors.py import failed).")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".txt"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        text = extract_text_from_uploaded_file(tmp_path, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return {"filename": file.filename, "length": len(text), "text_preview": text[:2000], "text": text}

@app.post("/summarize")
async def summarize(payload: Dict):
    full_text = payload.get("text", "") or payload.get("url_text", "")
    if not full_text:
        raise HTTPException(status_code=400, detail="No text provided.")
    llama = get_llama()
    chunks = simple_chunk_text(full_text, max_chars=3200)

    bullets = []
    summary_paragraphs = []
    for c in chunks:
        prompt = SUMMARY_PROMPT.format(content=c)
        out = llama.generate(prompt, n_predict=256)
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("-"):
                b = line.lstrip("- ").strip()
                if b and b not in bullets:
                    bullets.append(b)
            elif line.lower().startswith("summary"):
                if ":" in line:
                    summary_paragraphs.append(line.split(":", 1)[1].strip())
                else:
                    summary_paragraphs.append(line.strip())
        if len(bullets) >= 5:
            break

    bullets = bullets[:5]
    summary_text = " ".join(summary_paragraphs)[:800] if summary_paragraphs else ""
    return {"bullets": bullets, "summary": summary_text}

@app.post("/quiz")
async def quiz(payload: Dict):
    full_text = payload.get("text", "")
    if not full_text:
        raise HTTPException(status_code=400, detail="No text provided.")
    llama = get_llama()
    chunk = simple_chunk_text(full_text, max_chars=3200)
    chunk_text = chunk[0] if chunk else full_text[:3200]
    prompt = QUIZ_PROMPT.format(content=chunk_text)
    out = llama.generate(prompt, n_predict=512)

    # Try to parse questions; fallback to raw text
    questions = []
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    current_q = None
    for l in lines:
        if l.lower().startswith("question") or l.endswith("?"):
            if current_q:
                questions.append(current_q)
            current_q = {"question": l, "options": [], "answer": None}
        elif l.upper().startswith(("A)", "B)", "C)", "D)", "A.", "B.", "C.", "D.", "A ", "B ", "C ", "D ")):
            if current_q is None:
                current_q = {"question": "", "options": [], "answer": None}
            current_q["options"].append(l)
            if "(Answer:" in l or "Answer:" in l:
                current_q["answer"] = l.split("Answer:")[-1].strip(" )")
        elif "(Answer:" in l:
            if current_q:
                current_q["answer"] = l.split("(Answer:",1)[1].strip(" )")
        else:
            if current_q is None:
                current_q = {"question": l, "options": [], "answer": None}
    if current_q:
        questions.append(current_q)

    if not questions:
        return {"quiz_raw": out}
    return {"quiz": questions, "quiz_raw": out}

# --- New /generate endpoint ---
@app.post("/generate")
async def generate_text(request: PromptRequest):
    llama = get_llama()
    try:
        response = llama.generate(request.prompt, n_predict=request.max_tokens)
        return {"prompt": request.prompt, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
