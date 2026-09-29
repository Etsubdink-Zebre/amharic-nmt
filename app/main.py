"""English → Amharic translation API + web UI.

Run:   uvicorn app.main:app --port 8000
       POST /translate  {"text": "I am going to the university."}
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.translator import Translator  # noqa: E402

TRANSLATORS: dict[str, Translator] = {}
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app):
    # Load both models (+ tokenizers) once, and warm them up so the first
    # user request does not pay for GPU kernel compilation.
    for name in ("bahdanau", "attention", "seq2seq"):
        TRANSLATORS[name] = Translator(name)
        TRANSLATORS[name].translate("hello.")
    yield


app = FastAPI(title="English → Amharic NMT", version="1.0", lifespan=lifespan)


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, examples=["I am going to the university."])
    # "bahdanau" = Attention-LSTM (additive attention, the deployed default);
    # "attention" = Luong-attention ablation; "seq2seq" = basic baseline.
    model: Literal["bahdanau", "attention", "seq2seq"] = "bahdanau"
    beam: int = Field(5, ge=1, le=10)
    return_attention: bool = False


class TranslateResponse(BaseModel):
    translation: str
    model: str
    latency_ms: float
    src_tokens: list[str] | None = None
    tgt_tokens: list[str] | None = None
    attention: list[list[float]] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "models": list(TRANSLATORS)}


@app.post("/translate", response_model=TranslateResponse, response_model_exclude_none=True)
def translate(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(422, "text is empty")
    r = TRANSLATORS[req.model].translate(text, beam=req.beam)
    if not req.return_attention:
        r = {k: r[k] for k in ("translation", "model", "latency_ms")}
    return r


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
