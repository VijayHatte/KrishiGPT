"""
rag/multilingual.py
-------------------
Marathi + English support for the KrishiGPT advisory chatbot.

Retrieval works best when the query is embedded in the same language as the
knowledge base (mostly English here), so we:

  1. Detect the query language (Marathi `mr` vs English `en`).
  2. Translate Marathi -> English for the retrieval step.
  3. Ask the LLM to answer back in the farmer's original language.

Everything is free / open-source:
  * Detection   : `langdetect` (+ a Devanagari-script short-circuit).
  * Translation : Helsinki-NLP OPUS-MT models via `transformers`
                  (opus-mt-mr-en / opus-mt-en-mr), downloaded once and cached.

Both degrade gracefully: if a dependency is missing or a call fails we fall
back to treating the text as English, so the pipeline never hard-fails.
"""

from __future__ import annotations

from functools import lru_cache

# Devanagari Unicode block U+0900-U+097F (Marathi/Hindi share the script).
_DEVANAGARI = range(0x0900, 0x0980)

LANG_NAMES = {"mr": "Marathi", "en": "English"}

# Open-source MarianMT translation checkpoints (no API key, run locally).
_OPUS_MODELS = {
    ("mr", "en"): "Helsinki-NLP/opus-mt-mr-en",
    ("en", "mr"): "Helsinki-NLP/opus-mt-en-mr",
}


def _has_devanagari(text: str) -> bool:
    return any(ord(ch) in _DEVANAGARI for ch in text)


def detect_language(text: str) -> str:
    """Return an ISO code: 'mr' for Marathi, otherwise 'en'.

    Devanagari script is the strongest signal, so we short-circuit on it
    before falling back to statistical detection.
    """
    if _has_devanagari(text):
        return "mr"
    try:
        from langdetect import detect  # lazy import; optional dependency

        return "mr" if detect(text) == "mr" else "en"
    except Exception:
        return "en"


@lru_cache(maxsize=4)
def _load_translator(source: str, target: str):
    """Lazily build and cache a HuggingFace translation pipeline."""
    from transformers import pipeline

    model_name = _OPUS_MODELS[(source, target)]
    return pipeline("translation", model=model_name)


def translate(text: str, source: str, target: str) -> str:
    """Translate ``text`` from ``source`` to ``target`` (ISO codes).

    Returns the original text unchanged if source == target or translation
    is unavailable, so a missing model / offline run never breaks the chat.
    """
    if source == target or not text.strip():
        return text
    if (source, target) not in _OPUS_MODELS:
        return text
    try:
        translator = _load_translator(source, target)
        return translator(text, max_length=512)[0]["translation_text"]
    except Exception:
        return text


def language_name(code: str) -> str:
    """Human-readable language name for a prompt (e.g. 'mr' -> 'Marathi')."""
    return LANG_NAMES.get(code, "English")
