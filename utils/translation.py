# utils/translation.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from functools import lru_cache

# Simple caching of loaded models to avoid reloading
@lru_cache(maxsize=8)
def load_marian(src, tgt):
    """
    Returns tokenizer, model for given src->tgt using Helsinki models.
    If src == tgt, returns None (no translation).
    """
    if src == tgt:
        return None, None
    # e.g., src='hi', tgt='en' -> 'Helsinki-NLP/opus-mt-hi-en'
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        # fallback: try en-src style or raise informative error
        raise RuntimeError(f"Translation model {model_name} not found: {e}")

def translate_text(text, src="auto", tgt="en"):
    """
    If src == 'auto', this function assumes the caller will set src explicitly.
    For demo, user will supply src_lang; if 'auto' we skip translation.
    """
    if src == tgt or src == "auto":
        # For 'auto' we do not attempt detection to keep dependencies low.
        # In practice use a language detection library (langdetect).
        return text
    tokenizer, model = load_marian(src, tgt)
    if tokenizer is None:
        return text
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    out = model.generate(**inputs, max_length=512)
    return tokenizer.decode(out[0], skip_special_tokens=True)
