# models_config.py
# Swap these names if you want to use different models later.

# Embeddings model (sentence-transformers)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Generator / Q/A / Summarization model - choose a small/flan-t5 for demo
GENERATION_MODEL = "google/flan-t5-base"   # text2text-generation (good demo model)

# Marian translation model mapping source->English. We'll dynamically pick model names.
# For many language pairs, we use Helsinki models like 'Helsinki-NLP/opus-mt-<src>-en'
MARIAN_MODEL_TEMPLATE = "Helsinki-NLP/opus-mt-{src}-en"
MARIAN_MODEL_TEMPLATE_REV = "Helsinki-NLP/opus-mt-en-{tgt}"
