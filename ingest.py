# ingest.py
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models_config import EMBEDDING_MODEL

INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_meta.pkl"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

def load_scenarios(path="data/medical_scenarios.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text):
    # Using simple splitter from langchain for robustness
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def build_faiss(chunks, emb_model_name=EMBEDDING_MODEL):
    print("Loading embedding model:", emb_model_name)
    embedder = SentenceTransformer(emb_model_name)
    print("Computing embeddings...")
    embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    print("Embedding dimension:", dim)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    # save index
    faiss.write_index(index, INDEX_PATH)
    # save metadata (chunks)
    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "model": emb_model_name}, f)
    print("Saved FAISS index ->", INDEX_PATH)
    print("Saved metadata ->", META_PATH)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    raw = load_scenarios("data/medical_scenarios.txt")
    chunks = chunk_text(raw)
    build_faiss(chunks)
