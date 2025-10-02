# chatbot_en.py

import pickle
import faiss
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from models_config import EMBEDDING_MODEL, GENERATION_MODEL


INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_meta.pkl"

class MedicalRAG:
    def __init__(self, index_path=INDEX_PATH, meta_path=META_PATH,
                 embedding_model=EMBEDDING_MODEL, generation_model=GENERATION_MODEL):
        # Load embedding model
        self.embedder = SentenceTransformer(embedding_model)

        # Load FAISS index + chunks
        if not os.path.exists(meta_path) or not os.path.exists(index_path):
            raise FileNotFoundError("Run ingest.py first to create FAISS index and metadata.")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.chunks = meta["chunks"]

        # Load text2text generation pipeline
        self.generator = pipeline("text2text-generation", model=generation_model, device=-1)  # CPU

        # Conversation history
        self.sessions = {}

    def _retrieve(self, query, k=4):
        """Retrieve top-k relevant chunks from FAISS"""
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        return [self.chunks[idx] for idx in I[0] if idx < len(self.chunks)]

    def start_session(self, session_id):
        self.sessions[session_id] = {"history": []}

    def ask(self, session_id, question: str):
        if session_id not in self.sessions:
            self.start_session(session_id)

        # Retrieve a very small context (or disable if it's noisy)
        context_chunks = self._retrieve(question, k=2)
        context_text = "\n".join(context_chunks[:1])

        # Strong prompt engineering
        prompt = f"""
You are a clinical triage assistant. 
Your job is to ask ONE focused, specific follow-up question based on the patient's complaint.
The question must explore: onset, duration, severity, location, frequency, associated symptoms, or risk factors.
Do not repeat the patient's words directly.
Do not ask vague questions like "What is the cause?"
Do not provide medical advice or diagnosis.
Keep your response short (max 1 sentence).

Patient input:
"{question}"

Relevant context (optional):
{context_text}

Your follow-up question:
"""

        inputs = self.generator.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        outputs = self.generator.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,             # enable some variation
            temperature=0.7,            # allow creativity
            top_p=0.9,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5
        )

        out = self.generator.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        self.sessions[session_id]["history"].append({
            "patient": question,
            "bot": out,
            "context": context_chunks
        })

        disclaimer = "\n\n⚠️ Note: This chatbot provides informational questions only; it is not a substitute for medical care."
        return out + disclaimer

    