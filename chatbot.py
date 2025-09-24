# chatbot.py
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from models_config import EMBEDDING_MODEL, GENERATION_MODEL
from utils.translation import translate_text
import os

INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_meta.pkl"

class MedicalRAG:
    def __init__(self, index_path=INDEX_PATH, meta_path=META_PATH,
                 embedding_model=EMBEDDING_MODEL, generation_model=GENERATION_MODEL):
        # Load embedding model
        self.embedder = SentenceTransformer(embedding_model)
        # Load FAISS index and chunks
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Run ingest.py first to create FAISS index and metadata.")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.chunks = meta["chunks"]
        # generation pipeline (text2text)
        self.generator = pipeline("text2text-generation", model=generation_model, device=-1)  # CPU
        # conversation history per session (simple dict; in prod persist to DB)
        self.sessions = {}

    def _retrieve(self, query, k=4):
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        results = [self.chunks[idx] for idx in I[0] if idx < len(self.chunks)]
        return results

    def start_session(self, session_id):
        self.sessions[session_id] = {"history": []}

    def ask(self, session_id, question, src_lang="en", translate_back=True):
        """
        question: raw text from patient (in src_lang)
        src_lang: language code like 'hi' for Hindi. If 'en', no translation performed.
        """
        if session_id not in self.sessions:
            self.start_session(session_id)

        # 1. translate patient input to English for retrieval/generation
        question_en = translate_text(question, src=src_lang, tgt="en") if src_lang != "en" else question

        # 2. Retrieve top chunks
        context_chunks = self._retrieve(question_en, k=4)
        context_text = "\n\n".join(context_chunks)

        # 3. Build prompt for follow-up question (we ask LLM to act like a clinician and ask the next question)
        prompt = f"""You are a clinician assistant. Given the patient's input below and the retrieved clinical Q&A context, generate the next single concise question to ask the patient to gather more clinical information. Use plain language that a patient will understand.

Patient input:
{question_en}

Context (relevant dialogues):
{context_text}

Ask one follow-up question that helps clarify the present complaint. Do NOT give medical advice or diagnosis. """
        out = self.generator(prompt, max_length=128, do_sample=False)[0]["generated_text"].strip()

        # store into session
        self.sessions[session_id]["history"].append({"patient": question_en, "bot": out, "context": context_chunks})

        # 4. translate back to patient's language if needed
        if translate_back and src_lang != "en":
            out_local = translate_text(out, src="en", tgt=src_lang)
        else:
            out_local = out

        # append standard disclaimer in patient's language (simple English fallback)
        disclaimer = ""
        if src_lang == "hi":
            disclaimer = "\n\n⚠️ यह चैटबॉट केवल जानकारी के लिए है; यह किसी चिकित्सीय निदान का विकल्प नहीं है।"
        else:
            disclaimer = "\n\n⚠️ Note: This chatbot provides informational questions only; it is not a substitute for medical care."

        return out_local + disclaimer

    def summarize_session(self, session_id):
        if session_id not in self.sessions:
            return "No session found."

        history = self.sessions[session_id]["history"]
        convo_text = ""
        for turn in history:
            convo_text += f"Patient said: {turn['patient']}\nBot asked: {turn['bot']}\n\n"
        # Summarize into structured report
        summary_prompt = f"""
Summarize the following clinician-patient dialog into a structured medical report for the doctor.
Include: Chief complaint, Onset & duration, Location, Severity (if given), Associated symptoms, Past history & risk factors, Any red flags (call emergency), and a short recommended next step for the physician.

Dialog:
{convo_text}
"""
        summary = self.generator(summary_prompt, max_length=512, do_sample=False)[0]["generated_text"].strip()
        return summary
