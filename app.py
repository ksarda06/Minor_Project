# app.py
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from chatbot import MedicalRAG
from utils.report import save_report_pdf
import uuid
import os

app = FastAPI(title="Medical Appointment Chatbot (RAG Demo)")
@app.get("/")
def read_root():
    return {"message": "Hello! Your API is working 🚀"}
# Ensure ingest.py has been run
rag = MedicalRAG()

class ChatRequest(BaseModel):
    session_id: str = None
    text: str
    lang: str = "en"  # language code of the user's message (e.g., 'hi' for Hindi)

@app.post("/start_session")
def start_session():
    sid = str(uuid.uuid4())
    rag.start_session(sid)
    return {"session_id": sid}

@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or "default"
    if session_id not in rag.sessions:
        rag.start_session(session_id)
    reply = rag.ask(session_id, req.text, src_lang=req.lang)
    return {"reply": reply, "session_id": session_id}

@app.post("/summary")
def get_summary(session_id: str = Form(...), patient_name: str = Form("Unknown")):
    if session_id not in rag.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    summary = rag.summarize_session(session_id)
    # save pdf
    out = save_report_pdf(summary, patient_name=patient_name)
    return {"summary": summary, "report_file": out}

@app.post("/ingest")
def ingest_endpoint():
    # run ingestion from file (dangerous in prod, but OK for demo)
    from ingest import build_faiss, chunk_text, load_scenarios
    text = load_scenarios("data/medical_scenarios.txt")
    chunks = chunk_text(text)
    build_faiss(chunks)
    return {"status": "ingested"}
