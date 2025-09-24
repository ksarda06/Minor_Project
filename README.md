# Medical Appointment Chatbot (RAG Demo)

This project demonstrates a Retrieval-Augmented Generation (RAG) chatbot for medical appointment triage:
- patient speaks in native language -> translated -> RAG retrieval -> LLM asks follow-ups -> summarized report for doctor (PDF).

## Quick start
1. Put dialogues into `data/medical_scenarios.txt`
2. Create venv, install with `pip install -r requirements.txt`
3. Run `python ingest.py`
4. Start server: `uvicorn app:app --reload`
5. Open demo UI: `streamlit run frontend_streamlit.py`

## Swap models
Edit `models_config.py` to use different embedding and generation models.
