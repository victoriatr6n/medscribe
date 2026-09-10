# AI Medical Scribe

A pipeline that turns a doctor-patient conversation (audio) into a structured,
guardrailed SOAP note.

Audio -> Whisper (transcription) -> pyannote.audio (speaker diarization)
-> Claude (structured SOAP extraction with confidence flags) -> reviewable note

## Why this project

Most "call an LLM API" projects don't have much depth. This one does, because it
forces you to solve:

1. **Speech-to-text** on real, messy audio (pauses, overlaps, background noise)
2. **Speaker diarization** — telling doctor from patient (not solved by Whisper alone)
3. **Structured extraction under ambiguity** — mapping conversational speech into a
   rigid clinical schema (SOAP: Subjective, Objective, Assessment, Plan)
4. **Hallucination guardrails** — the note must be traceable back to transcript spans,
   or flagged as low-confidence. This is the single most interview-worthy feature.
5. **Evaluation** — how do you know if a generated note is any good? (see `tests/`)

## Project layout

```
medscribe/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entrypoint
│   │   ├── routers/
│   │   │   └── notes.py         # /transcribe, /diarize, /generate-note endpoints
│   │   ├── services/
│   │   │   ├── transcription.py # Whisper wrapper
│   │   │   ├── diarization.py   # pyannote.audio wrapper
│   │   │   └── note_generation.py # Claude API call + schema validation
│   │   └── models/
│   │       └── soap_note.py     # Pydantic schema for the SOAP note
│   └── requirements.txt
├── data/
│   ├── audio_samples/           # put test .wav/.mp3 files here
│   └── transcripts/             # cached transcripts land here
├── frontend/
│   └── app.py                   # Streamlit UI (build this last)
├── tests/
│   └── test_note_generation.py  # start your eval harness here
└── .env.example
```

## Setup

```bash
cd medscribe/backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env      # then fill in your ANTHROPIC_API_KEY
```

## Build order (do these in sequence — don't skip ahead)

1. **Get transcription working on one file.**
   Drop any short audio file into `data/audio_samples/`, then run:
   ```bash
   python -m app.services.transcription data/audio_samples/your_file.wav
   ```
   This should print a raw transcript. Get this rock solid before touching anything else.

2. **Add diarization.** `app/services/diarization.py` has a stub — this is the
   fiddliest part of the whole project. It requires a (free) Hugging Face token
   and accepting pyannote's model terms — instructions are in the file's docstring.

3. **Hardcode a transcript, get note generation working.** Don't wait on audio —
   use the `SAMPLE_TRANSCRIPT` in `note_generation.py` to iterate on your prompt
   and schema fast.

4. **Wire the pipeline end-to-end** via the FastAPI routes in `routers/notes.py`.

5. **Add the confidence-flagging / span-grounding logic** (already scaffolded in
   `note_generation.py` — this is the guardrail feature, don't skip it).

6. **Build the Streamlit frontend last.**

## Run the API

```bash
cd backend
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs (FastAPI gives you this for free).

## Run the frontend (once backend works)

```bash
cd frontend
streamlit run app.py
```

## A note on data

Do NOT use real patient data. Use:
- Self-recorded mock conversations (you + a friend acting out a script)
- LLM-generated synthetic dialogue (see `data/transcripts/synthetic_example.txt`)
- Public research datasets intended for this purpose (e.g. search for PriMock57 —
  verify licensing/availability yourself before using)
