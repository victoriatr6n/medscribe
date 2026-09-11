"""
API routes. Build/test each service standalone first (see their __main__ blocks),
then wire them together here.
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.transcription import transcribe
from app.services.diarization import diarize
from app.services.alignment import align_transcript_with_speakers
from app.services.note_generation import generate_soap_note
from app.models.soap_note import SoapNote

router = APIRouter()


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Phase 1: audio in, raw transcript out. Test this route first."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = transcribe(tmp_path)
        return {"text": result["text"], "segments": result["segments"]}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/diarize")
async def diarize_audio(file: UploadFile = File(...)):
    """Phase 2: audio in, speaker segments out."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        segments = diarize(tmp_path)
        return {"segments": segments}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/generate-note", response_model=SoapNote)
async def generate_note(transcript: str):
    """
    Phase 3: labeled transcript in (e.g. "[DOCTOR]: ... [PATIENT]: ..."),
    structured SOAP note out.

    NOTE: this takes transcript text directly so you can test note generation
    without needing working audio/diarization first. Once phases 1+2 work,
    build a helper that merges their outputs into this same labeled-transcript
    format (see align_transcript_with_speakers, which you'll write next) and
    call it from a combined /process-visit endpoint.
    """
    try:
        return generate_soap_note(transcript)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))

@router.post("/process-visit")
async def process_visit(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        transcript_result = transcribe(tmp_path)          # service, not route
        diarization_segments = diarize(tmp_path)           # service, not route
        labeled_transcript = align_transcript_with_speakers(
            transcript_result["segments"], diarization_segments
        )
        print("=== LABELED TRANSCRIPT ===")
        print(labeled_transcript)
        return generate_soap_note(labeled_transcript)       # service, not route
    finally:
        Path(tmp_path).unlink(missing_ok=True)