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

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # want to call transcription -> store that in a string? or dict of text, segment (value of 'segment' key is a lsit of {start,end,text})
    # want to call diarization -> store list of segments {start,end,speaker}
        # good format bc we can just align the start and end times + merge speaker and text

    # want a new line for each speaker so will probably base when to write a new line based on result from diarization
     # for each segment, find the time it overlaps with for transcription can pull those texts and concatentate them, label under appropriate speaker from diarization results
        # append to a string? with escape characters

    # give that transcript to the generate note script and return SOAP
    transcript = transcribe_audio(file)
    diarized = diarize_audio(file)
    str_to_return = ""

    for start_speaker,end_speaker,speaker in diarized:
        tup = (speaker, "")
        for text, segments in transcript:
            for start_text,end_text,text in segments:
                if start_text >= start_speaker and start_text <= end_speaker and end_text <= end_speaker and end_text >= start_speaker:
                    tup[1] += text
                    #should add all text that belongs to this speaker
        str_to_return += f"{speaker} {tup[1]} \n"

    soap_notes = generate_note(str_to_return)

    return soap_notes