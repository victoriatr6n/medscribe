"""
Step 2 of the pipeline: figure out WHO said WHAT (doctor vs patient).

This is the fiddliest part of the whole project. Budget real time for it.

Setup required before this will run:
  1. Create a free Hugging Face account: https://huggingface.co/join
  2. Create an access token: https://huggingface.co/settings/tokens
  3. Accept the model terms (required, it's a click-through) at:
       https://huggingface.co/pyannote/speaker-diarization-3.1
       https://huggingface.co/pyannote/segmentation-3.0
  4. Put your token in .env as HUGGINGFACE_TOKEN=...

What this gives you: a list of (start_time, end_time, speaker_label) segments,
e.g. SPEAKER_00 from 0.0s-4.2s, SPEAKER_01 from 4.2s-9.8s, etc.
pyannote does NOT know which speaker is the doctor vs the patient — that's a
separate, easier heuristic you'll add in `align.py` (e.g. whoever asks more
questions, or just let the user label it once in the UI).

Next step after this works: write the alignment logic that merges these
speaker segments with Whisper's word/segment timestamps from transcription.py,
so each transcript line gets a speaker label.
"""

import os
from dotenv import load_dotenv

load_dotenv()

_PIPELINE_CACHE = None


def get_pipeline():
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is None:
        from pyannote.audio import Pipeline

        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise RuntimeError(
                "HUGGINGFACE_TOKEN not set. See this file's docstring for setup steps."
            )
        _PIPELINE_CACHE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
    return _PIPELINE_CACHE


def diarize(audio_path: str) -> list[dict]:
    """
    Returns a list of segments like:
      [{"start": 0.0, "end": 4.2, "speaker": "SPEAKER_00"}, ...]
    """
    pipeline = get_pipeline()
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })
    return segments


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.services.diarization <path_to_audio_file>")
        sys.exit(1)

    for seg in diarize(sys.argv[1]):
        print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}")
