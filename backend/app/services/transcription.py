"""
Step 1 of the pipeline: audio -> raw transcript.

Run this standalone first, before touching diarization or note generation:

    python -m app.services.transcription data/audio_samples/your_file.wav

Once this reliably prints a clean transcript for a few different audio files,
move on to diarization.
"""

import sys
import whisper

_MODEL_CACHE = {}


def get_model(model_size: str = "base"):
    """
    Loads (and caches) a Whisper model.

    Model size tradeoffs (speed vs accuracy):
      tiny   - fastest, least accurate. Fine for pipeline plumbing/testing.
      base   - good default for a laptop CPU.
      small  - noticeably better accuracy, still workable on CPU.
      medium/large - best accuracy, you likely want a GPU for these.
    """
    if model_size not in _MODEL_CACHE:
        _MODEL_CACHE[model_size] = whisper.load_model(model_size)
    return _MODEL_CACHE[model_size]


def transcribe(audio_path: str, model_size: str = "base") -> dict:
    """
    Returns Whisper's raw result dict, which includes:
      - "text": full transcript as a single string
      - "segments": list of {start, end, text} — you'll need these timestamps
        later to align with diarization output.
    """
    model = get_model(model_size)
    result = model.transcribe(audio_path)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.services.transcription <path_to_audio_file>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Transcribing {path} ...")
    result = transcribe(path)

    print("\n--- Full transcript ---")
    print(result["text"])

    print("\n--- Segments (needed later for diarization alignment) ---")
    for seg in result["segments"]:
        print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
