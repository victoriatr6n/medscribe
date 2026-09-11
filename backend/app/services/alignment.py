def _overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def align_transcript_with_speakers(whisper_segments, diarization_segments):
    """
    whisper_segments: list of {"start", "end", "text"} from transcription.transcribe()
    diarization_segments: list of {"start", "end", "speaker"} from diarization.diarize()

    Assigns each whisper segment to whichever diarization speaker it overlaps
    with the most, then merges consecutive same-speaker segments into one line.
    """
    labeled = []
    for seg in whisper_segments:
        best_speaker, best_overlap = "UNKNOWN", 0
        for d in diarization_segments:
            ov = _overlap(seg["start"], seg["end"], d["start"], d["end"])
            if ov > best_overlap:
                best_overlap, best_speaker = ov, d["speaker"]
        labeled.append({"speaker": best_speaker, "text": seg["text"].strip()})

    # merge consecutive lines from the same speaker so you don't get a new
    # line for every single whisper chunk
    lines = []
    for item in labeled:
        if lines and lines[-1]["speaker"] == item["speaker"]:
            lines[-1]["text"] += " " + item["text"]
        else:
            lines.append(item)

    return "\n".join(f"[{l['speaker']}]: {l['text']}" for l in lines)