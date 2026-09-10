"""
Step 3 of the pipeline: labeled transcript -> structured, grounded SOAP note.

This is where the actual "AI engineering" happens. Two things matter most:
  1. The prompt must force strict JSON matching our schema.
  2. Every extracted field must be grounded in a verbatim transcript span,
     so we can flag anything the model can't actually point to.

Iterate on this file using SAMPLE_TRANSCRIPT below — don't wait for the
audio/diarization pipeline to be finished before you start tuning this.
"""

import json
import os
from dotenv import load_dotenv
from anthropic import Anthropic

from app.models.soap_note import SoapNote

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# A hardcoded example so you can develop this file in isolation.
# Format matches what your diarization+transcription pipeline will eventually output.
SAMPLE_TRANSCRIPT = """
[DOCTOR]: So what brings you in today?
[PATIENT]: I've had this chest tightness for about three days now, mostly when I climb stairs.
[DOCTOR]: Any shortness of breath with it, or pain radiating anywhere?
[PATIENT]: A little short of breath, yeah. No radiating pain that I've noticed.
[DOCTOR]: Okay. Let's check your blood pressure and listen to your heart.
[DOCTOR]: Blood pressure's 138 over 88, heart rate 76, sounds normal on auscultation.
[DOCTOR]: Given the exertional pattern, I want to rule out cardiac causes. I'm going to order an EKG and some bloodwork, and I'd like you to avoid strenuous activity until we have results.
[PATIENT]: Okay, that sounds good.
"""

SYSTEM_PROMPT = """You are a clinical documentation assistant. You will be given a \
transcript of a doctor-patient conversation, with speakers labeled [DOCTOR] and \
[PATIENT].

Extract a structured SOAP note from it. Respond with ONLY valid JSON, no preamble, \
no markdown code fences, matching exactly this schema:

{
  "subjective": [{"value": str, "source_span": str or null, "confidence": "high"|"medium"|"low"}],
  "objective": [...same shape...],
  "assessment": [...same shape...],
  "plan": [...same shape...]
}

Rules you must follow:
- "source_span" must be a VERBATIM quote copied from the transcript that supports \
"value". If you cannot find a supporting quote, set source_span to null and \
confidence to "low".
- "confidence" reflects how directly the transcript supports the value:
  - "high": stated outright, unambiguous
  - "medium": reasonably inferred from context, not stated outright
  - "low": you are filling a gap or guessing — this will be flagged for human review
- Do NOT invent clinical details (dosages, diagnoses, vitals) that are not in the \
transcript. If something is not mentioned, do not include a field for it.
- Keep "value" strings concise and clinical, not verbatim conversational speech.
"""


def generate_soap_note(transcript: str, model: str = "claude-sonnet-4-6") -> SoapNote:
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript}],
    )

    raw_text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    # Defensive cleanup in case the model wraps output in code fences anyway
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON:\n{raw_text}") from e

    return SoapNote.model_validate(parsed)


if __name__ == "__main__":
    note = generate_soap_note(SAMPLE_TRANSCRIPT)

    print("=== SOAP NOTE ===\n")
    for section_name in ["subjective", "objective", "assessment", "plan"]:
        print(f"-- {section_name.upper()} --")
        for field in getattr(note, section_name):
            flag = " ⚠️ LOW CONFIDENCE" if field.confidence == "low" else ""
            print(f"  - {field.value}{flag}")
            print(f"    source: {field.source_span!r}")
        print()

    low_conf = note.low_confidence_fields()
    if low_conf:
        print(f"{len(low_conf)} field(s) flagged for human review.")
