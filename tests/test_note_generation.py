"""
Starter evaluation harness.

The hard question this project raises: how do you know if a generated SOAP
note is actually good? This file is meant to grow into your answer to that
question — it's genuinely one of the more impressive things you can bring up
in an interview, since most people never think past "it looks right to me."

Three approaches, roughly in order of effort:

1. Structural checks (below): does the output validate against the schema?
   Are source_spans actually verbatim substrings of the transcript? These are
   cheap, deterministic, and catch a lot of failure modes.

2. Rubric-based LLM-as-judge: write a second prompt that scores a generated
   note against the source transcript on specific axes (completeness,
   faithfulness/no invented facts, clinical usefulness). Log scores across
   your test transcripts over time as you tune your main prompt.

3. Human spot-checking: for a handful of transcripts, write out what YOU think
   the correct SOAP note should be, and diff your generated notes against it.
   Small sample, but ground truth.

Run with: pytest tests/test_note_generation.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.services.note_generation import generate_soap_note, SAMPLE_TRANSCRIPT
from app.models.soap_note import SoapNote


def test_output_validates_against_schema():
    note = generate_soap_note(SAMPLE_TRANSCRIPT)
    assert isinstance(note, SoapNote)


def test_source_spans_are_verbatim_or_null():
    """
    Catches the most important failure mode: a field claiming grounding in a
    span that doesn't actually appear in the transcript (i.e. the model
    fabricated a citation for a fabricated fact).
    """
    note = generate_soap_note(SAMPLE_TRANSCRIPT)
    all_fields = note.subjective + note.objective + note.assessment + note.plan

    for field in all_fields:
        if field.source_span is not None:
            assert field.source_span in SAMPLE_TRANSCRIPT, (
                f"Source span not found verbatim in transcript: {field.source_span!r}"
            )


def test_no_medication_dosage_invented():
    """
    Domain-specific safety check: the SAMPLE_TRANSCRIPT never mentions a
    specific drug or dosage, so the plan section shouldn't invent one.
    Add more checks like this as you find real failure modes during testing.
    """
    note = generate_soap_note(SAMPLE_TRANSCRIPT)
    plan_text = " ".join(f.value.lower() for f in note.plan)
    # crude check — replace with a real drug-name list if you build this out
    suspicious_terms = ["mg", "milligram", "aspirin", "ibuprofen", "lisinopril"]
    for term in suspicious_terms:
        assert term not in plan_text, f"Possibly invented medication detail: {term!r}"


if __name__ == "__main__":
    # Quick manual run without pytest, useful while iterating
    note = generate_soap_note(SAMPLE_TRANSCRIPT)
    print(note.model_dump_json(indent=2))
