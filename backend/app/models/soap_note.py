"""
Schema for a generated SOAP note.

The key design decision here: every extracted field carries the transcript
text it was derived from (`source_span`) and a confidence label. If the model
can't point to a real span, that's your signal the field may be hallucinated.
This is the guardrail feature — don't simplify this away.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Confidence(str, Enum):
    high = "high"       # directly and unambiguously stated in the transcript
    medium = "medium"   # inferred from context, not stated outright
    low = "low"         # model is guessing / filling a gap — flag for human review


class GroundedField(BaseModel):
    """A single piece of extracted clinical info, grounded in the transcript."""
    value: str = Field(..., description="The extracted clinical content")
    source_span: Optional[str] = Field(
        None, description="Verbatim quote from the transcript that supports this value"
    )
    confidence: Confidence = Field(
        Confidence.low, description="How directly the source_span supports the value"
    )


class SoapNote(BaseModel):
    subjective: List[GroundedField] = Field(
        default_factory=list,
        description="Patient-reported symptoms, history, concerns"
    )
    objective: List[GroundedField] = Field(
        default_factory=list,
        description="Observable/measurable findings mentioned (vitals, exam findings)"
    )
    assessment: List[GroundedField] = Field(
        default_factory=list,
        description="Clinical impressions / possible diagnoses discussed"
    )
    plan: List[GroundedField] = Field(
        default_factory=list,
        description="Next steps: medications, tests, follow-up, referrals"
    )

    def low_confidence_fields(self) -> List[GroundedField]:
        """Convenience method for the frontend to highlight fields needing review."""
        all_fields = self.subjective + self.objective + self.assessment + self.plan
        return [f for f in all_fields if f.confidence == Confidence.low]
