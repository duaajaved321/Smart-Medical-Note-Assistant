"""Pydantic schema and validation for generated SOAP notes."""

import json
from pydantic import BaseModel, ConfigDict, Field


class SOAPNote(BaseModel):
    """The four required sections of a SOAP clinical note."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    subjective: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    assessment: str = Field(min_length=1)
    plan: str = Field(min_length=1)


def validate_clinical_note(response_text: str) -> SOAPNote:
    """Parse model output as JSON and validate the SOAP structure."""
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("The model returned an empty response.")
    data = json.loads(response_text)
    return SOAPNote.model_validate(data)
