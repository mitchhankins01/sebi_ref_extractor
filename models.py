from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ReferenceHit(BaseModel):
    reference_text: str = Field(..., description="Verbatim citation text")
    reference_type: str = Field(
        ..., description="Type of reference: circular, regulation, section, act, notification, gazette, url, annexure, chapter, other"
    )
    page_number: int = Field(..., ge=1)
    context_snippet: str = Field(..., description="±120 chars around match; max 240 chars")
    method: str = Field(..., description="regex or llm")
    canonical_id: Optional[str] = Field(default=None)
    issuing_body: Optional[str] = Field(default="SEBI")
    date: Optional[str] = Field(default=None, description="ISO YYYY-MM-DD")
    scope: Literal["internal", "external"]
    confidence: Optional[float] = Field(default=None)
    cited_title: Optional[str] = Field(default=None)
    cited_date: Optional[str] = Field(default=None, description="ISO YYYY-MM-DD")
    cited_title_confidence: Optional[float] = Field(default=None)


class SourceDoc(BaseModel):
    title: str
    pages: int
    file_name: str


class ExtractionResult(BaseModel):
    source_document: SourceDoc
    references: List[ReferenceHit]


