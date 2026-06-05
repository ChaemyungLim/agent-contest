from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


SourceType = Literal["official", "history"]
AnswerType = Literal["full", "partial"]


class Hit(BaseModel):
    """검색된 단일 후보."""
    id: str
    question: str
    answer: str
    source: SourceType
    answered_at: date | None = None
    answered_by: str | None = None
    score: float = 0.0


class SourceInfo(BaseModel):
    type: SourceType
    answered_at: date | None = None
    answered_by: str | None = None


class DepartmentInfo(BaseModel):
    id: str
    name: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    matched: bool
    answer: str | None = None
    answer_type: AnswerType | None = None  # matched 시 full/partial 구분
    source: SourceInfo | None = None
    confidence: float | None = None
    alternatives: list[Hit] = Field(default_factory=list)
    department: DepartmentInfo | None = None
    message: str | None = None
