from typing import Any

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=100_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentBatch(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1, max_length=100)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


class DocumentListItem(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
