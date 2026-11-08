from pathlib import Path
from typing import Protocol

import numpy as np
from fastapi import FastAPI, HTTPException, Query

from app.config import settings
from app.embeddings import SentenceTransformerEncoder
from app.schemas import DocumentBatch, DocumentListItem, SearchRequest, SearchResponse
from app.store import SQLiteDocumentStore


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def create_app(database_path: str | Path | None = None, encoder: Encoder | None = None) -> FastAPI:
    store = SQLiteDocumentStore(str(database_path or settings.database_path))
    embedding_encoder = encoder or SentenceTransformerEncoder(settings.embedding_model)
    application = FastAPI(
        title="Python Data Search API",
        description="Semantic document search using sentence embeddings and SQLite persistence.",
        version="1.0.0",
    )

    @application.get("/")
    def root() -> dict:
        return {
            "name": "Python Data Search API",
            "version": application.version,
            "description": "Semantic retrieval API; it returns matching source documents and does not generate LLM answers.",
            "docs": "/docs",
            "endpoints": ["/health", "/documents", "/search"],
        }

    @application.get("/health")
    def health() -> dict:
        return {"status": "ok", "storage": "sqlite", "documents": store.count()}

    @application.post("/documents", status_code=201)
    def index_documents(batch: DocumentBatch) -> dict:
        documents = [document.model_dump() for document in batch.documents]
        try:
            vectors = np.asarray(
                embedding_encoder.encode([document["text"] for document in documents]),
                dtype=np.float32,
            )
            if vectors.ndim != 2 or vectors.shape[0] != len(documents):
                raise ValueError("Encoder returned an unexpected embedding shape.")
            indexed = store.upsert_many(documents, vectors)
            return {"status": "indexed", "count": indexed}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Document embedding or storage is temporarily unavailable.",
            ) from exc

    @application.get("/documents", response_model=list[DocumentListItem])
    def list_documents(limit: int = Query(default=100, ge=1, le=500)):
        return store.list_documents(limit=limit)

    @application.delete("/documents/{document_id}")
    def delete_document(document_id: str) -> dict:
        deleted = store.delete(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"status": "deleted", "id": document_id}

    @application.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest):
        try:
            vectors = np.asarray(embedding_encoder.encode([request.query]), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != 1:
                raise ValueError("Encoder returned an unexpected query embedding shape.")
            results = store.search(vectors[0], request.top_k)
            return {"query": request.query, "results": results}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Query embedding or search is temporarily unavailable.",
            ) from exc

    return application


app = create_app()
