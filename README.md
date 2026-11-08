# Python Data Search API

A locally runnable semantic-search API that indexes text documents, generates sentence embeddings, stores them in SQLite, and ranks results by cosine similarity.

## Technology

- Python 3.11+
- FastAPI and Pydantic for the HTTP API and validation
- Sentence Transformers using all-MiniLM-L6-v2 for embeddings
- NumPy for vector normalization and cosine similarity
- SQLite for persistent document and embedding storage
- Pytest and HTTPX for automated API tests
- GitHub Actions for CI

No Pinecone account, API token, paid service, or generative-LLM key is required. The embedding model is downloaded by Sentence Transformers on first use.

## Architecture

~~~text
Client
  |
  | POST /documents             POST /search
  v                                  v
FastAPI --------------------> SearchService
                                   |
                            EmbeddingService
                         Sentence Transformers
                                   |
                              NumPy vectors
                                   |
                                SQLite
                          text + metadata + vector
~~~

This project implements semantic retrieval, not LLM answer generation. Search responses return source documents and similarity scores so a downstream RAG application can use the retrieved context.

## Quick start

### Requirements
- Python 3.11 or later
- Internet access for the first Sentence Transformers model download

### Setup

~~~bash
git clone https://github.com/abdul657/abdul657-python-data-search-api.git
cd abdul657-python-data-search-api
python -m venv .venv
~~~

Activate the virtual environment:

~~~bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
~~~

Install dependencies and create the local environment file:

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Copy .env.example to .env if you want to customize the model or database path.

### Run the API

~~~bash
uvicorn app.main:app --reload
~~~

Open the interactive API documentation at http://127.0.0.1:8000/docs.

### Load example documents

~~~bash
python scripts/load_samples.py
~~~

### Search

~~~bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Python API framework","top_k":3}'
~~~

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Liveness and storage status |
| POST | /documents | Embed and upsert one or more documents |
| GET | /documents | List stored document metadata |
| DELETE | /documents/{document_id} | Delete a document |
| POST | /search | Return semantically similar documents |

### Index documents

~~~json
{
  "documents": [
    {
      "id": "doc-1",
      "text": "FastAPI is a Python framework for building APIs.",
      "metadata": {"topic": "backend"}
    }
  ]
}
~~~

Document IDs are caller supplied. Re-indexing an existing ID updates that record.

### Search documents

~~~json
{"query":"web framework","top_k":5}
~~~

The API returns each matching document's ID, text, metadata, and cosine similarity score.

## Configuration

Environment variables (all optional):

| Variable | Default | Description |
|---|---|---|
| EMBEDDING_MODEL | sentence-transformers/all-MiniLM-L6-v2 | Hugging Face sentence embedding model |
| DATABASE_PATH | data/search.db | SQLite database file |
| CORS_ORIGINS | http://localhost:5173 | Comma-separated allowed browser origins |

The selected model's embedding output is stored as float32 vectors. SQLite stores the text, metadata JSON and embedding bytes together, keeping the demo self-contained.

## Tests

~~~bash
pytest -q
~~~

Tests use a deterministic fake embedding encoder, so they do not download the transformer model or require network access.

## Project layout

~~~text
app/
  config.py
  embeddings.py
  main.py
  schemas.py
  service.py
  store.py
data/
  .gitkeep
scripts/
  load_samples.py
tests/
  test_api.py
.env.example
.gitignore
requirements.txt
~~~

## Design boundaries

- This is a compact portfolio/demo service, not a distributed vector database.
- SQLite is appropriate for local development and small datasets; larger multi-user workloads should use a dedicated vector store.
- Semantic retrieval does not generate final natural-language answers.
- Add authentication, per-user data isolation, request quotas, observability, and deployment-specific secrets management before exposing a production service.

## License

No license is currently declared. Add one before accepting external contributions or permitting reuse.
