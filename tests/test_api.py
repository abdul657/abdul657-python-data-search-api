import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app


class FakeEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "python" in lowered or "fastapi" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "react" in lowered or "typescript" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def make_client(tmp_path):
    app = create_app(tmp_path / "test.db", encoder=FakeEncoder())
    return TestClient(app)


def test_health_endpoint(tmp_path):
    response = make_client(tmp_path).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "storage": "sqlite", "documents": 0}


def test_index_and_semantic_search(tmp_path):
    client = make_client(tmp_path)
    indexed = client.post("/documents", json={"documents": [
        {"id": "python", "text": "FastAPI is a Python API framework.", "metadata": {"tag": "backend"}},
        {"id": "react", "text": "React and TypeScript build web interfaces.", "metadata": {"tag": "frontend"}},
    ]})
    assert indexed.status_code == 201
    assert indexed.json()["count"] == 2

    result = client.post("/search", json={"query": "Python framework", "top_k": 2})
    assert result.status_code == 200
    assert result.json()["results"][0]["id"] == "python"
    assert result.json()["results"][0]["score"] > result.json()["results"][1]["score"]


def test_upsert_replaces_existing_document(tmp_path):
    client = make_client(tmp_path)
    client.post("/documents", json={"documents": [{"id": "one", "text": "Python", "metadata": {}}]})
    response = client.post("/documents", json={"documents": [{"id": "one", "text": "React", "metadata": {"v": 2}}]})
    assert response.status_code == 201
    assert response.json()["count"] == 1
    listed = client.get("/documents").json()
    assert len(listed) == 1
    assert listed[0]["text"] == "React"


def test_delete_document(tmp_path):
    client = make_client(tmp_path)
    client.post("/documents", json={"documents": [{"id": "one", "text": "Python", "metadata": {}}]})
    assert client.delete("/documents/one").status_code == 200
    assert client.delete("/documents/one").status_code == 404


def test_search_validates_top_k(tmp_path):
    response = make_client(tmp_path).post("/search", json={"query": "Python", "top_k": 0})
    assert response.status_code == 422
