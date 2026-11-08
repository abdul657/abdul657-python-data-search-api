import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "sample_documents.json"
API_URL = "http://127.0.0.1:8000/documents"


def main() -> None:
    documents = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    payload = json.dumps({"documents": documents}).encode("utf-8")
    request = Request(API_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=120) as response:
        print(response.status)
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
