import json
import os
from pathlib import Path

import httpx


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    root = Path(__file__).resolve().parent
    load_env(root / ".env")

    file_url = os.getenv("TEST_FILE_URL")
    if not file_url:
        raise SystemExit("TEST_FILE_URL is not set. Provide a public https URL to a PDF.")

    payload = {
        "file_url": file_url,
        "prompt": "Extract structured data from the document and return JSON",
    }

    resp = httpx.post(
        "http://localhost:8005/extract",
        json=payload,
        timeout=300,
        trust_env=False,
    )
    if resp.status_code >= 400:
        print(f"Request failed: {resp.status_code}")
        print(f"Response headers: {dict(resp.headers)}")
        print(resp.text)
        resp.raise_for_status()
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
