"""Client for the MinerU cloud document-parsing API (mineru.net, v4).

MinerU parses PDFs into Markdown that preserves table structure as HTML
(``<table>`` with ``rowspan``/``colspan``), which is far more faithful than
PyMuPDF's ``find_tables()`` for multi-level headers. This module wraps the
async upload → poll → download flow and returns the parsed Markdown plus the
structured ``content_list`` JSON.

IMPORTANT (privacy): the source document is uploaded to MinerU's object storage
(``oss-cn-shanghai.aliyuncs.com``), i.e. a third-party server in China. Callers
must only enable this backend for documents cleared for external processing.

The flow (verified against the live API):
  1. POST {base}/file-urls/batch  -> {batch_id, file_urls:[upload_url]}
  2. PUT upload_url (raw bytes, NO Authorization header) -> stores the file
  3. GET {base}/extract-results/batch/{batch_id} until state == "done"
  4. download full_zip_url, unzip in memory -> full.md + *_content_list.json
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import zipfile
from dataclasses import dataclass, field

from anyio import to_thread
from typing import Any, Optional

import httpx

logger = logging.getLogger("extraction_service")


class MineruError(Exception):
    """Raised when the MinerU cloud pipeline fails or times out."""


@dataclass
class MineruResult:
    markdown: str
    content_list: Optional[Any] = None
    # layout.json: {"pdf_info": [{"page_size":[w,h], "para_blocks":[{type,bbox,blocks...}]}]}
    # bbox блоков — в координатах page_size; используется для точной привязки координат.
    layout: Optional[Any] = None
    batch_id: Optional[str] = None
    page_count: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


async def extract_tables_mineru(
    pdf_bytes: bytes,
    filename: str,
    *,
    token: str,
    api_base: str = "https://mineru.net/api/v4",
    language: str = "cyrillic",
    model_version: str = "vlm",
    page_ranges: Optional[str] = None,
    enable_formula: bool = False,
    enable_table: bool = True,
    timeout: int = 300,
    interval: int = 5,
    data_id: str = "ivolga-extraction",
) -> MineruResult:
    """Parse ``pdf_bytes`` via the MinerU cloud API and return Markdown + JSON.

    Raises :class:`MineruError` on any failure (bad response code, upload
    failure, remote task failure, or timeout while polling). The auth token is
    never logged.
    """

    base = api_base.rstrip("/")
    auth_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    started = time.monotonic()

    # Общий timeout щедрый: загрузка большого файла на OSS (Shanghai) и скачивание
    # zip идут через китайский CDN и могут быть медленными. Отдельные операции
    # переопределяют timeout ниже. connect держим коротким, read — длинным.
    client_timeout = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=300.0)
    async with httpx.AsyncClient(timeout=client_timeout, trust_env=False) as client:
        batch_id, upload_url = await _request_upload_slot(
            client,
            base=base,
            headers=auth_headers,
            filename=filename,
            data_id=data_id,
            language=language,
            model_version=model_version,
            page_ranges=page_ranges,
            enable_formula=enable_formula,
            enable_table=enable_table,
        )
        logger.info("MinerU batch created: %s", batch_id)

        await _upload_file(client, upload_url, pdf_bytes)
        logger.info("MinerU upload done (%d kB)", len(pdf_bytes) // 1024)

        zip_url, page_count = await _poll_result(
            client,
            base=base,
            headers=auth_headers,
            batch_id=batch_id,
            timeout=timeout,
            interval=interval,
        )
        logger.info(
            "MinerU parsing finished in %.1fs (pages=%s)",
            time.monotonic() - started,
            page_count,
        )

        markdown, content_list, layout = await _download_and_unzip(client, zip_url)

    if not markdown.strip():
        raise MineruError("MinerU returned empty markdown")

    return MineruResult(
        markdown=markdown,
        content_list=content_list,
        layout=layout,
        batch_id=batch_id,
        page_count=page_count,
        metadata={
            "language": language,
            "model_version": model_version,
            "zip_url": zip_url,
        },
    )


async def _request_upload_slot(
    client: httpx.AsyncClient,
    *,
    base: str,
    headers: dict[str, str],
    filename: str,
    data_id: str,
    language: str,
    model_version: str,
    page_ranges: Optional[str],
    enable_formula: bool,
    enable_table: bool,
) -> tuple[str, str]:
    body: dict[str, Any] = {
        "files": [{"name": filename or "document.pdf", "data_id": data_id}],
        "model_version": model_version,
        "enable_formula": enable_formula,
        "enable_table": enable_table,
        "language": language,
    }
    # page_ranges может игнорироваться сервером (наблюдалось на 68-стр. файле),
    # но передаём — на случай, если для конкретного тарифа он работает.
    if page_ranges:
        body["page_ranges"] = page_ranges

    resp = await client.post(f"{base}/file-urls/batch", headers=headers, json=body)
    if resp.status_code != 200:
        raise MineruError(f"batch request HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    if data.get("code") != 0:
        raise MineruError(f"batch request error: {data.get('msg')} ({data.get('code')})")
    payload = data.get("data") or {}
    batch_id = payload.get("batch_id")
    file_urls = payload.get("file_urls") or []
    if not batch_id or not file_urls:
        raise MineruError(f"batch response missing batch_id/file_urls: {data}")
    return batch_id, file_urls[0]


def _sync_put(upload_url: str, pdf_bytes: bytes) -> tuple[int, str]:
    # Загрузка идёт на presigned OSS-URL (Alibaba, Shanghai) — БЕЗ Authorization.
    # Используем СИНХРОННЫЙ httpx: async-транспорт httpx/httpcore на некоторых
    # окружениях (anyio backend) даёт ложный ReadTimeout при PUT большого тела на
    # медленный CDN, тогда как sync-путь стабилен. Вызываем в отдельном потоке.
    resp = httpx.put(upload_url, content=pdf_bytes, timeout=httpx.Timeout(300.0))
    return resp.status_code, resp.text[:300]


async def _upload_file(client: httpx.AsyncClient, upload_url: str, pdf_bytes: bytes) -> None:
    status, text = await to_thread.run_sync(_sync_put, upload_url, pdf_bytes)
    if status not in (200, 201, 204):
        raise MineruError(f"file upload HTTP {status}: {text}")


async def _poll_result(
    client: httpx.AsyncClient,
    *,
    base: str,
    headers: dict[str, str],
    batch_id: str,
    timeout: int,
    interval: int,
) -> tuple[str, Optional[int]]:
    result_url = f"{base}/extract-results/batch/{batch_id}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(interval)
        resp = await client.get(result_url, headers=headers)
        if resp.status_code != 200:
            logger.warning("MinerU poll HTTP %s: %s", resp.status_code, resp.text[:200])
            continue
        data = resp.json()
        results = (data.get("data") or {}).get("extract_result") or []
        if not results:
            continue
        item = results[0]
        state = item.get("state")
        progress = item.get("extract_progress") or {}
        page_count = progress.get("total_pages")
        if state == "done":
            zip_url = item.get("full_zip_url")
            if not zip_url:
                raise MineruError("MinerU state=done but no full_zip_url")
            return zip_url, page_count
        if state in ("failed", "error"):
            raise MineruError(f"MinerU task failed: {item}")
        logger.info(
            "MinerU state=%s progress=%s/%s",
            state,
            progress.get("extracted_pages"),
            page_count,
        )
    raise MineruError(f"MinerU polling timed out after {timeout}s (batch {batch_id})")


def _sync_get_bytes(url: str) -> tuple[int, bytes]:
    resp = httpx.get(url, timeout=httpx.Timeout(300.0))
    return resp.status_code, resp.content


async def _download_and_unzip(
    client: httpx.AsyncClient, zip_url: str
) -> tuple[str, Optional[Any], Optional[Any]]:
    # Скачивание zip с CDN — синхронно в потоке (та же причина, что для upload).
    status, content = await to_thread.run_sync(_sync_get_bytes, zip_url)
    if status != 200:
        raise MineruError(f"zip download HTTP {status}")
    zf = zipfile.ZipFile(io.BytesIO(content))
    markdown = ""
    content_list: Optional[Any] = None
    layout: Optional[Any] = None
    import json as _json

    for name in zf.namelist():
        if name.endswith("full.md") or (name.endswith(".md") and not markdown):
            markdown = zf.read(name).decode("utf-8", "replace")
        if name.endswith("_content_list.json") and content_list is None:
            try:
                content_list = _json.loads(zf.read(name).decode("utf-8", "replace"))
            except Exception:  # noqa: BLE001 — content_list опционален
                content_list = None
        if name.endswith("layout.json") and layout is None:
            try:
                layout = _json.loads(zf.read(name).decode("utf-8", "replace"))
            except Exception:  # noqa: BLE001 — layout опционален
                layout = None
    return markdown, content_list, layout
