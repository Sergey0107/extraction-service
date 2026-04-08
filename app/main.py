import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import unquote, urlparse

import httpx
from anyio import to_thread
from docx import Document as WordDocument
from docx.table import Table as WordTable
from docx.text.paragraph import Paragraph as WordParagraph
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field

try:
    import fitz

    PYMUPDF_INSTALLED = True
except ImportError:  # pragma: no cover - dependency may be omitted in some builds
    fitz = None
    PYMUPDF_INSTALLED = False

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
    from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.document_extractor import DocumentExtractor
    from docling.pipeline.vlm_pipeline import VlmPipeline

    try:
        from docling.document_converter import ImageFormatOption
    except ImportError:  # pragma: no cover - depends on docling version
        ImageFormatOption = None
    DOCLING_INSTALLED = True
except ImportError:  # pragma: no cover - docling is optional
    InputFormat = None
    VlmConvertOptions = Any
    VlmPipelineOptions = Any
    ApiVlmEngineOptions = Any
    VlmEngineType = Any
    DocumentExtractor = Any
    DocumentConverter = Any
    PdfFormatOption = Any
    VlmPipeline = Any
    ImageFormatOption = None
    DOCLING_INSTALLED = False

logger = logging.getLogger("extraction_service")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _get_int_env(name: str, default: int) -> int:
    value = _get_env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r, using %s", name, value, default)
        return default


def _get_list_env(name: str) -> list[str]:
    value = _get_env(name)
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


DOC_PARSE_MODEL = _get_env("DOCLING_PARSE_MODEL", "docling")
DOC_EXTRACT_MODEL = _get_env("DOCLING_EXTRACT_MODEL", "docling")
EXTRACTION_BACKEND = (_get_env("EXTRACTION_BACKEND", "openrouter") or "openrouter").strip().lower()
DEBUG_ERRORS = _get_env("DEBUG_ERRORS", "0") == "1"
FILE_DOWNLOAD_TIMEOUT_SECONDS = _get_int_env("FILE_DOWNLOAD_TIMEOUT_SECONDS", 300)
REMOTE_API_TIMEOUT_SECONDS = _get_int_env("REMOTE_API_TIMEOUT_SECONDS", 600)

DOCLING_REMOTE_API_URL = _get_env("DOCLING_REMOTE_API_URL")
DOCLING_REMOTE_VLM_URL = _get_env("DOCLING_REMOTE_VLM_URL", DOCLING_REMOTE_API_URL)
DOCLING_REMOTE_LLM_URL = _get_env("DOCLING_REMOTE_LLM_URL", DOCLING_REMOTE_API_URL)
DOCLING_REMOTE_API_KEY = _get_env("DOCLING_REMOTE_API_KEY")
DOCLING_REMOTE_MODEL = _get_env("DOCLING_REMOTE_MODEL", "ibm-granite/granite-docling-258M")
DOCLING_REMOTE_EXTRACTION_MODEL = (
    _get_env("DOCLING_REMOTE_EXTRACTION_MODEL", DOCLING_REMOTE_MODEL)
    or DOCLING_REMOTE_MODEL
)
DOCLING_REMOTE_PRESET = _get_env("DOCLING_REMOTE_PRESET", "granite_docling")
DOCLING_REMOTE_MAX_TOKENS = _get_int_env("DOCLING_REMOTE_MAX_TOKENS", 4000)

OPENROUTER_API_KEY = _get_env("OPENROUTER_API_KEY")
OPENROUTER_MODEL = _get_env("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = _get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_PDF_ENGINE = _get_env("OPENROUTER_PDF_ENGINE", "mistral-ocr")
OPENROUTER_MAX_TOKENS = _get_int_env("OPENROUTER_MAX_TOKENS", 4000)
OPENROUTER_APP_NAME = _get_env("OPENROUTER_APP_NAME", "extraction-service")
OPENROUTER_SITE_URL = _get_env("OPENROUTER_SITE_URL", "http://localhost:8005")
OPENROUTER_PROVIDER_ORDER = _get_list_env("OPENROUTER_PROVIDER_ORDER")
OPENROUTER_PROVIDER_IGNORE = _get_list_env("OPENROUTER_PROVIDER_IGNORE")
GEOMETRY_ENRICHMENT_ENABLED = _get_env("GEOMETRY_ENRICHMENT_ENABLED", "1") != "0"

LLAMAPARSE_API_KEY = _get_env("LLAMAPARSE_API_KEY")
LLAMAPARSE_BASE_URL = _get_env("LLAMAPARSE_BASE_URL", "https://api.cloud.llamaindex.ai")
LLAMAPARSE_RESULT_TYPE = _get_env("LLAMAPARSE_RESULT_TYPE", "markdown")
LLAMAPARSE_POLLING_INTERVAL = _get_int_env("LLAMAPARSE_POLLING_INTERVAL", 3)
LLAMAPARSE_MAX_WAIT_SECONDS = _get_int_env("LLAMAPARSE_MAX_WAIT_SECONDS", 180)
LLAMAPARSE_LANGUAGE = _get_env("LLAMAPARSE_LANGUAGE", "ru")
LLAMAPARSE_REQUEST_TIMEOUT_SECONDS = _get_int_env("LLAMAPARSE_REQUEST_TIMEOUT_SECONDS", 120)
LLAMAPARSE_MAX_RETRIES = _get_int_env("LLAMAPARSE_MAX_RETRIES", 2)

SUPPORTED_BACKENDS = {"docling_local", "docling_remote", "openrouter", "llamaparse"}
ACTIVE_BACKENDS = {"openrouter", "llamaparse"}
STUBBED_BACKENDS = SUPPORTED_BACKENDS - ACTIVE_BACKENDS
if EXTRACTION_BACKEND not in ACTIVE_BACKENDS:
    logger.warning(
        "Backend %s is disabled in the current build; falling back to openrouter",
        EXTRACTION_BACKEND,
    )
    EXTRACTION_BACKEND = "openrouter"
ALLOWED_FORMATS = (InputFormat.IMAGE, InputFormat.PDF) if DOCLING_INSTALLED else tuple()
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
GENERIC_MIME_TYPES = {"application/octet-stream", "binary/octet-stream"}


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_url: str = Field(..., description="Public or service-reachable file URL")
    prompt: Optional[str] = None
    schema_payload: Optional[Union[dict, str]] = Field(default=None, alias="schema")
    analysis_id: Optional[str] = None
    file_id: Optional[str] = None
    file_type: Optional[str] = None
    backend: Optional[str] = Field(
        default=None,
        description="docling_local | docling_remote | openrouter | llamaparse",
    )


@dataclass
class DownloadedFile:
    filename: str
    content_type: Optional[str]
    local_path: str
    file_bytes: Optional[bytes] = None


@dataclass
class PdfWord:
    text: str
    normalized: str
    rect: Any


@dataclass
class PdfPageIndex:
    page_number: int
    page: Any
    words: list[PdfWord]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.docling_extractor = None
    logger.info("Extraction service started; default backend=%s", EXTRACTION_BACKEND)
    try:
        yield
    finally:
        extractor = getattr(app.state, "docling_extractor", None)
        close = getattr(extractor, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close Docling extractor")


app = FastAPI(lifespan=lifespan, title="extraction-service", version="0.3.0")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "default_backend": EXTRACTION_BACKEND,
        "active_backends": sorted(ACTIVE_BACKENDS),
        "stubbed_backends": sorted(STUBBED_BACKENDS),
        "docling_installed": DOCLING_INSTALLED,
        "pymupdf_installed": PYMUPDF_INSTALLED,
        "geometry_enrichment_enabled": GEOMETRY_ENRICHMENT_ENABLED,
        "llamaparse_configured": bool(LLAMAPARSE_API_KEY),
    }


def _get_docling_version() -> Optional[str]:
    if not DOCLING_INSTALLED:
        return None
    try:
        return version("docling")
    except PackageNotFoundError:
        return None


def _get_local_extractor() -> DocumentExtractor:
    if not DOCLING_INSTALLED:
        raise RuntimeError("Docling is not installed in this build")
    extractor = getattr(app.state, "docling_extractor", None)
    if extractor is None:
        extractor = DocumentExtractor(allowed_formats=list(ALLOWED_FORMATS))
        app.state.docling_extractor = extractor
        logger.info(
            "Local Docling extractor initialized with formats: %s",
            [fmt.name for fmt in ALLOWED_FORMATS],
        )
    return extractor


def _select_backend(requested_backend: Optional[str]) -> str:
    backend = (requested_backend or EXTRACTION_BACKEND).strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported backend: {requested_backend or backend}",
        )
    return backend


def _raise_docling_backend_unavailable(backend: str) -> None:
    raise HTTPException(
        status_code=501,
        detail=(
            f"Backend '{backend}' is temporarily disabled in this build. "
            "Use 'openrouter'. Docling integration is kept as a stub for future re-enable."
        ),
    )


def _build_default_template(prompt: Optional[str]) -> dict:
    return {"result": "string"}


def _build_default_json_schema(prompt: Optional[str]) -> dict[str, Any]:
    description = prompt or "Structured extraction result"
    return {
        "type": "object",
        "required": ["result"],
        "additionalProperties": False,
        "properties": {
            "result": {
                "type": "string",
                "description": description,
            }
        },
    }


def _is_json_schema(schema: dict) -> bool:
    return schema.get("type") == "object" and isinstance(schema.get("properties"), dict)


def _json_schema_to_template(schema: dict) -> dict:
    properties = schema.get("properties", {})
    template: dict[str, Any] = {}
    for key, prop_schema in properties.items():
        template[key] = _json_schema_value_to_template(prop_schema)
    return template


def _json_schema_value_to_template(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return "string"

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next(
            (value for value in schema_type if value != "null"),
            schema_type[0] if schema_type else None,
        )

    if schema_type == "object":
        return _json_schema_to_template(schema)
    if schema_type == "array":
        items = schema.get("items", {})
        return [_json_schema_value_to_template(items)]
    if schema_type == "number":
        return "float"
    if schema_type == "integer":
        return "integer"
    if schema_type == "boolean":
        return "boolean"
    if schema_type == "string":
        return "string"
    if "enum" in schema:
        return "string"
    return "string"


def _normalize_template(
    schema: Optional[Union[dict, str]],
    prompt: Optional[str],
) -> Union[dict, str]:
    if isinstance(schema, dict):
        return _json_schema_to_template(schema) if _is_json_schema(schema) else schema
    if isinstance(schema, str):
        try:
            parsed = json.loads(schema)
        except json.JSONDecodeError:
            return schema
        if isinstance(parsed, dict):
            return _json_schema_to_template(parsed) if _is_json_schema(parsed) else parsed
        return schema
    return _build_default_template(prompt)


def _normalize_json_schema(
    schema: Optional[Union[dict, str]],
    prompt: Optional[str],
) -> dict[str, Any]:
    if isinstance(schema, dict) and _is_json_schema(schema):
        return schema
    if isinstance(schema, str):
        try:
            parsed = json.loads(schema)
        except json.JSONDecodeError:
            return _build_default_json_schema(prompt)
        if isinstance(parsed, dict) and _is_json_schema(parsed):
            return parsed
    return _build_default_json_schema(prompt)


def _looks_like_image(filename: str, content_type: str | None) -> bool:
    if content_type and content_type.startswith("image/"):
        return True
    return Path(filename).suffix.lower() in IMAGE_SUFFIXES


def _looks_like_docx(filename: str, content_type: str | None) -> bool:
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    suffix = Path(filename).suffix.lower()
    return (
        normalized
        in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        }
        or suffix in {".docx", ".doc"}
    )


def _guess_filename_from_url(file_url: str, content_type: Optional[str]) -> str:
    parsed_url = urlparse(file_url)
    raw_name = unquote(Path(parsed_url.path).name)
    if raw_name:
        return raw_name

    extension = mimetypes.guess_extension((content_type or "").split(";", 1)[0].strip())
    extension = extension or ".pdf"
    return f"document{extension}"


def _normalized_content_type(filename: str, content_type: Optional[str]) -> Optional[str]:
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized and normalized not in GENERIC_MIME_TYPES:
        return normalized
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or normalized or None


def _message_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(part for part in text_parts if part)
    return ""


def _strip_json_fences(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    return candidate


def _extract_message_json_text(data: dict[str, Any], *, provider_name: str) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{provider_name} response has no choices")

    message = choices[0].get("message", {})
    content = _message_text_from_content(message.get("content"))
    content = _strip_json_fences(content)
    if not content:
        raise RuntimeError(f"{provider_name} returned empty content")
    return content


def _parse_json_from_chat_response(data: dict[str, Any], *, provider_name: str) -> dict[str, Any]:
    content = _extract_message_json_text(data, provider_name=provider_name)
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{provider_name} returned non-object JSON")
    return parsed


def _extract_error_text_from_response(response: httpx.Response, provider_name: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return text[:800] if text else f"{provider_name} returned HTTP {response.status_code}"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("metadata")
            if isinstance(message, str) and message.strip():
                return message.strip()[:800]
            if message is not None:
                return str(message)[:800]
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()[:800]
        if detail is not None:
            return str(detail)[:800]

    text = json.dumps(payload, ensure_ascii=False)[:800]
    return text or f"{provider_name} returned HTTP {response.status_code}"


def _raise_provider_http_error(response: httpx.Response, provider_name: str) -> None:
    detail = _extract_error_text_from_response(response, provider_name)
    raise RuntimeError(
        f"{provider_name} HTTP {response.status_code}: {detail}"
    )


def _build_openrouter_provider_preferences(*, require_parameters: bool) -> dict[str, Any]:
    provider: dict[str, Any] = {}
    if OPENROUTER_PROVIDER_ORDER:
        provider["order"] = OPENROUTER_PROVIDER_ORDER
    if OPENROUTER_PROVIDER_IGNORE:
        provider["ignore"] = OPENROUTER_PROVIDER_IGNORE
    if require_parameters:
        provider["require_parameters"] = True
    return provider


async def _download_file(file_url: str) -> DownloadedFile:
    temp_path = ""
    try:
        async with httpx.AsyncClient(
            timeout=FILE_DOWNLOAD_TIMEOUT_SECONDS,
            follow_redirects=True,
            trust_env=False,
        ) as client:
            async with client.stream("GET", file_url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type")
                filename = _guess_filename_from_url(file_url, content_type)
                suffix = Path(filename).suffix or (
                    mimetypes.guess_extension((content_type or "").split(";", 1)[0].strip())
                    or ".pdf"
                )
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    temp_path = handle.name
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Failed to download file: {exc}") from exc
    except Exception:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Failed to remove temp file %s", temp_path)
        raise

    return DownloadedFile(
        filename=filename,
        content_type=_normalized_content_type(filename, content_type),
        local_path=temp_path,
    )


def _normalize_match_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s./%-]+", " ", value.lower().replace("ё", "е"))).strip()


def _tokenize_match_text(value: str | None) -> list[str]:
    normalized = _normalize_match_text(value)
    if not normalized:
        return []
    return [token for token in normalized.split(" ") if len(token) >= 2]


def _safe_page_number(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _reference_iter(node: Any):
    if isinstance(node, dict):
        references = node.get("references")
        if isinstance(references, list):
            yield node, references
        for key, value in node.items():
            if key == "references":
                continue
            yield from _reference_iter(value)
    elif isinstance(node, list):
        for value in node:
            yield from _reference_iter(value)


def _inject_synthetic_references(node: Any) -> int:
    injected = 0
    if isinstance(node, dict):
        label_context = _collect_label_context(node)
        value_context = _collect_value_context(node)
        has_reference_key = "references" in node
        references = node.get("references")
        if (
            has_reference_key
            and isinstance(references, list)
            and not references
            and (label_context or value_context)
        ):
            synthetic_text = next((value for value in value_context if value), "") or " ".join(label_context[:2]).strip()
            if synthetic_text:
                node["references"] = [
                    {
                        "quote_text": synthetic_text,
                        "anchor_text": synthetic_text,
                        "locator_text": synthetic_text,
                        "synthetic_reference": True,
                    }
                ]
                injected += 1
        elif (
            not has_reference_key
            and label_context
            and value_context
            and any(key in node for key in ("name", "label", "title", "characteristic"))
            and any(key in node for key in ("value", "text", "answer", "content"))
        ):
            synthetic_text = next((value for value in value_context if value), "") or " ".join(label_context[:2]).strip()
            if synthetic_text:
                node["references"] = [
                    {
                        "quote_text": synthetic_text,
                        "anchor_text": synthetic_text,
                        "locator_text": synthetic_text,
                        "synthetic_reference": True,
                    }
                ]
                injected += 1
        for key, value in list(node.items()):
            if key == "references":
                continue
            injected += _inject_synthetic_references(value)
    elif isinstance(node, list):
        for value in node:
            injected += _inject_synthetic_references(value)
    return injected


def _collect_label_context(node: dict[str, Any]) -> list[str]:
    context_values: list[str] = []
    for key in (
        "name",
        "label",
        "title",
        "characteristic",
        "field_name",
        "product_name",
        "section",
        "key",
    ):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            context_values.append(value.strip())
    return context_values


def _collect_value_context(node: dict[str, Any]) -> list[str]:
    context_values: list[str] = []
    for key in ("value", "text", "answer", "content", "quote", "summary"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            context_values.append(value.strip())
    return context_values


def _collect_reference_candidates(
    reference: Any,
    *,
    label_context: list[str],
    value_context: list[str],
) -> list[dict[str, Any]]:
    raw_candidates: list[dict[str, Any]] = []
    page_hint = None
    if isinstance(reference, dict):
        page_hint = _safe_page_number(reference.get("page"))
        if page_hint is None:
            page_hint = _safe_page_number(reference.get("page_number"))
        for key in ("quote_text", "anchor_text", "locator_text", "text"):
            value = reference.get(key)
            if isinstance(value, str) and value.strip():
                raw_candidates.append(
                    {
                        "text": value.strip(),
                        "kind": key,
                        "weight": {
                            "quote_text": 1.0,
                            "anchor_text": 0.88,
                            "locator_text": 0.83,
                            "text": 0.8,
                        }.get(key, 0.75),
                        "page_hint": page_hint,
                    }
                )
    elif isinstance(reference, str) and reference.strip():
        raw_candidates.append(
            {
                "text": reference.strip(),
                "kind": "raw_reference",
                "weight": 0.92,
                "page_hint": page_hint,
            }
        )

    primary_label = " ".join(label_context[:2]).strip()
    primary_value = next((value for value in value_context if value), "")
    if primary_label and primary_value:
        raw_candidates.append(
            {
                "text": f"{primary_label} {primary_value}",
                "kind": "label_plus_value",
                "weight": 0.72,
                "page_hint": page_hint,
            }
        )
    if primary_value:
        raw_candidates.append(
            {
                "text": primary_value,
                "kind": "value",
                "weight": 0.8,
                "page_hint": page_hint,
            }
        )
    if primary_label:
        raw_candidates.append(
            {
                "text": primary_label,
                "kind": "label",
                "weight": 0.45,
                "page_hint": page_hint,
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        normalized = _normalize_match_text(candidate["text"])
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append({**candidate, "normalized": normalized})
    return deduped[:8]


def _rect_to_bbox(rect: Any) -> dict[str, float]:
    return {
        "x": round(float(rect.x0), 3),
        "y": round(float(rect.y0), 3),
        "width": round(float(rect.x1 - rect.x0), 3),
        "height": round(float(rect.y1 - rect.y0), 3),
        "x0": round(float(rect.x0), 3),
        "y0": round(float(rect.y0), 3),
        "x1": round(float(rect.x1), 3),
        "y1": round(float(rect.y1), 3),
        "left": round(float(rect.x0), 3),
        "top": round(float(rect.y0), 3),
        "right": round(float(rect.x1), 3),
        "bottom": round(float(rect.y1), 3),
    }


def _union_rects(rects: list[Any]) -> Any | None:
    if not rects:
        return None
    current = fitz.Rect(rects[0])
    for rect in rects[1:]:
        current.include_rect(rect)
    return current


def _build_pdf_page_index(local_path: str) -> tuple[Any, list[PdfPageIndex]]:
    document = fitz.open(local_path)
    pages: list[PdfPageIndex] = []
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        raw_words = page.get_text("words", sort=True)
        words: list[PdfWord] = []
        for raw_word in raw_words:
            text = str(raw_word[4]).strip()
            normalized = _normalize_match_text(text)
            if not normalized:
                continue
            words.append(
                PdfWord(
                    text=text,
                    normalized=normalized,
                    rect=fitz.Rect(raw_word[0], raw_word[1], raw_word[2], raw_word[3]),
                )
            )
        pages.append(PdfPageIndex(page_number=page_index + 1, page=page, words=words))
    return document, pages


def _search_exact_candidate(page: Any, candidate: str) -> Any | None:
    snippet = candidate.strip()
    if len(snippet) < 3:
        return None
    if len(snippet) > 220:
        snippet = snippet[:220].rsplit(" ", 1)[0].strip() or snippet[:220]
    try:
        rects = page.search_for(snippet)
    except Exception:
        return None
    return _union_rects(rects) if rects else None


def _search_token_candidate(page_index: PdfPageIndex, candidate: str) -> tuple[Any | None, float]:
    tokens = _tokenize_match_text(candidate)
    if len(tokens) < 2 or not page_index.words:
        return None, 0.0

    best_indices: list[int] = []
    best_coverage = 0.0
    max_window = min(len(page_index.words), max(len(tokens) + 12, 18))

    for start in range(len(page_index.words)):
        seen_indices: list[int] = []
        token_cursor = 0
        for index in range(start, min(len(page_index.words), start + max_window)):
            if token_cursor >= len(tokens):
                break
            word = page_index.words[index].normalized
            if word == tokens[token_cursor]:
                seen_indices.append(index)
                token_cursor += 1
                continue
            if token_cursor < len(tokens) and tokens[token_cursor] in word:
                seen_indices.append(index)
                token_cursor += 1
        matched_count = len(seen_indices)
        coverage = matched_count / len(tokens)
        if matched_count > len(best_indices) or (
            matched_count == len(best_indices) and coverage > best_coverage
        ):
            best_indices = seen_indices
            best_coverage = coverage

    matched_count = len(best_indices)
    if matched_count < min(3, len(tokens)) and best_coverage < 0.75:
        return None, 0.0
    if best_coverage < 0.55:
        return None, 0.0

    rect = _union_rects([page_index.words[index].rect for index in best_indices])
    return rect, best_coverage


def _find_reference_location(
    pages: list[PdfPageIndex],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best_match: dict[str, Any] | None = None

    for candidate in candidates:
        candidate_text = candidate["text"]
        candidate_weight = float(candidate.get("weight", 0.5))
        page_hint = _safe_page_number(candidate.get("page_hint"))
        ordered_pages = sorted(
            pages,
            key=lambda page_index: (
                0 if page_hint and page_index.page_number == page_hint else 1,
                abs(page_index.page_number - page_hint) if page_hint else 0,
                page_index.page_number,
            ),
        )
        for page_index in ordered_pages:
            rect = _search_exact_candidate(page_index.page, candidate_text)
            if not rect:
                continue
            page_bonus = 18.0 if page_hint and page_index.page_number == page_hint else 0.0
            score = len(candidate.get("normalized", _normalize_match_text(candidate_text))) * candidate_weight + page_bonus
            if not best_match or score > best_match["score"]:
                best_match = {
                    "page": page_index.page_number,
                    "bbox": _rect_to_bbox(rect),
                    "score": score,
                    "locator_strategy": "pymupdf_exact",
                    "matched_text": candidate_text,
                }

    if best_match:
        return best_match

    for candidate in candidates:
        candidate_text = candidate["text"]
        candidate_weight = float(candidate.get("weight", 0.5))
        page_hint = _safe_page_number(candidate.get("page_hint"))
        ordered_pages = sorted(
            pages,
            key=lambda page_index: (
                0 if page_hint and page_index.page_number == page_hint else 1,
                abs(page_index.page_number - page_hint) if page_hint else 0,
                page_index.page_number,
            ),
        )
        for page_index in ordered_pages:
            rect, coverage = _search_token_candidate(page_index, candidate_text)
            if not rect:
                continue
            page_bonus = 12.0 if page_hint and page_index.page_number == page_hint else 0.0
            score = coverage * 100 * candidate_weight + min(len(candidate_text), 120) / 10 + page_bonus
            if not best_match or score > best_match["score"]:
                best_match = {
                    "page": page_index.page_number,
                    "bbox": _rect_to_bbox(rect),
                    "score": score,
                    "locator_strategy": "pymupdf_tokens",
                    "matched_text": candidate_text,
                }
    return best_match


def _normalize_reference_pages(root: dict[str, Any]) -> bool:
    raw_pages: list[int] = []
    for _, references in _reference_iter(root):
        for reference in references:
            if isinstance(reference, dict):
                raw_page = _safe_page_number(reference.get("page"))
                if raw_page is None:
                    raw_page = _safe_page_number(reference.get("page_number"))
                if raw_page is not None:
                    raw_pages.append(raw_page)
            elif isinstance(reference, str):
                match = re.search(r"(?:стр\.?|page|p\.)\s*(\d+)", reference, flags=re.IGNORECASE)
                if match:
                    raw_pages.append(int(match.group(1)))

    zero_based = any(page == 0 for page in raw_pages)
    if not zero_based:
        return False

    for _, references in _reference_iter(root):
        for reference in references:
            if not isinstance(reference, dict):
                continue
            raw_page = _safe_page_number(reference.get("page"))
            if raw_page is not None:
                reference["page"] = raw_page + 1
            raw_page_number = _safe_page_number(reference.get("page_number"))
            if raw_page_number is not None:
                reference["page_number"] = raw_page_number + 1
    return True


def _enrich_references_with_pdf_geometry(
    extracted_data: dict[str, Any],
    *,
    local_path: str,
    content_type: str | None,
) -> dict[str, Any]:
    metadata = {
        "enabled": GEOMETRY_ENRICHMENT_ENABLED,
        "provider": "pymupdf" if PYMUPDF_INSTALLED else None,
        "applied": False,
        "searchable_pdf": False,
        "page_count": 1,
        "reference_count": 0,
        "matched_reference_count": 0,
        "converted_string_reference_count": 0,
        "synthetic_reference_count": 0,
        "zero_based_normalized": False,
        "errors": [],
    }

    if not GEOMETRY_ENRICHMENT_ENABLED:
        return metadata
    if not PYMUPDF_INSTALLED:
        metadata["errors"].append("PyMuPDF is not installed")
        return metadata
    if (content_type or "").split(";", 1)[0].strip().lower() != "application/pdf":
        metadata["errors"].append("Geometry enrichment currently supports PDF only")
        return metadata

    document = None
    try:
        document, pages = _build_pdf_page_index(local_path)
        metadata["page_count"] = document.page_count
        metadata["searchable_pdf"] = any(page.words for page in pages)
        if not metadata["searchable_pdf"]:
            metadata["errors"].append("PDF has no searchable text layer")
            return metadata

        metadata["zero_based_normalized"] = _normalize_reference_pages(extracted_data)
        metadata["synthetic_reference_count"] = _inject_synthetic_references(extracted_data)

        for holder, references in _reference_iter(extracted_data):
            label_context = _collect_label_context(holder)
            value_context = _collect_value_context(holder)
            for index, reference in enumerate(references):
                metadata["reference_count"] += 1
                original_reference = reference
                if isinstance(reference, str):
                    reference = {
                        "quote_text": reference,
                        "anchor_text": reference,
                        "locator_text": reference,
                    }
                    references[index] = reference
                    metadata["converted_string_reference_count"] += 1
                if not isinstance(reference, dict):
                    continue

                candidates = _collect_reference_candidates(
                    reference,
                    label_context=label_context,
                    value_context=value_context,
                )
                if not candidates:
                    continue

                match = _find_reference_location(pages, candidates)
                if not match:
                    continue

                reference["page"] = match["page"]
                reference["page_number"] = match["page"]
                reference["bbox"] = match["bbox"]
                reference["locator_strategy"] = match["locator_strategy"]
                reference.setdefault("quote_text", candidates[0]["text"])
                reference.setdefault("anchor_text", candidates[0]["text"])
                reference["geometry_source"] = "pymupdf"
                reference["matched_text"] = match.get("matched_text")
                reference["match_score"] = round(float(match.get("score", 0.0)), 3)
                metadata["matched_reference_count"] += 1

                if isinstance(original_reference, dict):
                    original_reference.update(reference)

        metadata["applied"] = metadata["matched_reference_count"] > 0
        return metadata
    except Exception as exc:
        logger.exception("PyMuPDF geometry enrichment failed")
        metadata["errors"].append(str(exc))
        return metadata
    finally:
        if document is not None:
            try:
                document.close()
            except Exception:
                logger.warning("Failed to close PDF document during geometry enrichment")


def _get_downloaded_file_bytes(downloaded_file: DownloadedFile) -> bytes:
    if downloaded_file.file_bytes is None:
        downloaded_file.file_bytes = Path(downloaded_file.local_path).read_bytes()
    return downloaded_file.file_bytes


def _cleanup_downloaded_file(downloaded_file: DownloadedFile) -> None:
    if not downloaded_file.local_path:
        return
    try:
        os.remove(downloaded_file.local_path)
    except OSError:
        logger.warning("Failed to remove temp file %s", downloaded_file.local_path)


def _build_result_page(
    extracted_data: dict[str, Any],
    *,
    raw_text: Optional[str] = None,
    page_no: int = 1,
) -> dict[str, Any]:
    return {
        "page_no": page_no,
        "extracted_data": extracted_data,
        "raw_text": raw_text,
        "errors": None,
    }


def _run_local_docling_extraction(
    extractor: DocumentExtractor,
    file_url: str,
    template: Union[dict, str],
):
    return extractor.extract(source=file_url, template=template)


def _serialize_pages(result: Any) -> list[dict[str, Any]]:
    pages = getattr(result, "pages", None)
    if not pages:
        return []
    serialized = []
    for page in pages:
        serialized.append(
            {
                "page_no": getattr(page, "page_no", None),
                "extracted_data": getattr(page, "extracted_data", None),
                "raw_text": getattr(page, "raw_text", None),
                "errors": getattr(page, "errors", None),
            }
        )
    return serialized


def _build_openai_headers(api_key: Optional[str]) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _chat_completion_json(
    *,
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    provider_name: str,
    repair_schema: Optional[dict[str, Any]] = None,
    repair_model: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    used_fallback = False
    async with httpx.AsyncClient(
        timeout=REMOTE_API_TIMEOUT_SECONDS,
        trust_env=False,
    ) as client:
        response = await client.post(endpoint, json=payload, headers=headers)

        if response.status_code >= 400 and "response_format" in payload:
            fallback_payload = dict(payload)
            fallback_payload.pop("response_format", None)
            if isinstance(fallback_payload.get("provider"), dict):
                provider = dict(fallback_payload["provider"])
                provider.pop("require_parameters", None)
                if provider:
                    fallback_payload["provider"] = provider
                else:
                    fallback_payload.pop("provider", None)
            used_fallback = True
            retry_response = await client.post(endpoint, json=fallback_payload, headers=headers)
            if retry_response.status_code >= 400:
                _raise_provider_http_error(retry_response, provider_name)
            data = retry_response.json()
        else:
            if response.status_code >= 400:
                _raise_provider_http_error(response, provider_name)
            data = response.json()

    if not isinstance(data, dict):
        raise RuntimeError(f"{provider_name} response is not a JSON object")
    raw_content = _extract_message_json_text(data, provider_name=provider_name)
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        if repair_schema and repair_model:
            logger.warning(
                "%s returned invalid JSON, attempting repair: %s",
                provider_name,
                exc,
            )
            parsed = await _repair_raw_text_to_schema(
                endpoint=endpoint,
                headers=headers,
                model=repair_model,
                schema=repair_schema,
                raw_text=raw_content,
                provider_name=f"{provider_name} raw JSON repair",
            )
        else:
            raise RuntimeError(f"{provider_name} returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{provider_name} returned non-object JSON")
    return parsed, data, used_fallback


def _schema_required_keys(schema: dict[str, Any]) -> set[str]:
    required = schema.get("required")
    if not isinstance(required, list):
        return set()
    return {item for item in required if isinstance(item, str)}


def _needs_schema_repair(candidate: dict[str, Any], schema: dict[str, Any]) -> bool:
    required = _schema_required_keys(schema)
    if required and not required.issubset(candidate.keys()):
        return True
    return False


async def _repair_json_to_schema(
    *,
    endpoint: str,
    headers: dict[str, str],
    model: str,
    schema: dict[str, Any],
    candidate: dict[str, Any],
    provider_name: str,
) -> dict[str, Any]:
    repair_prompt = (
        "Преобразуй исходный JSON к целевой схеме. "
        "Сохрани все фактические значения. Не придумывай данные. "
        "Если значения нет, используй null, пустую строку или пустой массив в зависимости от схемы. "
        "Верни только валидный JSON без markdown.\n\n"
        f"Целевая схема:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Исходный JSON:\n{json.dumps(candidate, ensure_ascii=False)}"
    )
    repair_payload = {
        "model": model,
        "messages": [{"role": "user", "content": repair_prompt}],
        "temperature": 0,
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "stream": False,
    }
    repaired, _, _ = await _chat_completion_json(
        endpoint=endpoint,
        headers=headers,
        payload=repair_payload,
        provider_name=provider_name,
    )
    return repaired


async def _repair_raw_text_to_schema(
    *,
    endpoint: str,
    headers: dict[str, str],
    model: str,
    schema: dict[str, Any],
    raw_text: str,
    provider_name: str,
) -> dict[str, Any]:
    repair_prompt = (
        "Ниже невалидный или незавершенный JSON, полученный из document extraction. "
        "Преобразуй его в валидный JSON строго по целевой схеме. "
        "Сохрани все фактические данные. Не придумывай новые значения. "
        "Если значения нет, используй null, пустую строку или пустой массив по смыслу схемы. "
        "Верни только валидный JSON без markdown.\n\n"
        f"Целевая схема:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Сырой ответ:\n{raw_text}"
    )
    repair_payload = {
        "model": model,
        "messages": [{"role": "user", "content": repair_prompt}],
        "temperature": 0,
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "stream": False,
    }
    repaired, _, _ = await _chat_completion_json(
        endpoint=endpoint,
        headers=headers,
        payload=repair_payload,
        provider_name=provider_name,
    )
    return repaired


def _build_openrouter_messages(
    *,
    prompt: str,
    downloaded_file: DownloadedFile,
) -> list[dict[str, Any]]:
    base64_payload = base64.b64encode(_get_downloaded_file_bytes(downloaded_file)).decode("ascii")
    message_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

    if _looks_like_image(downloaded_file.filename, downloaded_file.content_type):
        image_mime = downloaded_file.content_type or "image/jpeg"
        message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{base64_payload}"},
            }
        )
    else:
        file_mime = downloaded_file.content_type or "application/pdf"
        message_content.append(
            {
                "type": "file",
                "file": {
                    "filename": downloaded_file.filename,
                    "file_data": f"data:{file_mime};base64,{base64_payload}",
                },
            }
        )

    return [{"role": "user", "content": message_content}]


def _iter_docx_blocks(document: Any):
    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield WordParagraph(child, document)
        elif tag == "tbl":
            yield WordTable(child, document)


def _clean_docx_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _convert_docx_to_structured_text(local_path: str) -> dict[str, Any]:
    document = WordDocument(local_path)
    lines: list[str] = []
    paragraph_count = 0
    table_count = 0

    for block in _iter_docx_blocks(document):
        if isinstance(block, WordParagraph):
            text = _clean_docx_text(block.text)
            if not text:
                continue
            paragraph_count += 1
            style_name = ""
            try:
                style_name = _clean_docx_text(block.style.name)
            except Exception:
                style_name = ""
            prefix = f"[P{paragraph_count}]"
            if style_name:
                prefix += f" ({style_name})"
            lines.append(f"{prefix} {text}")
            continue

        if isinstance(block, WordTable):
            table_count += 1
            lines.append(f"[T{table_count}] TABLE START")
            seen_rows: set[tuple[str, ...]] = set()
            for row_index, row in enumerate(block.rows, start=1):
                cells = tuple(_clean_docx_text(cell.text) for cell in row.cells)
                if not any(cells):
                    continue
                if cells in seen_rows:
                    continue
                seen_rows.add(cells)
                serialized_cells = " | ".join(cell if cell else "—" for cell in cells)
                lines.append(f"[T{table_count}R{row_index}] {serialized_cells}")
            lines.append(f"[T{table_count}] TABLE END")

    structured_text = "\n".join(lines).strip()
    return {
        "text": structured_text,
        "paragraph_count": paragraph_count,
        "table_count": table_count,
    }


def _build_openrouter_messages_from_text(
    *,
    prompt: str,
    document_text: str,
    filename: str,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": (
                f"{prompt}\n\n"
                "Ниже содержимое документа, предварительно извлеченное локально. "
                "Сохраняй структуру документа, учитывай параграфы, таблицы и маркеры блоков. "
                "Если возвращаешь references, опирайся на точные фрагменты текста или строки таблиц из этого представления. "
                "quote_text должен быть дословной цитатой из представления, без пересказа. "
                "Если в представлении есть маркеры [PAGE N], указывай page именно по ним. "
                "Если точная цитата не найдена, используй ближайший дословный фрагмент и anchor_text.\n\n"
                f"Имя файла: {filename}\n"
                "Формат представления:\n"
                "- [PAGE N] — номер страницы исходного документа\n"
                "- [P<N>] — параграф\n"
                "- [T<N>] — таблица\n"
                "- [T<N>R<M>] — строка таблицы\n\n"
                f"Содержимое документа:\n{document_text}"
            ),
        }
    ]


def _build_docling_remote_converter() -> DocumentConverter:
    if not DOCLING_INSTALLED:
        raise RuntimeError("Docling is not installed in this build")
    if not DOCLING_REMOTE_VLM_URL:
        raise RuntimeError("DOCLING_REMOTE_VLM_URL is not set")

    engine_options = ApiVlmEngineOptions(
        runtime_type=VlmEngineType.API,
        url=DOCLING_REMOTE_VLM_URL,
        headers=_build_openai_headers(DOCLING_REMOTE_API_KEY),
        params={
            "model": DOCLING_REMOTE_MODEL,
            "temperature": 0,
        },
    )
    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True,
        vlm_options=VlmConvertOptions.from_preset(
            DOCLING_REMOTE_PRESET,
            engine_options=engine_options,
        ),
    )
    format_options: dict[InputFormat, Any] = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    }
    if ImageFormatOption is not None:
        format_options[InputFormat.IMAGE] = ImageFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    return DocumentConverter(format_options=format_options)


def _run_docling_remote_conversion(local_path: str) -> str:
    converter = _build_docling_remote_converter()
    result = converter.convert(source=local_path)
    document = getattr(result, "document", None)
    if document is None:
        raise RuntimeError("Docling remote conversion returned no document")
    export_to_markdown = getattr(document, "export_to_markdown", None)
    if callable(export_to_markdown):
        markdown = export_to_markdown()
        if isinstance(markdown, str) and markdown.strip():
            return markdown
    export_to_text = getattr(document, "export_to_text", None)
    if callable(export_to_text):
        text = export_to_text()
        if isinstance(text, str) and text.strip():
            return text
    raise RuntimeError("Docling remote conversion returned empty document content")


async def _llamaparse_upload(client: httpx.AsyncClient, local_path: str, filename: str) -> str:
    url = f"{LLAMAPARSE_BASE_URL.rstrip('/')}/api/parsing/upload"
    headers = {"Authorization": f"Bearer {LLAMAPARSE_API_KEY}"}
    upload_mime = _normalized_content_type(filename, None) or "application/octet-stream"
    with open(local_path, "rb") as fh:
        files = {"file": (filename, fh, upload_mime)}
        data = {"language": LLAMAPARSE_LANGUAGE}
        response = await _llamaparse_request(
            client,
            "POST",
            url,
            headers=headers,
            files=files,
            data=data,
            operation_name="LlamaParse upload",
        )
    job_id = response.json().get("id")
    if not job_id:
        raise RuntimeError("LlamaParse upload returned no job id")
    return job_id


async def _llamaparse_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    operation_name: str,
    **kwargs: Any,
) -> httpx.Response:
    last_error: Exception | None = None
    retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
    attempts = max(1, LLAMAPARSE_MAX_RETRIES + 1)

    for attempt in range(1, attempts + 1):
        try:
            response = await client.request(method, url, **kwargs)
            if response.is_success:
                return response
            if response.status_code in retryable_statuses and attempt < attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 6))
                continue
            _raise_provider_http_error(response, operation_name)
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            await asyncio.sleep(min(2 ** (attempt - 1), 6))

    raise RuntimeError(f"{operation_name} failed after {attempts} attempts: {last_error}")


async def _llamaparse_wait(client: httpx.AsyncClient, job_id: str) -> None:
    url = f"{LLAMAPARSE_BASE_URL.rstrip('/')}/api/parsing/job/{job_id}"
    headers = {"Authorization": f"Bearer {LLAMAPARSE_API_KEY}"}
    deadline = time.monotonic() + LLAMAPARSE_MAX_WAIT_SECONDS
    while True:
        response = await _llamaparse_request(
            client,
            "GET",
            url,
            headers=headers,
            operation_name="LlamaParse status",
        )
        status = str(response.json().get("status", "")).upper()
        if status == "SUCCESS":
            return
        if status in {"ERROR", "FAILED", "CANCELED"}:
            raise RuntimeError(f"LlamaParse job {job_id} failed: {response.json()}")
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"LlamaParse job {job_id} timed out after {LLAMAPARSE_MAX_WAIT_SECONDS}s"
            )
        await asyncio.sleep(LLAMAPARSE_POLLING_INTERVAL)


def _serialize_llamaparse_result(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    pages_payload = data.get("pages")
    page_sections: list[str] = []
    page_count = 0

    if isinstance(pages_payload, list):
        for index, page in enumerate(pages_payload, start=1):
            if not isinstance(page, dict):
                continue
            page_no = (
                _safe_page_number(page.get("page"))
                or _safe_page_number(page.get("page_number"))
                or index
            )
            fragment = (
                page.get("md")
                or page.get("markdown")
                or page.get("text")
                or page.get("content")
                or ""
            )
            if not isinstance(fragment, str) or not fragment.strip():
                continue
            page_sections.append(f"[PAGE {page_no}]\n{fragment.strip()}")
        page_count = len(page_sections)

    top_level_content = (
        data.get(LLAMAPARSE_RESULT_TYPE)
        or data.get("markdown")
        or data.get("text")
        or ""
    )
    if isinstance(top_level_content, str):
        top_level_content = top_level_content.strip()
    else:
        top_level_content = ""

    if not page_sections and top_level_content:
        heuristic_pages = [
            chunk.strip()
            for chunk in re.split(r"(?:\n|^)\s*---+\s*(?:\n|$)", top_level_content)
            if chunk.strip()
        ]
        if len(heuristic_pages) > 1:
            page_sections = [
                f"[PAGE {index}]\n{chunk}"
                for index, chunk in enumerate(heuristic_pages, start=1)
            ]
            page_count = len(page_sections)

    if page_sections:
        content = "\n\n".join(page_sections).strip()
    else:
        content = top_level_content

    if not content:
        raise RuntimeError("LlamaParse returned empty document content")

    metadata = {
        "result_type": LLAMAPARSE_RESULT_TYPE,
        "page_count": page_count or None,
        "has_page_markers": bool(page_sections),
    }
    return content, metadata


async def _llamaparse_get_markdown(client: httpx.AsyncClient, job_id: str) -> tuple[str, dict[str, Any]]:
    url = f"{LLAMAPARSE_BASE_URL.rstrip('/')}/api/parsing/job/{job_id}/result/{LLAMAPARSE_RESULT_TYPE}"
    headers = {"Authorization": f"Bearer {LLAMAPARSE_API_KEY}"}
    response = await _llamaparse_request(
        client,
        "GET",
        url,
        headers=headers,
        operation_name="LlamaParse result",
    )
    return _serialize_llamaparse_result(response.json())


async def _extract_via_llamaparse(payload: ExtractionRequest) -> dict[str, Any]:
    if not LLAMAPARSE_API_KEY:
        raise HTTPException(status_code=500, detail="LLAMAPARSE_API_KEY is not set")
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is not set (required for structured extraction after LlamaParse parsing)",
        )
    if not payload.prompt and not payload.schema_payload:
        raise HTTPException(status_code=400, detail="schema or prompt is required")

    schema = _normalize_json_schema(payload.schema_payload, payload.prompt)
    prompt = payload.prompt or "Extract structured data from the document and return JSON."
    prompt = (
        f"{prompt}\n"
        "Верни результат строго как JSON schema response_format. "
        "Не добавляй markdown, пояснения или кодовые блоки."
    )

    downloaded_file = await _download_file(payload.file_url)
    started_at = time.monotonic()
    markdown_text = ""
    parse_metadata: dict[str, Any] = {
        "result_type": LLAMAPARSE_RESULT_TYPE,
        "page_count": None,
        "has_page_markers": False,
    }
    geometry_metadata: dict[str, Any] = {
        "enabled": GEOMETRY_ENRICHMENT_ENABLED,
        "provider": "pymupdf" if PYMUPDF_INSTALLED else None,
        "applied": False,
        "searchable_pdf": False,
        "page_count": 1,
        "reference_count": 0,
        "matched_reference_count": 0,
        "converted_string_reference_count": 0,
        "synthetic_reference_count": 0,
        "zero_based_normalized": False,
        "errors": [],
    }
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(LLAMAPARSE_REQUEST_TIMEOUT_SECONDS),
            trust_env=False,
        ) as client:
            job_id = await _llamaparse_upload(client, downloaded_file.local_path, downloaded_file.filename)
            logger.info("LlamaParse job created: %s", job_id)
            await _llamaparse_wait(client, job_id)
            markdown_text, parse_metadata = await _llamaparse_get_markdown(client, job_id)
            logger.info(
                "LlamaParse parsing finished in %.2fs, content length=%d",
                time.monotonic() - started_at,
                len(markdown_text),
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("LlamaParse parsing failed")
        raise HTTPException(status_code=502, detail=f"LlamaParse parsing failed: {exc}") from exc

    messages = _build_openrouter_messages_from_text(
        prompt=prompt,
        document_text=markdown_text,
        filename=downloaded_file.filename,
    )
    payload_json: dict[str, Any] = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "llamaparse_extraction",
                "strict": True,
                "schema": schema,
            },
        },
    }
    provider_preferences = _build_openrouter_provider_preferences(require_parameters=True)
    if provider_preferences:
        payload_json["provider"] = provider_preferences

    headers = _build_openai_headers(OPENROUTER_API_KEY)
    headers["HTTP-Referer"] = OPENROUTER_SITE_URL or "http://localhost:8005"
    headers["X-Title"] = OPENROUTER_APP_NAME or "extraction-service"

    try:
        extracted_data, provider_response, used_fallback = await _chat_completion_json(
            endpoint=f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
            headers=headers,
            payload=payload_json,
            provider_name="OpenRouter (llamaparse)",
            repair_schema=schema,
            repair_model=OPENROUTER_MODEL,
        )
        if used_fallback and _needs_schema_repair(extracted_data, schema):
            extracted_data = await _repair_json_to_schema(
                endpoint=f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
                headers=headers,
                model=OPENROUTER_MODEL,
                schema=schema,
                candidate=extracted_data,
                provider_name="OpenRouter schema repair (llamaparse)",
            )
        geometry_metadata = await to_thread.run_sync(
            lambda: _enrich_references_with_pdf_geometry(
                extracted_data,
                local_path=downloaded_file.local_path,
                content_type=downloaded_file.content_type,
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("OpenRouter structured extraction failed (llamaparse)")
        raise HTTPException(
            status_code=502,
            detail=f"Structured extraction failed after LlamaParse: {exc}",
        ) from exc
    finally:
        _cleanup_downloaded_file(downloaded_file)

    elapsed = time.monotonic() - started_at
    logger.info("LlamaParse full pipeline finished in %.2fs", elapsed)

    return jsonable_encoder({
        "analysis_id": payload.analysis_id,
        "file_id": payload.file_id,
        "file_type": payload.file_type,
        "backend": "llamaparse",
        "model_parse": "llamaparse",
        "model_extract": OPENROUTER_MODEL,
        "result": extracted_data,
        "extraction": {"pages": [_build_result_page(extracted_data, raw_text=markdown_text)]},
        "extraction_metadata": {
            "docling_version": None,
            "page_count": parse_metadata.get("page_count") or geometry_metadata.get("page_count"),
            "errors": [],
            "provider": "llamaparse+openrouter",
            "provider_usage": provider_response.get("usage"),
            "geometry": geometry_metadata,
            "llamaparse": parse_metadata,
            "docx_conversion": None,
        },
    })


async def _extract_via_docling_local(payload: ExtractionRequest) -> dict[str, Any]:
    _raise_docling_backend_unavailable("docling_local")


async def _extract_via_docling_remote(payload: ExtractionRequest) -> dict[str, Any]:
    _raise_docling_backend_unavailable("docling_remote")


async def _extract_via_openrouter(payload: ExtractionRequest) -> dict[str, Any]:
    if not payload.prompt and not payload.schema_payload:
        raise HTTPException(status_code=400, detail="schema or prompt is required")
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")

    schema = _normalize_json_schema(payload.schema_payload, payload.prompt)
    prompt = payload.prompt or "Extract structured data from the document and return JSON."
    prompt = (
        f"{prompt}\n"
        "Верни результат строго как JSON schema response_format. "
        "Не добавляй markdown, пояснения или кодовые блоки."
    )

    downloaded_file = await _download_file(payload.file_url)
    is_docx = _looks_like_docx(downloaded_file.filename, downloaded_file.content_type)
    docx_conversion_metadata: dict[str, Any] | None = None
    if is_docx:
        try:
            docx_conversion_metadata = await to_thread.run_sync(
                lambda: _convert_docx_to_structured_text(downloaded_file.local_path)
            )
        except Exception as exc:
            logger.exception("DOCX conversion failed")
            raise HTTPException(
                status_code=502,
                detail=f"DOCX conversion failed: {exc}",
            ) from exc

    messages = (
        _build_openrouter_messages_from_text(
            prompt=prompt,
            document_text=(docx_conversion_metadata or {}).get("text", ""),
            filename=downloaded_file.filename,
        )
        if is_docx
        else _build_openrouter_messages(
            prompt=prompt,
            downloaded_file=downloaded_file,
        )
    )
    payload_json: dict[str, Any] = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "openrouter_extraction",
                "strict": True,
                "schema": schema,
            },
        },
    }
    provider_preferences = _build_openrouter_provider_preferences(require_parameters=True)
    if provider_preferences:
        payload_json["provider"] = provider_preferences
    if not is_docx and not _looks_like_image(downloaded_file.filename, downloaded_file.content_type):
        payload_json["plugins"] = [
            {
                "id": "file-parser",
                "pdf": {"engine": OPENROUTER_PDF_ENGINE},
            }
        ]

    headers = _build_openai_headers(OPENROUTER_API_KEY)
    headers["HTTP-Referer"] = OPENROUTER_SITE_URL or "http://localhost:8005"
    headers["X-Title"] = OPENROUTER_APP_NAME or "extraction-service"

    started_at = time.monotonic()
    geometry_metadata: dict[str, Any] = {
        "enabled": GEOMETRY_ENRICHMENT_ENABLED,
        "provider": "pymupdf" if PYMUPDF_INSTALLED else None,
        "applied": False,
        "searchable_pdf": False,
        "page_count": 1,
        "reference_count": 0,
        "matched_reference_count": 0,
        "converted_string_reference_count": 0,
        "zero_based_normalized": False,
        "errors": [],
    }
    try:
        extracted_data, provider_response, used_fallback = await _chat_completion_json(
            endpoint=f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
            headers=headers,
            payload=payload_json,
            provider_name="OpenRouter",
            repair_schema=schema,
            repair_model=OPENROUTER_MODEL,
        )
        if used_fallback and _needs_schema_repair(extracted_data, schema):
            extracted_data = await _repair_json_to_schema(
                endpoint=f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
                headers=headers,
                model=OPENROUTER_MODEL,
                schema=schema,
                candidate=extracted_data,
                provider_name="OpenRouter schema repair",
            )
        geometry_metadata = await to_thread.run_sync(
            lambda: _enrich_references_with_pdf_geometry(
                extracted_data,
                local_path=downloaded_file.local_path,
                content_type=downloaded_file.content_type,
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("OpenRouter extraction failed")
        detail = f"OpenRouter extraction failed: {exc}"
        if not DEBUG_ERRORS and detail:
            detail = str(detail)
        raise HTTPException(status_code=502, detail=detail) from exc
    finally:
        _cleanup_downloaded_file(downloaded_file)
        elapsed = time.monotonic() - started_at
        logger.info("OpenRouter extraction finished in %.2fs", elapsed)

    response = {
        "analysis_id": payload.analysis_id,
        "file_id": payload.file_id,
        "file_type": payload.file_type,
        "backend": "openrouter",
        "model_parse": OPENROUTER_MODEL,
        "model_extract": OPENROUTER_MODEL,
        "result": extracted_data,
        "extraction": {"pages": [_build_result_page(extracted_data)]},
        "extraction_metadata": {
            "docling_version": _get_docling_version(),
            "page_count": geometry_metadata.get("page_count", 1),
            "errors": [],
            "provider": "openrouter",
            "provider_usage": provider_response.get("usage"),
            "geometry": geometry_metadata,
            "docx_conversion": docx_conversion_metadata,
        },
    }
    return jsonable_encoder(response)


@app.post("/extract")
async def extract_document(payload: ExtractionRequest) -> dict:
    if not payload.schema_payload and not payload.prompt:
        raise HTTPException(status_code=400, detail="schema or prompt is required")

    parsed_url = urlparse(payload.file_url)
    if parsed_url.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="file_url must be a public URL")

    backend = _select_backend(payload.backend)
    if backend == "docling_local":
        return await _extract_via_docling_local(payload)
    if backend == "docling_remote":
        return await _extract_via_docling_remote(payload)
    if backend == "openrouter":
        return await _extract_via_openrouter(payload)
    if backend == "llamaparse":
        return await _extract_via_llamaparse(payload)
    raise HTTPException(status_code=400, detail=f"Unsupported backend: {backend}")
