import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import subprocess
import tempfile
import time
import traceback
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
from fastapi.responses import Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field

try:
    import fitz

    PYMUPDF_INSTALLED = True
except ImportError:
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
    except ImportError:
        ImageFormatOption = None
    DOCLING_INSTALLED = True
except ImportError:
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
OPENROUTER_MAX_TOKENS = min(_get_int_env("OPENROUTER_MAX_TOKENS", 4000), 8000)
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


def _looks_like_pdf(filename: str, content_type: str | None) -> bool:
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    suffix = Path(filename).suffix.lower()
    return normalized == "application/pdf" or suffix == ".pdf"


def _convert_pdf_to_structured_text(local_path: str) -> dict[str, Any]:
    """Извлекает текст из PDF для передачи в LLM.

    Сначала пробует извлечь текстовый слой через PyMuPDF.
    Если PDF отсканированный (нет текстового слоя) — применяет Tesseract OCR.
    Возвращает структурированный текст с маркерами [PAGE N].
    """
    if not PYMUPDF_INSTALLED:
        return {"text": "", "page_count": 0, "ocr_applied": False, "error": "PyMuPDF not installed"}

    document = fitz.open(local_path)
    page_count = document.page_count
    lines: list[str] = []
    ocr_applied = False

    # Сначала пробуем обычный текстовый слой
    all_pages_text: list[str] = []
    for i in range(page_count):
        page = document.load_page(i)
        text = page.get_text("text", sort=True).strip()
        all_pages_text.append(text)

    has_text = any(len(t) > 20 for t in all_pages_text)

    if has_text:
        for i, text in enumerate(all_pages_text):
            if text:
                lines.append(f"[PAGE {i + 1}]")
                lines.append(text)
    else:
        # Scanned PDF — пробуем Tesseract OCR
        ocr_available = False
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            ocr_available = True
        except Exception:
            pass

        if ocr_available:
            from PIL import Image
            import io
            DPI = 200
            try:
                import pytesseract
                for i in range(page_count):
                    page = document.load_page(i)
                    mat = fitz.Matrix(DPI / 72, DPI / 72)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(img, lang="rus+eng").strip()
                    if text:
                        lines.append(f"[PAGE {i + 1}]")
                        lines.append(text)
                ocr_applied = True
            except Exception as exc:
                logger.warning("PDF OCR failed: %s", exc)

    document.close()
    raw_text = "\n".join(lines).strip()
    # Убираем суррогатные символы (могут возникать при OCR) — они невалидны в JSON/UTF-8
    clean_text = raw_text.encode("utf-8", errors="replace").decode("utf-8")
    return {
        "text": clean_text,
        "page_count": page_count,
        "ocr_applied": ocr_applied,
    }


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


def _json_error_snippet(raw_content: str, position: int, radius: int = 700) -> str:
    start = max(0, position - radius)
    end = min(len(raw_content), position + radius)
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(raw_content) else ""
    return prefix + raw_content[start:end] + suffix


def _provider_response_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("id", "model", "provider"):
        value = data.get(key)
        if value is not None:
            metadata[key] = value
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        for key in ("finish_reason", "native_finish_reason"):
            value = choice.get(key)
            if value is not None:
                metadata[key] = value
    usage = data.get("usage")
    if isinstance(usage, dict):
        metadata["usage"] = usage
    return metadata


def _json_decode_diagnostic(
    provider_name: str,
    raw_content: str,
    exc: json.JSONDecodeError,
    response_data: dict[str, Any],
) -> str:
    snippet = _json_error_snippet(raw_content, exc.pos)
    metadata = _provider_response_metadata(response_data)
    return (
        f"{provider_name} returned invalid JSON: {exc.msg} "
        f"at line={exc.lineno} column={exc.colno} char={exc.pos}; "
        f"response_length={len(raw_content)}; "
        f"response_metadata={json.dumps(metadata, ensure_ascii=True)}; "
        f"snippet_around_error={json.dumps(snippet, ensure_ascii=True)}"
    )


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


def _json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


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


def _normalize_references_in_place(node: Any) -> None:
    """Нормализует поле references везде в дереве:
    если модель вернула dict вместо list — оборачивает в list.
    Некоторые модели (gpt-4o-mini) возвращают один объект вместо массива.
    """
    if isinstance(node, dict):
        references = node.get("references")
        if isinstance(references, dict):
            node["references"] = [references]
        for key, value in node.items():
            if key != "references":
                _normalize_references_in_place(value)
    elif isinstance(node, list):
        for value in node:
            _normalize_references_in_place(value)


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


def _reference_iter_with_ancestors(node: Any, ancestors: tuple[dict[str, Any], ...] = ()):
    if isinstance(node, dict):
        references = node.get("references")
        if isinstance(references, list):
            yield node, references, ancestors
        next_ancestors = (*ancestors, node)
        for key, value in node.items():
            if key == "references":
                continue
            yield from _reference_iter_with_ancestors(value, next_ancestors)
    elif isinstance(node, list):
        for value in node:
            yield from _reference_iter_with_ancestors(value, ancestors)


def _collect_ancestor_context(ancestors: tuple[dict[str, Any], ...]) -> list[str]:
    context_values: list[str] = []
    for node in reversed(ancestors):
        product_name = node.get("product_name")
        product_model = node.get("product_model")
        if isinstance(product_name, str) and product_name.strip():
            if isinstance(product_model, str) and product_model.strip():
                context_values.append(f"{product_name.strip()} {product_model.strip()}")
            context_values.append(product_name.strip())
        if isinstance(product_model, str) and product_model.strip():
            context_values.append(product_model.strip())
        for key in ("title", "label", "name"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                context_values.append(value.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for value in context_values:
        normalized = _normalize_match_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value)
    return deduped[:8]


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


def _rect_to_bbox(rect: Any, page_rect: Any = None) -> dict[str, float]:
    result: dict[str, float] = {
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
    if page_rect is not None:
        pw = float(page_rect.width)
        ph = float(page_rect.height)
        if pw > 0 and ph > 0:
            result["norm_x0"] = round(min(1.0, max(0.0, float(rect.x0) / pw)), 6)
            result["norm_y0"] = round(min(1.0, max(0.0, float(rect.y0) / ph)), 6)
            result["norm_x1"] = round(min(1.0, max(0.0, float(rect.x1) / pw)), 6)
            result["norm_y1"] = round(min(1.0, max(0.0, float(rect.y1) / ph)), 6)
    return result


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




def _build_pdf_page_index_ocr(local_path: str) -> tuple[Any, list[PdfPageIndex]]:
    """
    For scanned PDFs: renders each page as an image, runs Tesseract OCR,
    and builds PdfPageIndex from the recognized words.
    Word bboxes are in PDF-point coordinates (at DPI=150 scale).
    """
    import pytesseract
    from PIL import Image
    import io

    DPI = 150
    document = fitz.open(local_path)
    pages: list[PdfPageIndex] = []

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        try:
            ocr_data = pytesseract.image_to_data(
                img,
                lang="rus+eng",
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            logger.warning("OCR failed for page %d: %s", page_index + 1, exc)
            pages.append(PdfPageIndex(page_number=page_index + 1, page=page, words=[]))
            continue

        scale = 72.0 / DPI  # pixels -> PDF points
        words: list[PdfWord] = []
        n = len(ocr_data["text"])
        for i in range(n):
            text = str(ocr_data["text"][i]).strip()
            conf = int(ocr_data["conf"][i])
            if not text or conf < 30:
                continue
            normalized = _normalize_match_text(text)
            if not normalized:
                continue
            x = ocr_data["left"][i] * scale
            y = ocr_data["top"][i] * scale
            w = ocr_data["width"][i] * scale
            h = ocr_data["height"][i] * scale
            rect = fitz.Rect(x, y, x + w, y + h)
            words.append(PdfWord(text=text, normalized=normalized, rect=rect))

        pages.append(PdfPageIndex(page_number=page_index + 1, page=page, words=words))

    return document, pages

def _search_exact_candidate_rects(page: Any, candidate: str) -> list[Any]:
    snippet = candidate.strip()
    if len(snippet) < 3:
        return []
    if len(snippet) > 220:
        snippet = snippet[:220].rsplit(" ", 1)[0].strip() or snippet[:220]
    try:
        rects = page.search_for(snippet)
    except Exception:
        return []
    return sorted(rects, key=lambda rect: (float(rect.y0), float(rect.x0))) if rects else []


def _rect_center(rect: Any) -> tuple[float, float]:
    return ((float(rect.x0) + float(rect.x1)) / 2, (float(rect.y0) + float(rect.y1)) / 2)


def _select_best_exact_rect(
    rects: list[Any],
    page_index: PdfPageIndex,
    context_terms: list[str],
) -> tuple[Any | None, float]:
    if not rects:
        return None, 0.0
    if len(rects) == 1:
        return rects[0], 0.0

    context_rects: list[Any] = []
    for term in context_terms:
        context_rects.extend(_search_exact_candidate_rects(page_index.page, term))
        if context_rects:
            break

    if not context_rects:
        return rects[0], 0.0

    def distance_to_context(rect: Any) -> float:
        rect_x, rect_y = _rect_center(rect)
        best = float("inf")
        for context_rect in context_rects:
            context_x, context_y = _rect_center(context_rect)
            vertical_distance = abs(rect_y - context_y)
            horizontal_distance = abs(rect_x - context_x) * 0.15
            best = min(best, vertical_distance + horizontal_distance)
        return best

    best_rect = min(rects, key=distance_to_context)
    context_bonus = max(0.0, 80.0 - distance_to_context(best_rect))
    return best_rect, context_bonus


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




def _token_overlap_ratio(a: str, b: str) -> float:
    """Fraction of tokens from a that are found in b."""
    tokens_a = set(_tokenize_match_text(a))
    tokens_b = set(_tokenize_match_text(b))
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def _search_table_row_candidate(
    page_index: PdfPageIndex,
    anchor_text: str,
) -> tuple[Any | None, float]:
    """
    Searches for the table row or text line containing anchor_text.
    Strategy:
      1. Use page.find_tables() — if anchor_text matches a cell, return bbox of the whole row.
      2. Otherwise find the text line whose words best overlap anchor tokens (Y-band grouping).
    """
    normalized_anchor = _normalize_match_text(anchor_text)
    if not normalized_anchor or len(normalized_anchor) < 2:
        return None, 0.0

    # --- Step 1: Search in tables ---
    try:
        tables = page_index.page.find_tables()
        for table in tables.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell is None:
                        continue
                    try:
                        cell_rect = fitz.Rect(cell) if not isinstance(cell, fitz.Rect) else cell
                    except Exception:
                        continue
                    cell_text = page_index.page.get_textbox(cell_rect).strip()
                    cell_normalized = _normalize_match_text(cell_text)
                    if not cell_normalized:
                        continue
                    if (
                        normalized_anchor in cell_normalized
                        or cell_normalized in normalized_anchor
                        or _token_overlap_ratio(normalized_anchor, cell_normalized) >= 0.6
                    ):
                        row_rect = _union_rects([
                            fitz.Rect(c) if not isinstance(c, fitz.Rect) else c
                            for c in row.cells
                            if c is not None
                        ])
                        if row_rect and not row_rect.is_empty:
                            overlap = _token_overlap_ratio(normalized_anchor, cell_normalized)
                            return row_rect, max(0.7, overlap)
    except Exception as exc:
        logger.debug("find_tables failed on page %d: %s", page_index.page_number, exc)

    # --- Step 2: Free-text Y-band search ---
    if not page_index.words:
        return None, 0.0

    line_tolerance = 4.0  # words within this y0 delta are on the same line
    lines: list[list[PdfWord]] = []
    current_line: list[PdfWord] = []
    current_y = None
    for word in page_index.words:
        word_y = float(word.rect.y0)
        if current_y is None or abs(word_y - current_y) <= line_tolerance:
            current_line.append(word)
            if current_y is None:
                current_y = word_y
        else:
            if current_line:
                lines.append(current_line)
            current_line = [word]
            current_y = word_y
    if current_line:
        lines.append(current_line)

    anchor_tokens = set(_tokenize_match_text(anchor_text))
    if not anchor_tokens:
        return None, 0.0

    best_line = None
    best_overlap = 0.0
    for line in lines:
        line_tokens = set(w.normalized for w in line)
        overlap = len(anchor_tokens & line_tokens) / len(anchor_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_line = line

    if best_line and best_overlap >= 0.5:
        rect = _union_rects([w.rect for w in best_line])
        return rect, best_overlap

    return None, 0.0

def _is_ambiguous_short_candidate(text: str, normalized: str) -> bool:
    """Returns True if candidate is short or purely numeric/units."""
    if len(normalized) <= 6:
        return True
    if re.fullmatch(r"[\d.,\-+/%\s]+", text.strip()):
        return True
    return False


def _candidate_rank(candidate: dict[str, Any], has_descriptive_candidate: bool = False) -> float:
    base = {
        "quote_text": 1000.0,
        "value": 950.0,
        "label_plus_value": 850.0,
        "raw_reference": 800.0,
        "locator_text": 550.0,
        "text": 520.0,
        "anchor_text": 450.0,
        "label": 250.0,
    }.get(str(candidate.get("kind") or ""), 300.0)

    # Lower rank for short/numeric quote_text or value when a descriptive candidate exists.
    if has_descriptive_candidate and candidate.get("kind") in {"quote_text", "value"}:
        text = str(candidate.get("text") or "")
        normalized = str(candidate.get("normalized") or _normalize_match_text(text))
        if _is_ambiguous_short_candidate(text, normalized):
            return 200.0
    return base


def _is_generic_table_anchor(candidate: dict[str, Any]) -> bool:
    kind = str(candidate.get("kind") or "")
    if kind not in {"anchor_text", "locator_text", "text"}:
        return False
    normalized = str(candidate.get("normalized") or _normalize_match_text(str(candidate.get("text") or "")))
    return bool(re.fullmatch(r"(таблица|table)\s*\d*", normalized, flags=re.IGNORECASE))


def _find_reference_location(
    pages: list[PdfPageIndex],
    candidates: list[dict[str, Any]],
    context_terms: list[str] | None = None,
) -> dict[str, Any] | None:
    best_match: dict[str, Any] | None = None
    context_terms = context_terms or []
    has_specific_candidate = any(
        candidate.get("kind") in {"quote_text", "value", "label_plus_value", "raw_reference"}
        for candidate in candidates
    )
    # Whether a descriptive (long) candidate exists — used to penalise short/numeric ones
    has_descriptive_candidate = any(
        c.get("kind") in {"anchor_text", "locator_text", "label_plus_value"}
        and len(str(c.get("normalized") or _normalize_match_text(str(c.get("text") or "")))) > 6
        for c in candidates
    )

    for candidate in candidates:
        if has_specific_candidate and _is_generic_table_anchor(candidate):
            continue
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
            rects = _search_exact_candidate_rects(page_index.page, candidate_text)
            rect, context_bonus = _select_best_exact_rect(rects, page_index, context_terms)
            if not rect:
                continue
            page_bonus = 18.0 if page_hint and page_index.page_number == page_hint else 0.0
            score = (
                _candidate_rank(candidate, has_descriptive_candidate=has_descriptive_candidate)
                + len(candidate.get("normalized", _normalize_match_text(candidate_text))) * candidate_weight
                + page_bonus
                + context_bonus
            )
            if not best_match or score > best_match["score"]:
                best_match = {
                    "page": page_index.page_number,
                    "bbox": _rect_to_bbox(rect, page_index.page.rect),
                    "score": score,
                    "locator_strategy": "pymupdf_exact",
                    "matched_text": candidate_text,
                }

    if best_match:
        return best_match

    for candidate in candidates:
        if has_specific_candidate and _is_generic_table_anchor(candidate):
            continue
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
            score = (
                _candidate_rank(candidate, has_descriptive_candidate=has_descriptive_candidate)
                + coverage * 100 * candidate_weight
                + min(len(candidate_text), 120) / 10
                + page_bonus
            )
            if not best_match or score > best_match["score"]:
                best_match = {
                    "page": page_index.page_number,
                    "bbox": _rect_to_bbox(rect, page_index.page.rect),
                    "score": score,
                    "locator_strategy": "pymupdf_tokens",
                    "matched_text": candidate_text,
                }
    # --- Third pass: table row / free-text line search ---
    anchor_candidates = [
        c for c in candidates
        if c.get("kind") in {"anchor_text", "locator_text", "label", "label_plus_value"}
        and len(c.get("normalized", "")) >= 3
    ]
    for candidate in anchor_candidates:
        if has_specific_candidate and _is_generic_table_anchor(candidate):
            continue
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
            rect, overlap = _search_table_row_candidate(page_index, candidate_text)
            if not rect:
                continue
            page_bonus = 10.0 if page_hint and page_index.page_number == page_hint else 0.0
            score = (
                _candidate_rank(candidate, has_descriptive_candidate=False)
                + overlap * 80 * candidate_weight
                + min(len(candidate_text), 120) / 10
                + page_bonus
            )
            if not best_match or score > best_match["score"]:
                best_match = {
                    "page": page_index.page_number,
                    "bbox": _rect_to_bbox(rect, page_index.page.rect),
                    "score": score,
                    "locator_strategy": "table_row",
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
        "ocr_applied": False,
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
            # Attempt OCR fallback for scanned PDFs
            ocr_available = False
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                ocr_available = True
            except Exception:
                pass

            if not ocr_available:
                metadata["errors"].append("PDF has no searchable text layer and tesseract is not available")
                return metadata

            try:
                if document is not None:
                    document.close()
                document, pages = _build_pdf_page_index_ocr(local_path)
                metadata["searchable_pdf"] = any(page.words for page in pages)
                metadata["ocr_applied"] = True
            except Exception as exc:
                logger.warning("OCR fallback failed: %s", exc)
                metadata["errors"].append(f"OCR failed: {exc}")
                return metadata

            if not metadata["searchable_pdf"]:
                metadata["errors"].append("OCR produced no words")
                return metadata

        metadata["zero_based_normalized"] = _normalize_reference_pages(extracted_data)
        metadata["synthetic_reference_count"] = _inject_synthetic_references(extracted_data)

        for holder, references, ancestors in _reference_iter_with_ancestors(extracted_data):
            label_context = _collect_label_context(holder)
            value_context = _collect_value_context(holder)
            ancestor_context = _collect_ancestor_context(ancestors)
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

                match = _find_reference_location(pages, candidates, ancestor_context)
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
        diagnostic = _json_decode_diagnostic(provider_name, raw_content, exc, data)
        if repair_schema and repair_model:
            logger.warning(
                "%s; attempting repair",
                diagnostic,
            )
            try:
                parsed = await _repair_raw_text_to_schema(
                    endpoint=endpoint,
                    headers=headers,
                    model=repair_model,
                    schema=repair_schema,
                    raw_text=raw_content,
                    provider_name=f"{provider_name} raw JSON repair",
                )
            except Exception as repair_exc:
                raise RuntimeError(
                    f"{provider_name} returned invalid JSON and repair failed. "
                    f"Original response diagnostic: {diagnostic}; "
                    f"repair_error={repair_exc}"
                ) from repair_exc
        else:
            raise RuntimeError(diagnostic) from exc
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
        "response_format": _json_schema_response_format("json_schema_repair", schema),
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
        "response_format": _json_schema_response_format("raw_json_repair", schema),
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


def _convert_office_document_to_pdf(local_path: str, output_dir: str) -> str:
    source_path = Path(local_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = target_dir / "lo-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "soffice",
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--nodefault",
        f"-env:UserInstallation=file://{profile_dir.as_posix()}",
        "--convert-to",
        "pdf",
        "--outdir",
        str(target_dir),
        str(source_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "LibreOffice conversion failed").strip()
        raise RuntimeError(detail)

    expected_path = target_dir / f"{source_path.stem}.pdf"
    if expected_path.exists():
        return str(expected_path)

    candidates = sorted(target_dir.glob("*.pdf"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("LibreOffice did not create a PDF preview")
    return str(candidates[0])


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
    is_docx = _looks_like_docx(downloaded_file.filename, downloaded_file.content_type)
    docx_pdf_temp_dir: tempfile.TemporaryDirectory | None = None
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
    geometry_local_path = downloaded_file.local_path
    geometry_content_type = downloaded_file.content_type
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
        raise HTTPException(
            status_code=502,
            detail=(
                f"LlamaParse parsing failed: {exc}\n\n"
                f"Extraction service traceback:\n{traceback.format_exc()}"
            ),
        ) from exc

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
        "response_format": _json_schema_response_format("llamaparse_extraction", schema),
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
        if is_docx:
            try:
                docx_pdf_temp_dir = tempfile.TemporaryDirectory()
                geometry_local_path = await to_thread.run_sync(
                    lambda: _convert_office_document_to_pdf(
                        downloaded_file.local_path,
                        docx_pdf_temp_dir.name,
                    )
                )
                geometry_content_type = "application/pdf"
                parse_metadata["pdf_preview"] = {"created": True}
            except Exception as exc:
                logger.exception("DOCX PDF preview conversion failed")
                parse_metadata["pdf_preview"] = {"created": False, "error": str(exc)}
        geometry_metadata = await to_thread.run_sync(
            lambda: _enrich_references_with_pdf_geometry(
                extracted_data,
                local_path=geometry_local_path,
                content_type=geometry_content_type,
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("OpenRouter structured extraction failed (llamaparse)")
        raise HTTPException(
            status_code=502,
            detail=(
                f"Structured extraction failed after LlamaParse: {exc}\n\n"
                f"Extraction service traceback:\n{traceback.format_exc()}"
            ),
        ) from exc
    finally:
        _cleanup_downloaded_file(downloaded_file)
        if docx_pdf_temp_dir is not None:
            docx_pdf_temp_dir.cleanup()

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
    is_pdf = _looks_like_pdf(downloaded_file.filename, downloaded_file.content_type)
    logger.info(
        "File detected: filename=%r content_type=%r is_docx=%s is_pdf=%s",
        downloaded_file.filename, downloaded_file.content_type, is_docx, is_pdf,
    )
    docx_conversion_metadata: dict[str, Any] | None = None
    pdf_conversion_metadata: dict[str, Any] | None = None
    docx_pdf_temp_dir: tempfile.TemporaryDirectory | None = None

    if is_docx:
        try:
            docx_conversion_metadata = await to_thread.run_sync(
                lambda: _convert_docx_to_structured_text(downloaded_file.local_path)
            )
        except Exception as exc:
            logger.exception("DOCX conversion failed")
            raise HTTPException(
                status_code=502,
                detail=(
                    f"DOCX conversion failed: {exc}\n\n"
                    f"Extraction service traceback:\n{traceback.format_exc()}"
                ),
            ) from exc
    elif is_pdf and PYMUPDF_INSTALLED:
        # Для PDF: сначала пробуем извлечь текст локально (searchable или OCR).
        # Если текст получен — отправляем как текст (надёжнее, чем file-parser плагин).
        # Если текст пустой — падаем обратно на file-parser (нативный путь OpenRouter).
        try:
            pdf_conversion_metadata = await to_thread.run_sync(
                lambda: _convert_pdf_to_structured_text(downloaded_file.local_path)
            )
            logger.info(
                "PDF text extraction: ocr_applied=%s text_len=%d",
                pdf_conversion_metadata.get("ocr_applied"),
                len(pdf_conversion_metadata.get("text") or ""),
            )
        except Exception as exc:
            logger.warning("PDF text extraction failed, falling back to file-parser: %s", exc)
            pdf_conversion_metadata = None

    # Определяем текст документа для text-пути
    use_text_path = False
    document_text = ""
    if is_docx and docx_conversion_metadata:
        use_text_path = True
        document_text = (docx_conversion_metadata or {}).get("text", "")
    elif is_pdf and pdf_conversion_metadata:
        extracted_text = pdf_conversion_metadata.get("text", "")
        if extracted_text.strip():
            use_text_path = True
            document_text = extracted_text

    logger.info("Extraction path: use_text_path=%s document_text_len=%d", use_text_path, len(document_text))

    messages = (
        _build_openrouter_messages_from_text(
            prompt=prompt,
            document_text=document_text,
            filename=downloaded_file.filename,
        )
        if use_text_path
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
        "response_format": _json_schema_response_format("openrouter_extraction", schema),
    }
    provider_preferences = _build_openrouter_provider_preferences(require_parameters=True)
    if provider_preferences:
        payload_json["provider"] = provider_preferences
    # file-parser плагин добавляем только если не используем text-путь
    if not use_text_path and not _looks_like_image(downloaded_file.filename, downloaded_file.content_type):
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
    geometry_local_path = downloaded_file.local_path
    geometry_content_type = downloaded_file.content_type
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
        # Нормализуем references: некоторые модели возвращают dict вместо list
        _normalize_references_in_place(extracted_data)
        if is_docx:
            try:
                docx_pdf_temp_dir = tempfile.TemporaryDirectory()
                geometry_local_path = await to_thread.run_sync(
                    lambda: _convert_office_document_to_pdf(
                        downloaded_file.local_path,
                        docx_pdf_temp_dir.name,
                    )
                )
                geometry_content_type = "application/pdf"
                if docx_conversion_metadata is not None:
                    docx_conversion_metadata["pdf_preview"] = {"created": True}
            except Exception as exc:
                logger.exception("DOCX PDF preview conversion failed")
                if docx_conversion_metadata is not None:
                    docx_conversion_metadata["pdf_preview"] = {
                        "created": False,
                        "error": str(exc),
                    }
        geometry_metadata = await to_thread.run_sync(
            lambda: _enrich_references_with_pdf_geometry(
                extracted_data,
                local_path=geometry_local_path,
                content_type=geometry_content_type,
            )
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("OpenRouter extraction failed")
        detail = (
            f"OpenRouter extraction failed: {exc}\n\n"
            f"Extraction service traceback:\n{traceback.format_exc()}"
        )
        if not DEBUG_ERRORS and detail:
            detail = str(detail)
        raise HTTPException(status_code=502, detail=detail) from exc
    finally:
        _cleanup_downloaded_file(downloaded_file)
        if docx_pdf_temp_dir is not None:
            docx_pdf_temp_dir.cleanup()
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
            "pdf_conversion": pdf_conversion_metadata,
        },
    }
    return jsonable_encoder(response)


@app.get("/render-pdf")
async def render_pdf(url: str) -> Response:
    """Скачивает PDF по URL, рендерит каждую страницу через PyMuPDF в изображение
    и возвращает новый PDF из растровых страниц. Решает проблему с нестандартными
    кириллическими шрифтами, которые PDF.js не умеет отображать."""
    if not PYMUPDF_INSTALLED:
        raise HTTPException(status_code=501, detail="PyMuPDF not available")

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="url must be http or https")

    try:
        downloaded = await _download_file(url)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to download file: {exc}") from exc

    try:
        def _rasterize(local_path: str) -> bytes:
            src = fitz.open(local_path)
            out = fitz.open()
            DPI = 150
            scale = DPI / 72
            mat = fitz.Matrix(scale, scale)
            for page in src:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                # Page size in pts at 72dpi
                pw = pix.width * 72 / DPI
                ph = pix.height * 72 / DPI
                out_page = out.new_page(width=pw, height=ph)
                out_page.insert_image(fitz.Rect(0, 0, pw, ph), pixmap=pix)
            src.close()
            import io
            buf = io.BytesIO()
            out.save(buf, deflate=True, garbage=2)
            out.close()
            return buf.getvalue()

        pdf_bytes = await to_thread.run_sync(lambda: _rasterize(downloaded.local_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF rasterization failed: {exc}") from exc
    finally:
        _cleanup_downloaded_file(downloaded)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=preview.pdf"},
    )


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
