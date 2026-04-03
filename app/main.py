import base64
import json
import logging
import mimetypes
import os
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
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
from docling.document_extractor import DocumentExtractor
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field

try:
    from docling.document_converter import ImageFormatOption
except ImportError:  # pragma: no cover - depends on docling version
    ImageFormatOption = None

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
EXTRACTION_BACKEND = _get_env("EXTRACTION_BACKEND", "docling_local") or "docling_local"
DEBUG_ERRORS = _get_env("DEBUG_ERRORS", "0") == "1"
FILE_DOWNLOAD_TIMEOUT_SECONDS = _get_int_env("FILE_DOWNLOAD_TIMEOUT_SECONDS", 300)
REMOTE_API_TIMEOUT_SECONDS = _get_int_env("REMOTE_API_TIMEOUT_SECONDS", 600)

DOCLING_REMOTE_API_URL = _get_env("DOCLING_REMOTE_API_URL")
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

SUPPORTED_BACKENDS = {"docling_local", "docling_remote", "openrouter"}
ALLOWED_FORMATS = (InputFormat.IMAGE, InputFormat.PDF)
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
        description="docling_local | docling_remote | openrouter",
    )


@dataclass
class DownloadedFile:
    filename: str
    content_type: Optional[str]
    file_bytes: bytes


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
    return {"status": "ok", "default_backend": EXTRACTION_BACKEND}


def _get_docling_version() -> Optional[str]:
    try:
        return version("docling")
    except PackageNotFoundError:
        return None


def _get_local_extractor() -> DocumentExtractor:
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


def _parse_json_from_chat_response(data: dict[str, Any], *, provider_name: str) -> dict[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{provider_name} response has no choices")

    message = choices[0].get("message", {})
    content = _message_text_from_content(message.get("content"))
    content = _strip_json_fences(content)
    if not content:
        raise RuntimeError(f"{provider_name} returned empty content")

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
    try:
        async with httpx.AsyncClient(
            timeout=FILE_DOWNLOAD_TIMEOUT_SECONDS,
            follow_redirects=True,
            trust_env=False,
        ) as client:
            response = await client.get(file_url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Failed to download file: {exc}") from exc

    content_type = response.headers.get("content-type")
    filename = _guess_filename_from_url(file_url, content_type)
    return DownloadedFile(
        filename=filename,
        content_type=_normalized_content_type(filename, content_type),
        file_bytes=response.content,
    )


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
    parsed = _parse_json_from_chat_response(data, provider_name=provider_name)
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


def _build_openrouter_messages(
    *,
    prompt: str,
    downloaded_file: DownloadedFile,
) -> list[dict[str, Any]]:
    base64_payload = base64.b64encode(downloaded_file.file_bytes).decode("ascii")
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


def _build_docling_remote_converter() -> DocumentConverter:
    if not DOCLING_REMOTE_API_URL:
        raise RuntimeError("DOCLING_REMOTE_API_URL is not set")

    engine_options = ApiVlmEngineOptions(
        runtime_type=VlmEngineType.API,
        url=DOCLING_REMOTE_API_URL,
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


async def _extract_via_docling_local(payload: ExtractionRequest) -> dict[str, Any]:
    template = _normalize_template(payload.schema_payload, payload.prompt)
    started_at = time.monotonic()
    try:
        result = await to_thread.run_sync(
            _run_local_docling_extraction,
            _get_local_extractor(),
            payload.file_url,
            template,
        )
    except Exception as exc:
        logger.exception("Local Docling extraction failed")
        detail = "Local Docling extraction failed"
        if DEBUG_ERRORS:
            detail = f"{detail}: {exc}"
        raise HTTPException(status_code=502, detail=detail) from exc
    finally:
        elapsed = time.monotonic() - started_at
        logger.info("Local Docling extraction finished in %.2fs", elapsed)

    pages = _serialize_pages(result)
    page_errors = [
        {"page_no": page.get("page_no"), "errors": page.get("errors")}
        for page in pages
        if page.get("errors")
    ]

    response = {
        "analysis_id": payload.analysis_id,
        "file_id": payload.file_id,
        "file_type": payload.file_type,
        "backend": "docling_local",
        "model_parse": DOC_PARSE_MODEL,
        "model_extract": DOC_EXTRACT_MODEL,
        "extraction": {"pages": pages},
        "extraction_metadata": {
            "docling_version": _get_docling_version(),
            "page_count": len(pages),
            "errors": page_errors,
        },
    }
    return jsonable_encoder(response)


async def _extract_via_docling_remote(payload: ExtractionRequest) -> dict[str, Any]:
    if not payload.prompt and not payload.schema_payload:
        raise HTTPException(status_code=400, detail="schema or prompt is required")
    if not DOCLING_REMOTE_API_URL:
        raise HTTPException(status_code=500, detail="DOCLING_REMOTE_API_URL is not set")

    schema = _normalize_json_schema(payload.schema_payload, payload.prompt)
    prompt = payload.prompt or "Extract structured data from the document and return JSON."
    downloaded_file = await _download_file(payload.file_url)

    suffix = Path(downloaded_file.filename).suffix or mimetypes.guess_extension(
        (downloaded_file.content_type or "").split(";", 1)[0].strip()
    ) or ".pdf"
    started_at = time.monotonic()
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(downloaded_file.file_bytes)
            temp_path = handle.name

        markdown = await to_thread.run_sync(_run_docling_remote_conversion, temp_path)

        extraction_payload = {
            "model": DOCLING_REMOTE_EXTRACTION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        "Документ уже распознан Docling. Ниже markdown-представление документа.\n"
                        "Верни только валидный JSON по заданной схеме.\n\n"
                        f"{markdown}"
                    ),
                }
            ],
            "temperature": 0,
            "max_tokens": DOCLING_REMOTE_MAX_TOKENS,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "docling_remote_extraction",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        extracted_data, provider_response, _ = await _chat_completion_json(
            endpoint=DOCLING_REMOTE_API_URL,
            headers=_build_openai_headers(DOCLING_REMOTE_API_KEY),
            payload=extraction_payload,
            provider_name="Docling remote extraction",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Docling remote extraction failed")
        detail = f"Docling remote extraction failed: {exc}"
        if not DEBUG_ERRORS and detail:
            detail = str(detail)
        raise HTTPException(status_code=502, detail=detail) from exc
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Failed to remove temp file %s", temp_path)
        elapsed = time.monotonic() - started_at
        logger.info("Docling remote extraction finished in %.2fs", elapsed)

    response = {
        "analysis_id": payload.analysis_id,
        "file_id": payload.file_id,
        "file_type": payload.file_type,
        "backend": "docling_remote",
        "model_parse": DOCLING_REMOTE_MODEL,
        "model_extract": DOCLING_REMOTE_EXTRACTION_MODEL,
        "result": extracted_data,
        "extraction": {"pages": [_build_result_page(extracted_data, raw_text=markdown)]},
        "extraction_metadata": {
            "docling_version": _get_docling_version(),
            "page_count": 1,
            "errors": [],
            "provider": "docling_remote_api",
            "provider_usage": provider_response.get("usage"),
        },
    }
    return jsonable_encoder(response)


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
    payload_json: dict[str, Any] = {
        "model": OPENROUTER_MODEL,
        "messages": _build_openrouter_messages(
            prompt=prompt,
            downloaded_file=downloaded_file,
        ),
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
    if not _looks_like_image(downloaded_file.filename, downloaded_file.content_type):
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
    try:
        extracted_data, provider_response, used_fallback = await _chat_completion_json(
            endpoint=f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
            headers=headers,
            payload=payload_json,
            provider_name="OpenRouter",
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
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("OpenRouter extraction failed")
        detail = f"OpenRouter extraction failed: {exc}"
        if not DEBUG_ERRORS and detail:
            detail = str(detail)
        raise HTTPException(status_code=502, detail=detail) from exc
    finally:
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
            "page_count": 1,
            "errors": [],
            "provider": "openrouter",
            "provider_usage": provider_response.get("usage"),
        },
    }
    return jsonable_encoder(response)


@app.post("/extract")
async def extract_document(payload: ExtractionRequest) -> dict:
    if not payload.schema_payload and not payload.prompt:
        raise HTTPException(status_code=400, detail="schema or prompt is required")

    parsed_url = urlparse(payload.file_url)
    if parsed_url.scheme not in {"http", "https", "ftp", "ftps"}:
        raise HTTPException(status_code=400, detail="file_url must be a public URL")

    backend = _select_backend(payload.backend)
    if backend == "docling_local":
        return await _extract_via_docling_local(payload)
    if backend == "docling_remote":
        return await _extract_via_docling_remote(payload)
    if backend == "openrouter":
        return await _extract_via_openrouter(payload)
    raise HTTPException(status_code=400, detail=f"Unsupported backend: {backend}")
