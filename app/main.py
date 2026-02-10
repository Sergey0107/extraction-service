import json
import logging
import os
import time
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional, Union
from urllib.parse import urlparse

from anyio import to_thread
from docling.datamodel.base_models import InputFormat
from docling.document_extractor import DocumentExtractor
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

logger = logging.getLogger("extraction_service")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


DOC_PARSE_MODEL = _get_env("DOCLING_PARSE_MODEL", "docling")
DOC_EXTRACT_MODEL = _get_env("DOCLING_EXTRACT_MODEL", "docling")
DEBUG_ERRORS = _get_env("DEBUG_ERRORS", "0") == "1"
ALLOWED_FORMATS = (InputFormat.IMAGE, InputFormat.PDF)


class ExtractionRequest(BaseModel):
    file_url: str = Field(..., description="Public file URL")
    prompt: Optional[str] = None
    schema: Optional[Union[dict, str]] = None
    analysis_id: Optional[str] = None
    file_id: Optional[str] = None
    file_type: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    extractor = DocumentExtractor(allowed_formats=list(ALLOWED_FORMATS))
    app.state.docling_extractor = extractor
    logger.info(
        "Docling extractor initialized with formats: %s",
        [fmt.name for fmt in ALLOWED_FORMATS],
    )
    try:
        yield
    finally:
        close = getattr(extractor, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close Docling extractor")


app = FastAPI(lifespan=lifespan, title="extraction-service", version="0.2.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _get_docling_version() -> Optional[str]:
    try:
        return version("docling")
    except PackageNotFoundError:
        return None


def _build_default_template(prompt: Optional[str]) -> dict:
    return {"result": "string"}


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


def _run_extraction(
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


@app.post("/extract")
async def extract_document(payload: ExtractionRequest) -> dict:
    if not payload.schema and not payload.prompt:
        raise HTTPException(status_code=400, detail="schema or prompt is required")

    parsed_url = urlparse(payload.file_url)
    if parsed_url.scheme not in {"http", "https", "ftp", "ftps"}:
        raise HTTPException(status_code=400, detail="file_url must be a public URL")

    template = _normalize_template(payload.schema, payload.prompt)

    started_at = time.monotonic()
    try:
        result = await to_thread.run_sync(
            _run_extraction,
            app.state.docling_extractor,
            payload.file_url,
            template,
        )
    except Exception as exc:
        logger.exception("Docling extraction failed")
        detail = "Docling extraction failed"
        if DEBUG_ERRORS:
            detail = f"{detail}: {exc}"
        raise HTTPException(status_code=502, detail=detail) from exc
    finally:
        elapsed = time.monotonic() - started_at
        logger.info("Extraction finished in %.2fs", elapsed)

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
