import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Union
from urllib.parse import urlparse

from anyio import to_thread
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
import landingai_ade
from landingai_ade import LandingAIADE
from pydantic import BaseModel, Field

logger = logging.getLogger("extraction_service")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


PARSE_MODEL = _get_env("ADE_PARSE_MODEL", "dpt-2-latest")
EXTRACT_MODEL = _get_env("ADE_EXTRACT_MODEL", "extract-latest")
ADE_ENV = _get_env("ADE_ENV")
DEBUG_ERRORS = _get_env("DEBUG_ERRORS", "0") == "1"


class ExtractionRequest(BaseModel):
    file_url: str = Field(..., description="S3 file URL")
    prompt: Optional[str] = None
    schema: Optional[Union[dict, str]] = None
    analysis_id: Optional[str] = None
    file_id: Optional[str] = None
    file_type: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = _get_env("VISION_AGENT_API_KEY")
    client_kwargs = {}
    if api_key:
        client_kwargs["apikey"] = api_key
    if ADE_ENV:
        client_kwargs["environment"] = ADE_ENV
    client = LandingAIADE(**client_kwargs)
    app.state.ade_client = client
    if api_key:
        logger.info("VISION_AGENT_API_KEY is set")
    else:
        logger.warning("VISION_AGENT_API_KEY is not set")
    try:
        yield
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close ADE client")


app = FastAPI(lifespan=lifespan, title="extraction-service", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _build_schema(prompt: Optional[str]) -> str:
    schema = {
        "type": "object",
        "required": ["result"],
        "properties": {
            "result": {
                "type": "string",
                "description": prompt or "Extracted data",
            }
        },
    }
    return json.dumps(schema, ensure_ascii=True)


def _run_extraction(client: LandingAIADE, file_url: str, schema_json: str):
    parse_response = client.parse(
        document_url=file_url,
        model=PARSE_MODEL,
    )
    return client.extract(
        schema=schema_json,
        markdown=parse_response.markdown,
        model=EXTRACT_MODEL,
    )


@app.post("/extract")
async def extract_document(payload: ExtractionRequest) -> dict:
    if not payload.schema and not payload.prompt:
        raise HTTPException(status_code=400, detail="schema or prompt is required")

    parsed_url = urlparse(payload.file_url)
    if parsed_url.scheme not in {"http", "https", "ftp", "ftps"}:
        raise HTTPException(status_code=400, detail="file_url must be a public URL")

    if isinstance(payload.schema, dict):
        schema_json = json.dumps(payload.schema, ensure_ascii=True)
    elif isinstance(payload.schema, str):
        schema_json = payload.schema
    else:
        schema_json = _build_schema(payload.prompt)

    started_at = time.monotonic()
    try:
        extract_response = await to_thread.run_sync(
            _run_extraction,
            app.state.ade_client,
            payload.file_url,
            schema_json,
        )
    except landingai_ade.APIError as exc:
        logger.exception("ADE API error")
        status = getattr(exc, "status_code", None)
        if status:
            detail = f"ADE API error ({status}): {exc}"
        else:
            detail = f"ADE API error: {exc}"
        raise HTTPException(status_code=502, detail=detail) from exc
    except Exception as exc:
        logger.exception("Extraction failed")
        detail = "Extraction failed"
        if DEBUG_ERRORS:
            detail = f"{detail}: {exc}"
        raise HTTPException(status_code=500, detail=detail) from exc
    finally:
        elapsed = time.monotonic() - started_at
        logger.info("Extraction finished in %.2fs", elapsed)

    response = {
        "analysis_id": payload.analysis_id,
        "file_id": payload.file_id,
        "file_type": payload.file_type,
        "model_parse": PARSE_MODEL,
        "model_extract": EXTRACT_MODEL,
        "extraction": extract_response.extraction,
        "extraction_metadata": extract_response.extraction_metadata,
    }
    return jsonable_encoder(response)
