"""Environment-driven configuration and optional-dependency probing.

All runtime tunables live here as module-level constants resolved from the
environment at import time. Optional heavy dependencies (PyMuPDF, Docling) are
probed once and exposed as ``*_INSTALLED`` flags so the rest of the service can
degrade gracefully when they are absent.
"""

import logging
import os
from typing import Any, Optional

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


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def get_int_env(name: str, default: int) -> int:
    value = get_env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r, using %s", name, value, default)
        return default


def get_list_env(name: str) -> list[str]:
    value = get_env(name)
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


DOC_PARSE_MODEL = get_env("DOCLING_PARSE_MODEL", "docling")
DOC_EXTRACT_MODEL = get_env("DOCLING_EXTRACT_MODEL", "docling")
EXTRACTION_BACKEND = (get_env("EXTRACTION_BACKEND", "openrouter") or "openrouter").strip().lower()
DEBUG_ERRORS = get_env("DEBUG_ERRORS", "0") == "1"
FILE_DOWNLOAD_TIMEOUT_SECONDS = get_int_env("FILE_DOWNLOAD_TIMEOUT_SECONDS", 300)
REMOTE_API_TIMEOUT_SECONDS = get_int_env("REMOTE_API_TIMEOUT_SECONDS", 600)

DOCLING_REMOTE_API_URL = get_env("DOCLING_REMOTE_API_URL")
DOCLING_REMOTE_VLM_URL = get_env("DOCLING_REMOTE_VLM_URL", DOCLING_REMOTE_API_URL)
DOCLING_REMOTE_LLM_URL = get_env("DOCLING_REMOTE_LLM_URL", DOCLING_REMOTE_API_URL)
DOCLING_REMOTE_API_KEY = get_env("DOCLING_REMOTE_API_KEY")
DOCLING_REMOTE_MODEL = get_env("DOCLING_REMOTE_MODEL", "ibm-granite/granite-docling-258M")
DOCLING_REMOTE_EXTRACTION_MODEL = (
    get_env("DOCLING_REMOTE_EXTRACTION_MODEL", DOCLING_REMOTE_MODEL)
    or DOCLING_REMOTE_MODEL
)
DOCLING_REMOTE_PRESET = get_env("DOCLING_REMOTE_PRESET", "granite_docling")
DOCLING_REMOTE_MAX_TOKENS = get_int_env("DOCLING_REMOTE_MAX_TOKENS", 4000)

OPENROUTER_API_KEY = get_env("OPENROUTER_API_KEY")
OPENROUTER_MODEL = get_env("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_PDF_ENGINE = get_env("OPENROUTER_PDF_ENGINE", "mistral-ocr")
OPENROUTER_MAX_TOKENS = min(get_int_env("OPENROUTER_MAX_TOKENS", 4000), 8000)
OPENROUTER_APP_NAME = get_env("OPENROUTER_APP_NAME", "extraction-service")
OPENROUTER_SITE_URL = get_env("OPENROUTER_SITE_URL", "http://localhost:8005")
OPENROUTER_PROVIDER_ORDER = get_list_env("OPENROUTER_PROVIDER_ORDER")
OPENROUTER_PROVIDER_IGNORE = get_list_env("OPENROUTER_PROVIDER_IGNORE")
GEOMETRY_ENRICHMENT_ENABLED = get_env("GEOMETRY_ENRICHMENT_ENABLED", "1") != "0"

LLAMAPARSE_API_KEY = get_env("LLAMAPARSE_API_KEY")
LLAMAPARSE_BASE_URL = get_env("LLAMAPARSE_BASE_URL", "https://api.cloud.llamaindex.ai")
LLAMAPARSE_RESULT_TYPE = get_env("LLAMAPARSE_RESULT_TYPE", "markdown")
LLAMAPARSE_POLLING_INTERVAL = get_int_env("LLAMAPARSE_POLLING_INTERVAL", 3)
LLAMAPARSE_MAX_WAIT_SECONDS = get_int_env("LLAMAPARSE_MAX_WAIT_SECONDS", 180)
LLAMAPARSE_LANGUAGE = get_env("LLAMAPARSE_LANGUAGE", "ru")
LLAMAPARSE_REQUEST_TIMEOUT_SECONDS = get_int_env("LLAMAPARSE_REQUEST_TIMEOUT_SECONDS", 120)
LLAMAPARSE_MAX_RETRIES = get_int_env("LLAMAPARSE_MAX_RETRIES", 2)

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
