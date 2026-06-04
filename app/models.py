"""Request schema and internal data containers for the extraction service."""

from dataclasses import dataclass
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


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
