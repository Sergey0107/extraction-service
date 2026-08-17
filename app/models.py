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
    # job_id — id extraction_job на стороне api-gateway. Используется только
    # backend'ом yandex_vision_ocr в async-режиме (см. _extract_via_paddleocr_vl):
    # передаётся дальше в paddleocr-vl-service как job_id для идемпотентного
    # старта job'а (тот же job_id => тот же job, не дублируется при повторной
    # отправке), и не имеет смысла для остальных синхронных backend'ов.
    job_id: Optional[str] = None
    # True — извлечение асинхронное: /extract вернёт {"async": true, "job_id": ...}
    # немедленно вместо ожидания полного результата. Поддерживается только
    # backend=yandex_vision_ocr; для остальных backend'ов игнорируется (они
    # остаются полностью синхронными — нет длинной облачной OCR-цепочки,
    # которая оправдывала бы усложнение).
    async_mode: bool = False
    # Модель изделия, указанная пользователем при загрузке (необязательно).
    # Для openrouter уже используется в промпте через _build_product_model_appendix.
    # Для yandex_vision_ocr/paddleocr_vl прокидывается в paddleocr-vl-service как
    # fallback-источник variant, когда явного заголовка модели на странице нет и
    # document_outline не даёт однозначного ответа — см. _extract_via_paddleocr_vl.
    product_model: Optional[str] = None
    # Имена характеристик, одобренных пользователем в ТЗ (только для паспорта).
    # openrouter получает их внутри prompt; yandex_vision_ocr/paddleocr_vl промпт
    # не читают, поэтому список идёт отдельным полем и передаётся дальше в
    # paddleocr-vl-service как контракт «верни ровно эти имена дословно».
    # Без него паспорт извлекается «вслепую», модель придумывает свои
    # формулировки, и сопоставление с ТЗ становится вероятностным (замер на
    # ДЖАМБО 60/35: 7 из 15 требований ТЗ не доходили до таблицы сравнения).
    target_characteristic_names: Optional[list[str]] = None


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
    # Прямоугольник системы координат, в которой заданы rect слов. Для обычного
    # текстового слоя — None (берётся page.rect). Для OCR-индекса с авто-поворотом
    # страницы — прямоугольник ВЫПРЯМЛЕННОГО изображения (в PDF-пунктах), чтобы
    # нормализованные координаты bbox совпали с тем, что показывает вьювер.
    page_rect: Any = None
