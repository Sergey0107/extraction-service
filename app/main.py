import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from app.logging_setup import configure_logging

configure_logging("extraction-service")

import httpx
from anyio import to_thread
from docx import Document as WordDocument
from docx.table import Table as WordTable
from docx.text.paragraph import Paragraph as WordParagraph
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.encoders import jsonable_encoder

from app.config import (
    ACTIVE_BACKENDS,
    DEBUG_ERRORS,
    DOCLING_INSTALLED,
    EXTRACTION_BACKEND,
    FILE_DOWNLOAD_TIMEOUT_SECONDS,
    GENERIC_MIME_TYPES,
    HEAVY_WORK_CONCURRENCY,
    RENDER_PDF_DPI,
    RENDER_PDF_JPEG_QUALITY,
    GEOMETRY_ENRICHMENT_ENABLED,
    IMAGE_SUFFIXES,
    LLAMAPARSE_API_KEY,
    LLAMAPARSE_BASE_URL,
    LLAMAPARSE_LANGUAGE,
    LLAMAPARSE_MAX_RETRIES,
    LLAMAPARSE_MAX_WAIT_SECONDS,
    LLAMAPARSE_POLLING_INTERVAL,
    LLAMAPARSE_REQUEST_TIMEOUT_SECONDS,
    LLAMAPARSE_RESULT_TYPE,
    LLM_IS_YANDEX,
    MINERU_API_BASE,
    MINERU_ENABLED,
    MINERU_LANGUAGE,
    MINERU_MODEL_VERSION,
    MINERU_PAGE_RANGES,
    MINERU_POLL_INTERVAL,
    MINERU_POLL_TIMEOUT,
    MINERU_TOKEN,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_MAX_TOKENS,
    OPENROUTER_MODEL,
    OPENROUTER_PDF_ENGINE,
    OPENROUTER_PROVIDER_IGNORE,
    OPENROUTER_PROVIDER_ORDER,
    OPENROUTER_SITE_URL,
    PADDLEOCR_VL_ENABLED,
    PADDLEOCR_VL_REQUEST_TIMEOUT_SECONDS,
    PADDLEOCR_VL_SERVICE_URL,
    PDFPLUMBER_INSTALLED,
    PYMUPDF_INSTALLED,
    REMOTE_API_TIMEOUT_SECONDS,
    STUBBED_BACKENDS,
    SUPPORTED_BACKENDS,
    VISION_DPI,
    VISION_MAX_TABLE_PAGES,
    VISION_MODEL,
    VISION_TABLES_ENABLED,
    YANDEX_VISION_OCR_ENABLED,
    fitz,
    logger,
)
from app.mineru_client import MineruError, extract_tables_mineru
from app.models import DownloadedFile, ExtractionRequest, PdfPageIndex, PdfWord
from app.pdfplumber_extractor import build_llm_payload as build_pdfplumber_llm_payload
from app.pdfplumber_extractor import extract_pdf_geometry
from app.pdfplumber_extractor import filter_relevant_pages as filter_pdfplumber_pages
from app.pdfplumber_extractor import plain_text_for_quality_check
from app.url_guard import UrlNotAllowed, validate_public_url
from app.schema_utils import normalize_json_schema


_strict_schema_supported: dict[str, bool] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.docling_extractor = None
    try:
        to_thread.current_default_thread_limiter().total_tokens = HEAVY_WORK_CONCURRENCY
        logger.info("Heavy-work concurrency limited to %s", HEAVY_WORK_CONCURRENCY)
    except Exception:
        logger.warning("Failed to set thread limiter", exc_info=True)
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
        "mineru_configured": bool(MINERU_ENABLED and MINERU_TOKEN),
        "vision_hybrid_enabled": VISION_TABLES_ENABLED,
        "pdfplumber_installed": PDFPLUMBER_INSTALLED,
    }


def _get_docling_version() -> Optional[str]:
    if not DOCLING_INSTALLED:
        return None
    try:
        return version("docling")
    except PackageNotFoundError:
        return None


def _redact_url(url: str) -> str:
    """Drop query string (presigned S3 signatures) before logging a URL."""
    return url.split("?", 1)[0] if url else url


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


def _looks_like_excel(filename: str, content_type: str | None) -> bool:
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    suffix = Path(filename).suffix.lower()
    return (
        normalized
        in {
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        or suffix in {".xls", ".xlsx", ".xlsm"}
    )


def _looks_like_pdf(filename: str, content_type: str | None) -> bool:
    normalized = (content_type or "").split(";", 1)[0].strip().lower()
    suffix = Path(filename).suffix.lower()
    return normalized == "application/pdf" or suffix == ".pdf"


# Таймаут на ОДИН вызов Tesseract (на страницу). Нормальный OCR страницы — секунды;
# но на некоторых сканах с шумом/паттернами Tesseract LSTM зацикливается и висит
# минутами, блокируя всю задачу извлечения навсегда. Лимит превращает зависание в
# обрабатываемую ошибку (страница пропускается / падаем на другой путь).
OCR_PAGE_TIMEOUT_SECONDS = 45
OCR_OSD_TIMEOUT_SECONDS = 20


def _auto_orient_for_ocr(img):
    """Доворачивает изображение страницы в правильную ориентацию перед OCR.

    Некоторые PDF имеют повёрнутые страницы (rotation 90/270): после рендера в
    картинку текст оказывается боком/вверх ногами, и Tesseract выдаёт мусор.
    Через OSD определяем угол и доворачиваем. При ошибке/низкой уверенности —
    возвращаем как есть (не хуже прежнего поведения)."""
    try:
        import pytesseract

        osd = pytesseract.image_to_osd(
            img, output_type=pytesseract.Output.DICT, timeout=OCR_OSD_TIMEOUT_SECONDS
        )
        rotate = int(osd.get("rotate", 0) or 0)
        conf = float(osd.get("orientation_conf", 0) or 0)
        if rotate and conf >= 1.0:
            # PIL.rotate крутит против часовой; OSD 'rotate' — на сколько повернуть
            # ПО часовой, чтобы выпрямить → используем expand и отрицательный угол.
            return img.rotate(-rotate, expand=True)
    except Exception:
        pass
    return img


def _text_layer_is_usable(text: str) -> bool:
    """Проверяет, что извлечённый текстовый слой ОСМЫСЛЕННЫЙ, а не «мусор».

    У некоторых PDF шрифт со сломанной кодировкой (нет/битый ToUnicode): визуально
    документ читается, но get_text() возвращает мешанину одиночных символов вроде
    'u E e 5 ^s * 2 FE'. Такой текст нельзя слать в LLM — он не найдёт ни одной
    характеристики. Признак мусора: мало «слов» (последовательностей букв >=3).
    Тогда лучше упасть на OCR (он читает отрисованные глифы, а не битую кодировку).
    """
    stripped = text.strip()
    if len(stripped) < 200:
        # Слишком мало текста, чтобы судить о качестве — не блокируем (OCR решит по длине).
        return len(stripped) > 20
    letters = re.findall(r"[^\W\d_]", stripped, re.UNICODE)
    if not letters:
        return False
    words = re.findall(r"[^\W\d_]{3,}", stripped, re.UNICODE)
    letters_in_words = sum(len(w) for w in words)
    word_ratio = letters_in_words / len(letters)
    # У нормального текста (RU/EN) word_ratio ~0.9; у мусора одиночных символов ~0.4-0.5.
    if word_ratio < 0.6:
        return False

    # Битая кодировка кириллического шрифта (нет/сломан ToUnicode): get_text() даёт
    # «слова», визуально читаемые как русский, но это мусор. word_ratio при этом
    # высокий, и такой текст проходил проверку выше. Ловим двумя признаками:
    if len(words) >= 20:
        def _is_cyr(ch: str) -> bool:
            return "Ѐ" <= ch <= "ӿ"

        def _is_lat(ch: str) -> bool:
            return ("a" <= ch <= "z") or ("A" <= ch <= "Z")

        cyrillic = sum(1 for ch in letters if _is_cyr(ch))
        cyrillic_ratio = cyrillic / len(letters)

        mixed_case = 0      # 'HanuenoaaHne' — заглавная не в начале слова
        mixed_script = 0    # 'ВзiпIеll' — кириллица И латиница в одном слове
        for w in words:
            inner = w[1:]
            if any(ch.isupper() for ch in inner) and any(ch.islower() for ch in inner):
                mixed_case += 1
            has_cyr = any(_is_cyr(ch) for ch in w)
            has_lat = any(_is_lat(ch) for ch in w)
            if has_cyr and has_lat:
                mixed_script += 1
        mixed_case_ratio = mixed_case / len(words)
        mixed_script_ratio = mixed_script / len(words)

        # 1. Латиница-вместо-кириллицы ('HanuenoaaHne'): кириллицы почти нет,
        #    но много «рваного» регистра.
        if cyrillic_ratio < 0.15 and mixed_case_ratio >= 0.2:
            return False
        # 2. Кириллица + латиница намешаны в одних словах ('ВзiпIеll', 'llсilrr'):
        #    в нормальном тексте смешение скриптов внутри слова — единичные случаи
        #    (бренды вроде 'iPhone'), у битой кодировки — массово.
        if mixed_script_ratio >= 0.15:
            return False

    return True


def _text_layer_has_enough_content(all_pages_text: list[str]) -> bool:
    """Проверяет, что текстового слоя ДОСТАТОЧНО для извлечения, а не несколько
    символов на скан-страницах (типичный даташит-картинка с одним артикулом в
    углу как текст, остальное — изображение). Без этой проверки система считает
    «150TBS11 × 3» годным слоем и НЕ запускает OCR → LLM получает пустоту.

    Критерий: уникального текста должно быть достаточно относительно числа
    непустых страниц. Учитываем дублирование (на каждой странице один артикул)."""
    non_empty = [t.strip() for t in all_pages_text if t and t.strip()]
    if not non_empty:
        return False
    # Уникальные строки контента (убираем повторяющиеся колонтитулы/артикулы)
    unique_lines: set[str] = set()
    for page_text in non_empty:
        for line in page_text.splitlines():
            line = line.strip()
            if len(line) >= 3:
                unique_lines.add(line)
    unique_chars = sum(len(line) for line in unique_lines)
    page_count = len(all_pages_text) or 1
    # Минимум ~60 уникальных символов на страницу — иначе это скан без текста.
    return unique_chars >= 60 * page_count and unique_chars >= 120


def _page_words_are_usable(pages: list["PdfPageIndex"]) -> bool:
    """Проверяет, что слова в текстовом индексе страниц — осмысленные, а не мусор.

    Используется геометрией: текстовый слой может «существовать» (слова есть), но
    из-за битой кодировки шрифта это мешанина одиночных символов. По такому индексу
    цитаты не находятся → 0 координат → клик по характеристике ничего не делает.
    Если слова мусорные — геометрия должна перейти на OCR-индекс."""
    sample = " ".join(
        word.text
        for page in pages
        for word in page.words
    )
    return _text_layer_is_usable(sample)


def _page_words_are_enough(pages: list["PdfPageIndex"]) -> bool:
    """Достаточно ли уникальных слов в индексе. Скан-PDF с одним артикулом-текстом
    на странице (остальное — картинка) даёт пару слов на лист — этого мало для
    привязки цитат, нужен OCR-индекс."""
    unique_words = {
        word.normalized
        for page in pages
        for word in page.words
        if len(word.normalized) >= 2
    }
    page_count = len(pages) or 1
    return len(unique_words) >= 8 * page_count and len(unique_words) >= 15


def _extract_target_names_from_prompt(prompt: str) -> list[str] | None:
    """Extracts target characteristic names from the prompt JSON list, if present."""
    match = re.search(r"characteristic names to return.*?\n\s*\[", prompt, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    start = prompt.index("[", match.start())
    try:
        names = json.loads(prompt[start:prompt.index("]", start) + 1])
        if isinstance(names, list):
            return [str(n) for n in names if isinstance(n, str) and n.strip()]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


_RELEVANCE_KEYWORDS = re.compile(
    r"характерист|параметр|показател|спецификац|specification|parameter|"
    r"техничес|nominal|dimension|габарит|размер|масс[аы]|вес\b|"
    r"мощност|напор|подач[аи]|производител|давлен|температур|"
    r"частот|оборот|диаметр|длин[аы]|ширин[аы]|высот[аы]|"
    r"марк[аи]|модел[ьи]|тип\b|обозначен",
    re.IGNORECASE,
)


def _page_is_relevant(text: str, target_names: list[str] | None = None) -> bool:
    if _RELEVANCE_KEYWORDS.search(text):
        return True
    if target_names:
        text_lower = text.lower().replace("ё", "е")
        for name in target_names:
            name_words = [w for w in re.split(r"[\s:,]+", name.lower().replace("ё", "е")) if len(w) >= 3]
            if name_words and sum(1 for w in name_words if w in text_lower) >= max(1, len(name_words) // 2):
                return True
    return False


def _page_has_table(text: str) -> bool:
    """Страница содержит Markdown-таблицу (вставленную _page_text_with_tables):
    есть строка-разделитель «| --- | ...». Самый дешёвый и точный признак."""
    return bool(re.search(r"^\s*\|[\s|:-]*---[\s|:-]*\|", text, re.MULTILINE))


def _page_continues_table(text: str) -> bool:
    """Страница ВЫГЛЯДИТ как продолжение таблицы: начинается со строк «| ... |»,
    но без строки-заголовка-разделителя «| --- |». Так бывает, когда таблица
    переходит на следующую страницу — шапка осталась на предыдущей."""
    stripped = text.lstrip()
    starts_with_row = stripped.startswith("|")
    return starts_with_row and not _page_has_table(text)


def _render_table_pages_to_images(
    local_path: str,
    *,
    dpi: int = 200,
    max_pages: int = 8,
) -> list[dict[str, Any]]:
    """Рендерит страницы PDF, содержащие таблицы, в base64-PNG для vision-модели.

    Отбирает страницы, где PyMuPDF находит таблицы, ранжирует по «табличной
    плотности» (число ячеек) и берёт не более ``max_pages`` самых насыщенных.
    Возвращает список content-частей вида {type:image_url, image_url:{url:data...},
    page_number:N} для добавления в user-message. При любой ошибке — пустой список
    (гибрид деградирует к текстовому пути)."""
    if not PYMUPDF_INSTALLED or fitz is None:
        return []
    try:
        doc = fitz.open(local_path)
    except Exception:  # noqa: BLE001
        logger.warning("vision-hybrid: cannot open PDF %s", local_path)
        return []
    scored: list[tuple[int, int]] = []  # (cell_count, page_index)
    try:
        for i in range(doc.page_count):
            try:
                tables = doc.load_page(i).find_tables().tables
            except Exception:  # noqa: BLE001
                continue
            cells = 0
            for t in tables:
                try:
                    cells += len(t.rows) * len(t.header.names or [])
                except Exception:  # noqa: BLE001
                    cells += 1
            if cells > 0:
                scored.append((cells, i))
        scored.sort(reverse=True)
        selected = sorted(idx for _, idx in scored[:max_pages])
        parts: list[dict[str, Any]] = []
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for idx in selected:
            pix = doc.load_page(idx).get_pixmap(matrix=mat, alpha=False)
            b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
                "page_number": idx + 1,
            })
        return parts
    except Exception:  # noqa: BLE001
        logger.exception("vision-hybrid: rendering table pages failed")
        return []
    finally:
        doc.close()


def _filter_relevant_pages(
    all_pages_text: list[str],
    target_names: list[str] | None = None,
    max_chars: int = 30000,
) -> tuple[list[tuple[int, str]], bool]:
    """Returns ([(page_number, text), ...], was_filtered).
    Keeps first page, last page, and any page with characteristic-related keywords.
    Falls back to all pages if total text is small enough.

    Пункт 4 — НЕ разрывать таблицы фильтром: страница с таблицей всегда релевантна,
    а её соседи (предыдущая/следующая) подтягиваются целиком, чтобы шапку таблицы
    не оторвало от строк-продолжения на соседней странице."""
    total_len = sum(len(t) for t in all_pages_text)
    if total_len <= max_chars:
        return [(i + 1, t) for i, t in enumerate(all_pages_text) if t.strip()], False

    n = len(all_pages_text)
    keep: set[int] = set()
    for i, text in enumerate(all_pages_text):
        if not text.strip():
            continue
        if i == 0 or i == n - 1:
            keep.add(i)
            continue
        if _page_has_table(text) or _page_is_relevant(text, target_names):
            keep.add(i)

    # Целостность таблиц: для каждой страницы с таблицей или её продолжением
    # подтягиваем непустых соседей, чтобы заголовок и строки не разорвались.
    for i, text in enumerate(all_pages_text):
        if not text.strip():
            continue
        if _page_has_table(text) or _page_continues_table(text):
            for j in (i - 1, i + 1):
                if 0 <= j < n and all_pages_text[j].strip():
                    keep.add(j)

    relevant = [(i + 1, all_pages_text[i]) for i in sorted(keep)]

    if not relevant:
        return [(i + 1, t) for i, t in enumerate(all_pages_text) if t.strip()], False

    # If filtering cut too aggressively (< 3 pages), fall back to full text
    if len(relevant) < 3 and n > 5:
        return [(i + 1, t) for i, t in enumerate(all_pages_text) if t.strip()], False

    return relevant, True


def _augment_sparse_pages_with_ocr(
    document: Any,
    all_pages_text: list[str],
    *,
    sparse_threshold: int = 150,
    min_ocr_gain: int = 200,
) -> tuple[list[str], int]:
    """Для страниц с бедным текстовым слоем прогоняет OCR и подменяет их текст.

    Кейс: документ в основном текстовый (описание читается), но таблицы ТТХ
    нарисованы векторной графикой/картинкой без ToUnicode — get_text() для такой
    страницы почти пуст, и характеристики теряются. Если OCR такой страницы даёт
    существенно больше текста (min_ocr_gain), берём OCR-вариант.

    Возвращает (обновлённый список текстов, число заменённых страниц)."""
    sparse_indices = [
        i for i, t in enumerate(all_pages_text)
        if len((t or "").strip()) < sparse_threshold
    ]
    if not sparse_indices:
        return all_pages_text, 0
    try:
        import pytesseract
        from PIL import Image
        import io
        pytesseract.get_tesseract_version()
    except Exception:
        return all_pages_text, 0

    DPI = 200
    updated = list(all_pages_text)
    replaced = 0
    for i in sparse_indices:
        try:
            page = document.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img = _auto_orient_for_ocr(img)
            ocr_text = pytesseract.image_to_string(
                img, lang="rus+eng", timeout=OCR_PAGE_TIMEOUT_SECONDS
            ).strip()
        except Exception:
            continue
        if len(ocr_text) >= len((updated[i] or "").strip()) + min_ocr_gain:
            updated[i] = ocr_text
            replaced += 1
    return updated, replaced


def _cell_is_numeric(value: str) -> bool:
    """Ячейка — числовое значение (а не подпись): после удаления единиц/скобок
    осталось в основном число."""
    s = value.strip()
    if not s:
        return False
    return bool(re.fullmatch(r"[\d.,()\s/x×*+\-±]+", s))


def _serialize_table_markdown(rows: list[list[Any]]) -> str:
    """Сериализует таблицу (от find_tables().extract()) в Markdown.

    Многоуровневые заголовки СХЛОПЫВАЮТСЯ в один: верхний уровень растягивается
    на «протянутые» (None) ячейки справа, затем уровни склеиваются по столбцам.
    Пример: ('Подача', None, 'Напор') + ('м3/ч', 'л/с', 'м') →
    'Подача, м3/ч | Подача, л/с | Напор, м'. Иначе LLM путает подколонки
    (брал л/с вместо напора). Пустые ячейки (None) → ''; '|' внутри → '/'."""
    def _cell(value: Any) -> str:
        text = "" if value is None else str(value)
        return re.sub(r"\s+", " ", text).replace("|", "/").strip()

    raw = [row for row in rows if row]
    if not raw:
        return ""
    width = max(len(r) for r in raw)
    raw = [list(r) + [None] * (width - len(r)) for r in raw]

    # Сколько верхних строк — заголовки: строка-заголовок почти не содержит
    # числовых ячеек (это подписи колонок/единицы). Максимум 2 уровня.
    def _is_header_row(row: list[Any]) -> bool:
        cells = [_cell(c) for c in row if _cell(c)]
        if not cells:
            return True  # пустая «протяжка» — часть шапки
        numeric = sum(1 for c in cells if _cell_is_numeric(c))
        return numeric <= max(0, len(cells) // 3)

    # Первая строка — всегда заголовок. Вторая считается ВТОРЫМ уровнем шапки
    # только если она похожа на под-заголовки (мало чисел) И её первая ячейка
    # пустая — т.е. под колонкой-ключом («Типоразмер») нет под-подписи. Иначе это
    # уже строка данных (на стр.5 первая «ячейка» второго ряда = коды моделей).
    header_depth = 1
    if len(raw) >= 2 and _is_header_row(raw[1]) and not _cell(raw[1][0]):
        header_depth = 2

    # «Протягиваем» верхний уровень вправо по None-ячейкам (объединённые шапки).
    header_levels: list[list[str]] = []
    for level in raw[:header_depth]:
        filled: list[str] = []
        last = ""
        for c in level:
            txt = _cell(c)
            if txt:
                last = txt
                filled.append(txt)
            else:
                # None в исходнике = продолжение объединённой ячейки слева.
                filled.append(last if c is None else "")
        header_levels.append(filled)

    # Склеиваем уровни по столбцам, убирая дубли (если уровни совпали).
    merged_header: list[str] = []
    for col in range(width):
        parts: list[str] = []
        for lvl in header_levels:
            val = lvl[col] if col < len(lvl) else ""
            if val and val not in parts:
                parts.append(val)
        merged_header.append(", ".join(parts))

    out = [
        "| " + " | ".join(merged_header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in raw[header_depth:]:
        out.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(out)


def _page_text_with_tables(page: Any) -> str:
    """Текст страницы, где таблицы заменены Markdown-представлением.

    Плоский get_text() в таблицах склеивает значения пробелами, заголовки
    оторваны от строк — LLM угадывает «модель→колонка→значение» и ошибается на
    широких таблицах. find_tables() распознаёт сетку, и мы подаём её как Markdown.
    Нетабличный текст сохраняется в исходном вертикальном порядке. При любой
    ошибке — фолбэк на плоский sorted-текст (надёжность важнее)."""
    try:
        tables_finder = page.find_tables()
        tables = list(tables_finder.tables)
    except Exception:
        return page.get_text("text", sort=True)

    if not tables:
        return page.get_text("text", sort=True)

    table_bboxes = []
    table_segments = []  # (y0, markdown)
    for table in tables:
        try:
            rows = table.extract()
            md = _serialize_table_markdown(rows)
            bbox = fitz.Rect(table.bbox)
        except Exception:
            continue
        if md:
            table_bboxes.append(bbox)
            table_segments.append((float(bbox.y0), md))

    if not table_segments:
        return page.get_text("text", sort=True)

    # Текстовые блоки вне областей таблиц (чтобы не дублировать табличный текст).
    text_segments = []  # (y0, text)
    try:
        blocks = page.get_text("blocks", sort=True)
    except Exception:
        blocks = []
    for block in blocks:
        if len(block) < 5:
            continue
        x0, y0, x1, y1, btext = block[0], block[1], block[2], block[3], block[4]
        if not isinstance(btext, str) or not btext.strip():
            continue
        block_rect = fitz.Rect(x0, y0, x1, y1)
        # Пропускаем блоки, существенно пересекающиеся с любой таблицей.
        inside_table = False
        for tb in table_bboxes:
            inter = block_rect & tb
            if inter.is_valid and inter.get_area() > 0.5 * block_rect.get_area():
                inside_table = True
                break
        if not inside_table:
            text_segments.append((float(y0), btext.strip()))

    # Собираем в вертикальном порядке: текстовые блоки и таблицы вперемешку по Y.
    combined = sorted(text_segments + table_segments, key=lambda s: s[0])
    parts = [seg[1] for seg in combined]
    result = "\n".join(parts).strip()
    return result or page.get_text("text", sort=True)


def _convert_pdf_to_structured_text(
    local_path: str,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    """Извлекает текст из PDF для передачи в LLM.

    Для больших документов (>30K символов) фильтрует только страницы с
    релевантным содержимым (таблицы характеристик), чтобы сократить prompt.
    """
    if not PYMUPDF_INSTALLED:
        return {"text": "", "page_count": 0, "ocr_applied": False, "error": "PyMuPDF not installed"}

    document = fitz.open(local_path)
    page_count = document.page_count
    lines: list[str] = []
    ocr_applied = False

    all_pages_text: list[str] = []
    for i in range(page_count):
        page = document.load_page(i)
        # Таблицы подаём как Markdown (явная привязка значение→столбец), остальной
        # текст — как обычно. На сканах/OCR ветка ниже не затрагивается.
        text = _page_text_with_tables(page).strip()
        all_pages_text.append(text)

    combined_text = "\n".join(t for t in all_pages_text if t)
    has_enough = _text_layer_has_enough_content(all_pages_text)
    has_text = (
        len(combined_text) > 20
        and _text_layer_is_usable(combined_text)
        and has_enough
    )
    if len(combined_text) > 20 and not has_text:
        if not has_enough:
            logger.info(
                "PDF text layer too sparse (%d chars over %d pages — likely scanned) — falling back to OCR",
                len(combined_text), page_count,
            )
        else:
            logger.info("PDF text layer looks garbled (broken font encoding) — falling back to OCR")

    pages_filtered = False
    if has_text:
        # «Смешанный» PDF: часть страниц имеет нормальный текстовый слой (описание),
        # а ключевые таблицы характеристик нарисованы как векторная графика/картинка
        # без ToUnicode — их get_text() почти пуст, и LLM не видит значений. Для таких
        # «бедных» страниц прогоняем OCR и подменяем их текст распознанным.
        all_pages_text, augmented = _augment_sparse_pages_with_ocr(
            document, all_pages_text
        )
        if augmented:
            ocr_applied = True
            logger.info(
                "PDF mixed-content: OCR-augmented %d sparse page(s) with image/vector tables",
                augmented,
            )
        relevant_pages, pages_filtered = _filter_relevant_pages(
            all_pages_text, target_names
        )
        if pages_filtered:
            logger.info(
                "PDF page filter: %d/%d pages selected (total %d chars → %d chars)",
                len(relevant_pages), page_count,
                sum(len(t) for t in all_pages_text),
                sum(len(t) for _, t in relevant_pages),
            )
        for page_num, text in relevant_pages:
            lines.append(f"[PAGE {page_num}]")
            lines.append(text)
    else:
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
                    img = _auto_orient_for_ocr(img)
                    try:
                        text = pytesseract.image_to_string(
                            img, lang="rus+eng", timeout=OCR_PAGE_TIMEOUT_SECONDS
                        ).strip()
                    except RuntimeError as exc:
                        logger.warning("OCR timeout/error on page %d: %s", i + 1, exc)
                        text = ""
                    if text:
                        lines.append(f"[PAGE {i + 1}]")
                        lines.append(text)
                ocr_applied = True
            except Exception as exc:
                logger.warning("PDF OCR failed: %s", exc)

    document.close()
    raw_text = "\n".join(lines).strip()
    clean_text = raw_text.encode("utf-8", errors="replace").decode("utf-8")
    return {
        "text": clean_text,
        "page_count": page_count,
        "ocr_applied": ocr_applied,
        "pages_filtered": pages_filtered,
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
    if LLM_IS_YANDEX:
        return {"type": "json_object"}
    # strict=False: схема передаётся модели как ОРИЕНТИР, без жёсткой валидации
    # структуры. Это критично для gpt-4.1 (и новее): в strict-режиме OpenAI требует
    # 'additionalProperties': false и полный 'required' в КАЖДОМ объекте схемы,
    # а наша схема с references/bbox/anyOf этим требованиям не удовлетворяет —
    # strict-запрос отклонялся с 400, падал в fallback без схемы, и модель
    # возвращала голый массив (non-object JSON → весь анализ failed).
    # Со strict=False gpt-4.1 корректно возвращает {products: [...]}.
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": False,
            "schema": schema,
        },
    }


def _build_openrouter_provider_preferences(*, require_parameters: bool) -> dict[str, Any]:
    if LLM_IS_YANDEX:
        return {}
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
    started_at = time.monotonic()
    redacted_url = _redact_url(file_url)
    # SSRF: file_url приходит в теле запроса /extract. Раньше проверялась
    # только схема, из-за чего URL мог вести на метаданные облака или на
    # внутренние сервисы. Валидация до запроса + отключённые редиректы:
    # с follow_redirects=True разрешённый хост мог увести на запрещённый
    # уже после проверки.
    validate_public_url(file_url)
    try:
        async with httpx.AsyncClient(
            timeout=FILE_DOWNLOAD_TIMEOUT_SECONDS,
            follow_redirects=False,
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
                size_bytes = 0
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    temp_path = handle.name
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)
                        size_bytes += len(chunk)
    except httpx.HTTPError as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        logger.error(
            "File download failed url=%s status=%s elapsed=%.2fs error=%s",
            redacted_url, status_code, time.monotonic() - started_at, exc,
        )
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise RuntimeError(f"Failed to download file: {exc}") from exc
    except Exception:
        logger.exception(
            "File download crashed url=%s elapsed=%.2fs", redacted_url, time.monotonic() - started_at,
        )
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Failed to remove temp file %s", temp_path)
        raise

    logger.info(
        "File downloaded url=%s filename=%s size_bytes=%d elapsed=%.2fs",
        redacted_url, filename, size_bytes, time.monotonic() - started_at,
    )
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


def _matched_text_consistent_with_quote(
    matched_text: str | None, quote_text: str | None
) -> bool:
    """True, если matched_text согласуется с quote_text (один — подстрока другого).

    Зеркалит проверку фронтенда (isFallbackAnchor): если геометрия нашла текст,
    не совпадающий с цитатой, фронт прячет такую подсветку как «обманчивую».
    Возвращаем True только когда matched_text реально соответствует цитате —
    тогда его безопасно выставлять. Для таблиц (нашли строку по названию, а
    quote — это значение) вернём False, и matched_text не сохранится, чтобы
    корректный bbox не был отброшен фронтендом."""
    if not matched_text or not quote_text:
        return False
    norm_matched = _normalize_match_text(matched_text)
    norm_quote = _normalize_match_text(quote_text)
    if not norm_matched or not norm_quote:
        return False
    if norm_matched == norm_quote:
        return True
    return norm_quote in norm_matched or norm_matched in norm_quote


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


def _collect_generic_anchor_texts(root: Any) -> set[str]:
    """Находит «общие» anchor/locator-тексты, которые нельзя использовать как
    привязку к месту в PDF.

    Типичная проблема: LLM ставит одинаковый anchor_text (название изделия,
    например «Гидрант пожарный») для ВСЕХ характеристик. Геометрия тогда находит
    этот заголовок один раз на стр. 1 и сажает туда все характеристики.
    Считаем такими «общими» якорями: (1) названия изделий (product_name) и
    (2) anchor/locator-тексты, повторяющиеся в >=2 референсах."""
    generic: set[str] = set()
    counts: dict[str, int] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("product_name", "product_model"):
                value = node.get(key)
                if isinstance(value, str):
                    norm = _normalize_match_text(value)
                    if norm:
                        generic.add(norm)
            references = node.get("references")
            if isinstance(references, list):
                for reference in references:
                    if not isinstance(reference, dict):
                        continue
                    for key in ("anchor_text", "locator_text"):
                        value = reference.get(key)
                        if isinstance(value, str):
                            norm = _normalize_match_text(value)
                            if norm:
                                counts[norm] = counts.get(norm, 0) + 1
            for key, value in node.items():
                if key != "references":
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(root)
    generic.update(text for text, count in counts.items() if count >= 2)
    return generic


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


def _build_synthetic_reference(label_context: list[str], value_context: list[str]) -> dict[str, Any] | None:
    primary_value = next((v for v in value_context if v), "")
    primary_label = " ".join(label_context[:2]).strip()
    if not primary_value and not primary_label:
        return None
    value_is_ambiguous = primary_value and _is_ambiguous_short_candidate(
        primary_value, _normalize_match_text(primary_value)
    )
    if value_is_ambiguous and primary_label:
        quote_text = f"{primary_label} {primary_value}"
        anchor_text = primary_label
    else:
        quote_text = primary_value or primary_label
        anchor_text = quote_text
    return {
        "quote_text": quote_text,
        "anchor_text": anchor_text,
        "locator_text": anchor_text,
        "synthetic_reference": True,
    }


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
            ref = _build_synthetic_reference(label_context, value_context)
            if ref:
                node["references"] = [ref]
                injected += 1
        elif (
            not has_reference_key
            and label_context
            and value_context
            and any(key in node for key in ("name", "label", "title", "characteristic"))
            and any(key in node for key in ("value", "text", "answer", "content"))
        ):
            ref = _build_synthetic_reference(label_context, value_context)
            if ref:
                node["references"] = [ref]
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


def _looks_like_markdown_table_row(text: str) -> bool:
    """True, если строка похожа на ячейки Markdown-таблицы (есть разделители «|»).
    LLM теперь получает таблицы как Markdown (см. _page_text_with_tables), поэтому
    quote_text характеристики из таблицы — это строка вида 'модель | v1 | v2 | ...'.
    В реальном тексте PDF символов «|» нет, и точный поиск такой цитаты проваливался
    → геометрия привязывалась к заголовку. Поэтому такие цитаты разбираем на ячейки."""
    return text.count("|") >= 2


def _markdown_row_cell_candidates(
    text: str, page_hint: int | None
) -> list[dict[str, Any]]:
    """Из Markdown-строки таблицы делает кандидаты для геометрии: код модели (первая
    значимая ячейка), отдельные значения-ячейки и очищенный текст без «|».
    Markdown-разделители «| --- |» игнорируются."""
    cells = [c.strip() for c in text.split("|")]
    cells = [c for c in cells if c and not re.fullmatch(r"[-:\s]+", c)]
    if not cells:
        return []
    out: list[dict[str, Any]] = []
    # Первая ячейка — обычно код модели / название строки: ценный якорь.
    out.append({
        "text": cells[0], "kind": "anchor_text", "weight": 0.9, "page_hint": page_hint,
    })
    # Остальные ячейки — значения; короткие числа ambiguous, но в паре с моделью
    # дают точную привязку строки.
    for cell in cells[1:]:
        if len(cell) >= 2:
            out.append({
                "text": cell, "kind": "value", "weight": 0.75, "page_hint": page_hint,
            })
    # Очищенная строка целиком (числа через пробел) — иногда совпадает с PDF-строкой.
    cleaned = " ".join(cells)
    if cleaned and cleaned != text:
        out.append({
            "text": cleaned, "kind": "text", "weight": 0.7, "page_hint": page_hint,
        })
    return out


def _collect_reference_candidates(
    reference: Any,
    *,
    label_context: list[str],
    value_context: list[str],
    generic_anchors: set[str] | None = None,
) -> list[dict[str, Any]]:
    raw_candidates: list[dict[str, Any]] = []
    generic_anchors = generic_anchors or set()
    page_hint = None
    if isinstance(reference, dict):
        page_hint = _safe_page_number(reference.get("page"))
        if page_hint is None:
            page_hint = _safe_page_number(reference.get("page_number"))
        for key in ("quote_text", "anchor_text", "locator_text", "text"):
            value = reference.get(key)
            if isinstance(value, str) and value.strip():
                # Пропускаем «общие» якоря (название изделия и т.п.): по ним нельзя
                # позиционировать конкретную характеристику — иначе все они сядут
                # в одно место (заголовок на стр. 1).
                if key in {"anchor_text", "locator_text"} and _normalize_match_text(value) in generic_anchors:
                    continue
                # Markdown-строка таблицы: «|» в реальном PDF нет — разбираем на ячейки
                # (код модели + значения), иначе точный поиск цитаты провалится и
                # геометрия привяжется к заголовку таблицы.
                if key == "quote_text" and _looks_like_markdown_table_row(value):
                    raw_candidates.extend(
                        _markdown_row_cell_candidates(value, page_hint)
                    )
                    continue
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
    value_is_ambiguous = primary_value and _is_ambiguous_short_candidate(
        primary_value, _normalize_match_text(primary_value)
    )
    if primary_label and primary_value:
        raw_candidates.append(
            {
                "text": f"{primary_label} {primary_value}",
                "kind": "label_plus_value",
                "weight": 0.95 if value_is_ambiguous else 0.82,
                "page_hint": page_hint,
            }
        )
    if primary_value:
        raw_candidates.append(
            {
                "text": primary_value,
                "kind": "value",
                "weight": 0.5 if value_is_ambiguous else 0.8,
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


def _page_index_rect(page_index: "PdfPageIndex") -> Any:
    """Система координат для нормализации bbox страницы: для OCR-индекса с
    авто-поворотом — прямоугольник выпрямленного изображения, иначе — page.rect."""
    if getattr(page_index, "page_rect", None) is not None:
        return page_index.page_rect
    return page_index.page.rect


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


def _build_single_page_index(document: Any, page_number: int) -> PdfPageIndex | None:
    """Строит индекс слов ОДНОЙ страницы (1-based) по текстовому слою.

    Быстрая точечная альтернатива _build_pdf_page_index для гибрида MinerU: нам
    нужны координаты слов лишь на страницах, где MinerU нашёл таблицы, а не по
    всему документу. Возвращает None, если на странице нет текстового слоя."""
    idx = page_number - 1
    if idx < 0 or idx >= document.page_count:
        return None
    page = document.load_page(idx)
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
    if not words:
        return None
    return PdfPageIndex(page_number=page_number, page=page, words=words)


def _build_pdf_page_index_ocr(local_path: str) -> tuple[Any, list[PdfPageIndex]]:
    """
    For scanned PDFs: renders each page as an image, runs Tesseract OCR,
    and builds PdfPageIndex from the recognized words.
    Word bboxes are in PDF-point coordinates (at DPI=150 scale).
    """
    import pytesseract
    from PIL import Image
    import io

    # DPI=150 даёт самые стабильные заголовки столбцов моделей ('50-110' и т.п.),
    # что критично для привязки ячеек таблиц к нужной модели. Более высокий DPI
    # иногда лучше читает плотные строки значений, но дробит/теряет заголовки
    # моделей — а без якоря модели вся табличная привязка рушится.
    DPI = 150
    document = fitz.open(local_path)
    pages: list[PdfPageIndex] = []

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        # Выпрямляем страницу (повёрнутые сканы) — иначе OCR читает текст боком/
        # вверх ногами и выдаёт мусор, а координаты не совпадут с вьювером.
        img = _auto_orient_for_ocr(img)
        # Прямоугольник системы координат = размер ВЫПРЯМЛЕННОГО изображения в
        # PDF-пунктах. Именно в этой системе фронт нормализует bbox.
        scale = 72.0 / DPI  # pixels -> PDF points
        page_rect = fitz.Rect(0, 0, img.width * scale, img.height * scale)

        try:
            ocr_data = pytesseract.image_to_data(
                img,
                lang="rus+eng",
                output_type=pytesseract.Output.DICT,
                timeout=OCR_PAGE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("OCR failed for page %d: %s", page_index + 1, exc)
            pages.append(
                PdfPageIndex(page_number=page_index + 1, page=page, words=[], page_rect=page_rect)
            )
            continue

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

        pages.append(
            PdfPageIndex(page_number=page_index + 1, page=page, words=words, page_rect=page_rect)
        )

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
        rects = []
    if rects:
        return sorted(rects, key=lambda rect: (float(rect.y0), float(rect.x0)))

    # Fallback: если точный поиск не нашёл (из-за множественных пробелов в таблицах),
    # нормализуем пробелы и ищем ещё раз
    normalized = re.sub(r"\s+", " ", snippet).strip()
    if normalized != snippet and len(normalized) >= 3:
        try:
            rects = page.search_for(normalized)
        except Exception:
            rects = []
        if rects:
            return sorted(rects, key=lambda rect: (float(rect.y0), float(rect.x0)))

    # Fallback 2: ищем только первую значимую часть (до первого большого пробела)
    parts = re.split(r"\s{2,}", snippet)
    if len(parts) > 1 and len(parts[0].strip()) >= 4:
        try:
            rects = page.search_for(parts[0].strip())
        except Exception:
            rects = []
        if rects:
            return sorted(rects, key=lambda rect: (float(rect.y0), float(rect.x0)))

    return []


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


def _bbox_is_compact(word_rects: list[Any], page_index: PdfPageIndex) -> bool:
    """Rejects a union bbox that spans too large a portion of the page.
    Typical table row is ~2-5% of page height and rarely the full width; a union
    spanning >12% height OR >85% width means matched tokens are scattered across
    multiple rows/columns → wrong match (e.g. value in one column, label in another)."""
    if not word_rects:
        return True
    page_rect = _page_index_rect(page_index)
    page_height = float(page_rect.height) if page_rect else 842.0  # A4 default
    page_width = float(page_rect.width) if page_rect else 595.0
    if page_height <= 0 or page_width <= 0:
        return True
    y_min = min(float(r.y0) for r in word_rects)
    y_max = max(float(r.y1) for r in word_rects)
    x_min = min(float(r.x0) for r in word_rects)
    x_max = max(float(r.x1) for r in word_rects)
    if (y_max - y_min) / page_height > 0.12:
        return False
    # Wide spans are only suspicious for multi-token matches: a single matched
    # phrase is naturally narrow; a >85%-width union means tokens jumped columns.
    if len(word_rects) >= 3 and (x_max - x_min) / page_width > 0.85:
        return False
    return True


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

    word_rects = [page_index.words[index].rect for index in best_indices]
    if not _bbox_is_compact(word_rects, page_index):
        return None, 0.0

    rect = _union_rects(word_rects)
    return rect, best_coverage


def _search_fuzzy_token_candidate(
    page_index: PdfPageIndex, candidate: str
) -> tuple[Any | None, float]:
    """OCR-устойчивый поиск с привязкой к ОДНОМУ компактному месту на странице.

    Берём только различительные токены цитаты (длиной >= 4: имена характеристик,
    а не служебные «не»/«до» и не голые числа, которые встречаются по всей странице),
    и ищем тесное непрерывное окно слов, где встречается большинство из них.
    Так мы избегаем ложных совпадений «по разбросанным общим словам», которые
    давали одинаковый счёт на разных страницах. Возвращает прямоугольник найденного
    места и долю совпавших различительных токенов."""
    distinctive = [token for token in _tokenize_match_text(candidate) if len(token) >= 4]
    if len(distinctive) < 2 or not page_index.words:
        return None, 0.0

    token_set = set(distinctive)
    word_norms = [word.normalized for word in page_index.words]

    def token_hits(word: str) -> str | None:
        for token in token_set:
            if word == token or (len(word) >= 4 and (token in word or word in token)):
                return token
        return None

    # Узкое окно: совпадения должны идти кучно (одно место в документе),
    # а не быть рассыпаны по всей странице.
    window = len(distinctive) + 3
    best_indices: list[int] = []
    best_ratio = 0.0
    for start in range(len(page_index.words)):
        end = min(len(page_index.words), start + window)
        hit_indices: list[int] = []
        matched_tokens: set[str] = set()
        for index in range(start, end):
            token = token_hits(word_norms[index])
            if token is not None and token not in matched_tokens:
                matched_tokens.add(token)
                hit_indices.append(index)
        ratio = len(matched_tokens) / len(token_set)
        if ratio > best_ratio or (ratio == best_ratio and len(hit_indices) < len(best_indices)):
            best_ratio = ratio
            best_indices = hit_indices

    # Требуем уверенное совпадение: >=70% различительных токенов И минимум 2 из них.
    if best_ratio < 0.7 or len(best_indices) < 2:
        return None, 0.0

    word_rects = [page_index.words[index].rect for index in best_indices]
    if not _bbox_is_compact(word_rects, page_index):
        return None, 0.0

    rect = _union_rects(word_rects)
    return rect, best_ratio


def _token_overlap_ratio(a: str, b: str) -> float:
    """Fraction of tokens from a that are found in b."""
    tokens_a = set(_tokenize_match_text(a))
    tokens_b = set(_tokenize_match_text(b))
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


_tables_cache: dict[int, Any] = {}


def _get_page_tables(page_index: PdfPageIndex) -> Any:
    page_id = id(page_index.page)
    cached = _tables_cache.get(page_id)
    if cached is not None:
        return cached
    try:
        tables = page_index.page.find_tables()
    except Exception:
        tables = None
    _tables_cache[page_id] = tables
    return tables


def _model_numeric_cores(text: str) -> list[str]:
    """Числовое ядро типоразмера: '5Кс — 5х4 (КС 50-110/4)' → ['50-110/4','50-110'].
    Паспорт пишет модель как '1Кс50-110' — полную строку из ТЗ search_for не найдёт,
    а ядро '50-110' совпадает точно."""
    norm = text.lower().replace("–", "-").replace("—", "-").replace("х", "x")
    cores: list[str] = []
    for m in re.findall(r"\d+(?:[-/x]\d+)+", norm):
        if m not in cores:
            cores.append(m)
        base = m.split("/", 1)[0]
        if "-" in base and base not in cores:
            cores.append(base)
    return cores


def _model_term_rects(page_index: PdfPageIndex, context_terms: list[str]) -> list[Any]:
    """Прямоугольники, где на странице встречается код модели из контекста —
    по полной строке И по числовому ядру типоразмера (устойчиво к разным префиксам)."""
    rects: list[Any] = []
    search_terms: list[str] = []
    for term in context_terms:
        norm_term = _normalize_match_text(term)
        if not norm_term or len(norm_term) < 4:
            continue
        search_terms.append(term)
        # Числовое ядро (50-110) — паспорт обычно пишет модель именно так.
        for core in _model_numeric_cores(term):
            if len(core) >= 4 and core not in search_terms:
                search_terms.append(core)
    for term in search_terms:
        rects.extend(_search_exact_candidate_rects(page_index.page, term))
    return rects


def _model_rects_from_words(page_index: PdfPageIndex, context_terms: list[str]) -> list[Any]:
    """Прямоугольники кода модели, найденные по СЛОВАМ страницы (page_index.words).

    В отличие от _model_term_rects (через page.search_for), работает и на OCR-страницах,
    где реального текстового слоя нет. Ищем числовое ядро типоразмера ('50-110') как
    подстроку в нормализованном тексте каждого слова — паспорт пишет заголовок столбца
    модели как '50-110', '1Кс50-110', '50-110-...' и т.п."""
    cores: list[str] = []
    for term in context_terms:
        for core in _model_numeric_cores(term):
            norm_core = _normalize_match_text(core)
            if len(norm_core) >= 4 and norm_core not in cores:
                cores.append(norm_core)
    if not cores:
        return []

    rects: list[Any] = []
    matched_y: list[float] = []
    for word in page_index.words:
        wn = word.normalized
        for core in cores:
            if core in wn:
                rects.append(word.rect)
                matched_y.append(float(word.rect.y0))
                break
    if rects:
        return rects

    # OCR мог раздробить заголовок '50-110' на соседние слова '50' и '110'
    # (или '50-' + '110'). Восстанавливаем: для каждого ядра вида 'A-B' ищем
    # слово, начинающееся на A, и рядом (та же строка, правее, близко по X)
    # слово, заканчивающееся на B. Прямоугольник — объединение двух слов.
    for core in cores:
        parts = core.split("-")
        if len(parts) != 2 or not (parts[0].isdigit() and parts[1].isdigit()):
            continue
        a, b = parts[0], parts[1]
        a_words = [w for w in page_index.words if w.normalized.rstrip("-") == a or w.normalized == a]
        b_words = [w for w in page_index.words if w.normalized == b or w.normalized.lstrip("-") == b]
        for aw in a_words:
            ay = (float(aw.rect.y0) + float(aw.rect.y1)) / 2
            ah = float(aw.rect.y1) - float(aw.rect.y0)
            for bw in b_words:
                by = (float(bw.rect.y0) + float(bw.rect.y1)) / 2
                same_row = abs(ay - by) <= max(ah, 6.0)
                right_after = 0 <= (float(bw.rect.x0) - float(aw.rect.x1)) <= max(ah * 2, 20.0)
                if same_row and right_after:
                    rects.append(fitz.Rect(
                        min(float(aw.rect.x0), float(bw.rect.x0)),
                        min(float(aw.rect.y0), float(bw.rect.y0)),
                        max(float(aw.rect.x1), float(bw.rect.x1)),
                        max(float(aw.rect.y1), float(bw.rect.y1)),
                    ))
    return rects


def _value_cells_in_table(page_index: PdfPageIndex, norm_value: str) -> list[Any]:
    """Все ячейки таблиц, текст которых равен искомому значению."""
    tables = _get_page_tables(page_index)
    if not tables:
        return []
    cells: list[Any] = []
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
                if _normalize_match_text(cell_text) == norm_value:
                    cells.append(cell_rect)
    return cells


def _search_value_in_model_table(
    page_index: PdfPageIndex,
    value_text: str,
    context_terms: list[str],
    param_label: str | None = None,
) -> tuple[Any | None, float]:
    """Находит ячейку со значением `value_text` в модельной таблице.

    Поддерживает две раскладки:
      • обычная (строка = модель): значение на той же ГОРИЗОНТАЛИ, что и код модели;
      • транспонированная (столбец = модель, строка = параметр L/B/H): значение на
        пересечении СТОЛБЦА модели (по X) и СТРОКИ параметра (по Y).

    Anchor вида "Мощность, кВт 30" или "Типоразмер насоса 1Кс 50-110: L 1195" нельзя
    искать целиком — заголовок и значение лежат в разных ячейках. Ищем ровно ячейку
    значения, привязанную к строке/столбцу модели."""
    norm_value = _normalize_match_text(value_text)
    if not norm_value:
        return None, 0.0

    # Работаем НАПРЯМУЮ по словам страницы (page_index.words), а не через
    # page.search_for()/find_tables — на OCR-страницах (сканы) реального текстового
    # слоя нет, и оба не работают. Слова же есть всегда (из текстового слоя или OCR),
    # поэтому word-based поиск универсален.
    model_rects = _model_rects_from_words(page_index, context_terms)
    if not model_rects:
        return None, 0.0
    # Ячейки-кандидаты со значением — слова, чей нормализованный текст равен искомому.
    value_cells = [w.rect for w in page_index.words if w.normalized == norm_value]
    if not value_cells:
        return None, 0.0

    def _center(rect: Any) -> tuple[float, float]:
        return ((float(rect.x0) + float(rect.x1)) / 2, (float(rect.y0) + float(rect.y1)) / 2)

    # Высота строки модели — масштаб допусков (OCR-слова мельче ячеек таблиц).
    model_h = max(
        (float(mr.y1) - float(mr.y0) for mr in model_rects), default=12.0
    ) or 12.0
    y_tol = max(model_h * 0.8, 8.0)

    def _layout_same_row() -> Any | None:
        """Значение на ОДНОЙ СТРОКЕ с кодом модели (обычная раскладка)."""
        cell, dist = None, float("inf")
        for cell_rect in value_cells:
            _, cy = _center(cell_rect)
            for mr in model_rects:
                y0, y1 = float(mr.y0) - y_tol, float(mr.y1) + y_tol
                d = 0.0 if y0 <= cy <= y1 else min(abs(cy - y0), abs(cy - y1))
                if d < dist:
                    dist, cell = d, cell_rect
        return cell if (cell is not None and dist <= y_tol) else None

    def _layout_transposed() -> Any | None:
        """Транспонированная: пересечение СТОЛБЦА модели (X) и СТРОКИ параметра (Y).

        Выбираем ячейку значения, X которой ближе всего к X-столбцу модели, а Y —
        к Y-строке параметра. Это устраняет неоднозначность, когда одно и то же число
        (напр. '280') встречается в нескольких строках/столбцах."""
        if not param_label:
            return None
        norm_label = _normalize_match_text(param_label)
        # Для односимвольных меток (L/B/H) — точное совпадение (иначе 'l' ловит всё).
        # Для словесных ('масса') — точное ИЛИ префиксное (OCR склеивает 'Масса,кг').
        if len(norm_label) <= 2:
            label_words = [w for w in page_index.words if w.normalized == norm_label]
        else:
            label_words = [
                w for w in page_index.words
                if w.normalized == norm_label or w.normalized.startswith(norm_label)
            ]
        if not label_words:
            return None
        # Метка строки — самая левая (это подпись строки, а не значение в ячейке).
        leftmost_x = min(float(w.rect.x0) for w in label_words)
        param_rects = [w.rect for w in label_words if float(w.rect.x0) <= leftmost_x + 40]
        if not param_rects:
            return None
        model_xs = [(float(mr.x0) + float(mr.x1)) / 2 for mr in model_rects]
        x_tol = max(model_h * 4, 60.0)
        cell, best = None, float("inf")
        for cell_rect in value_cells:
            cx, cy = _center(cell_rect)
            x_dist = min(abs(cx - mx) for mx in model_xs)
            y_dist = min(
                0.0 if (float(pr.y0) - y_tol) <= cy <= (float(pr.y1) + y_tol)
                else min(abs(cy - float(pr.y0)), abs(cy - float(pr.y1)))
                for pr in param_rects
            )
            if y_dist <= y_tol and x_dist <= x_tol:
                score = y_dist * 3 + x_dist
                if score < best:
                    best, cell = score, cell_rect
        return cell

    def _layout_column_only() -> tuple[Any | None, float]:
        """Fallback: значение в СТОЛБЦЕ модели (ближайшее по X к заголовку модели),
        НИЖЕ заголовка. Применяется, когда метку строки (Масса/L/B/H) OCR не распознал —
        тогда нельзя выбрать точную строку, но можно хотя бы попасть в нужный столбец,
        что устраняет грубую ошибку (значение из чужой модели). Уверенность ниже."""
        model_xs = [(float(mr.x0) + float(mr.x1)) / 2 for mr in model_rects]
        header_bottom = max(float(mr.y1) for mr in model_rects)
        # Узкий допуск по X — иначе залезаем в соседнюю колонку модели. Колонка
        # обычно шире метки модели; берём ~ширину заголовка модели как полупорог.
        model_w = max((float(mr.x1) - float(mr.x0) for mr in model_rects), default=40.0)
        x_tol = max(model_w * 0.7, 25.0)
        cell, best = None, float("inf")
        for cell_rect in value_cells:
            cx, cy = _center(cell_rect)
            if cy < header_bottom:  # выше заголовка модели — не данные этой таблицы
                continue
            x_dist = min(abs(cx - mx) for mx in model_xs)
            if x_dist <= x_tol and x_dist < best:
                best, cell = x_dist, cell_rect
        return (cell, 0.5) if cell is not None else (None, 0.0)

    # Для габаритов и веса (есть метка строки) сначала пробуем транспонированную
    # раскладку — иначе значение случайно «прилипает» к заголовку модели по Y.
    if param_label:
        cell = _layout_transposed() or _layout_same_row()
    else:
        cell = _layout_same_row() or _layout_transposed()
    if cell is not None:
        return cell, 0.88
    # Метку строки не нашли (плохой OCR) — пробуем хотя бы попасть в столбец модели.
    if param_label:
        return _layout_column_only()
    return None, 0.0


_DIMENSION_LABELS = {
    "длина": ["l", "длина", "дл"],
    "ширина": ["b", "в", "ширина", "шир"],
    "высота": ["h", "н", "высота", "выс"],
}


def _dimension_param_label(anchor_text: str) -> str | None:
    """Из имени характеристики габарита возвращает короткую метку строки таблицы
    (L/B/H), по которой ищется строка параметра в транспонированной таблице.
    "Габаритные размеры: Длина" → 'L'."""
    norm = _normalize_match_text(anchor_text)
    for key, variants in _DIMENSION_LABELS.items():
        if key in norm:
            return variants[0].upper()
    return None


def _weight_param_label(anchor_text: str) -> str | None:
    """Для характеристик веса/массы насоса возвращает метку строки 'масса' —
    в таблицах габаритов масса идёт отдельной строкой 'Масса, кг', значения которой
    разнесены по столбцам моделей. Это позволяет искать ячейку массы на пересечении
    строки 'Масса' и столбца нужной модели, а не первое попавшееся число.

    Только для веса/массы НАСОСА — не для электродвигателя/плиты/агрегата, чьи
    значения в этой таблице отсутствуют."""
    norm = _normalize_match_text(anchor_text)
    if "электродвигател" in norm or "плит" in norm or "агрегат" in norm or "общий" in norm:
        return None
    if "вес насоса" in norm or "масса насоса" in norm or norm in {"вес насоса", "масса насоса"}:
        return "масса"
    # "Вес: Насоса" → нормализуется в "вес насоса"
    if ("вес" in norm or "масс" in norm) and "насос" in norm:
        return "масса"
    return None


def _search_table_row_candidate(
    page_index: PdfPageIndex,
    anchor_text: str,
    context_terms: list[str] | None = None,
    value_text: str | None = None,
) -> tuple[Any | None, float]:
    """
    Searches for the table row or text line containing anchor_text.
    Strategy:
      0. If value + model context given: find the value cell on the model's row.
      1. Use page.find_tables() — if anchor_text matches a cell, return bbox of the whole row.
      2. Otherwise find the text line whose words best overlap anchor tokens (Y-band grouping).
    """
    normalized_anchor = _normalize_match_text(anchor_text)
    if not normalized_anchor or len(normalized_anchor) < 2:
        return None, 0.0

    # --- Step 0: model-keyed table — find the value cell on the model's row/column ---
    if value_text and context_terms:
        norm_val = _normalize_match_text(value_text)
        if norm_val and _is_ambiguous_short_candidate(value_text, norm_val):
            param_label = _dimension_param_label(anchor_text)
            rect, conf = _search_value_in_model_table(
                page_index, value_text, context_terms, param_label=param_label
            )
            if rect is not None:
                return rect, conf

    # --- Step 1: Search in tables ---
    try:
        tables = _get_page_tables(page_index)
        for table in (tables.tables if tables else []):
            # Строки-заголовки таблицы (подписи колонок) НЕ содержат значений —
            # привязка к ним даёт выделение заголовка. Например название
            # характеристики «материал проточной части» совпадает токенами с
            # ЗАГОЛОВКОМ колонки «Материал». Определяем заголовочные строки и
            # пропускаем их ячейки. Заголовок — это верхние строки, где почти нет
            # числовых ячеек (подписи), по аналогии с _serialize_table_markdown.
            header_row_count = 0
            try:
                ext = table.extract()
                for r in ext[: min(2, len(ext))]:
                    cells_txt = [_normalize_match_text("" if c is None else str(c)) for c in r]
                    cells_txt = [c for c in cells_txt if c]
                    if not cells_txt:
                        header_row_count += 1
                        continue
                    numeric = sum(1 for c in cells_txt if _cell_is_numeric(c))
                    if numeric <= max(0, len(cells_txt) // 3):
                        header_row_count += 1
                    else:
                        break
            except Exception:
                header_row_count = 1
            header_row_count = max(1, header_row_count)

            for row_idx, row in enumerate(table.rows):
                if row_idx < header_row_count:
                    continue  # пропускаем строки-заголовки
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
                    # Совпадение должно быть ОСМЫСЛЕННЫМ. Substring-проверка
                    # «cell in anchor» ложно срабатывала на одиночных символах/цифрах
                    # (ячейка '5' — подстрока кода модели '...100-65-250'), привязывая
                    # к произвольной ячейке. Поэтому substring разрешаем только для
                    # достаточно длинных ячеек (>=4 симв.), а короткие сверяем строго
                    # по перекрытию токенов.
                    overlap_ratio = _token_overlap_ratio(normalized_anchor, cell_normalized)
                    long_enough = len(cell_normalized) >= 4
                    matched = (
                        (normalized_anchor in cell_normalized and len(normalized_anchor) >= 4)
                        or (cell_normalized in normalized_anchor and long_enough)
                        or overlap_ratio >= 0.6
                    )
                    if matched:
                        row_cells = [
                            fitz.Rect(c) if not isinstance(c, fitz.Rect) else c
                            for c in row.cells
                            if c is not None
                        ]
                        row_rect = _union_rects(row_cells)
                        if not row_rect or row_rect.is_empty:
                            continue
                        page_rect = _page_index_rect(page_index)
                        ph = float(page_rect.height) if page_rect else 842.0
                        pw = float(page_rect.width) if page_rect else 595.0
                        # Reject oversized rows (merged cells across visual rows)
                        if ph > 0 and float(row_rect.height) / ph > 0.12:
                            continue
                        overlap = _token_overlap_ratio(normalized_anchor, cell_normalized)
                        # For very wide rows (multi-column tables like "Размеры в мм"),
                        # the whole-row highlight is misleading — narrow to the matched
                        # cell so the user sees the actual value, not the full table width.
                        if pw > 0 and float(row_rect.width) / pw > 0.6:
                            return cell_rect, max(0.7, overlap)
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
        "label_plus_value": 950.0,
        "value": 900.0,
        "raw_reference": 800.0,
        "locator_text": 550.0,
        "text": 520.0,
        "anchor_text": 450.0,
        "label": 250.0,
    }.get(str(candidate.get("kind") or ""), 300.0)

    text = str(candidate.get("text") or "")
    normalized = str(candidate.get("normalized") or _normalize_match_text(text))

    # Ambiguous short/numeric candidates are unreliable — they match everywhere
    # in multi-page documents. Demote them heavily so that descriptive candidates
    # (label_plus_value, anchor_text) win.
    if candidate.get("kind") in {"quote_text", "value"}:
        if _is_ambiguous_short_candidate(text, normalized):
            return 150.0 if has_descriptive_candidate else 200.0
    return base


def _is_generic_table_anchor(candidate: dict[str, Any]) -> bool:
    kind = str(candidate.get("kind") or "")
    if kind not in {"anchor_text", "locator_text", "text"}:
        return False
    normalized = str(candidate.get("normalized") or _normalize_match_text(str(candidate.get("text") or "")))
    return bool(re.fullmatch(r"(таблица|table)\s*\d*", normalized, flags=re.IGNORECASE))


def _page_has_context(page_index: PdfPageIndex, context_terms: list[str]) -> bool:
    """Checks whether any of the context terms appear on the given page."""
    if not context_terms or not page_index.words:
        return False
    page_text_set = {w.normalized for w in page_index.words}
    for term in context_terms:
        term_tokens = _tokenize_match_text(term)
        if not term_tokens:
            continue
        if sum(1 for t in term_tokens if t in page_text_set) >= max(1, len(term_tokens) // 2):
            return True
    return False


def _find_reference_location(
    pages: list[PdfPageIndex],
    candidates: list[dict[str, Any]],
    context_terms: list[str] | None = None,
    *,
    label: str = "",
) -> dict[str, Any] | None:
    best_match: dict[str, Any] | None = None
    context_terms = context_terms or []
    logger.debug(
        "geometry: searching reference label=%r candidates=%s context_terms=%s",
        label,
        [(c.get("kind"), (c.get("text") or "")[:60]) for c in candidates],
        context_terms,
    )
    has_specific_candidate = any(
        candidate.get("kind") in {"quote_text", "value", "label_plus_value", "raw_reference"}
        for candidate in candidates
    )
    has_descriptive_candidate = any(
        c.get("kind") in {"anchor_text", "locator_text", "label_plus_value"}
        and len(str(c.get("normalized") or _normalize_match_text(str(c.get("text") or "")))) > 6
        for c in candidates
    )

    def _sorted_pages(page_hint: int | None, nearby_only: bool = False) -> list[PdfPageIndex]:
        ordered = sorted(
            pages,
            key=lambda page_index: (
                0 if page_hint and page_index.page_number == page_hint else 1,
                abs(page_index.page_number - page_hint) if page_hint else 0,
                page_index.page_number,
            ),
        )
        if nearby_only and page_hint:
            return [p for p in ordered if abs(p.page_number - page_hint) <= 3]
        return ordered

    def _ambiguity_penalty(candidate: dict[str, Any], page_index: PdfPageIndex) -> float:
        """Heavy penalty for ambiguous short candidates that land on a page
        where none of the context terms (characteristic name, product name) appear.
        This prevents "280" from matching on page 58 when the characteristic
        "Масса насоса" is only on page 6."""
        text = str(candidate.get("text") or "")
        normalized = str(candidate.get("normalized") or _normalize_match_text(text))
        if not _is_ambiguous_short_candidate(text, normalized):
            return 0.0
        if not context_terms:
            return 0.0
        if _page_has_context(page_index, context_terms):
            return 0.0
        return -500.0

    any_has_page_hint = any(_safe_page_number(c.get("page_hint")) for c in candidates)
    large_doc = len(pages) > 10

    # --- Pass 0: модель-aware поиск ячейки значения для ГАБАРИТОВ ---
    # Габаритные quote вида "Типоразмер насоса 1Кс 50-110: L 1195" иначе ловятся
    # token-поиском за заголовок "Типоразмер насоса". Если есть модель и метка
    # габарита (L/B/H), приоритетно ищем ячейку значения в строке параметра ×
    # столбце модели — это даёт точную ячейку, а не заголовок.
    value_cand = next((c for c in candidates if c.get("kind") == "value"), None)
    anchor_cand = next(
        (c for c in candidates if c.get("kind") in {"anchor_text", "label", "label_plus_value"}),
        None,
    )
    if value_cand and context_terms:
        param_label = None
        for c in candidates:
            lbl = _dimension_param_label(str(c.get("text") or ""))
            if lbl:
                param_label = lbl
                break
        # Контекст характеристики (название) тоже может содержать метку — для веса
        # ("Вес: Насоса") она не лежит в candidates, поэтому проверяем context_terms.
        if param_label is None:
            for term in context_terms:
                lbl = _dimension_param_label(term) or _weight_param_label(term)
                if lbl:
                    param_label = lbl
                    break
        if param_label:
            page_hint = _safe_page_number(value_cand.get("page_hint"))
            for page_index in _sorted_pages(page_hint, nearby_only=bool(page_hint and large_doc)):
                rect, conf = _search_value_in_model_table(
                    page_index, value_cand["text"], context_terms, param_label=param_label
                )
                if rect is not None:
                    # Высокий score, чтобы token-поиск не перебил точную ячейку
                    # заголовком "Типоразмер насоса".
                    best_match = {
                        "page": page_index.page_number,
                        "bbox": _rect_to_bbox(rect, _page_index_rect(page_index)),
                        "score": 1300.0 + conf * 100,
                        "locator_strategy": "table_cell",
                        "matched_text": value_cand["text"],
                    }
                    break
    logger.debug(
        "geometry: pass0 (model table cell) label=%r result=%s",
        label,
        best_match["locator_strategy"] if best_match else None,
    )

    def _do_exact_pass(page_list_fn):
        nonlocal best_match
        for candidate in candidates:
            if has_specific_candidate and _is_generic_table_anchor(candidate):
                continue
            candidate_text = candidate["text"]
            candidate_weight = float(candidate.get("weight", 0.5))
            page_hint = _safe_page_number(candidate.get("page_hint"))
            for page_index in page_list_fn(page_hint):
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
                    + _ambiguity_penalty(candidate, page_index)
                )
                if not best_match or score > best_match["score"]:
                    best_match = {
                        "page": page_index.page_number,
                        "bbox": _rect_to_bbox(rect, _page_index_rect(page_index)),
                        "score": score,
                        "locator_strategy": "pymupdf_exact",
                        "matched_text": candidate_text,
                    }

    def _do_token_pass(page_list_fn):
        nonlocal best_match
        for candidate in candidates:
            if has_specific_candidate and _is_generic_table_anchor(candidate):
                continue
            candidate_text = candidate["text"]
            candidate_weight = float(candidate.get("weight", 0.5))
            page_hint = _safe_page_number(candidate.get("page_hint"))
            for page_index in page_list_fn(page_hint):
                rect, coverage = _search_token_candidate(page_index, candidate_text)
                if not rect:
                    continue
                page_bonus = 12.0 if page_hint and page_index.page_number == page_hint else 0.0
                score = (
                    _candidate_rank(candidate, has_descriptive_candidate=has_descriptive_candidate)
                    + coverage * 100 * candidate_weight
                    + min(len(candidate_text), 120) / 10
                    + page_bonus
                    + _ambiguity_penalty(candidate, page_index)
                )
                if not best_match or score > best_match["score"]:
                    best_match = {
                        "page": page_index.page_number,
                        "bbox": _rect_to_bbox(rect, _page_index_rect(page_index)),
                        "score": score,
                        "locator_strategy": "pymupdf_tokens",
                        "matched_text": candidate_text,
                    }

    # --- Pass 1+2: exact + token search ---
    # For large docs with page hints: try nearby pages first, expand only if needed
    if large_doc and any_has_page_hint:
        _do_exact_pass(lambda hint: _sorted_pages(hint, nearby_only=True))
        _do_token_pass(lambda hint: _sorted_pages(hint, nearby_only=True))
        if not best_match or best_match["score"] <= 500:
            _do_exact_pass(lambda hint: _sorted_pages(hint))
            _do_token_pass(lambda hint: _sorted_pages(hint))
    else:
        _do_exact_pass(lambda hint: _sorted_pages(hint))
        _do_token_pass(lambda hint: _sorted_pages(hint))

    logger.debug(
        "geometry: pass1+2 (exact+token) label=%r result=%s page=%s score=%s",
        label,
        best_match["locator_strategy"] if best_match else None,
        best_match["page"] if best_match else None,
        round(best_match["score"], 1) if best_match else None,
    )

    # --- Pass 3: table row / free-text line search ---
    # find_tables() is expensive on large PDFs — skip if passes 1-2 found a confident match
    if not best_match or best_match["score"] <= 500:
        anchor_candidates = [
            c for c in candidates
            if c.get("kind") in {"anchor_text", "locator_text", "label", "label_plus_value"}
            and len(c.get("normalized", "")) >= 3
        ]
        # Значение характеристики (для модельных таблиц: ищем ячейку значения на
        # строке нужной модели, а не заголовок в чужой колонке).
        value_candidate = next(
            (c["text"] for c in candidates if c.get("kind") == "value"), None
        )
        for candidate in anchor_candidates:
            if has_specific_candidate and _is_generic_table_anchor(candidate):
                continue
            candidate_text = candidate["text"]
            candidate_weight = float(candidate.get("weight", 0.5))
            page_hint = _safe_page_number(candidate.get("page_hint"))
            for page_index in _sorted_pages(page_hint):
                rect, overlap = _search_table_row_candidate(
                    page_index,
                    candidate_text,
                    context_terms=context_terms,
                    value_text=value_candidate,
                )
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
                        "bbox": _rect_to_bbox(rect, _page_index_rect(page_index)),
                        "score": score,
                        "locator_strategy": "table_row",
                        "matched_text": candidate_text,
                    }

    logger.debug(
        "geometry: pass3 (table row) label=%r result=%s page=%s score=%s",
        label,
        best_match["locator_strategy"] if best_match else None,
        best_match["page"] if best_match else None,
        round(best_match["score"], 1) if best_match else None,
    )

    # --- Pass 4: OCR-robust fuzzy search ---
    if best_match is None:
        fuzzy_candidates = [
            c
            for c in candidates
            if c.get("kind") in {"quote_text", "value", "label_plus_value", "raw_reference"}
            and not _is_generic_table_anchor(c)
        ]
        for candidate in fuzzy_candidates:
            candidate_text = candidate["text"]
            page_ratios: list[tuple[float, Any, Any]] = []
            for page_index in pages:
                rect, ratio = _search_fuzzy_token_candidate(page_index, candidate_text)
                if rect:
                    page_ratios.append((ratio, rect, page_index))
            if not page_ratios:
                continue
            page_ratios.sort(key=lambda item: item[0], reverse=True)
            top_ratio, top_rect, top_page_index = page_ratios[0]
            runner_up = page_ratios[1][0] if len(page_ratios) > 1 else 0.0
            if top_ratio - runner_up < 0.2:
                continue
            candidate_weight = float(candidate.get("weight", 0.5))
            score = top_ratio * 60 * candidate_weight
            if not best_match or score > best_match["score"]:
                best_match = {
                    "page": top_page_index.page_number,
                    "bbox": _rect_to_bbox(top_rect, _page_index_rect(top_page_index)),
                    "score": score,
                    "locator_strategy": "pymupdf_fuzzy",
                    "matched_text": candidate_text,
                }

    if best_match:
        logger.debug(
            "geometry: reference resolved label=%r strategy=%s page=%s score=%s matched_text=%r",
            label, best_match["locator_strategy"], best_match["page"],
            round(best_match["score"], 1), (best_match.get("matched_text") or "")[:80],
        )
    else:
        logger.info(
            "geometry: reference NOT found label=%r candidates=%d pages_searched=%d context_terms=%s",
            label, len(candidates), len(pages), context_terms,
        )

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
    log_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    log_extra = {**(log_context or {}), "step": "geometry_enrichment"}
    unmatched_labels: list[str] = []
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
    _tables_cache.clear()
    try:
        document, pages = _build_pdf_page_index(local_path)
        metadata["page_count"] = document.page_count
        # Текстовый слой годен, только если слова осмысленные. У PDF с битой
        # кодировкой шрифта слова «есть», но это мусор — по нему цитаты не
        # находятся (0 координат, клик не работает). Тогда падаем на OCR.
        has_words = any(page.words for page in pages)
        # Достаточно ли слов: скан с парой слов на странице («осмысленных», но
        # пустых по сути) тоже надо отправить на OCR, иначе цитаты не находятся.
        enough_words = _page_words_are_enough(pages)
        text_layer_usable = has_words and _page_words_are_usable(pages) and enough_words
        if has_words and not text_layer_usable:
            if not enough_words:
                logger.info("Geometry: text layer too sparse (scanned) — falling back to OCR index")
            else:
                metadata["garbled_text_layer"] = True
                logger.info("Geometry: text layer garbled — falling back to OCR index")
        metadata["searchable_pdf"] = text_layer_usable
        if not metadata["searchable_pdf"]:
            # Attempt OCR fallback for scanned PDFs OR garbled text layer
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
        generic_anchors = _collect_generic_anchor_texts(extracted_data)

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
                    generic_anchors=generic_anchors,
                )
                reference_label = (
                    holder.get("name") if isinstance(holder, dict) else None
                ) or (label_context[0] if label_context else "") or "?"
                if not candidates:
                    logger.info(
                        "geometry: reference skipped (no candidates) label=%r",
                        reference_label,
                        extra=log_extra,
                    )
                    continue

                match = _find_reference_location(
                    pages, candidates, ancestor_context + label_context, label=reference_label,
                )
                if not match:
                    unmatched_labels.append(reference_label)
                    logger.info(
                        "geometry: reference NOT matched label=%r candidates=%d",
                        reference_label, len(candidates),
                        extra=log_extra,
                    )
                    # Геометрия не смогла привязать цитату к месту в PDF.
                    # Для сканов (OCR) номер страницы от LLM — это догадка
                    # (обычно дефолтная "1"), показывать её как точную позицию нельзя:
                    # это и приводило к тому, что все характеристики падали на стр. 1.
                    # Помечаем референс как непроверенный и убираем недостоверную страницу.
                    if metadata["ocr_applied"]:
                        reference["page"] = None
                        reference["page_number"] = None
                    reference["position_unverified"] = True
                    if isinstance(original_reference, dict):
                        original_reference.update(reference)
                    continue

                reference["page"] = match["page"]
                reference["page_number"] = match["page"]
                reference["bbox"] = match["bbox"]
                reference["locator_strategy"] = match["locator_strategy"]
                reference.setdefault("quote_text", candidates[0]["text"])
                reference.setdefault("anchor_text", candidates[0]["text"])
                reference["geometry_source"] = "pymupdf"
                reference["position_unverified"] = False
                # matched_text сохраняем только если он согласуется с quote_text.
                # В таблицах геометрия часто находит строку по НАЗВАНИЮ характеристики
                # ("Максимальный напор, м"), тогда как quote_text — это ЗНАЧЕНИЕ ("35").
                # Это валидная привязка (нашли нужную строку, где значение и стоит),
                # но фронтенд считает matched!=quote «обманчивым якорем» и прячет такие
                # совпадения — из-за этого в паспорте отображалась лишь 1 характеристика.
                # Чтобы не терять корректные bbox, в таком случае matched_text не выставляем.
                matched_text = match.get("matched_text")
                quote_text = reference.get("quote_text")
                if _matched_text_consistent_with_quote(matched_text, quote_text):
                    reference["matched_text"] = matched_text
                else:
                    reference.pop("matched_text", None)
                reference["match_score"] = round(float(match.get("score", 0.0)), 3)
                metadata["matched_reference_count"] += 1
                logger.info(
                    "geometry: reference matched label=%r page=%s strategy=%s score=%.1f",
                    reference_label, match["page"], match["locator_strategy"],
                    reference["match_score"],
                    extra=log_extra,
                )

                if isinstance(original_reference, dict):
                    original_reference.update(reference)

        metadata["applied"] = metadata["matched_reference_count"] > 0
        metadata["unmatched_labels"] = unmatched_labels
        logger.info(
            "geometry enrichment summary: references=%d matched=%d unmatched=%s",
            metadata["reference_count"], metadata["matched_reference_count"],
            unmatched_labels,
            extra=log_extra,
        )
        return metadata
    except Exception as exc:
        logger.exception("PyMuPDF geometry enrichment failed", extra=log_extra)
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


def _build_openai_headers(api_key: Optional[str]) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# Число попыток и базовая задержка (экспоненциальный бэкофф) для сетевых
# сбоев (ReadTimeout/ConnectError/ConnectTimeout) при вызове LLM-провайдера
# извлечения (OpenRouter/AI Tunnel — backend'ы openrouter/mineru/pdfplumber/
# llamaparse все проходят через эту функцию). Раньше единичный ReadTimeout
# ронял весь Celery-таск extract_file, который затем ретраился ЦЕЛИКОМ
# (заново скачивание файла и весь запрос) — тот же класс проблемы, что
# чинили для Yandex-пути в paddleocr-vl-service/yandex_structurer.py, только
# локальный retry здесь на порядок дешевле, чем retry всего таска.
_CHAT_COMPLETION_MAX_ATTEMPTS = int(os.environ.get("CHAT_COMPLETION_MAX_ATTEMPTS", "3"))
_CHAT_COMPLETION_RETRY_BASE_DELAY_SECONDS = float(
    os.environ.get("CHAT_COMPLETION_RETRY_BASE_DELAY_SECONDS", "5")
)


async def _post_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str],
    provider_name: str,
) -> httpx.Response:
    """POST с ретраем только сетевых сбоев (таймаут/обрыв соединения) —
    HTTP-ошибки провайдера (4xx/5xx) возвращаются как есть и обрабатываются
    существующей логикой fallback/ошибок в _chat_completion_json."""
    last_exc: Exception | None = None
    for attempt in range(1, _CHAT_COMPLETION_MAX_ATTEMPTS + 1):
        try:
            return await client.post(endpoint, json=json, headers=headers)
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            if attempt == _CHAT_COMPLETION_MAX_ATTEMPTS:
                break
            delay = _CHAT_COMPLETION_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "%s request failed on attempt %d/%d (%s), retrying in %.0fs",
                provider_name, attempt, _CHAT_COMPLETION_MAX_ATTEMPTS, exc, delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


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
    model_key = str(payload.get("model") or "")
    skip_strict = model_key and _strict_schema_supported.get(model_key) is False

    # Некоторые провайдеры (замечено на AI Tunnel для claude-sonnet-5) сами
    # включают extended thinking на своё усмотрение даже без явного запроса —
    # это резко увеличивает время ответа на больших документах и может увести
    # модель от точного соблюдения структуры/схемы. Извлечению reasoning не
    # нужен, поэтому глушим его явно, если вызывающий код не задал иное.
    if "reasoning" not in payload:
        payload = dict(payload)
        payload["reasoning"] = {"enabled": False}

    actual_payload = payload
    if skip_strict and "response_format" in payload:
        actual_payload = dict(payload)
        actual_payload.pop("response_format", None)
        if isinstance(actual_payload.get("provider"), dict):
            provider = dict(actual_payload["provider"])
            provider.pop("require_parameters", None)
            actual_payload["provider"] = provider if provider else actual_payload.pop("provider", None)
        used_fallback = True

    async with httpx.AsyncClient(
        timeout=REMOTE_API_TIMEOUT_SECONDS,
        trust_env=False,
    ) as client:
        response = await _post_with_retry(
            client, endpoint, json=actual_payload, headers=headers, provider_name=provider_name,
        )

        if response.status_code >= 400 and "response_format" in actual_payload:
            logger.warning(
                "%s first attempt HTTP %s (with response_format); body=%s",
                provider_name, response.status_code, response.text[:1000],
            )
            if model_key:
                _strict_schema_supported[model_key] = False
            fallback_payload = dict(actual_payload)
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
                logger.warning(
                    "%s retry attempt (no response_format) HTTP %s; body=%s",
                    provider_name, retry_response.status_code, retry_response.text[:1000],
                )
                _raise_provider_http_error(retry_response, provider_name)
            data = retry_response.json()
        else:
            if response.status_code >= 400:
                _raise_provider_http_error(response, provider_name)
            if model_key and not skip_strict and "response_format" in payload:
                _strict_schema_supported[model_key] = True
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
        # Некоторые модели (gpt-4.1 в fallback без схемы) возвращают JSON-массив
        # на верхнем уровне вместо объекта {products: [...]}. Не падаем, а
        # оборачиваем: список изделий или плоский список характеристик — это
        # валидные данные, дальше их нормализует gateway.
        if isinstance(parsed, list):
            logger.info(
                "%s returned a top-level JSON array — wrapping into {products: [...]}",
                provider_name,
            )
            return {"products": parsed}, data, used_fallback
        raise RuntimeError(f"{provider_name} returned non-object JSON")
    parsed = _normalize_flat_characteristics_payload(parsed, provider_name=provider_name)
    return parsed, data, used_fallback


def _looks_like_characteristic_entry(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "name" in value
        and "value" in value
        and isinstance(value.get("name"), str)
    )


def _normalize_flat_characteristics_payload(
    parsed: dict[str, Any], *, provider_name: str
) -> dict[str, Any]:
    """Некоторые модели (например Qwen3.7-Max) иногда возвращают характеристики
    как плоский словарь {имя: {name, value, references}} на верхнем уровне
    вместо ожидаемого {"products": [{"characteristics": [...]}]}. Срабатывает
    только когда products действительно отсутствует/некорректен — в остальных
    случаях payload не трогается."""
    products = parsed.get("products")
    if isinstance(products, list):
        return parsed

    candidate_entries = [
        value for value in parsed.values() if _looks_like_characteristic_entry(value)
    ]
    if not candidate_entries or len(candidate_entries) != len(parsed):
        return parsed

    logger.info(
        "%s returned a flat characteristics dict at top level — wrapping into "
        "{products: [{characteristics: [...]}]}",
        provider_name,
    )
    return {"products": [{"product_name": None, "characteristics": candidate_entries}]}


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
    missing_keys = _schema_required_keys(schema) - candidate.keys()
    started_at = time.monotonic()
    logger.info(
        "%s: repairing JSON, missing required keys=%s", provider_name, sorted(missing_keys),
    )
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
    try:
        repaired, _, _ = await _chat_completion_json(
            endpoint=endpoint,
            headers=headers,
            payload=repair_payload,
            provider_name=provider_name,
        )
    except Exception:
        logger.exception(
            "%s: repair failed after %.2fs", provider_name, time.monotonic() - started_at,
        )
        raise
    still_missing = _schema_required_keys(schema) - repaired.keys()
    logger.info(
        "%s: repair finished in %.2fs, still_missing_keys=%s",
        provider_name, time.monotonic() - started_at, sorted(still_missing),
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


def _convert_xlsx_to_structured_text(local_path: str) -> dict[str, Any]:
    """Извлекает текст из Excel-книги для передачи в LLM.

    Каждый лист сериализуется как таблица: строки с непустыми ячейками,
    разделённые ' | '. Формат повторяет DOCX-таблицы ('[T..R..] ...'), чтобы
    LLM одинаково понимал табличную структуру независимо от исходного формата.
    Пустые строки и полностью пустые листы пропускаются."""
    import openpyxl

    workbook = openpyxl.load_workbook(local_path, read_only=True, data_only=True)
    lines: list[str] = []
    sheet_count = 0
    row_count = 0
    try:
        for sheet in workbook.worksheets:
            rows_serialized: list[str] = []
            seen_rows: set[tuple[str, ...]] = set()
            for row in sheet.iter_rows(values_only=True):
                cells = tuple(
                    re.sub(r"\s+", " ", str(value)).strip() if value is not None else ""
                    for value in row
                )
                # Обрезаем хвост пустых ячеек, чтобы '—' не плодились до конца листа.
                while cells and cells[-1] == "":
                    cells = cells[:-1]
                if not any(cells):
                    continue
                if cells in seen_rows:
                    continue
                seen_rows.add(cells)
                rows_serialized.append(cells)
            if not rows_serialized:
                continue
            sheet_count += 1
            sheet_name = _clean_docx_text(sheet.title) or f"Sheet{sheet_count}"
            lines.append(f"[S{sheet_count}: {sheet_name}] SHEET START")
            for row_index, cells in enumerate(rows_serialized, start=1):
                row_count += 1
                serialized = " | ".join(cell if cell else "—" for cell in cells)
                lines.append(f"[S{sheet_count}R{row_index}] {serialized}")
            lines.append(f"[S{sheet_count}] SHEET END")
    finally:
        workbook.close()

    structured_text = "\n".join(lines).strip()
    return {
        "text": structured_text,
        "sheet_count": sheet_count,
        "row_count": row_count,
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
    # Изолируем HOME/GNUPGHOME в temp-профиль: иначе LibreOffice трогает общий
    # gpg-agent и плодит фоновые gpg-процессы, которые становятся зомби.
    env = dict(os.environ)
    env["HOME"] = str(profile_dir)
    env["GNUPGHOME"] = str(profile_dir / "gnupg")
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=90, check=False, env=env
    )
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

    schema = normalize_json_schema(payload.schema_payload, payload.prompt)
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
    if not LLM_IS_YANDEX:
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


def _mineru_layout_blocks(layout: Any) -> list[dict[str, Any]]:
    """Разворачивает layout.json MinerU в плоский список блоков с геометрией.

    Возвращает список {page_idx, page_size:[w,h], bbox:[x0,y0,x1,y1], text},
    где text — нормализованный текст блока (для таблиц — извлечённый из HTML).
    bbox уже в координатах page_size (в отличие от content_list, где иная шкала)."""
    blocks: list[dict[str, Any]] = []
    if not isinstance(layout, dict):
        return blocks
    pdf_info = layout.get("pdf_info")
    if not isinstance(pdf_info, list):
        return blocks
    for page in pdf_info:
        if not isinstance(page, dict):
            continue
        page_idx = page.get("page_idx")
        page_size = page.get("page_size")
        if not (isinstance(page_size, (list, tuple)) and len(page_size) == 2):
            continue
        para_blocks = page.get("para_blocks") or page.get("preproc_blocks") or []
        for blk in para_blocks:
            if not isinstance(blk, dict):
                continue
            bbox = blk.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            text = _mineru_block_text(blk)
            blocks.append({
                "page_idx": page_idx,
                "page_size": [float(page_size[0]), float(page_size[1])],
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "text": text,
            })
    return blocks


def _mineru_block_text(blk: dict[str, Any]) -> str:
    """Собирает весь текст блока layout (включая HTML таблиц) в одну строку."""
    parts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("html", "content", "text"):
                v = node.get(key)
                if isinstance(v, str) and v.strip():
                    parts.append(v)
            for v in node.values():
                if isinstance(v, (list, dict)):
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(blk)
    joined = " ".join(parts)
    # вычищаем HTML-теги, чтобы матчить по тексту
    return re.sub(r"<[^>]+>", " ", joined)


def _split_table_row_cells(quote: str | None) -> list[str]:
    """Разбивает строку-ряд таблицы на ячейки. MinerU/LLM непоследовательны в
    разделителе: иногда Markdown-пайпы ('| a | b |'), иногда табуляция ('a\\tb').
    Поддерживаем оба."""
    if not quote:
        return []
    s = str(quote)
    if "|" in s:
        cells = s.split("|")
    elif "\t" in s:
        cells = s.split("\t")
    else:
        return []
    cells = [c.strip() for c in cells]
    return [c for c in cells if c and not re.fullmatch(r"[-:\s]+", c)]


def _model_code_from_markdown_quote(quote: str | None) -> str | None:
    """Извлекает код модели из строки-ряда таблицы: первая непустая ячейка.

    LLM для табличных характеристик кладёт в quote_text ВСЮ строку таблицы
    ('| XM 0,25/1,2П-0,06 | 0,25 | ...' или через табы). Первая ячейка — код
    модели, надёжный якорь строки. None, если quote не похож на ряд таблицы."""
    cells = _split_table_row_cells(quote)
    if not cells:
        return None
    first = cells[0]
    # первая ячейка должна выглядеть как код модели (буквы+цифры), а не число-значение.
    if re.search(r"[A-Za-zА-Яа-я]", first) and re.search(r"\d", first):
        return first
    return None


_MEASURE_UNITS = (
    "мм", "см", "м3/час", "м3/ч", "м³/час", "м³/ч", "л/с", "об/мин", "квт", "вт",
    "кг", "г", "гц", "в", "а", "бар", "мпа", "°c", "м.в.ст", "м", "%",
)


def _value_core_for_cell_search(value: Any) -> str:
    """Готовит значение характеристики к поиску ячейки: аккуратно убирает единицы.

    LLM возвращает value по-разному: '6,0 Вт', '150x90x100 мм', 'G1', 'G3/4',
    '14/14 мм', 'Резьбовое'. В таблице ячейка — без единиц. НЕЛЬЗЯ грубо тянуть
    первое число ('G1' → '1' сломает поиск). Логика по приоритету:
      1) размер AxBxC → как есть без пробелов;
      2) '<токен> <единица>' → снять только хвост-единицу (по пробелу);
      3) иначе — вернуть значение как есть (для 'G1', 'G3/4' и т.п.)."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    # 1) размер вида 150x90x100 / 150х90х100 (лат./кир. x)
    m = re.search(r"\d+(?:[.,]\d+)?\s*[xх*]\s*\d+(?:[.,]\d+)?(?:\s*[xх*]\s*\d+(?:[.,]\d+)?)?", s)
    if m:
        return re.sub(r"\s+", "", m.group(0))
    # 2) хвост-единица, отделённая пробелом: '6,0 Вт' → '6,0', '180,0 Вт' → '180,0'
    parts = s.split()
    if len(parts) >= 2 and parts[-1].lower().strip(".,") in _MEASURE_UNITS:
        core = " ".join(parts[:-1]).strip()
        if core:
            return core
    # 3) как есть (G1, G3/4, 14/14, Резьбовое — ищем дословно)
    return s


def _model_rects_by_anchor_tokens(
    page_index: PdfPageIndex,
    context_terms: list[str],
    inside_fn,
) -> list[Any]:
    """Строки кода модели по прямому совпадению слов с anchor (код модели).

    Для форматов без числового ядра AAA-BBB (напр. 'ХМ 3,2/4Т-0,18-G1') ищем
    слово таблицы, чей нормализованный текст сильно пересекается с anchor —
    например слово '3,2/4Т-0,18-G1' покрывает большую часть кода модели."""
    # берём самые длинные/специфичные термы (код модели, а не короткие названия)
    anchor_norms = [
        _normalize_match_text(t) for t in context_terms if t and len(str(t)) >= 5
    ]
    anchor_norms = [a for a in anchor_norms if a]
    if not anchor_norms:
        return []
    rects: list[Any] = []
    for word in page_index.words:
        if not inside_fn(word.rect):
            continue
        wn = word.normalized
        if len(wn) < 4:
            continue
        for anchor in anchor_norms:
            # слово таблицы — подстрока anchor (частичный код модели) или наоборот,
            # и достаточно длинное, чтобы не ловить случайные короткие токены.
            if (wn in anchor and len(wn) >= 6) or (anchor in wn and len(anchor) >= 6):
                rects.append(word.rect)
                break
            ratio = _token_overlap_ratio(anchor, wn)
            if ratio >= 0.6 and len(wn) >= 6:
                rects.append(word.rect)
                break
    return rects


def _locate_value_in_table_bbox(
    page_index: PdfPageIndex,
    table_bbox_norm: tuple[float, float, float, float],
    value_text: str,
    context_terms: list[str],
) -> Any | None:
    """Находит ячейку значения ВНУТРИ bbox таблицы (нормализованного) на странице.

    MinerU знает bbox таблицы, откуда взято значение. Ищем слово == value внутри
    этого bbox, лежащее на СТРОКЕ кода модели (та же горизонталь). Это точная
    ячейка нужной строки — без поиска по всему документу и без привязки к
    названию характеристики (которое может не совпадать с заголовком столбца)."""
    page_rect = _page_index_rect(page_index)
    pw, ph = float(page_rect.width), float(page_rect.height)
    if pw <= 0 or ph <= 0:
        return None
    tx0, ty0, tx1, ty1 = table_bbox_norm
    # абсолютные границы таблицы (с небольшим запасом на неточность bbox)
    ax0, ay0 = tx0 * pw - 4, ty0 * ph - 4
    ax1, ay1 = tx1 * pw + 4, ty1 * ph + 4

    def _inside(r: Any) -> bool:
        cx = (float(r.x0) + float(r.x1)) / 2
        cy = (float(r.y0) + float(r.y1)) / 2
        return ax0 <= cx <= ax1 and ay0 <= cy <= ay1

    # Унифицируем разделитель размеров: в PDF габариты пишут кириллической 'х'
    # (340х245х205), а LLM часто возвращает латинскую 'x' (340x245x205) — без
    # приведения слово != значение. Обе 'x/х' и '*' → общий маркер.
    def _unify(s: str) -> str:
        return re.sub(r"[xх*]", "x", _normalize_match_text(s))

    norm_value = _unify(value_text)
    if not norm_value:
        return None
    # слова-значения внутри таблицы (с унификацией разделителя размеров)
    value_cells = [
        w.rect for w in page_index.words
        if _unify(w.text) == norm_value and _inside(w.rect)
    ]
    if not value_cells:
        return None
    # строки кода модели внутри таблицы: сначала штатный поиск по числовому ядру,
    # затем — по прямому совпадению слова с кодом модели из anchor (форматы вроде
    # 'ХМ 3,2/4Т-0,18-G1', где числового ядра AAA-BBB нет).
    model_rects = [
        mr for mr in _model_rects_from_words(page_index, context_terms) if _inside(mr)
    ]
    if not model_rects:
        model_rects = _model_rects_by_anchor_tokens(page_index, context_terms, _inside)
    if not model_rects:
        # без кода модели: если значение в таблице единственное — берём его,
        # иначе неоднозначно — отказываемся (пусть отработает fallback на таблицу).
        return value_cells[0] if len(value_cells) == 1 else None

    model_h = max((float(mr.y1) - float(mr.y0) for mr in model_rects), default=12.0) or 12.0
    y_tol = max(model_h * 0.8, 8.0)
    best, best_d = None, float("inf")
    for cell in value_cells:
        cy = (float(cell.y0) + float(cell.y1)) / 2
        for mr in model_rects:
            y0, y1 = float(mr.y0) - y_tol, float(mr.y1) + y_tol
            d = 0.0 if y0 <= cy <= y1 else min(abs(cy - y0), abs(cy - y1))
            if d < best_d:
                best_d, best = d, cell
    return best if (best is not None and best_d <= y_tol) else None


def _enrich_references_with_mineru_layout(
    extracted_data: dict[str, Any],
    layout: Any,
    *,
    local_path: str,
    content_type: str | None,
) -> dict[str, Any]:
    """Гибридная геометрия для MinerU: layout определяет НУЖНУЮ таблицу/страницу,
    PyMuPDF находит точную ЯЧЕЙКУ внутри.

    MinerU надёжно знает, в какой таблице лежит значение (по HTML), но координат
    ячейки не отдаёт (весь bbox — на всю таблицу). Поэтому: (1) по layout находим
    таблицу и корректируем страницу reference; (2) PyMuPDF по этой странице ищет
    точную строку/ячейку модель×столбец; (3) если PyMuPDF не смог — падаем на bbox
    таблицы MinerU (грубее, но на нужной таблице)."""
    metadata: dict[str, Any] = {
        "enabled": True,
        "provider": "mineru_layout",
        "applied": False,
        "page_count": 0,
        "reference_count": 0,
        "matched_reference_count": 0,
        "page_corrections": 0,
        "table_bbox_fallback": 0,
        "errors": [],
    }
    blocks = _mineru_layout_blocks(layout)
    if not blocks:
        # MinerU не дал геометрии (напр. DOCX/текстовый TZ — только текст, без bbox).
        # Сигнализируем вызывающему коду, что нужен PyMuPDF-энричмент (он умеет
        # DOCX→PDF→координаты и текстовый поиск цитат).
        metadata["errors"].append("no layout blocks")
        metadata["needs_pymupdf_fallback"] = True
        return metadata
    metadata["page_count"] = len({b["page_idx"] for b in blocks if b["page_idx"] is not None})

    _normalize_references_in_place(extracted_data)
    generic_anchors = _collect_generic_anchor_texts(extracted_data)

    # Шаг 1: по layout MinerU для каждой reference находим НУЖНЫЙ блок (таблицу),
    # запоминаем его bbox и корректируем страницу reference.
    # Для табличных references важно взять ПРАВИЛЬНЫЕ значение и код модели:
    #  • значение — из ЗНАЧЕНИЯ характеристики (holder.value), а НЕ из reference
    #    (в reference quote_text — целая Markdown-строка таблицы);
    #  • код модели — первая ячейка Markdown-строки quote (если quote табличный),
    #    т.к. anchor_text от LLM — это caption таблицы, не модель.
    ref_info: list[dict[str, Any]] = []
    pages_needed: set[int] = set()
    for holder, references, ancestors in _reference_iter_with_ancestors(extracted_data):
        label_context = _collect_label_context(holder)
        value_context = _collect_value_context(holder)
        ctx_terms = _collect_ancestor_context(ancestors) + label_context
        holder_value = holder.get("value") if isinstance(holder, dict) else None
        for reference in references:
            if not isinstance(reference, dict):
                continue
            metadata["reference_count"] += 1
            candidates = _collect_reference_candidates(
                reference,
                label_context=label_context,
                value_context=value_context,
                generic_anchors=generic_anchors,
            )
            best = _match_mineru_block(candidates, value_context, blocks, reference)
            if not best:
                continue
            block = best["block"]
            mineru_page = (block["page_idx"] or 0) + 1
            if _safe_page_number(reference.get("page")) != mineru_page:
                metadata["page_corrections"] += 1
            reference["page"] = mineru_page
            reference["page_number"] = mineru_page
            # код модели из Markdown-строки quote (первая ячейка) — приоритетно
            model_code = _model_code_from_markdown_quote(reference.get("quote_text"))
            model_terms = [model_code] if model_code else []
            model_terms += [reference.get("anchor_text") or ""]
            model_terms += ctx_terms
            ref_info.append({
                "reference": reference,
                "block": block,
                "value": holder_value if holder_value is not None else reference.get("value"),
                "model_terms": [t for t in model_terms if t],
            })
            pages_needed.add(mineru_page)

    if not ref_info:
        return metadata

    # Шаг 2: строим PyMuPDF-индекс ТОЛЬКО нужных страниц (не всех 68 — быстро) и
    # ищем точную ячейку значения ВНУТРИ bbox таблицы MinerU. Если ячейку не нашли
    # (значение в тексте, не в таблице; или не распознано) — падаем на bbox таблицы.
    page_index_cache: dict[int, PdfPageIndex | None] = {}
    document = None
    try:
        if content_type and content_type.split(";", 1)[0].strip().lower() == "application/pdf" \
                and PYMUPDF_INSTALLED and fitz is not None:
            document = fitz.open(local_path)
            for pno in pages_needed:
                page_index_cache[pno] = _build_single_page_index(document, pno)

        for info in ref_info:
            reference = info["reference"]
            block = info["block"]
            page_no = (block["page_idx"] or 0) + 1
            w, h = block["page_size"]
            x0, y0, x1, y1 = block["bbox"]
            table_norm = (
                x0 / w if w else 0.0, y0 / h if h else 0.0,
                x1 / w if w else 1.0, y1 / h if h else 1.0,
            )
            # значение характеристики без единиц («6,0 Вт» → «6,0», «150x90x100 мм» → «150x90x100»)
            value_text = _value_core_for_cell_search(info.get("value"))
            pi = page_index_cache.get(page_no)
            cell_rect = None
            if pi is not None and value_text:
                cell_rect = _locate_value_in_table_bbox(
                    pi, table_norm, value_text, info["model_terms"]
                )

            if cell_rect is not None:
                reference["bbox"] = _rect_to_bbox(cell_rect, _page_index_rect(pi))
                reference["locator_strategy"] = "mineru_cell"
                reference["geometry_source"] = "mineru_cell"
                reference["position_unverified"] = False
                metadata["matched_reference_count"] += 1
            else:
                # fallback: bbox всей таблицы MinerU (грубее, но на нужной таблице)
                reference["bbox"] = {
                    "x": round(x0, 3), "y": round(y0, 3),
                    "width": round(x1 - x0, 3), "height": round(y1 - y0, 3),
                    "x0": round(x0, 3), "y0": round(y0, 3),
                    "x1": round(x1, 3), "y1": round(y1, 3),
                    "left": round(x0, 3), "top": round(y0, 3),
                    "right": round(x1, 3), "bottom": round(y1, 3),
                    "norm_x0": round(min(1.0, max(0.0, table_norm[0])), 6),
                    "norm_y0": round(min(1.0, max(0.0, table_norm[1])), 6),
                    "norm_x1": round(min(1.0, max(0.0, table_norm[2])), 6),
                    "norm_y1": round(min(1.0, max(0.0, table_norm[3])), 6),
                }
                reference["locator_strategy"] = "mineru_table_bbox"
                reference["geometry_source"] = "mineru_table_bbox"
                reference["position_unverified"] = False
                metadata["table_bbox_fallback"] += 1
                metadata["matched_reference_count"] += 1
    finally:
        if document is not None:
            try:
                document.close()
            except Exception:
                pass

    metadata["applied"] = metadata["matched_reference_count"] > 0
    return metadata


def _match_mineru_block(
    candidates: list[dict[str, Any]],
    value_context: list[str],
    blocks: list[dict[str, Any]],
    reference: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Находит блок layout, лучше всего соответствующий reference.

    Скоринг: приоритет блокам, чей текст содержит и код модели (якорь), и
    значение характеристики — так значение привязывается к нужной таблице, а не
    к первой попавшейся. page_hint из reference повышает вес совпадений на той
    же странице."""
    # тексты-кандидаты якоря (код модели, цитата) и значения
    anchor_texts = [
        _normalize_match_text(c.get("text"))
        for c in candidates
        if c.get("text")
    ]
    anchor_texts = [t for t in anchor_texts if len(t) >= 3]
    value_texts = [_normalize_match_text(v) for v in value_context if v and len(str(v)) >= 1]
    value_texts = [t for t in value_texts if t]
    page_hint = _safe_page_number(reference.get("page")) or _safe_page_number(
        reference.get("page_number")
    )
    # Отдельные ячейки-значения из Markdown-строки quote. Когда одна и та же модель
    # есть в нескольких таблицах (параметры vs габариты), правильную таблицу выдаёт
    # именно совпадение ЯЧЕЕК ряда: '303','155','142' есть только в таблице габаритов.
    row_cells: list[str] = []
    for c in _split_table_row_cells(reference.get("quote_text")):
        cn = _normalize_match_text(c)
        # значимые ячейки: числа/размеры длиной >=2 (не одиночные цифры-шум)
        if cn and len(cn) >= 2 and re.search(r"\d", cn):
            row_cells.append(cn)

    best: Optional[dict[str, Any]] = None
    best_score = 0.0
    best_area = float("inf")
    best_page = float("inf")
    for block in blocks:
        norm_block = _normalize_match_text(block["text"])
        if not norm_block:
            continue
        score = 0.0
        anchor_hit = any(a in norm_block for a in anchor_texts)
        value_hit = any(v in norm_block for v in value_texts) if value_texts else False
        if anchor_hit:
            score += 0.6
        if value_hit:
            score += 0.3
        # доля ячеек ряда, реально присутствующих в блоке — отражает, что это ИМЕННО
        # та таблица, откуда взят ряд (а не другая с тем же кодом модели).
        if row_cells:
            hit_cells = sum(1 for c in row_cells if c in norm_block)
            score += 0.5 * (hit_cells / len(row_cells))
        if not anchor_hit and not value_hit and not row_cells:
            continue
        if score < 0.3:
            continue
        # бонус за совпадение страницы-подсказки от LLM: указание страницы обычно
        # надёжно, поэтому вес значимый — перевешивает совпадение по значению на
        # другой странице (одно и то же значение встречается в разных таблицах).
        if page_hint is not None and block["page_idx"] == page_hint - 1:
            score += 0.35
        x0, y0, x1, y1 = block["bbox"]
        area = max(1.0, (x1 - x0) * (y1 - y0))
        page = block["page_idx"] if block["page_idx"] is not None else 10**9
        # При равном score предпочитаем компактный блок (точнее ячейки),
        # при равной компактности — более раннюю страницу (детерминизм).
        better = (
            score > best_score
            or (abs(score - best_score) < 1e-9 and area < best_area)
            or (abs(score - best_score) < 1e-9 and abs(area - best_area) < 1e-9 and page < best_page)
        )
        if better:
            best_score = score
            best_area = area
            best_page = page
            best = {"block": block, "score": score}
    return best


async def _extract_via_mineru(payload: ExtractionRequest) -> dict[str, Any]:
    """Извлечение через облачный MinerU (mineru.net).

    MinerU меняет ТОЛЬКО источник таблиц: документ парсится в Markdown, где
    таблицы представлены как HTML (<table> с rowspan/colspan) — это точнее
    передаёт двухуровневые шапки, чем PyMuPDF. Дальше идёт тот же LLM-слой
    структурного извлечения (gpt-4.1), что и в openrouter-бэкенде.

    Геометрия/подсветка на этом этапе НЕ переписывается под MinerU-bbox:
    references привязываются к странице (page-anchor) через существующий
    PyMuPDF-энричмент по quote_text. Точный bbox из content_list.json —
    отдельная задача (известное ограничение).
    """
    if not (MINERU_ENABLED and MINERU_TOKEN):
        raise HTTPException(
            status_code=501,
            detail="Backend 'mineru' is disabled: set MINERU_ENABLED=1 and MINERU_TOKEN.",
        )
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is not set (required for structured extraction after MinerU parsing)",
        )
    if not payload.prompt and not payload.schema_payload:
        raise HTTPException(status_code=400, detail="schema or prompt is required")

    schema = normalize_json_schema(payload.schema_payload, payload.prompt)
    prompt = payload.prompt or "Extract structured data from the document and return JSON."
    prompt = (
        f"{prompt}\n"
        "Таблицы ниже даны как HTML (<table> с rowspan/colspan). "
        "Значение бери СТРОГО из ячейки на пересечении строки нужной модели и нужного столбца; "
        "учитывай объединённые ячейки (rowspan/colspan) при определении, к каким строкам относится значение. "
        "В quote_text помещай ТОЛЬКО код модели и искомое значение (напр. 'XM 3,2/4Т-0,18-G1 | 4,0'), "
        "а НЕ всю строку таблицы со всеми столбцами — это раздувает ответ. "
        "Верни результат строго как JSON schema response_format, без markdown и пояснений."
    )

    downloaded_file = await _download_file(payload.file_url)
    started_at = time.monotonic()
    markdown_text = ""
    mineru_meta: dict[str, Any] = {}
    mineru_layout: Any = None
    geometry_metadata: dict[str, Any] = {
        "enabled": GEOMETRY_ENRICHMENT_ENABLED,
        "provider": "mineru_layout",
        "applied": False,
        "page_count": 1,
        "errors": [],
    }
    try:
        with open(downloaded_file.local_path, "rb") as fh:
            pdf_bytes = fh.read()
        mineru_result = await extract_tables_mineru(
            pdf_bytes,
            downloaded_file.filename,
            token=MINERU_TOKEN,
            api_base=MINERU_API_BASE,
            language=MINERU_LANGUAGE,
            model_version=MINERU_MODEL_VERSION,
            page_ranges=MINERU_PAGE_RANGES,
            timeout=MINERU_POLL_TIMEOUT,
            interval=MINERU_POLL_INTERVAL,
        )
        markdown_text = mineru_result.markdown
        mineru_layout = mineru_result.layout
        mineru_meta = {
            "batch_id": mineru_result.batch_id,
            "page_count": mineru_result.page_count,
            "layout_available": mineru_layout is not None,
            **mineru_result.metadata,
        }
        logger.info(
            "MinerU parsing finished in %.2fs, content length=%d",
            time.monotonic() - started_at,
            len(markdown_text),
        )
    except MineruError as exc:
        logger.exception("MinerU parsing failed")
        _cleanup_downloaded_file(downloaded_file)
        raise HTTPException(status_code=502, detail=f"MinerU parsing failed: {exc}") from exc
    except HTTPException:
        _cleanup_downloaded_file(downloaded_file)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("MinerU parsing failed (unexpected)")
        _cleanup_downloaded_file(downloaded_file)
        raise HTTPException(
            status_code=502,
            detail=(
                f"MinerU parsing failed: {exc}\n\n"
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
        "response_format": _json_schema_response_format("mineru_extraction", schema),
    }
    provider_preferences = _build_openrouter_provider_preferences(require_parameters=True)
    if provider_preferences:
        payload_json["provider"] = provider_preferences

    headers = _build_openai_headers(OPENROUTER_API_KEY)
    if not LLM_IS_YANDEX:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL or "http://localhost:8005"
        headers["X-Title"] = OPENROUTER_APP_NAME or "extraction-service"

    mineru_pdf_tmp: tempfile.TemporaryDirectory | None = None
    try:
        extracted_data, provider_response, used_fallback = await _chat_completion_json(
            endpoint=f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
            headers=headers,
            payload=payload_json,
            provider_name="OpenRouter (mineru)",
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
                provider_name="OpenRouter schema repair (mineru)",
            )
        _normalize_references_in_place(extracted_data)
        # Геометрия для PyMuPDF работает только по PDF. DOCX/Excel сначала
        # конвертируем в PDF (как в openrouter-бэкенде), иначе координат не будет.
        geom_local_path = downloaded_file.local_path
        geom_content_type = downloaded_file.content_type
        if _looks_like_docx(downloaded_file.filename, downloaded_file.content_type) or \
                _looks_like_excel(downloaded_file.filename, downloaded_file.content_type):
            try:
                mineru_pdf_tmp = tempfile.TemporaryDirectory()
                geom_local_path = await to_thread.run_sync(
                    lambda: _convert_office_document_to_pdf(
                        downloaded_file.local_path, mineru_pdf_tmp.name
                    )
                )
                geom_content_type = "application/pdf"
            except Exception as exc:  # noqa: BLE001
                logger.warning("MinerU: office→PDF for geometry failed: %s", exc)

        if mineru_layout is not None:
            # Гибрид: MinerU layout определяет таблицу/страницу, PyMuPDF — ячейку.
            geometry_metadata = await to_thread.run_sync(
                lambda: _enrich_references_with_mineru_layout(
                    extracted_data,
                    mineru_layout,
                    local_path=geom_local_path,
                    content_type=geom_content_type,
                )
            )
            # MinerU не дал геометрии (DOCX/текстовый документ без bbox) —
            # падаем на полный PyMuPDF-энричмент по цитатам.
            if geometry_metadata.get("needs_pymupdf_fallback"):
                logger.info("MinerU layout has no geometry; falling back to PyMuPDF")
                geometry_metadata = await to_thread.run_sync(
                    lambda: _enrich_references_with_pdf_geometry(
                        extracted_data,
                        local_path=geom_local_path,
                        content_type=geom_content_type,
                    )
                )
        else:
            geometry_metadata = await to_thread.run_sync(
                lambda: _enrich_references_with_pdf_geometry(
                    extracted_data,
                    local_path=geom_local_path,
                    content_type=geom_content_type,
                )
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenRouter structured extraction failed (mineru)")
        raise HTTPException(
            status_code=502,
            detail=(
                f"Structured extraction failed after MinerU: {exc}\n\n"
                f"Extraction service traceback:\n{traceback.format_exc()}"
            ),
        ) from exc
    finally:
        _cleanup_downloaded_file(downloaded_file)
        if mineru_pdf_tmp is not None:
            mineru_pdf_tmp.cleanup()

    elapsed = time.monotonic() - started_at
    logger.info("MinerU full pipeline finished in %.2fs", elapsed)

    return jsonable_encoder({
        "analysis_id": payload.analysis_id,
        "file_id": payload.file_id,
        "file_type": payload.file_type,
        "backend": "mineru",
        "model_parse": "mineru",
        "model_extract": OPENROUTER_MODEL,
        "result": extracted_data,
        "extraction": {"pages": [_build_result_page(extracted_data, raw_text=markdown_text)]},
        "extraction_metadata": {
            "docling_version": None,
            "page_count": mineru_meta.get("page_count") or geometry_metadata.get("page_count"),
            "errors": [],
            "provider": "mineru+openrouter",
            "provider_usage": provider_response.get("usage"),
            "geometry": geometry_metadata,
            "mineru": mineru_meta,
            "docx_conversion": None,
        },
    })


def _paddle_bbox_to_reference_bbox(
    bbox_px: list[float], page_width: float, page_height: float
) -> dict[str, float]:
    """Конвертирует [x0,y0,x1,y1] в пикселях Paddle-рендера страницы в формат
    bbox, ожидаемый фронтендом ivolga (см. _rect_to_bbox выше в этом файле —
    та функция работает с fitz.Rect, эта — с плоским списком пикселей).
    Нормализованные norm_x0..norm_y1 (0..1) имеют наивысший приоритет в
    pdf-analyzer/src/services/mappers/bbox.ts::bboxWithUnits(), поэтому
    именно они, а не абсолютные пиксели/points, гарантируют корректную
    подсветку независимо от DPI рендера на стороне paddleocr-vl-service."""
    x0, y0, x1, y1 = bbox_px
    return {
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "left": x0, "top": y0, "right": x1, "bottom": y1,
        "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
        "norm_x0": x0 / page_width, "norm_y0": y0 / page_height,
        "norm_x1": x1 / page_width, "norm_y1": y1 / page_height,
    }


# Строки, которыми LLM обозначает "значения нет" — без отсева доезжают до
# интерфейса в виде "None кВт". Копия из api-gateway/paddleocr_vl_convert.py.
_EMPTY_VALUE_TOKENS = {"none", "null", "n/a", "na", "nan", "-", "—", "–", ""}


def _clean_value(value: Any) -> str:
    """Значение характеристики, где "пустышки" от LLM приведены к пустой строке."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _EMPTY_VALUE_TOKENS else text


def _spec_page_index(spec: dict[str, Any]) -> int | None:
    """Номер страницы, на которой найдена характеристика (0-based, как отдаёт
    paddleocr-vl-service). None, если источник не проставлен."""
    source = spec.get("source") or {}
    page_index = source.get("page_index")
    return page_index if isinstance(page_index, int) else None


def _build_page_variant_map(
    specifications: list[dict[str, Any]],
) -> dict[int, str]:
    """Карта "страница -> изделие" по характеристикам, у которых variant задан
    явно. Страница-якорь попадает в карту, только если ВСЕ её
    variant-размеченные характеристики принадлежат одному изделию:
    страница-переход, где встречаются метки сразу двух моделей, якорем не
    становится — там безопаснее старое поведение (дублирование), чем угадывание.

    От якорей принадлежность РАСПРОСТРАНЯЕТСЯ на последующие страницы до
    следующего якоря: в каталоге изделие описано разделом на несколько
    страниц, а явный заголовок модели (единственный источник variant для LLM)
    стоит только на первой странице раздела.

    Страницы ДО первого якоря намеренно остаются вне карты (общая шапка
    документа, титульный лист). Пустая карта = документ не даёт сигнала о
    разделении по страницам, поведение остаётся прежним.

    Копия логики из api-gateway/app/services/paddleocr_vl_convert.py — правки
    обязаны идти в обе копии одновременно."""
    variants_per_page: dict[int, set[str]] = {}
    for spec in specifications:
        variant = spec.get("variant")
        page_index = _spec_page_index(spec)
        if not variant or page_index is None:
            continue
        variants_per_page.setdefault(page_index, set()).add(variant)

    anchors = {
        page_index: next(iter(variants))
        for page_index, variants in variants_per_page.items()
        if len(variants) == 1
    }
    if not anchors:
        return {}

    max_page = max(
        [p for p in (_spec_page_index(s) for s in specifications) if p is not None]
        or [max(anchors)]
    )

    page_variants: dict[int, str] = {}
    current: str | None = None
    for page_index in range(max_page + 1):
        if page_index in anchors:
            current = anchors[page_index]
        elif page_index in variants_per_page:
            current = None
        if current is not None:
            page_variants[page_index] = current
    return page_variants


def _pick_single_product_name(specifications: list[dict[str, Any]]) -> str | None:
    """Имя единственного изделия документа среди размеченных LLM вариантов.

    Берём вариант с наибольшим числом характеристик: настоящее изделие описано
    десятками строк, а фантомы — единицами (на реальном ТЗ насоса НП-1 рядом
    с ним оказались "изделия" Мр.20, Мр.25 и 'A' — маркировки металлорукавов
    и буква с чертежа). Копия из api-gateway/paddleocr_vl_convert.py."""
    counts: dict[str, int] = {}
    for spec in specifications:
        variant = spec.get("variant")
        if isinstance(variant, str) and variant.strip():
            counts[variant] = counts.get(variant, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _paddleocr_vl_specs_to_products(
    specifications: list[dict[str, Any]],
    paddle_pages: list[dict[str, Any]],
    *,
    single_product: bool = False,
) -> list[dict[str, Any]]:
    """Конвертирует yandex.specifications[] (формат paddleocr-vl-service) в
    result.products[].characteristics[] (формат ivolga), группируя по variant.
    Характеристики БЕЗ variant (общие, не привязанные к конкретной модели)
    никогда не образуют отдельный продукт "Общее" — они приписываются к уже
    известным именованным моделям:

    1) если именованная модель ровно одна — общие характеристики уходят в неё;
    2) если моделей несколько (каталог паспорта на 2+ изделия) — характеристика
       уходит в изделие СВОЕЙ страницы (см. _build_page_variant_map);
       дублирование во все модели остаётся только как fallback, когда страница
       ничего не говорит о принадлежности;
    3) если именованных моделей нет вовсе, общие характеристики образуют
       единственный безымянный продукт (product_name=None)."""
    page_dims: dict[int, tuple[float, float]] = {}
    for idx, page in enumerate(paddle_pages):
        pruned = (page or {}).get("prunedResult") or {}
        width, height = pruned.get("width"), pruned.get("height")
        if width and height:
            page_dims[idx] = (float(width), float(height))

    if single_product:
        # Документ описывает ровно одно изделие (ТЗ): разметка вариантов от
        # LLM здесь шум, а не структура каталога.
        single_name = _pick_single_product_name(specifications)
        named_variants: list[str] = [single_name] if single_name else []
        page_variants: dict[int, str] = {}
    else:
        named_variants = []
        for spec in specifications:
            variant = spec.get("variant")
            if variant and variant not in named_variants:
                named_variants.append(variant)

        page_variants = _build_page_variant_map(specifications)

    products_by_variant: dict[str | None, dict[str, Any]] = {}
    order: list[str | None] = []

    def _ensure_product(key: str | None) -> dict[str, Any]:
        if key not in products_by_variant:
            products_by_variant[key] = {"product_name": key, "characteristics": []}
            order.append(key)
        return products_by_variant[key]

    for spec in specifications:
        variant = spec.get("variant")
        if single_product:
            # Всё изделие целиком — одна карточка, вне зависимости от variant.
            target_keys = named_variants or [None]
        elif variant:
            target_keys = [variant]
        elif len(named_variants) > 1:
            page_variant = page_variants.get(_spec_page_index(spec))
            target_keys = [page_variant] if page_variant else list(named_variants)
        else:
            target_keys = named_variants or [None]

        value = _clean_value(spec.get("value"))
        unit = spec.get("unit")
        # Единица без значения — мусор ("None кВт").
        value_text = f"{value} {unit}".strip() if (unit and value) else value

        source = spec.get("source") or {}
        page_index = source.get("page_index")
        blocks = source.get("blocks") or []

        references: list[dict[str, Any]] = []
        for block in blocks:
            block_bbox = block.get("precise_bbox") or block.get("block_bbox")
            if block_bbox is None or page_index is None:
                continue
            dims = page_dims.get(page_index)
            reference: dict[str, Any] = {
                # +1: paddleocr-vl-service использует 0-based page_index,
                # ivolga (references[].page) ожидает 1-based (см. Этап 0
                # аудита — _reference_to_span в api-gateway игнорирует page,
                # если он не положительный int).
                "page": page_index + 1,
                "page_number": page_index + 1,
                "quote_text": value,
                "anchor_text": spec.get("name"),
                "matched_text": value,
                "locator_strategy": "bbox",
                "geometry_source": "paddleocr_vl",
                "position_unverified": False,
            }
            if dims is not None:
                reference["bbox"] = _paddle_bbox_to_reference_bbox(block_bbox, *dims)
            references.append(reference)

        for key in target_keys:
            product = _ensure_product(key)
            product["characteristics"].append(
                {
                    "name": spec.get("name"),
                    "value": value_text,
                    "references": list(references),
                }
            )

    return [products_by_variant[v] for v in order]


async def _extract_via_paddleocr_vl(payload: ExtractionRequest) -> dict[str, Any]:
    """Извлечение через внешний микросервис paddleocr-vl-service (PaddleOCR-VL
    на GPU, либо Yandex Cloud Vision OCR — выбор зависит от того, какое из
    двух backend-имён запрошено: "paddleocr_vl" или "yandex_vision_ocr").

    В отличие от openrouter/mineru, geometry здесь НЕ требует
    _enrich_references_with_pdf_geometry — paddleocr-vl-service уже отдаёт
    точный bbox каждого значения (precise_locator на его стороне), поэтому
    эта функция ограничивается конвертацией формата ответа. Файл НЕ
    скачивается на этой стороне (в отличие от _extract_via_mineru) —
    paddleocr-vl-service сам скачивает по payload.file_url через свой
    /extract-specs-by-url эндпоинт.
    """
    backend = (payload.backend or "").strip().lower()
    if backend == "yandex_vision_ocr":
        if not YANDEX_VISION_OCR_ENABLED:
            raise HTTPException(
                status_code=501,
                detail="Backend 'yandex_vision_ocr' is disabled: set YANDEX_VISION_OCR_ENABLED=1.",
            )
        ocr_provider = "yandex_vision"
    else:
        if not PADDLEOCR_VL_ENABLED:
            raise HTTPException(
                status_code=501,
                detail="Backend 'paddleocr_vl' is disabled: set PADDLEOCR_VL_ENABLED=1.",
            )
        ocr_provider = "paddle"

    filename = _guess_filename_from_url(payload.file_url, None)
    started_at = time.monotonic()

    # async_mode: только для yandex_vision_ocr — облачная OCR+LLM цепочка на
    # больших документах (60-70+ страниц) может занимать дольше, чем разумно
    # держать один синхронный HTTP-запрос открытым (см. обсуждение с
    # пользователем: ReadTimeout после 30 минут ронял всю Celery-задачу и
    # терял весь прогресс). paddleocr_vl (GPU, локальный, быстрый) остаётся
    # синхронным — там этой проблемы нет, усложнение не оправдано.
    if payload.async_mode and backend == "yandex_vision_ocr":
        try:
            # Короткий таймаут: этот запрос лишь СТАВИТ job в очередь на
            # стороне paddleocr-vl-service и получает job_id обратно — сама
            # обработка документа идёт в фоне там же, см. job_queue.py.
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{PADDLEOCR_VL_SERVICE_URL.rstrip('/')}/extract-specs-by-url",
                    json={
                        "file_url": payload.file_url,
                        "filename": filename,
                        "ocr_provider": ocr_provider,
                        "async_mode": True,
                        "job_id": payload.job_id,
                        "product_model": payload.product_model,
                        "target_characteristic_names": payload.target_characteristic_names,
                    },
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.exception("paddleocr-vl-service returned an error while starting async job")
            raise HTTPException(
                status_code=502,
                detail=f"paddleocr-vl-service failed: {exc.response.status_code} {exc.response.text[:500]}",
            ) from exc
        except httpx.RequestError as exc:
            logger.exception("Failed to reach paddleocr-vl-service to start async job")
            raise HTTPException(status_code=502, detail=f"Failed to reach paddleocr-vl-service: {exc!r}") from exc

        logger.info(
            "paddleocr_vl async job started: job_id=%s elapsed=%.2fs",
            data.get("job_id"), time.monotonic() - started_at,
        )
        # Маркер "работа продолжается асинхронно" — вызывающая сторона
        # (api-gateway/tasks.py) распознаёт это по ключу "async" и не
        # трактует как готовый результат извлечения. Финальный результат
        # придёт позже через callback на /internal/extraction-callback.
        return {"async": True, "job_id": data.get("job_id")}

    try:
        async with httpx.AsyncClient(timeout=PADDLEOCR_VL_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{PADDLEOCR_VL_SERVICE_URL.rstrip('/')}/extract-specs-by-url",
                json={
                    "file_url": payload.file_url,
                    "filename": filename,
                    "ocr_provider": ocr_provider,
                    "product_model": payload.product_model,
                    "target_characteristic_names": payload.target_characteristic_names,
                },
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.exception("paddleocr-vl-service returned an error")
        raise HTTPException(
            status_code=502,
            detail=f"paddleocr-vl-service failed: {exc.response.status_code} {exc.response.text[:500]}",
        ) from exc
    except httpx.RequestError as exc:
        logger.exception("Failed to reach paddleocr-vl-service")
        raise HTTPException(status_code=502, detail=f"Failed to reach paddleocr-vl-service: {exc!r}") from exc

    paddle_pages = (data.get("paddle") or {}).get("pages") or []
    yandex_result = data.get("yandex") or {}
    specifications = yandex_result.get("specifications") or []

    # ТЗ — заказ на ОДНО изделие: все его характеристики относятся к нему.
    products = _paddleocr_vl_specs_to_products(
        specifications, paddle_pages, single_product=(payload.file_type == "tz")
    )
    elapsed = time.monotonic() - started_at
    logger.info(
        "paddleocr_vl extraction finished in %.2fs, %d specifications -> %d products",
        elapsed, len(specifications), len(products),
    )

    return jsonable_encoder({
        "analysis_id": payload.analysis_id,
        "file_id": payload.file_id,
        "file_type": payload.file_type,
        "backend": backend or "paddleocr_vl",
        "model_parse": ocr_provider,
        "model_extract": "yandex-deepseek-v4-flash",
        "result": {"products": products},
        "extraction": {"pages": [_build_result_page({"products": products})]},
        # Сырой ответ paddleocr-vl-service ДО конвертации в products — Paddle
        # layout-parsing (bbox/текст по блокам) + LLM-structuring specifications
        # (имя/значение/variant до группировки по изделиям). Нужен отдельно от
        # "result", чтобы можно было скачать и посмотреть, что именно вернул
        # OCR+LLM этап, не путая с уже причёсанным результатом конвертации
        # (см. postprocess_extraction_result, которая кладёт это в debug-дамп).
        "raw_ocr": data,
        "extraction_metadata": {
            "docling_version": None,
            "page_count": len(paddle_pages),
            "errors": [],
            "provider": f"paddleocr-vl-service/{ocr_provider}",
            "provider_usage": None,
            "geometry": {
                "enabled": True,
                "provider": "paddleocr_vl_precise_locator",
                "applied": True,
                "page_count": len(paddle_pages),
            },
            "ocr_provider": ocr_provider,
            "pages_processed": yandex_result.get("pages_processed"),
            "pages_failed": yandex_result.get("pages_failed"),
            "has_text_layer": yandex_result.get("has_text_layer"),
            "ocr_fallback_used": yandex_result.get("ocr_fallback_used"),
            "docx_conversion": None,
        },
    })


# Большие технические паспорта/руководства (60-70+ страниц) почти целиком
# релевантны по ключевым словам характеристик — фильтрация страниц не может
# сильно сократить такой документ. Чтобы не терять модели/характеристики с
# "хвоста" документа, лимит подобран так, чтобы вместить типичный крупный
# паспорт целиком (наблюдался реальный payload ~254k символов на 67 страниц).
PDFPLUMBER_MAX_PAYLOAD_CHARS = 400000
# Completion-лимит для pdfplumber-бэкенда отдельный от общего OPENROUTER_MAX_TOKENS:
# документ с большим payload обычно содержит МНОГО моделей/характеристик, ответ
# JSON получается длиннее, чем у остальных бэкендов — иначе ответ обрывается
# (finish_reason=length) и весь JSON становится невалидным.
PDFPLUMBER_MAX_TOKENS = 32000

PDFPLUMBER_BBOX_PROMPT_RULES = (
    "\n\nВАЖНО: ФОРМАТ ДОКУМЕНТА НИЖЕ — НЕ MARKDOWN. Если в инструкциях выше "
    "упоминались Markdown-таблицы со строками '|' и разделителем '| --- |' — "
    "это НЕ ПРИМЕНИМО к тексту ниже. Реальный формат другой (см. правила координат).\n\n"
    "ПРАВИЛА ЧТЕНИЯ ДОКУМЕНТА И КООРДИНАТ (bbox):\n"
    "Документ разбит на страницы ('=== СТРАНИЦА N ==='), внутри — на таблицы "
    "('-- Таблица K --') и текст вне таблиц. Каждый фрагмент помечен точными "
    'координатами в формате [bbox=[x0, top, x1, bottom]] "текст". Эти координаты '
    "извлечены программно из PDF (через pdfplumber) — они АБСОЛЮТНО ТОЧНЫЕ.\n"
    "• Внутри '-- Таблица K --' фрагменты идут по ЯЧЕЙКАМ строка за строкой — "
    "каждая непустая ячейка таблицы выведена как отдельный [bbox=...] с "
    "текстом именно этой ячейки, а НЕ всей строки и не всей таблицы. "
    "ТАБЛИЦА МОЖЕТ СОДЕРЖАТЬ МНОГО МОДЕЛЕЙ — извлекай характеристики КАЖДОЙ "
    "модели, не только первой попавшейся или той, что кажется 'главной'. Если "
    "пользователь явно не указал конкретную модель — извлеки ВСЕ модели.\n"
    "• Таблица может быть организована ДВУМЯ способами — определи по первым "
    "1-2 строкам:\n"
    "  (а) ОБЫЧНАЯ: одна СТРОКА = одна модель (первый столбец — код модели, "
    "остальные столбцы — характеристики с единицами в заголовке).\n"
    "  (б) ТРАНСПОНИРОВАННАЯ: одна СТРОКА = одна характеристика (первый "
    "столбец — название параметра типа 'Подача, м3/ч', 'Напор, м', "
    "'Масса, кг'), а КОДЫ МОДЕЛЕЙ идут в шапке как отдельная строка-заголовок "
    "('Типоразмер насоса', далее сама строка с кодами '20-50', '20-110', "
    "'32-150', ...) — тогда каждый СТОЛБЕЦ (кроме первого) = одна модель, а "
    "значение характеристики для этой модели — ячейка на пересечении СТРОКИ "
    "параметра и СТОЛБЦА модели. В таком случае извлеки ВСЕ модели-столбцы "
    "(их часто 5-10+ в одной таблице), а не 1-2 первых.\n"
    "• Одна и та же таблица физически может продолжаться в нескольких "
    "'-- Таблица K --' на СОСЕДНИХ страницах (одна широкая таблица с моделями "
    "разбита pdfplumber на несколько блоков, т.к. не помещается на одной "
    "странице) — тогда это ОДНА логическая таблица, объедини все её столбцы/"
    "модели при извлечении.\n"
    "• Если под ячейкой таблицы дополнительно идут строки с пометкой "
    "'(строка внутри ячейки выше)' — это значит сама ячейка МНОГОСТРОЧНАЯ "
    "(содержит несколько характеристик, перечисленных через перенос строки, "
    "как бывает в ТЗ: одна ячейка 'Характеристики Товара' содержит все "
    "параметры сразу). В этом случае ВСЕГДА предпочитай bbox КОНКРЕТНОЙ "
    "под-строки (более узкий, точно содержащий нужное значение), а НЕ bbox "
    "всей ячейки целиком — иначе подсветка растянется на весь список "
    "характеристик вместо одной.\n"
    "ТЫ НИКОГДА НЕ ВЫЧИСЛЯЕШЬ И НЕ ПЕРЕСЧИТЫВАЕШЬ bbox. Для каждой reference "
    "найди ближайший входной фрагмент [bbox=...] (по правилам выше — предпочитая "
    "самый узкий/точный из подходящих), который буквально содержит нужное "
    "значение (или код модели), и скопируй его bbox ПОБАЙТОВО в виде строки "
    "со следующим точным форматом (без пробелов, без лишних символов):\n"
    "PDFPLUMBER_BBOX|<номер_страницы>|<x0>,<top>,<x1>,<bottom>\n"
    "Например: PDFPLUMBER_BBOX|2|399.12,679.45,430.88,691.02\n"
    "ЭТУ СТРОКУ ПОМЕЩАЙ ИМЕННО В ПОЛЕ locator_text — НЕ в поле bbox! Поле "
    "bbox в схеме — это отдельный объект с числовыми полями (x0,y0,x1,y1 и "
    "т.п.), а не строка PDFPLUMBER_BBOX|...; если ты не уверена в конкретных "
    "значениях этого объекта — оставь поле bbox полностью пустым/null и "
    "положи маркер ТОЛЬКО в locator_text (постобработка сама построит "
    "правильный объект bbox из этой строки).\n"
    "Числа x0/top/x1/bottom — скопированы дословно из входного [bbox=...], "
    "БЕЗ округления, БЕЗ придумывания новых значений. Номер страницы — из "
    "заголовка '=== СТРАНИЦА N ...' того блока, где найден фрагмент. "
    "Если для характеристики нет точного совпадающего фрагмента в исходном "
    "тексте — оставь locator_text пустым/null, НЕ придумывай координаты.\n\n"
    "‼️ ЭТО ПРАВИЛО ОТМЕНЯЕТ пример и описание поля locator_text из инструкций "
    "ВЫШЕ. Там мог быть показан пример вида "
    '\'{"locator_text": null, "bbox": null, "locator_strategy": "table"}\' — '
    "НЕ копируй этот пример буквально для locator_text/bbox: для ЭТОГО "
    "документа (обработка через pdfplumber) locator_text ВСЕГДА должен "
    "содержать строку PDFPLUMBER_BBOX|... (если подходящий фрагмент найден), "
    "а НЕ null. Правильный пример ответа для ЭТОГО документа:\n"
    '{"name": "Напор", "value": "80 м", "references": [{"quote_text": '
    '"80 м", "anchor_text": "Напор", "page": 2, "locator_strategy": "table", '
    '"confidence": 1.0, "locator_text": "PDFPLUMBER_BBOX|2|399.12,679.45,'
    '430.88,691.02", "bbox": null}]}\n'
    "Заполняй locator_text маркером для КАЖДОЙ reference, где во входном "
    "тексте есть совпадающий фрагмент [bbox=...] — это не опция, а "
    "обязательная часть ответа."
)

_PDFPLUMBER_BBOX_RE = re.compile(
    r"PDFPLUMBER_BBOX\|(\d+)\|(-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)"
)


def _find_pdfplumber_bbox_marker(
    reference: dict[str, Any]
) -> "tuple[re.Match, str] | None":
    """Ищет маркер координат PDFPLUMBER_BBOX|... среди ЛЮБОГО строкового
    значения в reference — не в конкретном ожидаемом поле.

    LLM непредсказуемо кладёт маркер в разные поля схемы в зависимости от
    документа: наблюдалось в locator_text (как задумано), bbox (по смыслу
    имени поля), locator_strategy (тоже по смыслу — 'как искать позицию').
    Вместо того чтобы угадывать очередное поле, сканируем ВСЕ строковые
    значения объекта — маркер достаточно специфичен (жёсткий префикс
    'PDFPLUMBER_BBOX|'), чтобы не давать ложных срабатываний на обычном
    тексте цитат. Возвращает (match, имя_поля) — имя поля нужно вызывающему,
    чтобы очистить его от служебной строки в финальном ответе."""
    for field_name, value in reference.items():
        if isinstance(value, str) and "PDFPLUMBER_BBOX|" in value:
            m = _PDFPLUMBER_BBOX_RE.search(value)
            if m:
                return m, field_name
    return None


def _apply_pdfplumber_bbox_copies(
    extracted_data: dict[str, Any],
    page_sizes: dict[int, tuple[float, float]],
) -> dict[str, Any]:
    """Постобработка ответа LLM для pdfplumber-бэкенда.

    LLM НЕ ищет и не вычисляет координаты — она только скопировала строку
    'PDFPLUMBER_BBOX|page|x0,top,x1,bottom' в locator_text (см. промпт). Здесь
    мы её парсим, проверяем на вменяемость (число, страница существует, bbox
    внутри границ страницы) и собираем стандартный bbox-объект. Если строка
    отсутствует/невалидна — References остаётся БЕЗ bbox (position_unverified),
    мы намеренно НЕ подключаем сюда эвристический поиск (в этом и есть чистота
    подхода: либо bbox дословно скопирован, либо его нет)."""
    metadata: dict[str, Any] = {
        "enabled": True,
        "provider": "pdfplumber",
        "applied": False,
        "reference_count": 0,
        "matched_reference_count": 0,
        "clipped_count": 0,
        "errors": [],
    }

    _normalize_references_in_place(extracted_data)

    for holder, references, _ancestors in _reference_iter_with_ancestors(extracted_data):
        for index, reference in enumerate(references):
            if isinstance(reference, str):
                reference = {"quote_text": reference, "anchor_text": reference}
                references[index] = reference
            if not isinstance(reference, dict):
                continue
            metadata["reference_count"] += 1

            # LLM должна класть маркер строго в locator_text, но на практике
            # непредсказуемо кладёт его в разные поля (bbox, locator_strategy
            # и т.п. — по смыслу имени поля, несмотря на явную инструкцию).
            # Сканируем ВСЕ строковые значения reference вместо угадывания
            # конкретного поля (см. _find_pdfplumber_bbox_marker).
            found = _find_pdfplumber_bbox_marker(reference)
            if not found:
                reference["position_unverified"] = True
                continue
            match, source_field = found
            # bbox/locator_strategy перезаписываются ниже безусловно; если
            # маркер оказался в другом поле (quote_text, anchor_text и т.п.),
            # очищаем его — иначе служебная строка попадёт во фронт как якобы
            # цитата/якорь.
            if source_field not in ("bbox", "locator_strategy"):
                reference[source_field] = None

            page_number = int(match.group(1))
            x0, top, x1, bottom = (float(match.group(i)) for i in range(2, 6))
            page_size = page_sizes.get(page_number)
            if page_size is None:
                reference["position_unverified"] = True
                continue
            w, h = page_size

            clipped = False
            nx0 = max(0.0, min(x0, w))
            nx1 = max(0.0, min(x1, w))
            ntop = max(0.0, min(top, h))
            nbottom = max(0.0, min(bottom, h))
            if (nx0, ntop, nx1, nbottom) != (x0, top, x1, bottom):
                clipped = True
            if nx1 <= nx0 or nbottom <= ntop:
                reference["position_unverified"] = True
                continue

            reference["page"] = page_number
            reference["page_number"] = page_number
            reference["bbox"] = {
                "x": round(nx0, 3), "y": round(ntop, 3),
                "width": round(nx1 - nx0, 3), "height": round(nbottom - ntop, 3),
                "x0": round(nx0, 3), "y0": round(ntop, 3),
                "x1": round(nx1, 3), "y1": round(nbottom, 3),
                "left": round(nx0, 3), "top": round(ntop, 3),
                "right": round(nx1, 3), "bottom": round(nbottom, 3),
                "norm_x0": round(min(1.0, max(0.0, nx0 / w)), 6) if w else 0.0,
                "norm_y0": round(min(1.0, max(0.0, ntop / h)), 6) if h else 0.0,
                "norm_x1": round(min(1.0, max(0.0, nx1 / w)), 6) if w else 1.0,
                "norm_y1": round(min(1.0, max(0.0, nbottom / h)), 6) if h else 1.0,
            }
            reference["locator_strategy"] = "pdfplumber_llm_copy"
            reference["geometry_source"] = "pdfplumber"
            reference["position_unverified"] = False
            metadata["matched_reference_count"] += 1
            if clipped:
                metadata["clipped_count"] += 1

    metadata["applied"] = metadata["matched_reference_count"] > 0
    return metadata


async def _extract_via_pdfplumber(payload: ExtractionRequest) -> dict[str, Any]:
    """Bbox-first геометрия: координаты ячеек вычисляются ГЕОМЕТРИЧЕСКИ через
    pdfplumber (find_tables + within_bbox) ДО обращения к LLM. LLM получает уже
    готовые пары «текст ↔ bbox» и только копирует нужный bbox дословно в ответ —
    она не ищет и не вычисляет координаты постфактум.

    Это принципиально отличается от openrouter/mineru: там LLM сначала
    переформулирует значение характеристики в текст, а геометрия потом
    эвристически ищет совпадение этого текста в PDF — что ломается на разнице
    единиц измерения, кириллице/латинице, коротких неоднозначных значениях и
    т.п. Здесь эвристики нет вообще: либо LLM скопировала верный bbox, либо
    reference остаётся без координат (см. _apply_pdfplumber_bbox_copies).
    """
    if not PDFPLUMBER_INSTALLED:
        raise HTTPException(
            status_code=501,
            detail="Backend 'pdfplumber' is disabled: pdfplumber package is not installed.",
        )
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")
    if not payload.prompt and not payload.schema_payload:
        raise HTTPException(status_code=400, detail="schema or prompt is required")

    schema = normalize_json_schema(payload.schema_payload, payload.prompt)
    prompt = payload.prompt or "Extract structured data from the document and return JSON."
    prompt = (
        f"{prompt}\n"
        "Верни результат строго как JSON schema response_format. "
        "Не добавляй markdown, пояснения или кодовые блоки."
        f"{PDFPLUMBER_BBOX_PROMPT_RULES}"
    )

    downloaded_file = await _download_file(payload.file_url)
    started_at = time.monotonic()
    geometry_metadata: dict[str, Any] = {
        "enabled": True,
        "provider": "pdfplumber",
        "applied": False,
        "reference_count": 0,
        "matched_reference_count": 0,
        "errors": [],
    }
    pdfplumber_pdf_tmp: tempfile.TemporaryDirectory | None = None
    try:
        pdf_local_path = downloaded_file.local_path
        if _looks_like_docx(downloaded_file.filename, downloaded_file.content_type) or \
                _looks_like_excel(downloaded_file.filename, downloaded_file.content_type):
            pdfplumber_pdf_tmp = tempfile.TemporaryDirectory()
            pdf_local_path = await to_thread.run_sync(
                lambda: _convert_office_document_to_pdf(
                    downloaded_file.local_path, pdfplumber_pdf_tmp.name
                )
            )

        pages = await to_thread.run_sync(lambda: extract_pdf_geometry(pdf_local_path))
        # page_sizes — из ВСЕХ страниц (нужны для валидации bbox независимо от
        # того, какие страницы попали в payload после фильтрации ниже).
        page_sizes = {p.page_number: (p.width, p.height) for p in pages}
        # Большие документы (много титульных/сервисных страниц — техника
        # безопасности, гарантии, содержание) иначе обрезаются посередине
        # таблицы характеристик из-за лимита max_chars. Отбираем страницы с
        # таблицами/ключевыми словами (как делает openrouter-бэкенд), остальное
        # не отправляем в LLM вообще — не в обрезке, а в целевом отборе.
        relevant_pages, pages_filtered = await to_thread.run_sync(
            lambda: filter_pdfplumber_pages(pages, max_chars=PDFPLUMBER_MAX_PAYLOAD_CHARS)
        )
        document_text, truncated = build_pdfplumber_llm_payload(
            relevant_pages, max_chars=PDFPLUMBER_MAX_PAYLOAD_CHARS
        )
        logger.info(
            "pdfplumber geometry extracted: %d/%d pages kept (filtered=%s), payload_len=%d, truncated=%s",
            len(relevant_pages), len(pages), pages_filtered, len(document_text), truncated,
        )

        # pdfplumber не умеет OCR: на сканах (нет текстового слоя) и PDF с битой
        # кодировкой шрифта (текст — мешанина символов вроде 'u E', '5 g FE')
        # он либо не находит ничего, либо LLM получает нечитаемый мусор — итог
        # в обоих случаях 0 характеристик. openrouter-бэкенд уже умеет
        # детектировать оба случая и делать OCR-fallback, поэтому передаём
        # документ туда вместо того, чтобы возвращать пустой результат.
        plain_text = await to_thread.run_sync(lambda: plain_text_for_quality_check(pages))
        if not _text_layer_is_usable(plain_text):
            logger.info(
                "pdfplumber: text layer unusable (scan or garbled font) — "
                "falling back to openrouter backend (has OCR)"
            )
            _cleanup_downloaded_file(downloaded_file)
            if pdfplumber_pdf_tmp is not None:
                pdfplumber_pdf_tmp.cleanup()
            fallback_result = await _extract_via_openrouter(payload)
            fallback_result["backend"] = "pdfplumber"
            extraction_metadata = fallback_result.get("extraction_metadata")
            if isinstance(extraction_metadata, dict):
                extraction_metadata["pdfplumber_fallback"] = "openrouter (unusable text layer)"
            return fallback_result
    except HTTPException:
        _cleanup_downloaded_file(downloaded_file)
        if pdfplumber_pdf_tmp is not None:
            pdfplumber_pdf_tmp.cleanup()
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("pdfplumber geometry extraction failed")
        _cleanup_downloaded_file(downloaded_file)
        if pdfplumber_pdf_tmp is not None:
            pdfplumber_pdf_tmp.cleanup()
        raise HTTPException(
            status_code=502,
            detail=(
                f"pdfplumber extraction failed: {exc}\n\n"
                f"Extraction service traceback:\n{traceback.format_exc()}"
            ),
        ) from exc

    messages = _build_openrouter_messages_from_text(
        prompt=prompt,
        document_text=document_text,
        filename=downloaded_file.filename,
    )
    payload_json: dict[str, Any] = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": PDFPLUMBER_MAX_TOKENS,
        "stream": False,
        "response_format": _json_schema_response_format("pdfplumber_extraction", schema),
    }
    provider_preferences = _build_openrouter_provider_preferences(require_parameters=True)
    if provider_preferences:
        payload_json["provider"] = provider_preferences

    headers = _build_openai_headers(OPENROUTER_API_KEY)
    if not LLM_IS_YANDEX:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL or "http://localhost:8005"
        headers["X-Title"] = OPENROUTER_APP_NAME or "extraction-service"

    try:
        extracted_data, provider_response, used_fallback = await _chat_completion_json(
            endpoint=f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
            headers=headers,
            payload=payload_json,
            provider_name="OpenRouter (pdfplumber)",
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
                provider_name="OpenRouter schema repair (pdfplumber)",
            )
        geometry_metadata = await to_thread.run_sync(
            lambda: _apply_pdfplumber_bbox_copies(extracted_data, page_sizes)
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenRouter structured extraction failed (pdfplumber)")
        raise HTTPException(
            status_code=502,
            detail=(
                f"Structured extraction failed after pdfplumber: {exc}\n\n"
                f"Extraction service traceback:\n{traceback.format_exc()}"
            ),
        ) from exc
    finally:
        _cleanup_downloaded_file(downloaded_file)
        if pdfplumber_pdf_tmp is not None:
            pdfplumber_pdf_tmp.cleanup()

    elapsed = time.monotonic() - started_at
    logger.info("pdfplumber full pipeline finished in %.2fs", elapsed)

    return jsonable_encoder({
        "analysis_id": payload.analysis_id,
        "file_id": payload.file_id,
        "file_type": payload.file_type,
        "backend": "pdfplumber",
        "model_parse": "pdfplumber",
        "model_extract": OPENROUTER_MODEL,
        "result": extracted_data,
        "extraction": {"pages": [_build_result_page(extracted_data, raw_text=document_text)]},
        "extraction_metadata": {
            "docling_version": None,
            "page_count": len(page_sizes),
            "pages_sent_to_llm": len(relevant_pages),
            "pages_filtered": pages_filtered,
            "payload_truncated": truncated,
            "errors": [],
            "provider": "pdfplumber+openrouter",
            "provider_usage": provider_response.get("usage"),
            "geometry": geometry_metadata,
            "docx_conversion": None,
        },
    })


async def _extract_via_docling_local(payload: ExtractionRequest) -> dict[str, Any]:
    _raise_docling_backend_unavailable("docling_local")


async def _extract_via_docling_remote(payload: ExtractionRequest) -> dict[str, Any]:
    _raise_docling_backend_unavailable("docling_remote")


async def _extract_via_openrouter(payload: ExtractionRequest) -> dict[str, Any]:
    log_extra = {
        "analysis_id": payload.analysis_id,
        "file_id": payload.file_id,
        "file_type": payload.file_type,
    }
    stage_timings: dict[str, float] = {}

    if not payload.prompt and not payload.schema_payload:
        raise HTTPException(status_code=400, detail="schema or prompt is required")
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")

    schema = normalize_json_schema(payload.schema_payload, payload.prompt)
    prompt = payload.prompt or "Extract structured data from the document and return JSON."
    prompt = (
        f"{prompt}\n"
        "Верни результат строго как JSON schema response_format. "
        "Не добавляй markdown, пояснения или кодовые блоки."
    )

    download_started_at = time.monotonic()
    downloaded_file = await _download_file(payload.file_url)
    stage_timings["download"] = time.monotonic() - download_started_at
    is_docx = _looks_like_docx(downloaded_file.filename, downloaded_file.content_type)
    is_excel = _looks_like_excel(downloaded_file.filename, downloaded_file.content_type)
    is_pdf = _looks_like_pdf(downloaded_file.filename, downloaded_file.content_type)
    logger.info(
        "File detected: filename=%r content_type=%r is_docx=%s is_excel=%s is_pdf=%s downloaded_in=%.2fs",
        downloaded_file.filename, downloaded_file.content_type, is_docx, is_excel, is_pdf,
        stage_timings["download"],
        extra={**log_extra, "step": "download"},
    )
    docx_conversion_metadata: dict[str, Any] | None = None
    excel_conversion_metadata: dict[str, Any] | None = None
    pdf_conversion_metadata: dict[str, Any] | None = None
    docx_pdf_temp_dir: tempfile.TemporaryDirectory | None = None

    text_conversion_started_at = time.monotonic()
    if is_docx:
        try:
            docx_conversion_metadata = await to_thread.run_sync(
                lambda: _convert_docx_to_structured_text(downloaded_file.local_path)
            )
        except Exception as exc:
            logger.exception(
                "DOCX conversion failed after %.2fs",
                time.monotonic() - text_conversion_started_at,
                extra={**log_extra, "step": "docx_conversion"},
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"DOCX conversion failed: {exc}\n\n"
                    f"Extraction service traceback:\n{traceback.format_exc()}"
                ),
            ) from exc
    elif is_excel:
        try:
            excel_conversion_metadata = await to_thread.run_sync(
                lambda: _convert_xlsx_to_structured_text(downloaded_file.local_path)
            )
        except Exception as exc:
            logger.exception(
                "Excel conversion failed after %.2fs",
                time.monotonic() - text_conversion_started_at,
                extra={**log_extra, "step": "excel_conversion"},
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Excel conversion failed: {exc}\n\n"
                    f"Extraction service traceback:\n{traceback.format_exc()}"
                ),
            ) from exc
    elif is_pdf and PYMUPDF_INSTALLED:
        # Для PDF: сначала пробуем извлечь текст локально (searchable или OCR).
        # Если текст получен — отправляем как текст (надёжнее, чем file-parser плагин).
        # Если текст пустой — падаем обратно на file-parser (нативный путь OpenRouter).
        # Для Yandex: file-parser недоступен, поэтому всегда используем текстовый путь.
        try:
            pdf_conversion_metadata = await to_thread.run_sync(
                lambda: _convert_pdf_to_structured_text(
                    downloaded_file.local_path,
                    target_names=_extract_target_names_from_prompt(prompt),
                )
            )
            logger.info(
                "PDF text extraction: ocr_applied=%s text_len=%d page_count=%s elapsed=%.2fs",
                pdf_conversion_metadata.get("ocr_applied"),
                len(pdf_conversion_metadata.get("text") or ""),
                pdf_conversion_metadata.get("page_count"),
                time.monotonic() - text_conversion_started_at,
                extra={**log_extra, "step": "pdf_text_extraction"},
            )
        except Exception as exc:
            logger.warning(
                "PDF text extraction failed after %.2fs, falling back to file-parser: %s",
                time.monotonic() - text_conversion_started_at, exc,
                exc_info=True,
                extra={**log_extra, "step": "pdf_text_extraction"},
            )
            pdf_conversion_metadata = None
    stage_timings["text_conversion"] = time.monotonic() - text_conversion_started_at

    # Определяем текст документа для text-пути
    use_text_path = False
    document_text = ""
    if is_docx and docx_conversion_metadata:
        use_text_path = True
        document_text = (docx_conversion_metadata or {}).get("text", "")
    elif is_excel and excel_conversion_metadata:
        use_text_path = True
        document_text = (excel_conversion_metadata or {}).get("text", "")
    elif is_pdf and pdf_conversion_metadata:
        extracted_text = pdf_conversion_metadata.get("text", "")
        ocr_was_applied = pdf_conversion_metadata.get("ocr_applied", False)
        if extracted_text.strip() and not ocr_was_applied:
            use_text_path = True
            document_text = extracted_text
        elif extracted_text.strip() and ocr_was_applied and LLM_IS_YANDEX:
            use_text_path = True
            document_text = extracted_text
        elif extracted_text.strip() and ocr_was_applied:
            logger.info("Scanned PDF: OCR text available but preferring file-parser for better table reading")
        elif LLM_IS_YANDEX:
            # Yandex doesn't support file content parts — if text extraction yielded nothing,
            # we still can't send the raw file; raise an informative error.
            raise HTTPException(
                status_code=422,
                detail="Yandex AI: PDF has no extractable text layer and file upload is not supported.",
            )
    elif LLM_IS_YANDEX and _looks_like_image(downloaded_file.filename, downloaded_file.content_type):
        raise HTTPException(
            status_code=422,
            detail="Yandex AI: image extraction is not supported (no vision API).",
        )

    logger.info(
        "Extraction path: use_text_path=%s document_text_len=%d",
        use_text_path, len(document_text),
        extra={**log_extra, "step": "path_selection"},
    )

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

    # VLM-гибрид: для PDF на текстовом пути дополнительно прикладываем изображения
    # страниц с таблицами, чтобы модель сверяла значения по картинке (устойчивее к
    # сложным двухуровневым шапкам). При ошибке — молча остаёмся на тексте.
    vision_pages_sent = 0
    if (
        VISION_TABLES_ENABLED
        and use_text_path
        and is_pdf
        and not LLM_IS_YANDEX
        and PYMUPDF_INSTALLED
    ):
        try:
            image_parts = await to_thread.run_sync(
                lambda: _render_table_pages_to_images(
                    downloaded_file.local_path,
                    dpi=VISION_DPI,
                    max_pages=VISION_MAX_TABLE_PAGES,
                )
            )
            if image_parts:
                user_msg = messages[0]
                text_content = user_msg["content"]
                new_content: list[dict[str, Any]] = [
                    {"type": "text", "text": (
                        text_content
                        + "\n\nНиже приложены изображения страниц документа с таблицами. "
                        "Сверяй значения по картинке: ячейку бери на пересечении строки нужной "
                        "модели и нужного столбца шапки."
                    )},
                ]
                for part in image_parts:
                    new_content.append({"type": part["type"], "image_url": part["image_url"]})
                user_msg["content"] = new_content
                vision_pages_sent = len(image_parts)
                logger.info("vision-hybrid: attached %d table-page images", vision_pages_sent)
        except Exception:  # noqa: BLE001
            logger.exception("vision-hybrid failed; continuing with text-only path")

    payload_json: dict[str, Any] = {
        "model": VISION_MODEL if vision_pages_sent else OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "stream": False,
        "response_format": _json_schema_response_format("openrouter_extraction", schema),
    }
    provider_preferences = _build_openrouter_provider_preferences(require_parameters=True)
    if provider_preferences:
        payload_json["provider"] = provider_preferences
    # file-parser плагин — только для OpenRouter, когда не используем text-путь
    if not LLM_IS_YANDEX and not use_text_path and not _looks_like_image(downloaded_file.filename, downloaded_file.content_type):
        payload_json["plugins"] = [
            {
                "id": "file-parser",
                "pdf": {"engine": OPENROUTER_PDF_ENGINE},
            }
        ]

    headers = _build_openai_headers(OPENROUTER_API_KEY)
    if not LLM_IS_YANDEX:
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
        llm_started_at = time.monotonic()
        extracted_data, provider_response, used_fallback = await _chat_completion_json(
            endpoint=f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
            headers=headers,
            payload=payload_json,
            provider_name="OpenRouter",
            repair_schema=schema,
            repair_model=OPENROUTER_MODEL,
        )
        stage_timings["llm_call"] = time.monotonic() - llm_started_at
        logger.info(
            "OpenRouter chat completion done in %.2fs used_fallback=%s usage=%s",
            stage_timings["llm_call"], used_fallback, provider_response.get("usage"),
            extra={**log_extra, "step": "llm_call"},
        )
        if used_fallback and _needs_schema_repair(extracted_data, schema):
            logger.info(
                "Schema repair triggered: fallback response missing required schema fields",
                extra={**log_extra, "step": "schema_repair"},
            )
            repair_started_at = time.monotonic()
            extracted_data = await _repair_json_to_schema(
                endpoint=f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
                headers=headers,
                model=OPENROUTER_MODEL,
                schema=schema,
                candidate=extracted_data,
                provider_name="OpenRouter schema repair",
            )
            logger.info(
                "Schema repair finished in %.2fs",
                time.monotonic() - repair_started_at,
                extra={**log_extra, "step": "schema_repair"},
            )
        # Нормализуем references: некоторые модели возвращают dict вместо list
        _normalize_references_in_place(extracted_data)
        # Office-документы (DOCX/Excel) конвертируем в PDF, чтобы геометрия привязала
        # координаты цитат к страницам PDF-превью (вьювер показывает именно его).
        if is_docx or is_excel:
            office_meta = docx_conversion_metadata if is_docx else excel_conversion_metadata
            doc_kind = "DOCX" if is_docx else "Excel"
            try:
                docx_pdf_temp_dir = tempfile.TemporaryDirectory()
                geometry_local_path = await to_thread.run_sync(
                    lambda: _convert_office_document_to_pdf(
                        downloaded_file.local_path,
                        docx_pdf_temp_dir.name,
                    )
                )
                geometry_content_type = "application/pdf"
                if office_meta is not None:
                    office_meta["pdf_preview"] = {"created": True}
            except Exception as exc:
                logger.exception(
                    "%s PDF preview conversion failed",
                    doc_kind,
                    extra={**log_extra, "step": "office_to_pdf_preview"},
                )
                if office_meta is not None:
                    office_meta["pdf_preview"] = {
                        "created": False,
                        "error": str(exc),
                    }
        geometry_started_at = time.monotonic()
        geometry_metadata = await to_thread.run_sync(
            lambda: _enrich_references_with_pdf_geometry(
                extracted_data,
                local_path=geometry_local_path,
                content_type=geometry_content_type,
                log_context=log_extra,
            )
        )
        stage_timings["geometry_enrichment"] = time.monotonic() - geometry_started_at
        logger.info(
            "Geometry enrichment done in %.2fs: references=%s matched=%s applied=%s",
            stage_timings["geometry_enrichment"],
            geometry_metadata.get("reference_count"),
            geometry_metadata.get("matched_reference_count"),
            geometry_metadata.get("applied"),
            extra={**log_extra, "step": "geometry_enrichment"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "OpenRouter extraction failed after %.2fs; stage_timings=%s",
            time.monotonic() - started_at, stage_timings,
            extra=log_extra,
        )
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
        logger.info(
            "OpenRouter extraction finished in %.2fs; stage_timings=%s",
            elapsed, stage_timings,
            extra={**log_extra, "step": "finished"},
        )

    response = {
        "analysis_id": payload.analysis_id,
        "file_id": payload.file_id,
        "file_type": payload.file_type,
        "backend": "openrouter",
        "model_parse": OPENROUTER_MODEL,
        "model_extract": VISION_MODEL if vision_pages_sent else OPENROUTER_MODEL,
        "result": extracted_data,
        "extraction": {"pages": [_build_result_page(extracted_data)]},
        "extraction_metadata": {
            "docling_version": _get_docling_version(),
            "page_count": geometry_metadata.get("page_count", 1),
            "errors": [],
            "provider": "openrouter",
            "provider_usage": provider_response.get("usage"),
            "geometry": geometry_metadata,
            "vision_hybrid": {
                "enabled": VISION_TABLES_ENABLED,
                "pages_sent": vision_pages_sent,
                "model": VISION_MODEL if vision_pages_sent else None,
            },
            "docx_conversion": docx_conversion_metadata,
            "excel_conversion": excel_conversion_metadata,
            "pdf_conversion": pdf_conversion_metadata,
        },
    }
    return jsonable_encoder(response)


RENDER_CACHE_DIR = Path(tempfile.gettempdir()) / "render_pdf_cache"
# Блокировки на ключ кэша: пока один запрос растеризует документ, остальные
# запросы того же документа ждут результат, а не запускают свою растеризацию.
_render_locks: dict[str, asyncio.Lock] = {}
_render_locks_guard = asyncio.Lock()


def _render_cache_key(url: str) -> str:
    """Ключ кэша по СТАБИЛЬНОЙ части URL (путь к объекту в S3), без подписи.
    Presigned-URL для одного файла каждый раз разный (меняется signature), но
    путь к объекту постоянен — кэшируем по нему."""
    path = unquote(urlparse(url).path)
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


async def _get_render_lock(key: str) -> asyncio.Lock:
    async with _render_locks_guard:
        lock = _render_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _render_locks[key] = lock
        return lock


_classify_cache: dict[str, dict[str, bool]] = {}


def _file_content_hash(local_path: str) -> str:
    """Быстрый хеш содержимого файла для кэша классификации: размер + первые/
    последние 256 КБ (достаточно для уникальности PDF без чтения всего файла)."""
    h = hashlib.sha256()
    size = os.path.getsize(local_path)
    h.update(str(size).encode())
    with open(local_path, "rb") as f:
        h.update(f.read(262144))
        if size > 524288:
            f.seek(-262144, os.SEEK_END)
            h.update(f.read(262144))
    return h.hexdigest()


def _classify_pdf_for_preview(local_path: str) -> dict[str, bool]:
    """Классифицирует PDF для превью: что с ним не так и нужна ли растеризация.

    Вердикт кэшируется по хешу содержимого: один и тот же файл, загруженный
    повторно (под другим storage_path/URL), не проходит дорогую классификацию
    (OSD скан-страниц) заново."""
    try:
        content_key = _file_content_hash(local_path)
        cached = _classify_cache.get(content_key)
        if cached is not None:
            return cached
    except Exception:
        content_key = None
    result = _classify_pdf_for_preview_uncached(local_path)
    if content_key is not None:
        # Ограничиваем рост кэша (простая защита от утечки памяти).
        if len(_classify_cache) > 512:
            _classify_cache.clear()
        _classify_cache[content_key] = result
    return result


def _classify_pdf_for_preview_uncached(local_path: str) -> dict[str, bool]:
    """Классифицирует PDF для превью: что с ним не так и нужна ли растеризация.

    Растеризуем (с выпрямлением), если документ показался бы криво или подсветка
    не совпала бы с координатами:
      - rotated:  есть страницы с поворотом (/Rotate != 0) → вьювер покажет боком;
      - garbled:  текстовый слой — мусор (битая кодировка шрифта);
      - non_embedded_fonts: есть шрифты, которые PDF.js не умеет рендерить.
    Чистый PDF (только web-safe шрифты, без поворота, читаемый текст) —
    отдаём оригинал как есть (быстро, постранично, текст выделяется)."""
    info = {
        "rotated": False,
        "garbled": False,
        "non_embedded_fonts": False,
        "scan_rotated": False,
        "needs_rasterization": False,
    }
    # Шрифты, которые PDF.js умеет рендерить сам без растеризации:
    # стандартные 14 PDF-шрифтов + распространённые web-safe (latin-only).
    # Любой другой встроенный шрифт (особенно кириллические TrueType вроде
    # FuturisC, Helios, PragmaticaC и т.п.) PDF.js не отрисует — нужна растеризация.
    pdfjs_safe_fonts = {
        "helvetica", "courier", "times", "symbol", "zapfdingbats", "arial",
        "georgia", "verdana", "tahoma", "trebuchet", "impact",
        "comic", "palatino", "garamond", "bookman", "avantgarde",
    }
    try:
        doc = fitz.open(local_path)
        try:
            pages_to_check = min(doc.page_count, 15)
            text_parts: list[str] = []
            scan_pages: list[int] = []
            for i in range(pages_to_check):
                page = doc.load_page(i)
                if page.rotation:
                    info["rotated"] = True
                page_text = page.get_text("text", sort=True)
                text_parts.append(page_text)
                # Страница-скан (нет текстового слоя) — кандидат на проверку
                # ориентации содержимого: PDF /Rotate=0, но картинка может быть боком.
                if len(page_text.strip()) < 10:
                    scan_pages.append(i)
                for font in doc.get_page_fonts(i):
                    # font = (xref, ext, type, basefont, name, encoding)
                    ext = (font[1] or "").strip().lower()
                    base = (font[3] or "").lower()
                    is_pdfjs_safe = any(s in base for s in pdfjs_safe_fonts)
                    # Растеризуем при любом шрифте вне web-safe списка: PDF.js не
                    # умеет рендерить произвольные встроенные TrueType/Type1 из
                    # PDF-потока — кириллические шрифты показываются квадратиками.
                    if not is_pdfjs_safe:
                        info["non_embedded_fonts"] = True
            combined = "\n".join(t for t in text_parts if t)
            if len(combined) > 50 and not _text_layer_is_usable(combined):
                info["garbled"] = True
            # Скан-страницы без текстового слоя: проверяем ориентацию содержимого
            # через Tesseract OSD. Если контент повёрнут (текст идёт боком) — нужна
            # растеризация с авто-поворотом, иначе превью покажет документ криво и
            # подсветка (привязанная к выпрямленному OCR-индексу) не совпадёт.
            if scan_pages and not info["rotated"]:
                if _scan_pages_are_rotated(doc, scan_pages[:2]):
                    info["scan_rotated"] = True
        finally:
            doc.close()
    except Exception:
        logger.warning("PDF classification failed, defaulting to rasterization", exc_info=True)
        info["needs_rasterization"] = True
        return info

    info["needs_rasterization"] = (
        info["rotated"]
        or info["garbled"]
        or info["non_embedded_fonts"]
        or info["scan_rotated"]
    )
    return info


def _scan_pages_are_rotated(doc: Any, page_indices: list[int]) -> bool:
    """Проверяет через Tesseract OSD, повёрнуто ли содержимое скан-страниц.
    Возвращает True, если хотя бы одна страница уверенно распознана повёрнутой
    (rotate != 0). Используется для превью: PDF /Rotate=0, но картинка боком.

    Оптимизация скорости: рендер при низком DPI (для OSD достаточно ~100), и
    останавливаемся на ПЕРВОЙ повёрнутой странице. OSD — дорогая операция (десятки
    секунд на документ при высоком DPI и нескольких страницах), а для определения
    «повёрнут ли скан» хватает грубого превью одной-двух страниц."""
    try:
        import pytesseract
        from PIL import Image
        import io
    except Exception:
        return False
    # 100 DPI достаточно для OSD и вдвое-втрое быстрее 150; psm=0 — только OSD.
    osd_config = "--psm 0"
    for i in page_indices:
        try:
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(100 / 72, 100 / 72), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            osd = pytesseract.image_to_osd(
                img,
                output_type=pytesseract.Output.DICT,
                config=osd_config,
                timeout=OCR_OSD_TIMEOUT_SECONDS,
            )
            rotate = int(osd.get("rotate", 0) or 0)
            conf = float(osd.get("orientation_conf", 0) or 0)
            if rotate % 360 != 0 and conf >= 2.0:
                return True  # нашли повёрнутую — дальше проверять незачем
        except Exception:
            continue
    return False


def _pdf_needs_rasterization(local_path: str) -> bool:
    return _classify_pdf_for_preview(local_path)["needs_rasterization"]


def _rasterize_pdf(local_path: str, auto_orient: bool = False) -> bytes:
    """Растеризует каждую страницу PDF в JPEG и собирает новый PDF.

    Страницы вставляются как JPEG (а не lossless): размер итогового PDF в разы
    меньше, что критично для скорости загрузки превью на медленном интернете.
    DPI и качество JPEG настраиваются через RENDER_PDF_DPI / RENDER_PDF_JPEG_QUALITY.

    auto_orient=True — выпрямляет повёрнутые страницы (через OSD), чтобы вьювер
    показывал документ прямо И в той же ориентации, в которой геометрия привязала
    координаты (иначе подсветка не совпадёт)."""
    import io
    from PIL import Image

    src = fitz.open(local_path)
    out = fitz.open()
    dpi = RENDER_PDF_DPI
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)
    for page in src:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        if auto_orient:
            img = _auto_orient_for_ocr(img)
        # Размер страницы в пунктах при 72 dpi (после возможного поворота)
        pw = img.width * 72 / dpi
        ph = img.height * 72 / dpi
        jpeg_buf = io.BytesIO()
        img.convert("RGB").save(jpeg_buf, format="JPEG", quality=RENDER_PDF_JPEG_QUALITY)
        out_page = out.new_page(width=pw, height=ph)
        out_page.insert_image(fitz.Rect(0, 0, pw, ph), stream=jpeg_buf.getvalue())
    src.close()
    buf = io.BytesIO()
    # JPEG уже сжат — deflate на поток картинок не нужен, только чистим объекты.
    out.save(buf, garbage=2)
    out.close()
    return buf.getvalue()


def _render_response(pdf_bytes: bytes) -> Response:
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "inline; filename=preview.pdf",
            "Cache-Control": "private, max-age=3600",
        },
    )


@app.get("/render-pdf")
async def render_pdf(url: str) -> Response:
    """Скачивает PDF по URL, рендерит каждую страницу через PyMuPDF в изображение
    и возвращает новый PDF из растровых страниц. Решает проблему с нестандартными
    кириллическими шрифтами, которые PDF.js не умеет отображать.

    Результат кэшируется на диске по ключу объекта S3: повторные открытия того же
    документа отдаются из кэша мгновенно, без повторной растеризации (она грузит CPU)."""
    if not PYMUPDF_INSTALLED:
        raise HTTPException(status_code=501, detail="PyMuPDF not available")

    # Проверки схемы недостаточно: хост не валидировался, и URL мог указывать
    # на 169.254.169.254 (метаданные облака с IAM-токенами) или на внутренние
    # сервисы compose. validate_public_url резолвит хост и отклоняет
    # непубличные адреса, кроме явного allowlist объектного хранилища.
    try:
        validate_public_url(url)
    except UrlNotAllowed as exc:
        raise HTTPException(status_code=400, detail=f"url отклонён: {exc}")

    cache_key = _render_cache_key(url)
    cache_path = RENDER_CACHE_DIR / f"{cache_key}.pdf"
    if cache_path.exists():
        return _render_response(cache_path.read_bytes())

    # Сериализуем растеризацию одного и того же документа: параллельные открытия
    # не должны запускать N одновременных растеризаций (это и сатурировало CPU).
    lock = await _get_render_lock(cache_key)
    async with lock:
        if cache_path.exists():
            return _render_response(cache_path.read_bytes())

        try:
            downloaded = await _download_file(url)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to download file: {exc}") from exc

        try:
            # Растеризуем ТОЛЬКО если это реально нужно (есть невстроенные шрифты).
            # Для обычных PDF со встроенными шрифтами отдаём оригинал — PDF.js
            # покажет его сам, постранично и быстро (без тяжёлой растеризации и
            # без раздувания размера, которое тормозило превью на медленном инете).
            needs_raster = await to_thread.run_sync(
                lambda: _pdf_needs_rasterization(downloaded.local_path)
            )
            if needs_raster:
                # Растеризуем с авто-поворотом: вьювер покажет документ прямо и в
                # той же ориентации, в которой геометрия привязала координаты.
                pdf_bytes = await to_thread.run_sync(
                    lambda: _rasterize_pdf(downloaded.local_path, auto_orient=True)
                )
            else:
                pdf_bytes = await to_thread.run_sync(
                    lambda: Path(downloaded.local_path).read_bytes()
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"PDF rasterization failed: {exc}") from exc
        finally:
            _cleanup_downloaded_file(downloaded)

        try:
            RENDER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(".pdf.tmp")
            tmp_path.write_bytes(pdf_bytes)
            tmp_path.replace(cache_path)
        except Exception as exc:
            logger.warning("Failed to cache rendered PDF: %s", exc)

    return _render_response(pdf_bytes)


OCR_INDEX_CACHE_DIR = Path(tempfile.gettempdir()) / "ocr_index_cache"
_ocr_index_locks: dict[str, asyncio.Lock] = {}
_ocr_index_locks_guard = asyncio.Lock()


async def _get_ocr_index_lock(key: str) -> asyncio.Lock:
    async with _ocr_index_locks_guard:
        lock = _ocr_index_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _ocr_index_locks[key] = lock
        return lock


def _build_ocr_index_payload(local_path: str) -> dict:
    document, pages = _build_pdf_page_index_ocr(local_path)
    try:
        pages_payload = []
        for page_index in pages:
            page_rect = page_index.page_rect or page_index.page.rect
            words_payload = [
                {
                    "text": word.text,
                    "bbox": _rect_to_bbox(word.rect, page_rect),
                }
                for word in page_index.words
            ]
            pages_payload.append(
                {
                    "page_number": page_index.page_number,
                    "words": words_payload,
                }
            )
    finally:
        document.close()
    return {"page_count": len(pages_payload), "pages": pages_payload}


@app.get("/ocr-index")
async def ocr_index(url: str) -> dict:
    """Отдельный от основного /extract endpoint: строит постраничный OCR
    word-индекс (текст + нормализованные координаты каждого слова) для
    сканированного PDF по URL. Используется фронтендом лениво — только когда
    пользователь открывает поиск/выделение на документе без текстового слоя.
    Результат кэшируется на диске по стабильной части URL (как /render-pdf).

    Намеренно не переиспользует и не меняет _enrich_references_with_pdf_geometry
    и основной пайплайн /extract — это независимый read-only помощник."""
    # См. комментарий в /render-pdf: проверка одной лишь схемы оставляла SSRF.
    try:
        validate_public_url(url)
    except UrlNotAllowed as exc:
        raise HTTPException(status_code=400, detail=f"url отклонён: {exc}")
    if not PYMUPDF_INSTALLED:
        raise HTTPException(status_code=501, detail="PyMuPDF not available")

    try:
        import pytesseract

        await to_thread.run_sync(pytesseract.get_tesseract_version)
    except Exception as exc:
        raise HTTPException(status_code=501, detail=f"OCR is not available: {exc}") from exc

    cache_key = _render_cache_key(url)
    cache_path = OCR_INDEX_CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    lock = await _get_ocr_index_lock(cache_key)
    async with lock:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        try:
            downloaded = await _download_file(url)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to download file: {exc}") from exc

        try:
            payload = await to_thread.run_sync(_build_ocr_index_payload, downloaded.local_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc
        finally:
            _cleanup_downloaded_file(downloaded)

        try:
            OCR_INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(cache_path)
        except Exception as exc:
            logger.warning("Failed to cache OCR index: %s", exc)

    return payload


@app.post("/extract")
async def extract_document(payload: ExtractionRequest) -> dict:
    log_extra = {
        "analysis_id": payload.analysis_id,
        "file_id": payload.file_id,
        "file_type": payload.file_type,
        "step": "extract_document",
    }
    started_at = time.monotonic()

    if not payload.schema_payload and not payload.prompt:
        logger.warning("Rejecting /extract: no schema and no prompt", extra=log_extra)
        raise HTTPException(status_code=400, detail="schema or prompt is required")

    parsed_url = urlparse(payload.file_url)
    if parsed_url.scheme not in {"http", "https"}:
        logger.warning(
            "Rejecting /extract: file_url is not http(s): %s", parsed_url.scheme,
            extra=log_extra,
        )
        raise HTTPException(status_code=400, detail="file_url must be a public URL")

    backend = _select_backend(payload.backend)
    logger.info(
        "extract_document started backend=%s file_url=%s",
        backend, _redact_url(payload.file_url),
        extra=log_extra,
    )
    try:
        if backend == "docling_local":
            result = await _extract_via_docling_local(payload)
        elif backend == "docling_remote":
            result = await _extract_via_docling_remote(payload)
        elif backend == "openrouter":
            result = await _extract_via_openrouter(payload)
        elif backend == "llamaparse":
            result = await _extract_via_llamaparse(payload)
        elif backend == "mineru":
            result = await _extract_via_mineru(payload)
        elif backend == "pdfplumber":
            result = await _extract_via_pdfplumber(payload)
        elif backend in ("paddleocr_vl", "yandex_vision_ocr"):
            result = await _extract_via_paddleocr_vl(payload)
        else:
            result = None
    except HTTPException as exc:
        logger.error(
            "extract_document failed backend=%s status=%s detail=%s elapsed=%.2fs",
            backend, exc.status_code, exc.detail, time.monotonic() - started_at,
            extra=log_extra,
        )
        raise
    except Exception:
        logger.exception(
            "extract_document crashed backend=%s elapsed=%.2fs",
            backend, time.monotonic() - started_at,
            extra=log_extra,
        )
        raise
    else:
        if result is not None:
            logger.info(
                "extract_document finished backend=%s elapsed=%.2fs",
                backend, time.monotonic() - started_at,
                extra=log_extra,
            )
            return result
    raise HTTPException(status_code=400, detail=f"Unsupported backend: {backend}")
