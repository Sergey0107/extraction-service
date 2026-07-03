"""Bbox-first геометрия таблиц через pdfplumber.

Принцип (перенесён из pdf-spec-extractor, см. изучение в этой сессии):
LLM НИКОГДА не вычисляет координаты. ``find_tables()`` + ``within_bbox()``
находят для каждой ячейки таблицы точный bbox ГЕОМЕТРИЧЕСКИ, до всякого
обращения к LLM. LLM получает уже готовые пары «текст ↔ bbox» и должна
только скопировать нужный bbox дословно в свой ответ, разобравшись, какой
фрагмент относится к какой характеристике. Это устраняет класс ошибок,
свойственных постфактум-поиску координат по переформулированному LLM
тексту (единицы измерения, кириллица/латиница «х», короткие неоднозначные
значения вроде «G1» и т.п.) — здесь эвристика просто не нужна.

Система координат: pdfplumber отдаёт ``top``/``bottom`` от верхнего края
страницы вниз (origin — левый верхний угол), как и ``fitz.Rect`` в
PyMuPDF. Конвертация не нужна: bbox можно передавать напрямую в
формат, который строит ``_rect_to_bbox`` (main.py) — просто
``x0/page_width``, ``y0/page_height`` без инверсии оси.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Те же ключевые слова, что использует openrouter-бэкенд (_RELEVANCE_KEYWORDS в
# main.py) для отбора страниц с характеристиками в больших документах.
_RELEVANCE_KEYWORDS = re.compile(
    r"характерист|параметр|показател|спецификац|specification|parameter|"
    r"техничес|nominal|dimension|габарит|размер|масс[аы]|вес\b|"
    r"мощност|напор|подач[аи]|производител|давлен|температур|"
    r"частот|оборот|диаметр|длин[аы]|ширин[аы]|высот[аы]|"
    r"марк[аи]|модел[ьи]|тип\b|обозначен",
    re.IGNORECASE,
)


@dataclass
class CellGeometry:
    text: str
    bbox: tuple[float, float, float, float]  # (x0, top, x1, bottom) в points
    # Построчная разбивка ВНУТРИ этой ячейки — заполняется только для
    # многострочных ячеек (см. _extract_tables). Каждая под-строка имеет
    # собственный узкий bbox — LLM должна предпочитать его bbox всей ячейки,
    # когда характеристика соответствует ровно одной строке внутри неё.
    sub_lines: list["CellGeometry"] = field(default_factory=list)


@dataclass
class PageGeometry:
    page_number: int  # 1-based
    width: float
    height: float
    tables: list[list[list[Optional[CellGeometry]]]] = field(default_factory=list)
    lines: list[CellGeometry] = field(default_factory=list)


def extract_pdf_geometry(local_path: str) -> list[PageGeometry]:
    """Возвращает геометрию каждой страницы: ячейки таблиц + строки текста вне таблиц."""
    import pdfplumber  # локальный импорт: probe в config.py решает доступность

    pages_out: list[PageGeometry] = []
    with pdfplumber.open(local_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            table_bboxes: list[tuple[float, float, float, float]] = []
            tables = _extract_tables(page, table_bboxes)
            pages_out.append(
                PageGeometry(
                    page_number=page_number,
                    width=round(float(page.width), 2),
                    height=round(float(page.height), 2),
                    tables=tables,
                    # Строки вне таблиц: исключаем всё, что физически лежит внутри
                    # уже найденных таблиц (см. _extract_lines) — иначе одна и та же
                    # строка таблицы дублируется с ДВУМЯ bbox (узким — ячейка,
                    # широким — вся строка целиком), и LLM может скопировать
                    # широкий, из-за чего подсветка растягивается на всю строку/
                    # таблицу вместо конкретной ячейки.
                    lines=_extract_lines(page, table_bboxes),
                )
            )
    return pages_out


def _extract_tables(
    page: Any, table_bboxes_out: list[tuple[float, float, float, float]]
) -> list[list[list[Optional[CellGeometry]]]]:
    """Для каждой ячейки таблицы: within_bbox() вырезает область СТРОГО по
    границам ячейки, текст берётся только из символов внутри неё — привязка
    текст↔bbox устанавливается геометрически, не эвристикой.

    Для МНОГОСТРОЧНЫХ ячеек (текст с переносами — типично для ТЗ, где вся
    строка характеристик товара лежит в одной ячейке) дополнительно строится
    построчная разбивка внутри той же ячейки (см. _lines_within_bbox), чтобы
    LLM могла скопировать bbox конкретной СТРОКИ внутри ячейки, а не только
    bbox всей ячейки целиком."""
    tables_out: list[list[list[Optional[CellGeometry]]]] = []
    try:
        tables = page.find_tables()
    except Exception:  # noqa: BLE001
        tables = []

    for table in tables:
        table_bboxes_out.append(tuple(round(v, 2) for v in table.bbox))
        rows_out: list[list[Optional[CellGeometry]]] = []
        for row in table.rows:
            cells_out: list[Optional[CellGeometry]] = []
            for cell in row.cells:
                if cell is None:
                    cells_out.append(None)
                    continue
                x0, top, x1, bottom = cell
                try:
                    cropped = page.within_bbox((x0, top, x1, bottom))
                    text = (cropped.extract_text() or "").replace("\n", " ").strip()
                except Exception:  # noqa: BLE001
                    text = ""
                if text:
                    cell_geom = CellGeometry(
                        text=text,
                        bbox=(round(x0, 2), round(top, 2), round(x1, 2), round(bottom, 2)),
                    )
                    # Построчная разбивка внутри ячейки — только если ячейка
                    # достаточно высокая, чтобы содержать несколько строк текста.
                    sub_lines = _lines_within_bbox(page, (x0, top, x1, bottom))
                    if len(sub_lines) > 1:
                        cell_geom.sub_lines = sub_lines
                    cells_out.append(cell_geom)
                else:
                    cells_out.append(None)
            rows_out.append(cells_out)
        tables_out.append(rows_out)
    return tables_out


def _lines_within_bbox(
    page: Any, bbox: tuple[float, float, float, float]
) -> list[CellGeometry]:
    """Построчная разбивка текста СТРОГО внутри bbox (обычно — ячейка таблицы).

    Использует те же слова (extract_words), что и _extract_lines, но
    ограниченные внутри данной области — так многострочная ячейка (несколько
    характеристик через перенос строки в одной ячейке ТЗ) даёт отдельный узкий
    bbox для каждой строки, а не один bbox на весь текст ячейки."""
    x0, top, x1, bottom = bbox
    try:
        words = page.within_bbox((x0, top, x1, bottom)).extract_words()
    except Exception:  # noqa: BLE001
        return []
    return _group_words_into_lines(words)


def _extract_lines(
    page: Any, exclude_bboxes: list[tuple[float, float, float, float]]
) -> list[CellGeometry]:
    """Текст вне таблиц: слова группируются по совпадению округлённой 'top' —
    символы одной визуальной строки лежат на одной базовой линии.

    Слова, чей центр лежит внутри любого bbox из exclude_bboxes (найденные
    таблицы), исключаются — иначе строка таблицы дублируется здесь с широким
    bbox через всю ширину страницы, и LLM может ошибочно скопировать именно
    его вместо точного bbox ячейки."""
    try:
        words = page.extract_words()
    except Exception:  # noqa: BLE001
        words = []

    def _in_table(w: dict[str, Any]) -> bool:
        cx = (w["x0"] + w["x1"]) / 2
        cy = (w["top"] + w["bottom"]) / 2
        for bx0, btop, bx1, bbottom in exclude_bboxes:
            if bx0 <= cx <= bx1 and btop <= cy <= bbottom:
                return True
        return False

    words = [w for w in words if not _in_table(w)]
    return _group_words_into_lines(words)


def _group_words_into_lines(words: list[dict[str, Any]]) -> list[CellGeometry]:
    lines: dict[float, list[dict[str, Any]]] = {}
    for w in words:
        key = round(w["top"], 0)
        lines.setdefault(key, []).append(w)

    lines_out: list[CellGeometry] = []
    for top in sorted(lines.keys()):
        ws = sorted(lines[top], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in ws).strip()
        if not text:
            continue
        x0 = min(w["x0"] for w in ws)
        x1 = max(w["x1"] for w in ws)
        bottom = max(w["bottom"] for w in ws)
        lines_out.append(
            CellGeometry(text=text, bbox=(round(x0, 2), round(top, 2), round(x1, 2), round(bottom, 2)))
        )
    return lines_out


def _page_text_for_relevance(page: PageGeometry) -> str:
    """Весь текст страницы (ячейки таблиц + строки вне таблиц) — для проверки
    ключевых слов при решении "оставить/выбросить" страницу."""
    parts: list[str] = []
    for table in page.tables:
        for row in table:
            for cell in row:
                if cell is not None and cell.text:
                    parts.append(cell.text)
    for line in page.lines:
        parts.append(line.text)
    return " ".join(parts)


def plain_text_for_quality_check(pages: list[PageGeometry]) -> str:
    """Чистый текст документа (без bbox-разметки) — для внешней проверки
    качества текстового слоя (пуст/мусор из-за битой кодировки шрифта/скана).
    Используется вызывающим кодом (main.py), чтобы решить, нужен ли fallback
    на бэкенд с OCR."""
    return " ".join(_page_text_for_relevance(p) for p in pages)


def _page_payload_size(page: PageGeometry) -> int:
    """Оценка РЕАЛЬНОГО объёма, который страница добавит в payload для LLM —
    т.е. с учётом bbox-меток и служебных строк ('[bbox=[...]] "..."'), а не
    только длины текста. bbox-метки добавляют существенный объём (координаты
    + разметка на КАЖДУЮ ячейку/строку), поэтому чистая длина текста
    недооценивает итоговый размер в разы — из-за этого при пороге,
    рассчитанном по чистому тексту, реальный payload всё равно обрезался."""
    text, _ = build_llm_payload([page], max_chars=10**9)
    return len(text)


def filter_relevant_pages(
    pages: list[PageGeometry], max_chars: int = 150000
) -> tuple[list[PageGeometry], bool]:
    """Для больших документов (много страниц титульного/сервисного текста —
    техника безопасности, гарантии, содержание) отбирает страницы, вероятно
    содержащие характеристики, чтобы не обрезать payload посередине таблицы.

    Правила (симметрично _filter_relevant_pages в main.py для openrouter):
    всегда сохраняем первую и последнюю страницу; страницу с таблицей —
    ВСЕГДА (там могут быть характеристики независимо от ключевых слов);
    страницу с ключевыми словами характеристик — тоже; плюс СОСЕДЕЙ страниц
    с таблицами (чтобы не оторвать шапку таблицы от строк на следующей
    странице). max_chars сверяется с РЕАЛЬНЫМ размером payload (bbox-метки),
    а не голым текстом — иначе порог оказывается заниженным и всё равно
    обрезает документ посередине таблицы. Если после фильтрации остаётся
    мало страниц или общий объём и так укладывается в лимит — возвращаем
    все страницы без изменений."""
    total_len = sum(_page_payload_size(p) for p in pages)
    if total_len <= max_chars:
        return pages, False

    n = len(pages)
    keep: set[int] = set()
    for i, page in enumerate(pages):
        text = _page_text_for_relevance(page)
        if not text.strip():
            continue
        if i == 0 or i == n - 1:
            keep.add(i)
            continue
        if page.tables or _RELEVANCE_KEYWORDS.search(text):
            keep.add(i)

    # Целостность таблиц: подтягиваем соседей страниц с таблицами.
    for i, page in enumerate(pages):
        if page.tables:
            for j in (i - 1, i + 1):
                if 0 <= j < n and _page_text_for_relevance(pages[j]).strip():
                    keep.add(j)

    if not keep:
        return pages, False
    if len(keep) < 3 and n > 5:
        return pages, False

    filtered = [pages[i] for i in sorted(keep)]
    return filtered, True


def build_llm_payload(pages: list[PageGeometry], max_chars: int = 60000) -> tuple[str, bool]:
    """Компактное текстовое представление для LLM: только текст и bbox.

    Возвращает (текст, truncated) — truncated=True, если пришлось обрезать
    по max_chars (страховка от разрыва бюджета контекста на очень больших
    документах; координаты в обрезанной части просто не попадут в ответ)."""
    lines: list[str] = []
    for page in pages:
        lines.append(
            f"=== СТРАНИЦА {page.page_number} (ширина={page.width}, высота={page.height}) ==="
        )
        for t_idx, table in enumerate(page.tables):
            lines.append(f"-- Таблица {t_idx + 1} --")
            for row in table:
                for cell in row:
                    if cell is not None and cell.text:
                        lines.append(f'[bbox={list(cell.bbox)}] "{cell.text}"')
                        # Многострочная ячейка (напр. в ТЗ — все характеристики
                        # товара перечислены в одной ячейке через перенос строки):
                        # даём LLM узкие bbox КАЖДОЙ строки внутри, чтобы она могла
                        # привязать конкретную характеристику к её строке, а не ко
                        # всей ячейке целиком.
                        if cell.sub_lines:
                            for sub in cell.sub_lines:
                                lines.append(f'  [bbox={list(sub.bbox)}] (строка внутри ячейки выше) "{sub.text}"')
        for line in page.lines:
            lines.append(f'[bbox={list(line.bbox)}] "{line.text}"')

    text = "\n".join(lines)
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n... (обрезано, документ слишком большой)"
    return text, truncated
