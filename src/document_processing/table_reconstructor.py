"""
表格结构重建器 (Layer 3)
从 OCR 文字位置数据重建表格的行列结构
"""
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import fitz
import numpy as np

from src.common.logger import get_logger
from src.document_processing.text_cleaner import TextCleaner

logger = get_logger(__name__)


@dataclass
class Cell:
    """表格单元格"""
    text: str
    row: int
    col: int
    x: int; y: int; w: int; h: int
    confidence: float = 0.0


@dataclass
class StructuredTable:
    """重建后的结构化表格"""
    table_id: str
    page_num: int
    source_file: str
    headers: List[str]           # 表头行
    rows: List[List[str]]        # 数据行 (每行与表头列数对齐)
    raw_grid: List[List[Cell]]   # 原始网格单元格
    context: str = ""            # 表格前后文（标题等）
    column_count: int = 0
    row_count: int = 0
    table_type: str = ""         # 识别后的表格类型
    markdown: str = ""


class TableReconstructor:
    """
    表格结构重建器
    从 OCR 位置数据中重建表格的行列结构
    """

    def __init__(
        self,
        text_cleaner: Optional[TextCleaner] = None,
        ocr_lang: str = "chi_sim+eng",
        ocr_dpi: int = 400,
    ):
        self.text_cleaner = text_cleaner or TextCleaner()
        self.ocr_lang = ocr_lang
        self.ocr_dpi = ocr_dpi

    def extract_tables_from_page(
        self,
        doc: fitz.Document,
        page_num: int,
        pdf_path: str = "",
    ) -> List[StructuredTable]:
        """
        从单页 PDF 提取并重建表格

        Args:
            doc: PyMuPDF Document
            page_num: 页码 (1-indexed)
            pdf_path: PDF 文件路径（用于元信息）

        Returns:
            结构化表格列表
        """
        page = doc[page_num - 1]

        # 渲染页面为高分辨率图像
        zoom = self.ocr_dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        # Tesseract OCR with position data
        ocr_data = self._ocr_with_positions(img_bytes)
        if not ocr_data:
            return []

        # 从位置数据中聚类列和行
        tables = self._cluster_to_tables(ocr_data, page_num, pdf_path)
        return tables

    def extract_all_tables(
        self,
        doc: fitz.Document,
        pdf_path: str = "",
        page_types: Optional[Dict[int, str]] = None,
    ) -> List[StructuredTable]:
        """
        从整个 PDF 中提取所有表格

        Args:
            doc: PyMuPDF Document
            pdf_path: PDF 文件路径
            page_types: 页面分类结果 {page_num: "TABLE"/"MIXED"/...}

        Returns:
            结构化表格列表
        """
        all_tables = []
        for page_num in range(1, len(doc) + 1):
            ptype = (page_types or {}).get(page_num, "")
            if ptype in ("CHART",):
                continue  # 跳过图表页

            tables = self.extract_tables_from_page(doc, page_num, pdf_path)
            all_tables.extend(tables)

        logger.info(f"表格提取完成: {pdf_path}, {len(all_tables)} 个表格")
        return all_tables

    def _ocr_with_positions(self, img_bytes: bytes) -> List[Dict]:
        """
        Tesseract OCR 带位置数据

        Returns:
            [{text, left, top, width, height, conf, line_num, word_num}, ...]
        """
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(io.BytesIO(img_bytes))
            data = pytesseract.image_to_data(
                img,
                lang=self.ocr_lang,
                output_type=pytesseract.Output.DICT,
                config='--psm 6',  # 假设均匀文字块，适合表格
            )

            words = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                if not text or conf < 30:  # 低置信度跳过
                    continue

                words.append({
                    'text': text,
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'conf': conf / 100.0,
                    'line_num': data['line_num'][i],
                    'block_num': data['block_num'][i],
                    'word_num': data['word_num'][i],
                })

            return words

        except Exception as e:
            logger.error(f"OCR 位置提取失败: {e}")
            return []

    def _cluster_to_tables(
        self,
        words: List[Dict],
        page_num: int,
        source_file: str,
    ) -> List[StructuredTable]:
        """将 OCR 词汇按位置聚类为表格"""
        if len(words) < 6:
            return []

        # ---- 按 y 坐标分组为行 ----
        rows = self._cluster_rows(words)
        if len(rows) < 3:
            return []

        # ---- 对每行按 x 坐标排序 ----
        row_cells = []
        for row_words in rows:
            row_words.sort(key=lambda w: w['left'])
            row_cells.append(row_words)

        # ---- 确定列数 & 对齐 ----
        col_count = self._infer_column_count(row_cells)
        if col_count < 2:
            return []

        # ---- 构建网格 ----
        grid = self._build_grid(row_cells, col_count)

        # ---- 识别表头 ----
        headers = [c.text for c in grid[0]] if grid else []

        # ---- 提取数据行 ----
        data_rows = [[c.text for c in row] for row in grid[1:]]

        # ---- 生成 Markdown ----
        markdown = self._grid_to_markdown(grid)

        table = StructuredTable(
            table_id=f"tbl_p{page_num}_{len(grid)}x{col_count}",
            page_num=page_num,
            source_file=source_file,
            headers=headers,
            rows=data_rows,
            raw_grid=grid,
            column_count=col_count,
            row_count=len(grid),
            markdown=markdown,
        )

        # 清洗后重新生成 markdown
        table.headers = [self.text_cleaner.clean(h) for h in headers]
        table.rows = [[self.text_cleaner.clean(c) for c in row] for row in data_rows]

        return [table]

    def _cluster_rows(self, words: List[Dict]) -> List[List[Dict]]:
        """按 y 坐标聚类为行"""
        if not words:
            return []

        # 收集所有 y 坐标
        y_coords = sorted(set(w['top'] for w in words))

        # 简单的 DBSCAN 风格：间距小于阈值则同组
        rows = []
        current_row = [words[0]]
        row_y = words[0]['top']

        for w in sorted(words[1:], key=lambda x: x['top']):
            # 同一行的 y 坐标差应该小于平均行高
            if abs(w['top'] - row_y) < 15:  # 15px = ~1 行高在 300DPI
                current_row.append(w)
            else:
                if len(current_row) >= 2:
                    rows.append(current_row)
                current_row = [w]
                row_y = w['top']

        if len(current_row) >= 2:
            rows.append(current_row)

        return rows

    def _infer_column_count(self, row_cells: List[List[Dict]]) -> int:
        """推断表格列数（取最多列的那行）"""
        if not row_cells:
            return 0

        # 去重：同一 x 位置附近 ±20px 算同一列
        all_x = set()
        for row in row_cells:
            for w in row:
                # 量化 x 坐标
                quantized = round(w['left'] / 30) * 30
                all_x.add(quantized)

        # 取出现频率最高的列数
        col_counts = [len(row) for row in row_cells if len(row) >= 3]
        if not col_counts:
            return len(row_cells[0])

        # 众数
        from collections import Counter
        return Counter(col_counts).most_common(1)[0][0]

    def _build_grid(
        self,
        row_cells: List[List[Dict]],
        col_count: int,
    ) -> List[List[Cell]]:
        """构建行列对齐的表格网格"""
        grid = []
        for row_idx, row_words in enumerate(row_cells):
            # 按 x 坐标排序
            sorted_words = sorted(row_words, key=lambda w: w['left'])
            # 取前 col_count 列
            cells = []
            for col_idx in range(min(col_count, len(sorted_words))):
                w = sorted_words[col_idx]
                cells.append(Cell(
                    text=w['text'],
                    row=row_idx,
                    col=col_idx,
                    x=w['left'], y=w['top'],
                    w=w['width'], h=w['height'],
                    confidence=w['conf'],
                ))
            grid.append(cells)
        return grid

    def _grid_to_markdown(self, grid: List[List[Cell]]) -> str:
        """将 TableReconstructor 重建的网格转 Markdown"""
        if not grid:
            return ""

        lines = []
        # Header
        header_cells = [c.text for c in grid[0]]
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")

        for row in grid[1:]:
            row_cells = [c.text for c in row]
            lines.append("| " + " | ".join(row_cells) + " |")

        return "\n".join(lines)
