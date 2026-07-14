"""
PaddleOCR 封装 (PP-StructureV3)
替代 Tesseract，提供 PDF → Markdown 的高质量转换
支持中文文档、表格检测、版面分析、公式识别
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger

logger = get_logger(__name__)

# 全局单例 pipeline（避免重复加载 15+ 个模型）
_pipeline = None


def get_pipeline():
    """获取全局 PP-StructureV3 pipeline 单例"""
    global _pipeline
    if _pipeline is None:
        from paddlex import create_pipeline
        logger.info("加载 PP-StructureV3 pipeline...")
        _pipeline = create_pipeline(pipeline="PP-StructureV3")
        logger.info("PP-StructureV3 加载完成")
    return _pipeline


class PaddleOCREngine:
    """
    PP-StructureV3 OCR 引擎
    对 PDF 执行版面分析 + 文字识别 + 表格检测，输出结构化 Markdown
    """

    def __init__(
        self,
        use_doc_orientation: bool = False,
        use_doc_unwarping: bool = False,
        use_chart_parsing: bool = False,
    ):
        """
        Args:
            use_doc_orientation: 是否启用文档方向分类
            use_doc_unwarping: 是否启用文档展平
            use_chart_parsing: 是否启用图表解析（消耗更多显存）
        """
        self.use_doc_orientation = use_doc_orientation
        self.use_doc_unwarping = use_doc_unwarping
        self.use_chart_parsing = use_chart_parsing

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        处理整个 PDF，返回每页的结构化结果

        Returns:
            [{page_num, markdown_text, text_blocks, tables, images, is_empty}, ...]
        """
        pipeline = get_pipeline()

        logger.info(f"PaddleOCR 处理: {pdf_path}")
        raw_output = list(pipeline.predict(
            input=str(pdf_path),
            use_doc_orientation_classify=self.use_doc_orientation,
            use_doc_unwarping=self.use_doc_unwarping,
            use_chart_parsing=self.use_chart_parsing,
            skip_validation=True,
        ))

        pages = []
        for i, result in enumerate(raw_output):
            page_info = self._parse_page_result(result, i + 1, pdf_path)
            pages.append(page_info)

        text_pages = sum(1 for p in pages if not p["is_empty"])
        logger.info(f"PaddleOCR 完成: {text_pages}/{len(pages)} 页有内容")
        return pages

    def _parse_page_result(
        self,
        result: Any,
        page_num: int,
        source_file: str,
    ) -> Dict[str, Any]:
        """解析 PP-StructureV3 单页结果（LayoutParsingResultV2）"""
        # ---- 文本提取 ----
        # PP-StructureV3 结果对象是 dict-like，OCR 文本在 overall_ocr_res.rec_texts
        ocr_res = result.get("overall_ocr_res", {}) if hasattr(result, "get") else None
        if ocr_res and hasattr(ocr_res, "get"):
            rec_texts = ocr_res.get("rec_texts", []) or []
        else:
            rec_texts = []

        plain_text = "\n".join(str(t) for t in rec_texts if t)

        # ---- 表格提取 ----
        tables = []
        table_res_list = result.get("table_res_list", []) if hasattr(result, "get") else []
        for tbl in table_res_list:
            parsed = self._parse_table_result(tbl, page_num)
            if parsed:
                tables.append(parsed)

        # ---- 图片提取 ----
        images = []
        layout_res = result.get("layout_det_res", {}) if hasattr(result, "get") else {}
        # 提取图片区域信息
        if hasattr(layout_res, "get"):
            for item in layout_res.get("boxes", []) or []:
                if hasattr(item, "get") and item.get("label") in ("image", "figure"):
                    images.append({"bbox": item.get("coordinate", [])})

        # ---- 生成 Markdown 表示 ----
        md_parts = []
        if plain_text:
            md_parts.append(plain_text)
        for tbl in tables:
            md_parts.append(tbl.get("markdown", ""))

        is_empty = len(plain_text.strip()) < 10 and len(tables) == 0

        return {
            "page_num": page_num,
            "source_file": Path(source_file).name,
            "markdown_text": "\n\n".join(md_parts),
            "plain_text": plain_text,
            "tables": tables,
            "images": images,
            "is_empty": is_empty,
            "char_count": len(plain_text),
        }

    def _parse_table_result(
        self,
        tbl: Any,
        page_num: int,
    ) -> Optional[Dict[str, Any]]:
        """解析 PP-StructureV3 表格结果"""
        if not hasattr(tbl, "get"):
            return None

        # 表格 HTML/Markdown
        pred_html = tbl.get("pred_html", "") or ""
        pred_md = tbl.get("pred_markdown", "") or ""

        # 优先用 Markdown，没有再用 HTML 转换
        md = pred_md if pred_md else self._html_to_md(pred_html)

        if not md.strip():
            return None

        # 从 Markdown 解析表头和数据
        headers, rows = self._parse_simple_table(md)

        return {
            "table_id": f"tbl_p{page_num}_{len(tables) if hasattr(self, '_tables') else 0:02d}",
            "page_num": page_num,
            "headers": headers,
            "rows": rows,
            "markdown": md,
            "row_count": len(rows),
            "col_count": len(headers),
        }

    def _parse_simple_table(self, md: str) -> tuple:
        """从 Markdown 表格文本解析表头和数据行"""
        lines = [l.strip() for l in md.split("\n") if l.strip().startswith("|")]
        data_lines = [
            l for l in lines
            if not re.match(r'^\|[\s\-:|]+\|$', l.strip())
        ]

        def split_cells(line: str) -> list:
            return [c.strip() for c in line.strip().strip("|").split("|")]

        headers = split_cells(data_lines[0]) if data_lines else []
        rows = [split_cells(l) for l in data_lines[1:]] if len(data_lines) > 1 else []

        return headers, rows

    def _html_to_md(self, html: str) -> str:
        """简单的 HTML table → Markdown 转换"""
        if not html or "<table" not in html.lower():
            return html

        rows = []
        for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE):
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', tr, re.DOTALL | re.IGNORECASE)
            cells_clean = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            rows.append(cells_clean)

        if not rows:
            return ""

        md_lines = ["| " + " | ".join(rows[0]) + " |"]
        md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
        for row in rows[1:]:
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(md_lines)
