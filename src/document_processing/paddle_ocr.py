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
        """解析 PP-StructureV3 单页结果"""
        # 提取 markdown 文本
        md_text = getattr(result, "markdown_texts", "") or ""
        if isinstance(md_text, dict):
            md_text = md_text.get("markdown_texts", "")

        md_text = str(md_text).strip()

        # 提取图片信息
        images = []
        if hasattr(result, "markdown_images"):
            imgs = getattr(result, "markdown_images", {}) or {}
            for name, img in imgs.items():
                images.append({"name": name, "size": getattr(img, "size", (0, 0))})

        # 提取纯文本（去掉 markdown 标记）
        plain_text = self._md_to_text(md_text)

        # 提取表格
        tables = self._extract_tables(md_text)

        # 检测空页
        is_empty = len(plain_text.strip()) < 10 and len(tables) == 0

        return {
            "page_num": page_num,
            "source_file": Path(source_file).name,
            "markdown_text": md_text,
            "plain_text": plain_text,
            "tables": tables,
            "images": images,
            "is_empty": is_empty,
            "char_count": len(plain_text),
        }

    def _extract_tables(self, md_text: str) -> List[Dict[str, Any]]:
        """
        从 Markdown 中提取表格

        Returns:
            [{table_id, headers, rows, markdown, row_count, col_count}]
        """
        tables = []
        lines = md_text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # 检测 Markdown 表格开始：| ... | ... |
            if line.startswith("|") and line.endswith("|"):
                j = i + 1
                # 检查分隔行
                if j < len(lines) and re.match(r'^\|[\s\-:|]+\|$', lines[j].strip()):
                    # 收集表格全部行
                    table_lines = [line]
                    j += 1
                    while j < len(lines) and lines[j].strip().startswith("|"):
                        table_lines.append(lines[j].strip())
                        j += 1

                    if len(table_lines) >= 2:  # 表头 + 至少 1 行数据
                        headers, rows = self._parse_md_table(table_lines)
                        tbl_id = f"tbl_p{len(tables):02d}"
                        tables.append({
                            "table_id": tbl_id,
                            "headers": headers,
                            "rows": rows,
                            "markdown": "\n".join(table_lines),
                            "row_count": len(rows),
                            "col_count": len(headers),
                        })
                    i = j
                    continue
            i += 1

        return tables

    def _parse_md_table(
        self,
        md_lines: List[str],
    ) -> tuple:
        """解析 Markdown 表格行"""
        def split_cells(line: str) -> List[str]:
            # 去掉首尾的 |，按 | 分割
            s = line.strip().strip("|")
            return [c.strip() for c in s.split("|")]

        # 过滤掉分隔行
        data_lines = [
            split_cells(l) for l in md_lines
            if not re.match(r'^\|[\s\-:|]+\|$', l.strip())
        ]

        headers = data_lines[0] if data_lines else []
        rows = data_lines[1:] if len(data_lines) > 1 else []

        return headers, rows

    def _md_to_text(self, md: str) -> str:
        """Markdown 转纯文本"""
        if not md:
            return ""

        text = md
        # 移除 HTML 标签 (图片等)
        text = re.sub(r'<[^>]+>', '', text)
        # 移除 Markdown 标题标记
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        # 移除表格行（以 | 开头）
        text = re.sub(r'^\|.*\|$', '', text, flags=re.MULTILINE)
        # 移除分隔行
        text = re.sub(r'^\|[\s\-:|]+\|$', '', text, flags=re.MULTILINE)
        # 合并多余换行
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()
