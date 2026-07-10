"""
PDF 文档文本解析器
基于 PyMuPDF (fitz)，支持单文件与批量解析，自动过滤页眉页脚
"""
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF

from src.common.logger import get_logger

logger = get_logger(__name__)


class PDFParser:
    """PDF 文本解析器，将 PDF 转为结构化纯文本"""

    def __init__(
        self,
        skip_header_footer: bool = True,
        header_lines: int = 1,
        footer_lines: int = 1,
        min_font_size_for_heading: float = 14.0,
    ):
        """
        初始化解析器

        Args:
            skip_header_footer: 是否过滤页眉页脚
            header_lines: 页眉行数（从顶部算）
            footer_lines: 页脚行数（从底部算）
            min_font_size_for_heading: 识别为标题的最小字号（暂未实现）
        """
        self.skip_header_footer = skip_header_footer
        self.header_lines = header_lines
        self.footer_lines = footer_lines
        self.min_font_size_for_heading = min_font_size_for_heading

    def extract_text(self, pdf_path: str) -> str:
        """
        解析单个 PDF 文件为纯文本

        Args:
            pdf_path: PDF 文件路径

        Returns:
            全文文本（页间以双换行分隔）
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            logger.error(f"无法打开 PDF: {pdf_path}, 错误: {e}")
            raise RuntimeError(f"PDF 文件无法打开，可能已损坏或加密: {pdf_path}") from e

        page_texts = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if not text or not text.strip():
                continue

            if self.skip_header_footer:
                lines = text.strip().split("\n")
                if len(lines) > self.header_lines + self.footer_lines + 1:
                    h = self.header_lines
                    f = -self.footer_lines if self.footer_lines > 0 else None
                    text = "\n".join(lines[h:f])
                elif len(lines) <= 3:
                    # 短页面保留全部
                    pass

            page_texts.append(text)

        doc.close()
        full_text = "\n\n".join(page_texts)

        stats = {
            "path": str(pdf_path),
            "pages": len(page_texts),
            "chars": len(full_text),
            "lines": full_text.count("\n") + 1,
        }
        logger.info(f"PDF 解析完成: {stats}")
        return full_text

    def extract_text_with_meta(self, pdf_path: str) -> Dict:
        """
        解析 PDF 并返回带元信息的结果

        Returns:
            {
                "file_name": str,
                "file_path": str,
                "page_count": int,
                "pages": [{"page_num": int, "text": str}, ...],
                "full_text": str
            }
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise RuntimeError(f"PDF 文件无法打开: {pdf_path}") from e

        pages = []
        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            text = raw_text.strip() if raw_text else ""

            if self.skip_header_footer and text:
                lines = text.split("\n")
                if len(lines) > self.header_lines + self.footer_lines + 1:
                    h = self.header_lines
                    f = -self.footer_lines if self.footer_lines > 0 else None
                    text = "\n".join(lines[h:f])

            pages.append({"page_num": page_num, "text": text})

        doc.close()

        result = {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path.absolute()),
            "page_count": len(pages),
            "pages": pages,
            "full_text": "\n\n".join(p["text"] for p in pages),
        }

        logger.info(f"PDF 解析完成（带元信息）: {pdf_path.name}, {len(pages)} 页")
        return result

    def batch_extract(self, pdf_dir: str, output_dir: str) -> List[Dict]:
        """
        批量解析目录下的所有 PDF 文件

        Args:
            pdf_dir: 包含 PDF 文件的目录
            output_dir: 文本输出目录

        Returns:
            每个文件的解析结果列表
        """
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
        if not pdf_files:
            logger.warning(f"目录中无 PDF 文件: {pdf_dir}")
            return []

        results = []
        for pdf_file in pdf_files:
            logger.info(f"批量处理: {pdf_file.name}")
            try:
                result = self.extract_text_with_meta(str(pdf_file))

                # 保存全文
                txt_path = output_dir / f"{pdf_file.stem}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(result["full_text"])

                # 保存分页 JSON
                json_path = output_dir / f"{pdf_file.stem}_meta.json"
                import json
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                results.append(result)
                logger.info(f"保存完成: {txt_path}")
            except Exception as e:
                logger.error(f"解析失败: {pdf_file.name}, 错误: {e}")

        logger.info(f"批量解析完成: {len(results)}/{len(pdf_files)} 个文件成功")
        return results
