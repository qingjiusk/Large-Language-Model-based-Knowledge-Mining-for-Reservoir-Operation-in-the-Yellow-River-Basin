"""
PDF 文档文本解析器
基于 PyMuPDF (fitz) + Tesseract OCR，支持文本型/扫描型 PDF
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF

from src.common.logger import get_logger

logger = get_logger(__name__)


class PDFParser:
    """PDF 文本解析器，支持文本型 PDF 直接提取和扫描型 PDF OCR 回退"""

    def __init__(
        self,
        skip_header_footer: bool = True,
        header_lines: int = 1,
        footer_lines: int = 1,
        min_font_size_for_heading: float = 14.0,
        use_ocr: bool = True,
        ocr_lang: str = "chi_sim+eng",
        ocr_dpi: int = 300,
    ):
        """
        初始化解析器

        Args:
            skip_header_footer: 是否过滤页眉页脚
            header_lines: 页眉行数（从顶部算）
            footer_lines: 页脚行数（从底部算）
            min_font_size_for_heading: 识别为标题的最小字号（暂未实现）
            use_ocr: 是否为扫描型 PDF 启用 OCR 回退
            ocr_lang: Tesseract OCR 语言（默认 简体中文+英文）
            ocr_dpi: OCR 渲染 DPI（越高越清晰但越慢）
        """
        self.skip_header_footer = skip_header_footer
        self.header_lines = header_lines
        self.footer_lines = footer_lines
        self.min_font_size_for_heading = min_font_size_for_heading
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self.ocr_dpi = ocr_dpi

        # 延迟加载标记
        self._is_scanned = None

    def extract_text(self, pdf_path: str) -> str:
        """
        解析单个 PDF 文件为纯文本

        Args:
            pdf_path: PDF 文件路径

        Returns:
            全文文本（页间以双换行分隔）
        """
        result = self.extract_text_with_meta(pdf_path)
        return result["full_text"]

    def extract_text_with_meta(self, pdf_path: str) -> Dict:
        """
        解析 PDF 并返回带元信息的结果

        Returns:
            {
                "file_name": str,
                "file_path": str,
                "page_count": int,
                "is_scanned": bool,
                "pages": [{"page_num": int, "text": str, "method": "native"|"ocr"}, ...],
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

        # 检测是否为扫描版（检查前 3 页）
        self._detect_scanned(doc)

        pages = []
        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            text = raw_text.strip() if raw_text else ""

            if text:
                method = "native"
            elif self.use_ocr:
                # 扫描页 — 用 OCR
                logger.debug(f"第 {page_num} 页无嵌入文字，启用 OCR...")
                text = self._ocr_page(pdf_path, page_num)
                method = "ocr"
            else:
                method = "none"

            if self.skip_header_footer and text:
                lines = text.split("\n")
                if len(lines) > self.header_lines + self.footer_lines + 1:
                    h = self.header_lines
                    f = -self.footer_lines if self.footer_lines > 0 else None
                    text = "\n".join(lines[h:f])

            pages.append({"page_num": page_num, "text": text, "method": method})

        doc.close()

        result = {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path.absolute()),
            "page_count": len(pages),
            "is_scanned": self._is_scanned,
            "pages": pages,
            "full_text": "\n\n".join(p["text"] for p in pages),
        }

        ocr_pages = sum(1 for p in pages if p["method"] == "ocr")
        logger.info(
            f"PDF 解析完成: {pdf_path.name}, {len(pages)} 页 "
            f"(native: {len(pages) - ocr_pages}, ocr: {ocr_pages})"
        )
        return result

    def _detect_scanned(self, doc: fitz.Document):
        """检测 PDF 是否为扫描版（前 3 页无文字 = 扫描版）"""
        check_pages = min(3, len(doc))
        total_chars = 0
        for i in range(check_pages):
            total_chars += len(doc[i].get_text("text").strip())

        self._is_scanned = total_chars == 0
        if self._is_scanned:
            logger.info("检测到扫描型 PDF，将使用 OCR 回退")

    def _ocr_page(self, pdf_path: Path, page_num: int) -> str:
        """
        对单页 PDF 执行 OCR

        使用 PyMuPDF 直接渲染页面为图像，再用 pytesseract 识别文字
        不需要额外安装 poppler
        """
        try:
            import pytesseract
            from PIL import Image
            import io

            # 重新打开文档获取该页（避免与主循环的 doc 冲突）
            doc = fitz.open(str(pdf_path))
            page = doc[page_num - 1]

            # 渲染为高分辨率图像
            zoom = self.ocr_dpi / 72  # 72 DPI 为基准
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            doc.close()

            # 转为 PIL Image
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            text = pytesseract.image_to_string(img, lang=self.ocr_lang)
            return text.strip()

        except Exception as e:
            logger.error(f"OCR 失败: 第 {page_num} 页, 错误: {e}")
            return ""

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

                # 保存分页 JSON（不含全文，避免重复）
                meta = {k: v for k, v in result.items() if k != "full_text"}
                json_path = output_dir / f"{pdf_file.stem}_meta.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                results.append(result)
                logger.info(f"保存完成: {txt_path}")
            except Exception as e:
                logger.error(f"解析失败: {pdf_file.name}, 错误: {e}")

        logger.info(f"批量解析完成: {len(results)}/{len(pdf_files)} 个文件成功")
        return results
