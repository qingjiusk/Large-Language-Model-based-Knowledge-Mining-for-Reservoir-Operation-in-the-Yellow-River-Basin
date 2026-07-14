"""
PDF 文档文本解析器
OCR 引擎: PaddleOCR (PP-StructureV3) 优先 → Tesseract 兜底
支持文本/表格/版面分析
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF

from src.common.logger import get_logger
from src.document_processing.text_cleaner import TextCleaner

logger = get_logger(__name__)


class PDFParser:
    """PDF → 结构化文本 + 表格"""

    def __init__(
        self,
        skip_header_footer: bool = True,
        header_lines: int = 1,
        footer_lines: int = 1,
        use_ocr: bool = True,
        ocr_lang: str = "chi_sim+eng",
        ocr_dpi: int = 300,
        text_cleaner: Optional[TextCleaner] = None,
        use_paddle: bool = True,
    ):
        self.skip_header_footer = skip_header_footer
        self.header_lines = header_lines
        self.footer_lines = footer_lines
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self.ocr_dpi = ocr_dpi
        self.text_cleaner = text_cleaner or TextCleaner()
        self.use_paddle = use_paddle
        self._is_scanned = False

    def extract_text(self, pdf_path: str) -> str:
        result = self.extract_text_with_meta(pdf_path)
        return result["full_text"]

    def extract_text_with_meta(self, pdf_path: str) -> Dict:
        """
        Returns:
            {
                "file_name", "file_path", "page_count", "is_scanned",
                "pages": [{page_num, text, method, is_noise}],
                "tables": [{table_id, headers, rows, markdown, page_num}],
                "full_text"
            }
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise RuntimeError(f"PDF 文件无法打开: {pdf_path}") from e

        # ---- 检测扫描版 ----
        self._detect_scanned(doc)

        # ---- 选择 OCR 引擎 ----
        paddle_engine = None
        if self.use_paddle and self._is_scanned:
            try:
                from src.document_processing.paddle_ocr import PaddleOCREngine
                paddle_engine = PaddleOCREngine()
                logger.info("使用 PaddleOCR (PP-StructureV3) 引擎")
            except Exception as e:
                logger.warning(f"PaddleOCR 不可用，回退 Tesseract: {e}")
                paddle_engine = None

        if paddle_engine:
            pages, tables = self._parse_with_paddle(paddle_engine, pdf_path)
        else:
            pages, tables = self._parse_with_tesseract(doc, str(pdf_path))

        doc.close()

        # ---- 清洗 + 噪声过滤 ----
        for p in pages:
            if p["text"]:
                p["text"] = self.text_cleaner.clean(p["text"])
            p["is_noise"] = self.text_cleaner.is_noise(p["text"])

        full_text = "\n\n".join(
            p["text"] for p in pages
            if not p["is_noise"] and p["text"].strip()
        )

        # 统计表格来源页
        table_pages = set()
        for t in tables:
            t["source_file"] = pdf_path.name
            table_pages.add(t.get("page_num", 0))

        result = {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path.absolute()),
            "page_count": len(pages),
            "is_scanned": self._is_scanned,
            "pages": pages,
            "tables": tables,
            "full_text": full_text,
            "engine": "paddleocr" if paddle_engine else "tesseract",
        }

        noise_count = sum(1 for p in pages if p["is_noise"])
        logger.info(
            f"PDF 解析完成: {pdf_path.name}, "
            f"{len(pages)} 页 (噪声:{noise_count}), "
            f"{len(tables)} 表, "
            f"full_text={len(full_text)} chars"
        )
        return result

    def _parse_with_paddle(self, engine, pdf_path) -> tuple:
        """PaddleOCR 解析"""
        raw_pages = engine.process_pdf(str(pdf_path))

        pages = []
        all_tables = []

        for rp in raw_pages:
            text = rp["plain_text"]
            if self.skip_header_footer and text:
                lines = text.split("\n")
                if len(lines) > self.header_lines + self.footer_lines + 1:
                    text = "\n".join(lines[self.header_lines:-self.footer_lines or None])

            pages.append({
                "page_num": rp["page_num"],
                "text": text,
                "method": "paddleocr",
                "is_noise": rp["is_empty"],
            })

            # 合并表格 + 附加页码
            for tbl in rp["tables"]:
                tbl["page_num"] = rp["page_num"]
                all_tables.append(tbl)

        return pages, all_tables

    def _parse_with_tesseract(self, doc, pdf_path: str) -> tuple:
        """Tesseract OCR 兜底解析"""
        pages = []
        for page_num in range(1, len(doc) + 1):
            raw_text = self._ocr_page_tesseract(doc, page_num)
            if self.skip_header_footer and raw_text:
                lines = raw_text.split("\n")
                if len(lines) > self.header_lines + self.footer_lines + 1:
                    raw_text = "\n".join(lines[self.header_lines:-self.footer_lines or None])
            pages.append({
                "page_num": page_num,
                "text": raw_text,
                "method": "tesseract",
            })

        # Tesseract 无表格检测
        return pages, []

    def _ocr_page_tesseract(self, doc, page_num: int) -> str:
        """单页 Tesseract OCR"""
        try:
            import pytesseract
            from PIL import Image
            import io

            page = doc[page_num - 1]
            native = page.get_text("text").strip()
            if native and len(native) > 50:
                return native
            if not self.use_ocr:
                return native

            zoom = self.ocr_dpi / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img, lang=self.ocr_lang).strip()
        except Exception as e:
            logger.error(f"Tesseract OCR 失败: 第 {page_num} 页: {e}")
            return ""

    def _detect_scanned(self, doc: fitz.Document):
        check_pages = min(3, len(doc))
        total_chars = 0
        for i in range(check_pages):
            total_chars += len(doc[i].get_text("text").strip())
        self._is_scanned = total_chars == 0
        if self._is_scanned:
            logger.info("检测到扫描型 PDF，将使用 OCR")

    def batch_extract(self, pdf_dir: str, output_dir: str) -> List[Dict]:
        """批量解析"""
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
        if not pdf_files:
            logger.warning(f"目录中无 PDF: {pdf_dir}")
            return []

        results = []
        for pdf_file in pdf_files:
            logger.info(f"处理: {pdf_file.name}")
            try:
                result = self.extract_text_with_meta(str(pdf_file))
                txt_path = output_dir / f"{pdf_file.stem}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(result["full_text"])

                meta = {k: v for k, v in result.items() if k not in ("full_text",)}
                json_path = output_dir / f"{pdf_file.stem}_meta.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                if result.get("tables"):
                    tbl_path = output_dir / f"{pdf_file.stem}_tables.json"
                    with open(tbl_path, "w", encoding="utf-8") as f:
                        json.dump(result["tables"], f, ensure_ascii=False, indent=2)

                results.append(result)
            except Exception as e:
                logger.error(f"解析失败: {pdf_file.name}, {e}")

        logger.info(f"批量完成: {len(results)}/{len(pdf_files)}")
        return results
