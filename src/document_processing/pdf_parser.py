"""
PDF 文档文本解析器 (增强版)
集成页面分类 + OCR + 文本清洗 + 表格重建
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF

from src.common.logger import get_logger
from src.document_processing.page_classifier import PageClassifier, PageInfo
from src.document_processing.text_cleaner import TextCleaner

logger = get_logger(__name__)


class PDFParser:
    """PDF 文本解析器 — 页面分类 → 分类 OCR → 清洗 → 输出"""

    def __init__(
        self,
        skip_header_footer: bool = True,
        header_lines: int = 1,
        footer_lines: int = 1,
        use_ocr: bool = True,
        ocr_lang: str = "chi_sim+eng",
        ocr_dpi: int = 300,
        classifier: Optional[PageClassifier] = None,
        text_cleaner: Optional[TextCleaner] = None,
    ):
        self.skip_header_footer = skip_header_footer
        self.header_lines = header_lines
        self.footer_lines = footer_lines
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self.ocr_dpi = ocr_dpi
        self.classifier = classifier or PageClassifier()
        self.text_cleaner = text_cleaner or TextCleaner()

    def extract_text(self, pdf_path: str) -> str:
        result = self.extract_text_with_meta(pdf_path)
        return result["full_text"]

    def extract_text_with_meta(self, pdf_path: str) -> Dict:
        """
        增强版 PDF 解析：分类 → OCR → 清洗

        Returns:
            {
                "file_name", "file_path", "page_count", "is_scanned",
                "page_types": {page_num: "TEXT"/"TABLE"/"CHART"/"MIXED"},
                "noise_pages": [page_nums...],
                "pages": [{page_num, text, method, page_type, is_noise}],
                "tables": [StructuredTable...],
                "full_text": str (只含非噪声页的清洗后文本)
            }
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise RuntimeError(f"PDF 文件无法打开: {pdf_path}") from e

        # ---- 检测是否为扫描版 ----
        self._detect_scanned(doc)

        # ---- 全页 OCR（先提取文字，再根据内容分类） ----
        # 策略：OCR 全部页面，然后用文字内容判定是否为噪声
        pages = []
        tables = []
        table_reconstructor = None

        for page_num in range(1, len(doc) + 1):
            raw_text = self._ocr_page(doc, page_num)
            cleaned_text = self.text_cleaner.clean(raw_text)
            is_noise = self.text_cleaner.is_noise(cleaned_text)

            pages.append({
                "page_num": page_num,
                "text": cleaned_text if not is_noise else "",
                "raw_text": raw_text,
                "method": "ocr",
                "is_noise": is_noise,
            })

        # ---- 基于完整 OCR 文字做页面分类 ----
        ocr_texts = {p["page_num"]: p["raw_text"] for p in pages}
        page_infos = self.classifier.classify_all(doc, ocr_texts)

        # 合并分类结果
        for i, pi in enumerate(page_infos):
            pages[i]["page_type"] = pi.page_type
            # 合并噪声判定：分类器 or 文本清洗器 判定为噪声则跳过
            pages[i]["is_noise"] = pi.is_noise or pages[i]["is_noise"]

        page_type_map = {pi.page_num: pi.page_type for pi in page_infos}
        noise_set = {pi.page_num for pi in page_infos if pi.is_noise}

        # ---- 对 TABLE 页做表格重建 ----
        for i, pi in enumerate(page_infos):
            if pi.has_table and not pi.is_noise:
                page_num = pi.page_num
                if table_reconstructor is None:
                    from src.document_processing.table_reconstructor import TableReconstructor
                    table_reconstructor = TableReconstructor(
                        text_cleaner=self.text_cleaner,
                        ocr_lang=self.ocr_lang,
                        ocr_dpi=self.ocr_dpi,
                    )
                page_tables = table_reconstructor.extract_tables_from_page(
                    doc, page_num, str(pdf_path)
                )
                tables.extend(page_tables)

        doc.close()

        # 构建全文（仅非噪声页）
        full_text = "\n\n".join(
            p["text"] for p in pages
            if not p["is_noise"] and p["text"].strip()
        )

        result = {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path.absolute()),
            "page_count": len(pages),
            "is_scanned": self._is_scanned,
            "page_types": page_type_map,
            "noise_pages": sorted(noise_set),
            "pages": pages,
            "tables": tables,
            "full_text": full_text,
        }

        text_pages = sum(1 for p in pages if p["method"] != "skipped")
        noise = sum(1 for p in pages if p["is_noise"])
        table_count = len(tables)
        logger.info(
            f"PDF 解析完成: {pdf_path.name}, "
            f"{text_pages}/{len(pages)} 页有效, "
            f"噪声: {noise}, 表格: {table_count}"
        )
        return result

    def _fast_preview(self, doc: fitz.Document) -> Dict[int, str]:
        """中等 DPI OCR 预览（供页面分类用，太快了 OCR 效果差）"""
        previews = {}
        check_pages = min(6, len(doc))
        for page_num in range(1, check_pages + 1):
            try:
                page = doc[page_num - 1]
                zoom = 150 / 72  # 150 DPI 平衡速度和质量
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                import pytesseract
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, lang=self.ocr_lang)
                previews[page_num] = text
            except Exception:
                pass
        return previews

    def _detect_scanned(self, doc: fitz.Document):
        """检测 PDF 是否为扫描版"""
        check_pages = min(3, len(doc))
        total_chars = 0
        for i in range(check_pages):
            total_chars += len(doc[i].get_text("text").strip())
        self._is_scanned = total_chars == 0
        if self._is_scanned:
            logger.info("检测到扫描型 PDF，将使用 OCR")

    def _ocr_page(self, doc: fitz.Document, page_num: int) -> str:
        """对单页执行 OCR"""
        try:
            import pytesseract
            from PIL import Image
            import io

            page = doc[page_num - 1]

            # 先尝试原生文本
            native_text = page.get_text("text").strip()
            if native_text and len(native_text) > 50:
                return native_text

            if not self.use_ocr:
                return native_text

            zoom = self.ocr_dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang=self.ocr_lang)
            return text.strip()

        except Exception as e:
            logger.error(f"OCR 失败: 第 {page_num} 页, 错误: {e}")
            return ""

    def batch_extract(self, pdf_dir: str, output_dir: str) -> List[Dict]:
        """批量解析目录下的所有 PDF 文件"""
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
                meta = {k: v for k, v in result.items() if k not in ("full_text", "tables")}
                json_path = output_dir / f"{pdf_file.stem}_meta.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                # 保存表格
                if result.get("tables"):
                    tbl_path = output_dir / f"{pdf_file.stem}_tables.json"
                    # 只保存 markdown 和元信息（Cell 对象不可序列化）
                    tbl_data = [{
                        "table_id": t.table_id,
                        "page_num": t.page_num,
                        "headers": t.headers,
                        "rows": t.rows,
                        "markdown": t.markdown,
                        "column_count": t.column_count,
                        "row_count": t.row_count,
                    } for t in result["tables"]]
                    with open(tbl_path, "w", encoding="utf-8") as f:
                        json.dump(tbl_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"表格保存: {tbl_path}, {len(tbl_data)} 个")

                results.append(result)
                logger.info(f"保存完成: {txt_path}")
            except Exception as e:
                logger.error(f"解析失败: {pdf_file.name}, 错误: {e}")

        logger.info(f"批量解析完成: {len(results)}/{len(pdf_files)} 个文件成功")
        return results
