"""
图片解析器
支持 PNG/JPG/TIFF 等格式的图片 OCR + 表格识别
复用 PP-StructureV3 pipeline，输出与 PDFParser 一致的格式
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from src.common.logger import get_logger
from src.document_processing.paddle_ocr import PaddleOCREngine

logger = get_logger(__name__)

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


class ImageParser:
    """图片 → 结构化文本 + 表格，输出格式兼容 PDFParser"""

    def __init__(self, text_cleaner=None):
        self.engine: Optional[PaddleOCREngine] = None
        self.text_cleaner = text_cleaner

    def _get_engine(self) -> PaddleOCREngine:
        if self.engine is None:
            self.engine = PaddleOCREngine()
        return self.engine

    def extract_text_with_meta(self, image_path: str) -> Dict:
        """
        解析单张图片

        Returns 格式与 PDFParser.extract_text_with_meta() 一致:
            {
                "file_name", "file_path", "page_count": 1,
                "pages": [{page_num: 1, text, method: "paddleocr", is_noise}],
                "tables": [{table_id, headers, rows, markdown, page_num}],
                "full_text": str,
                "engine": "paddleocr"
            }
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        suffix = image_path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(f"不支持的图片格式: {suffix}，支持: {SUPPORTED_FORMATS}")

        # 验证图片可读
        try:
            img = Image.open(str(image_path))
            width, height = img.size
            logger.debug(f"图片: {image_path.name}, {width}x{height}")
        except Exception as e:
            raise RuntimeError(f"无法读取图片: {image_path}: {e}")

        # 用 PaddleOCR 处理（process_pdf 返回已解析的页面列表）
        engine = self._get_engine()
        pages = engine.process_pdf(str(image_path))

        if not pages:
            return self._empty_result(image_path)

        # process_pdf 已返回解析好的 dict，直接取第1页
        page_info = pages[0]

        # 构建与 PDFParser 兼容的输出
        text = page_info["plain_text"]
        if self.text_cleaner:
            text = self.text_cleaner.clean(text)
            is_noise = self.text_cleaner.is_noise(text)
        else:
            is_noise = page_info["is_empty"]

        if text and self.text_cleaner:
            lines = text.split("\n")
            if len(lines) > 3:
                text = "\n".join(lines[1:-1])  # 去首尾可能的噪声行

        tables = page_info["tables"]
        for t in tables:
            t["page_num"] = 1
            t["source_file"] = image_path.name

        pages = [{
            "page_num": 1,
            "text": text,
            "method": "paddleocr",
            "is_noise": is_noise,
        }]

        result = {
            "file_name": image_path.name,
            "file_path": str(image_path.absolute()),
            "page_count": 1,
            "is_scanned": True,
            "pages": pages,
            "tables": tables,
            "full_text": text if not is_noise else "",
            "engine": "paddleocr",
        }

        logger.info(
            f"图片解析完成: {image_path.name}, "
            f"{len(text)} chars, {len(tables)} 表"
        )
        return result

    def _empty_result(self, image_path: Path) -> Dict:
        return {
            "file_name": image_path.name,
            "file_path": str(image_path.absolute()),
            "page_count": 1,
            "is_scanned": True,
            "pages": [{"page_num": 1, "text": "", "method": "paddleocr", "is_noise": True}],
            "tables": [],
            "full_text": "",
            "engine": "paddleocr",
        }

    def batch_extract(self, image_dir: str, output_dir: str) -> List[Dict]:
        """批量解析目录下的所有图片"""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = []
        for fmt in SUPPORTED_FORMATS:
            files.extend(image_dir.glob(f"*{fmt}"))
            files.extend(image_dir.glob(f"*{fmt.upper()}"))

        if not files:
            logger.warning(f"目录中无支持的图片: {image_dir}")
            return []

        # 去重
        files = list(set(files))
        results = []
        for f in files:
            try:
                result = self.extract_text_with_meta(str(f))
                # 保存文本
                out_txt = output_dir / f"{f.stem}.txt"
                out_txt.write_text(result["full_text"], encoding="utf-8")
                # 保存表格
                if result["tables"]:
                    out_tbl = output_dir / f"{f.stem}_tables.json"
                    out_tbl.write_text(
                        json.dumps(result["tables"], ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )
                results.append(result)
                logger.info(f"  文字: {len(result['full_text'])} chars, 表格: {len(result['tables'])}")
            except Exception as e:
                logger.error(f"解析失败: {f.name}: {e}")

        logger.info(f"批量图片: {len(results)}/{len(files)}")
        return results
