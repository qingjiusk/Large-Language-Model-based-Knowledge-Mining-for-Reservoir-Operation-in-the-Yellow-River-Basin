"""
页面分类器 (Layer 0)
检测 PDF 页面类型: TEXT / TABLE / CHART / MIXED
决定后续 OCR 和处理策略
"""
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fitz
import numpy as np

from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PageRegion:
    """页面区域"""
    x: int; y: int; w: int; h: int
    region_type: str  # "text" / "table" / "chart"


@dataclass
class PageInfo:
    """页面分类结果"""
    page_num: int
    page_type: str      # "TEXT" / "TABLE" / "CHART" / "MIXED"
    confidence: float
    regions: List[PageRegion]   # MIXED 页的子区域
    has_table: bool
    is_noise: bool      # 是否为纯噪声（图表等不可文字化的内容）


class PageClassifier:
    """
    页面分类器
    使用图像分析检测页面类型，指导后续 OCR 策略
    """

    def __init__(
        self,
        min_text_ratio: float = 0.02,     # 有效中文字符最低占比
        table_num_density: float = 0.05,   # 数字密度阈值
        chart_empty_ratio: float = 0.40,   # 空白区域占比阈值（扫描件常有宽大边距）
    ):
        self.min_text_ratio = min_text_ratio
        self.table_num_density = table_num_density
        self.chart_empty_ratio = chart_empty_ratio

    def classify_page(
        self,
        doc: fitz.Document,
        page_num: int,
        ocr_preview_text: str = "",
    ) -> PageInfo:
        """
        分类单个页面

        Args:
            doc: PyMuPDF Document
            page_num: 页码 (1-indexed)
            ocr_preview_text: 可选的快速 OCR 预览文本（低 DPI）

        Returns:
            PageInfo 分类结果
        """
        page = doc[page_num - 1]

        # 渲染页面为图像进行分析
        pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))  # 72 DPI 足够分类
        img_bytes = pix.tobytes("png")

        try:
            import cv2
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        except ImportError:
            # 无 OpenCV，降级为纯文本分析
            return self._fallback_classify_text_only(page_num, ocr_preview_text)

        h, w = img.shape

        # ---- 特征 1: 空白区域占比 ----
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        total_pixels = w * h
        non_empty_pixels = cv2.countNonZero(binary)
        empty_ratio = 1.0 - (non_empty_pixels / total_pixels)

        # ---- 特征 2: 网格线密度 (表格检测) ----
        grid_score = self._detect_grid_lines(img)

        # ---- 特征 3: 文本区域占比 ----
        # 形态学闭运算检测文本块
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        text_blocks = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(text_blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_area = sum(cv2.contourArea(c) for c in contours)
        text_ratio = text_area / total_pixels

        # ---- 特征 4: 文本特征（OCR 预览辅助）----
        chinese_ratio = self._calc_chinese_ratio(ocr_preview_text)
        digit_ratio = self._calc_digit_ratio(ocr_preview_text)

        # ---- 判定逻辑 ----
        page_type, confidence, regions = self._decide_type(
            empty_ratio, grid_score, text_ratio, chinese_ratio, digit_ratio
        )

        is_noise = (
            page_type == "CHART" or
            (page_type == "TABLE" and chinese_ratio < 0.02 and digit_ratio < 0.01) or
            chinese_ratio < self.min_text_ratio
        )

        has_table = page_type in ("TABLE", "MIXED") or grid_score > 0.3

        result = PageInfo(
            page_num=page_num,
            page_type=page_type,
            confidence=confidence,
            regions=regions,
            has_table=has_table,
            is_noise=is_noise,
        )

        logger.debug(
            f"Page {page_num}: type={page_type} "
            f"(empty={empty_ratio:.2f}, grid={grid_score:.2f}, "
            f"text={text_ratio:.2f}, cn_ratio={chinese_ratio:.3f}, "
            f"digit_ratio={digit_ratio:.3f}, noise={is_noise})"
        )
        return result

    def classify_all(
        self,
        doc: fitz.Document,
        ocr_texts: Optional[Dict[int, str]] = None,
    ) -> List[PageInfo]:
        """批量分类所有页面"""
        results = []
        for page_num in range(1, len(doc) + 1):
            preview = (ocr_texts or {}).get(page_num, "")
            info = self.classify_page(doc, page_num, preview)
            results.append(info)

        stats = {
            "TEXT": sum(1 for r in results if r.page_type == "TEXT"),
            "TABLE": sum(1 for r in results if r.page_type == "TABLE"),
            "CHART": sum(1 for r in results if r.page_type == "CHART"),
            "MIXED": sum(1 for r in results if r.page_type == "MIXED"),
            "noise": sum(1 for r in results if r.is_noise),
        }
        logger.info(f"页面分类完成: {stats}")
        return results

    def _detect_grid_lines(self, img: np.ndarray) -> float:
        """检测表格网格线密度，返回 0-1 分数"""
        try:
            import cv2
            # 检测水平线
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            h_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, h_kernel)
            h_count = cv2.countNonZero(h_lines)

            # 检测垂直线
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            v_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, v_kernel)
            v_count = cv2.countNonZero(v_lines)

            total = img.shape[0] * img.shape[1]
            grid_score = (h_count + v_count) / total
            return min(grid_score * 50, 1.0)  # 归一化到 0-1
        except Exception:
            return 0.0

    def _decide_type(
        self,
        empty_ratio: float,
        grid_score: float,
        text_ratio: float,
        chinese_ratio: float,
        digit_ratio: float,
    ) -> Tuple[str, float, List[PageRegion]]:
        """综合判定页面类型：文本特征优先于图像特征"""
        regions = []

        # 规则 0: 文本特征强 → 直接判定（不依赖图像分析）
        if chinese_ratio > 0.05:
            if digit_ratio > self.table_num_density * 2:
                return ("MIXED", 0.8, regions)
            return ("TEXT", 0.9, regions)

        # 规则 1: 有中文 → 文本（优先于图像分析）
        if chinese_ratio > 0.02:
            return ("TEXT", 0.8, regions)

        # 规则 2: 图表 — 高空白 + 低文字（且上文未命中文本规则）
        if empty_ratio > self.chart_empty_ratio and text_ratio < 0.08 and chinese_ratio < 0.02:
            return ("CHART", 0.85, regions)

        # 规则 3: 纯表格 — 高网格 + 数字密集
        if grid_score > 0.25 and digit_ratio > self.table_num_density:
            return ("TABLE", 0.85, regions)

        # 规则 4: 表格 — 高数字密度（网格检测不到但有数字）
        if digit_ratio > self.table_num_density * 2:
            return ("TABLE", 0.7, regions)

        # 规则 5: 混合 — 有数字也有少量中文
        if chinese_ratio > 0.01 and digit_ratio > 0.01:
            return ("MIXED", 0.65, regions)

        # 规则 6: 有数字可能是表格
        if digit_ratio > 0.03:
            return ("TABLE", 0.5, regions)

        # 默认 → 图表噪声
        return ("CHART", 0.3, regions)

    def _calc_chinese_ratio(self, text: str) -> float:
        """计算中文字符占比"""
        if not text:
            return 0.0
        chinese = sum(1 for c in text if '一' <= c <= '鿿')
        return chinese / max(len(text), 1)

    def _calc_digit_ratio(self, text: str) -> float:
        """计算数字字符占比"""
        if not text:
            return 0.0
        digits = sum(1 for c in text if c.isdigit() or c in '.%-')
        return digits / max(len(text), 1)

    def _fallback_classify_text_only(self, page_num: int, text: str) -> PageInfo:
        """无 OpenCV 时的纯文本降级分类"""
        cn = self._calc_chinese_ratio(text)
        dg = self._calc_digit_ratio(text)

        if not text or len(text) < 20:
            ptype = "CHART"
            noise = True
        elif dg > 0.08 and cn < 0.05:
            ptype = "TABLE"
            noise = False
        elif cn > 0.01:
            ptype = "TEXT"
            noise = False
        elif dg > 0.03:
            ptype = "TABLE"
            noise = False
        else:
            ptype = "CHART"
            noise = True

        return PageInfo(
            page_num=page_num,
            page_type=ptype,
            confidence=0.6,
            regions=[],
            has_table=(ptype == "TABLE"),
            is_noise=noise,
        )
