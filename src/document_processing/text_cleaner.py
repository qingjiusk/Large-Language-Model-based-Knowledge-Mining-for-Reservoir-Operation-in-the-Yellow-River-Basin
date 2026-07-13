"""
文本清洗器 (Layer 2)
OCR 后处理：去噪声、修复常见 OCR 错误、字符级归一化
"""
import re
from typing import Dict, List

from src.common.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """OCR 文本后处理清洗器"""

    # ---- 常见 OCR 错误修复字典 ----
    # 格式: (错误片段, 正确文本)，按长度降序匹配避免子串误替换
    OCR_FIX_PAIRS: List[tuple] = [
        # 长字符串垃圾 → 删除
        ("ALT KR] = PA eT", ""),
        ("AAMT AR LAA ETT EEO", ""),
        ("REP nk ee, PE, TB eT THT Aaa AR", ""),
        ("LEU TI |", ""),
        ("Ga AAMT AR LAA", ""),
        ("Lap Se ERS", ""),
        ("ROT SIM ae", ""),
        ("Nhe SE se AL", ""),
        ("nan +FtrT ND OO CC oo", ""),
        ("cm t+ NM De he", ""),
        ("当本涯以本可", ""),
        ("XING A SER TE", ""),
        ("SHULD BE NCSL AP", ""),
        ("mR ARTIS mG RAR JIE AAA", ""),
        ("MUN Ese", ""),
        ("UR", ""),
        # OCR 常见数字/单位错误
        ("FR 79.58", "面积 79.58"),
        ("Ay 380.3 eK", "约 380.3 mm"),
        ("EK,", "mm,"),
        ("EK", "mm"),
        # 年份 OCR 偏移
        ("™ K  tz0Z", ""),
        ("tz0Z", "202"),
        # 商标符号噪声
        ("™", ""),
        ("©", ""),
        ("®", ""),
        # 常见破碎字符串
        ("Juk fal", ""),
        ("Jue gt", ""),
        ("Juke ge", ""),
        ("fuk lif", ""),
        ("Jak A", ""),
    ]

    # ---- 字符级过滤 ----
    # 保留的 Unicode 范围
    VALID_CHARS_PATTERN = re.compile(
        r'[^一-鿿'        # 中文
        r'　-〿'          # CJK 标点
        r'＀-￯'          # 全角字符
        r'\w'                      # 英文数字
        r'\s'                      # 空白
        r'\.\,\%\-\+\/\(\)\[\]'   # 常用符号
        r'℀-⅏'          # 字母式符号
        r']+'
    )

    # 中文字符检测
    CHINESE_PATTERN = re.compile(r'[一-鿿]')

    # 数字检测（含小数点、负号）
    NUMBER_PATTERN = re.compile(r'-?\d+\.?\d*')

    def __init__(self):
        # 按长度降序排列修复规则，优先匹配长字符串
        self.ocr_fix_map = dict(sorted(
            self.OCR_FIX_PAIRS, key=lambda x: -len(x[0])
        ))

    def clean(self, text: str) -> str:
        """
        清洗单段 OCR 文本

        Args:
            text: 原始 OCR 文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        # Step 1: 固定错误修复
        text = self._fix_known_errors(text)

        # Step 2: 行级清洗
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned = self._clean_line(line)
            if cleaned:
                cleaned_lines.append(cleaned)

        result = '\n'.join(cleaned_lines)

        # Step 3: 全局规范化
        result = self._normalize(result)

        return result.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """批量清洗"""
        return [self.clean(t) for t in texts]

    def is_noise(self, text: str) -> bool:
        """
        判断文本是否为噪声（不可用）

        Returns:
            True 表示该文本应该被丢弃
        """
        if not text or len(text) < 10:
            return True

        chinese_chars = len(self.CHINESE_PATTERN.findall(text))
        total_chars = len(text.strip())

        # 无中文字符且无有效数字 → 噪声
        if chinese_chars == 0:
            digits = len(self.NUMBER_PATTERN.findall(text))
            if digits < 2:
                return True

        # 中文字符占比极低 → 噪声
        chinese_ratio = chinese_chars / max(total_chars, 1)
        if chinese_ratio < 0.03 and len(text) < 100:
            return True

        return False

    def extract_numbers(self, text: str) -> List[float]:
        """从文本中提取数值列表"""
        return [float(m) for m in self.NUMBER_PATTERN.findall(text)]

    def _fix_known_errors(self, text: str) -> str:
        """应用已知错误修复字典"""
        for wrong, correct in self.ocr_fix_map.items():
            if wrong in text:
                text = text.replace(wrong, correct)
        return text

    def _clean_line(self, line: str) -> str:
        """清洗单行"""
        line = line.strip()
        if not line:
            return ""

        # 删除完全由特殊字符构成的行
        chinese = len(self.CHINESE_PATTERN.findall(line))
        alphanum = sum(1 for c in line if c.isalnum())
        total = len(line)

        if total == 0:
            return ""

        useful_ratio = (chinese + alphanum) / total
        if useful_ratio < 0.1 and total > 5:
            return ""  # 纯乱码行

        # 删除过短且无意义的行
        if total < 3 and chinese == 0 and alphanum == 0:
            return ""

        return line

    def _normalize(self, text: str) -> str:
        """全局规范化"""
        # 合并多个空白行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 清理多余空格（中文之间不应有空格）
        text = re.sub(r'(?<=[一-鿿])\s+(?=[一-鿿])', '', text)

        # 统一破折号
        text = text.replace('——', '—').replace('--', '—')

        return text
