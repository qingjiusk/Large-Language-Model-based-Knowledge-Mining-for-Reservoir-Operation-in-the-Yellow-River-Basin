"""
三元组抽取器单元测试 (Mock DeepSeek API)
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_pipeline.extractor import TripletExtractor


class TestTripletExtractor(unittest.TestCase):
    """三元组抽取器测试"""

    def setUp(self):
        # 用 Mock 替代 DeepSeekClient
        self.mock_client = MagicMock()
        self.extractor = TripletExtractor(self.mock_client, prompts_dir="prompts")

    def test_normalize_list_result(self):
        """测试列表格式结果归一化"""
        result = [
            {"subject": "小浪底水库", "relation": "汛限水位", "object": "275米", "context": "...", "confidence": 0.98},
        ]
        normalized = self.extractor._normalize_result(result)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["subject"], "小浪底水库")
        self.assertEqual(normalized[0]["data_type"], "text")

    def test_normalize_dict_wrapped_result(self):
        """测试 {'triplets': [...]} 包裹格式"""
        result = {
            "triplets": [
                {"subject": "A", "relation": "R", "object": "B"},
            ]
        }
        normalized = self.extractor._normalize_result(result)
        self.assertEqual(len(normalized), 1)

    def test_normalize_invalid_result(self):
        """测试无效结果过滤"""
        result = [
            {"subject": "", "relation": "R", "object": "B"},      # 空 subject
            {"subject": "A", "relation": "", "object": "B"},      # 空 relation
            {"subject": "A"},                                      # 缺少字段
            {"subject": "A", "relation": "R", "object": "B"},     # 有效
        ]
        normalized = self.extractor._normalize_result(result)
        self.assertEqual(len(normalized), 1)

    def test_normalize_empty_result(self):
        """测试空结果"""
        self.assertEqual(self.extractor._normalize_result([]), [])
        self.assertEqual(self.extractor._normalize_result({}), [])
        self.assertEqual(self.extractor._normalize_result(None), [])

    def test_tag_source(self):
        """测试来源标记"""
        triplets = [{"subject": "A", "relation": "R", "object": "B"}]
        tagged = self.extractor._tag_source(triplets, "tabular")
        self.assertEqual(tagged[0]["data_type"], "tabular")

    def test_default_fields(self):
        """测试默认字段补全"""
        result = [{"subject": "A", "relation": "R", "object": "B"}]
        normalized = self.extractor._normalize_result(result)
        self.assertIn("context", normalized[0])
        self.assertIn("confidence", normalized[0])
        self.assertEqual(normalized[0]["context"], "")
        self.assertEqual(normalized[0]["confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
