"""
冲突检测器单元测试
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_fusion.conflict_resolver import ConflictResolver


class TestConflictResolver(unittest.TestCase):
    """冲突检测与裁决测试"""

    def setUp(self):
        self.resolver = ConflictResolver(
            table_priority=True,
            enable_year_distinction=True,
            numeric_tolerance=0.05,
        )

    def test_no_conflict_with_single_entry(self):
        """单条记录无冲突"""
        triplets = [
            {"subject": "兰州水文站", "relation": "2024年实测径流量", "object": "362.80亿立方米", "data_type": "tabular", "confidence": 0.99},
        ]
        resolved, conflicts = self.resolver.resolve(triplets)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(len(conflicts), 0)

    def test_different_years_no_conflict(self):
        """不同年份不视为冲突"""
        triplets = [
            {"subject": "兰州水文站", "relation": "2024年实测径流量", "object": "362.80亿立方米", "data_type": "tabular"},
            {"subject": "兰州水文站", "relation": "2023年实测径流量", "object": "300.00亿立方米", "data_type": "tabular"},
        ]
        resolved, conflicts = self.resolver.resolve(triplets)
        # 不同年份，都应保留
        self.assertEqual(len(resolved), 2)
        self.assertEqual(len(conflicts), 0)

    def test_same_year_numeric_conflict_table_wins(self):
        """同年数值冲突：表格优先"""
        triplets = [
            {"subject": "兰州站", "relation": "2024年径流量", "object": "300亿立方米", "data_type": "text", "confidence": 0.7},
            {"subject": "兰州站", "relation": "2024年径流量", "object": "362.80亿立方米", "data_type": "tabular", "confidence": 0.99},
        ]
        resolved, conflicts = self.resolver.resolve(triplets)
        # table_priority=True，表格应胜出
        self.assertEqual(len(resolved), 1)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(resolved[0]["object"], "362.80亿立方米")
        self.assertEqual(resolved[0]["data_type"], "tabular")

    def test_similar_values_no_conflict(self):
        """相近数值不触发冲突（在容忍度内）"""
        triplets = [
            {"subject": "兰州站", "relation": "2024年径流量", "object": "362.80亿立方米", "data_type": "tabular"},
            {"subject": "兰州站", "relation": "2024年径流量", "object": "363.00亿立方米", "data_type": "text"},
        ]
        resolved, conflicts = self.resolver.resolve(triplets)
        # 相对误差 < 5%，不视为冲突
        self.assertEqual(len(resolved), 2)
        self.assertEqual(len(conflicts), 0)

    def test_extract_number(self):
        """测试数值提取"""
        self.assertAlmostEqual(self.resolver._extract_number("362.80亿立方米"), 362.80)
        self.assertAlmostEqual(self.resolver._extract_number("275米"), 275.0)
        self.assertAlmostEqual(self.resolver._extract_number("75.5%"), 75.5)
        self.assertIsNone(self.resolver._extract_number("无数字文本"))

    def test_has_number(self):
        """测试数值检测"""
        self.assertTrue(self.resolver._has_number("362.80亿立方米"))
        self.assertTrue(self.resolver._has_number("275米"))
        self.assertFalse(self.resolver._has_number("黄河干流"))


if __name__ == "__main__":
    unittest.main()
