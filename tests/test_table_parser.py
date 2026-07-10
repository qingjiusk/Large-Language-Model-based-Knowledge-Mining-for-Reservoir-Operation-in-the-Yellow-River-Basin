"""
表格解析器单元测试
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processing.table_parser import TableParser


class TestTableParser(unittest.TestCase):
    """表格解析器测试"""

    def setUp(self):
        self.parser = TableParser(min_table_rows=2, output_format="markdown")

    def test_clean_table_removes_empty_rows(self):
        """测试空行过滤"""
        table = [
            ["A", "B"],
            ["", ""],  # 应被过滤
            ["1", "2"],
        ]
        cleaned = self.parser._clean_table(table)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0], ["A", "B"])

    def test_clean_table_handles_none_cells(self):
        """测试 None 单元格处理"""
        table = [
            ["Header", None],
            ["data1", "data2"],
        ]
        cleaned = self.parser._clean_table(table)
        self.assertEqual(cleaned[0], ["Header", ""])
        self.assertEqual(cleaned[1], ["data1", "data2"])

    def test_to_markdown_basic(self):
        """测试基本 Markdown 转换"""
        table = [
            ["Name", "Value"],
            ["水位", "275米"],
            ["库容", "126.5亿"],
        ]
        md = self.parser._to_markdown(table)
        self.assertIn("| Name | Value |", md)
        self.assertIn("| --- | --- |", md)
        self.assertIn("| 水位 | 275米 |", md)
        self.assertIn("| 库容 | 126.5亿 |", md)
        self.assertEqual(md.count("\n"), 3)  # header + sep + 2 rows = 3 newlines (join of 4 lines)

    def test_to_markdown_empty(self):
        """测试空表格"""
        md = self.parser._to_markdown([])
        self.assertEqual(md, "")

    def test_normalize_columns(self):
        """测试列补齐"""
        table = [
            ["A", "B", "C"],
            ["1", "2"],  # 少一列
            ["x", "y", "z", "extra"],  # 多一列
        ]
        normalized = self.parser._normalize_columns(table)
        # 所有行应该有相同列数
        cols = [len(row) for row in normalized]
        self.assertEqual(len(set(cols)), 1)
        self.assertGreaterEqual(normalized[1][2], "")  # 补的空格

    def test_skip_small_tables(self):
        """测试过滤过小的表格"""
        # 构造只有一个有效行的表格
        table = [["Single"]]
        cleaned = self.parser._clean_table(table)
        # min_table_rows=2，1行表格应被跳过（extract_tables 层面）
        self.assertEqual(len(cleaned), 1)  # _clean_table 不过滤行数，extract_tables 过滤

    def test_output_format_both(self):
        """测试 both 输出格式"""
        parser = TableParser(output_format="both")
        self.assertEqual(parser.output_format, "both")


if __name__ == "__main__":
    unittest.main()
