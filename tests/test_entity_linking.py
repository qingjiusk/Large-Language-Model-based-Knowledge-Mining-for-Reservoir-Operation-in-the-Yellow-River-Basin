"""
实体链接器单元测试
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_fusion.entity_linking import EntityLinker


class TestEntityLinker(unittest.TestCase):
    """实体链接器测试"""

    def setUp(self):
        self.linker = EntityLinker(
            alias_dict_path="data/ontology/alias_dict.json",
            entity_types_path="data/ontology/entity_types.json",
        )

    def test_exact_match(self):
        """测试精确匹配"""
        result = self.linker.link_entity("龙羊峡水库")
        self.assertTrue(result["matched"])
        self.assertEqual(result["standardized"], "龙羊峡水库")
        self.assertEqual(result["method"], "exact")
        self.assertEqual(result["confidence"], 1.0)

    def test_alias_match(self):
        """测试别名匹配"""
        result = self.linker.link_entity("龙库")
        self.assertTrue(result["matched"])
        self.assertEqual(result["standardized"], "龙羊峡水库")
        self.assertEqual(result["method"], "alias")

    def test_alias_match_case_insensitive(self):
        """测试大小写不敏感"""
        # 别名存储为小写
        result = self.linker.link_entity("甘肃省")
        self.assertTrue(result["matched"])
        self.assertEqual(result["standardized"], "甘肃省")

    def test_province_full_to_short(self):
        """测试省份全称→简称"""
        result = self.linker.link_entity("青海省")
        self.assertTrue(result["matched"])
        self.assertEqual(result["standardized"], "青海省")

    def test_unknown_entity(self):
        """测试未知实体"""
        result = self.linker.link_entity("不存在的实体名称XYZ")
        self.assertFalse(result["matched"])
        self.assertEqual(result["method"], "none")

    def test_empty_entity(self):
        """测试空实体名"""
        result = self.linker.link_entity("")
        self.assertFalse(result["matched"])
        self.assertEqual(result["method"], "none")

    def test_water_resource_zone_alias(self):
        """测试水资源分区别名"""
        result = self.linker.link_entity("龙库以上")
        self.assertTrue(result["matched"])
        self.assertEqual(result["standardized"], "龙羊峡以上")

    def test_hydrological_station_alias(self):
        """测试水文站别名"""
        result = self.linker.link_entity("兰州站")
        self.assertTrue(result["matched"])
        self.assertEqual(result["standardized"], "兰州水文站")


if __name__ == "__main__":
    unittest.main()
