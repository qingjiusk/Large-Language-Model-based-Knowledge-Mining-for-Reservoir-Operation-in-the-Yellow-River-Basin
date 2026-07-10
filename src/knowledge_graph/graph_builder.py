"""
知识图谱构建器
串联实体链接 → 冲突检测 → Neo4j 写入的全流程
支持全量构建和增量追加两种模式
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_fusion.entity_linking import EntityLinker
from src.knowledge_fusion.conflict_resolver import ConflictResolver

logger = get_logger(__name__)


class GraphBuilder:
    """知识图谱构建器，串联融合与入库全流程"""

    # 实体类型推断规则：根据关系类型推断 subject/object 的节点标签
    ENTITY_TYPE_RULES = {
        "LOCATED_ON": ("Reservoir", "River"),
        "LOCATED_IN_ZONE": ("Reservoir", "WaterResourceZone"),
        "BELONGS_TO_PROVINCE": ("Reservoir", "Province"),
        "IS_TRIBUTARY_OF": ("River", "River"),
        "CONTROLS": ("HydrologicalStation", "River"),
        "HAS_ANNUAL_DATA": ("Reservoir", "AnnualHydrologyData"),
        "HAS_DISPATCH_RULE": ("Reservoir", "DispatchRule"),
        "HAS_CONSTRAINT": ("Reservoir", "Constraint"),
        "FLOOD_CONTROL_LEVEL": ("Reservoir", "Constraint"),
        "NORMAL_STORAGE_LEVEL": ("Reservoir", "Constraint"),
        "DEAD_STORAGE_LEVEL": ("Reservoir", "Constraint"),
        "TOTAL_CAPACITY": ("Reservoir", "Constraint"),
        "ANNUAL_RUNOFF": ("HydrologicalStation", "AnnualHydrologyData"),
        "ANNUAL_SEDIMENT": ("HydrologicalStation", "AnnualHydrologyData"),
        "ANNUAL_PRECIPITATION": ("WaterResourceZone", "AnnualHydrologyData"),
        "WATER_SUPPLY": ("Province", "AnnualHydrologyData"),
        "WATER_USE": ("Province", "AnnualHydrologyData"),
        "SOURCE_FROM": ("Reservoir", "Document"),
    }

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        entity_linker: Optional[EntityLinker] = None,
        conflict_resolver: Optional[ConflictResolver] = None,
    ):
        """
        初始化图谱构建器

        Args:
            neo4j_client: Neo4j 客户端实例
            entity_linker: 实体链接器（可选，不传则创建默认）
            conflict_resolver: 冲突检测器（可选）
        """
        self.neo4j = neo4j_client
        self.entity_linker = entity_linker or EntityLinker()
        self.conflict_resolver = conflict_resolver or ConflictResolver()

        # 从 entity_linker 获取别名反向索引，用于实体类型推测
        self._known_entities: Dict[str, str] = {}  # name → label
        self._build_entity_type_cache()

        self.stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "conflicts_resolved": 0,
            "triplets_processed": 0,
            "triplets_skipped": 0,
        }

    def _build_entity_type_cache(self):
        """从别名字典和实体类型定义中预构建实体→类型映射"""
        # 从别名字典推测类型
        category_to_label = {
            "reservoirs": "Reservoir",
            "hydrological_stations": "HydrologicalStation",
            "water_resource_zones": "WaterResourceZone",
            "provinces": "Province",
            "rivers": "River",
        }
        alias_dict = getattr(self.entity_linker, "alias_dict", {})
        for category, label in category_to_label.items():
            for std_name, aliases in alias_dict.get(category, {}).items():
                self._known_entities[std_name] = label
                for alias in aliases:
                    self._known_entities[alias] = label

        # 从 entity_types.json 补充
        entity_types = getattr(self.entity_linker, "entity_types", {})
        for type_name, info in entity_types.items():
            label = info.get("label", type_name)
            # 将 type_name 映射为可能出现在文本中的关键词
            self._known_entities[type_name] = label

        logger.debug(f"实体类型缓存: {len(self._known_entities)} 条")

    def build_from_triplets(
        self,
        triplets: List[Dict],
        source_doc: Optional[Dict] = None,
        mode: str = "append",
    ) -> Dict[str, int]:
        """
        从三元组列表构建图谱

        Args:
            triplets: 三元组列表
            source_doc: 来源文档信息 {"name": ..., "year": ..., "type": ...}
            mode: "full" 全量 / "append" 增量追加

        Returns:
            构建统计
        """
        logger.info(f"开始构建图谱: {len(triplets)} 条三元组, mode={mode}")

        if mode == "full":
            logger.warning("全量模式：清空现有图谱...")
            self.neo4j.delete_all(confirm=True)

        # Step 1: 实体链接
        linked_triplets = self.entity_linker.link_triplets(triplets)

        # Step 2: 冲突检测与解决
        resolved_triplets, conflicts = self.conflict_resolver.resolve(linked_triplets)
        self.stats["conflicts_resolved"] = len(conflicts)

        # Step 3: 写入来源文档节点
        if source_doc:
            doc_id = source_doc.get("name", "unknown")
            self.neo4j.upsert_node("Document", doc_id, source_doc)

        # Step 4: 写节点和关系
        for triplet in resolved_triplets:
            try:
                self._write_triplet(triplet, source_doc)
                self.stats["triplets_processed"] += 1
            except Exception as e:
                logger.error(f"三元组写入失败: {triplet.get('subject')} - {e}")
                self.stats["triplets_skipped"] += 1

        # 获取图谱统计
        db_stats = self.neo4j.get_stats()
        logger.info(f"图谱构建完成: {self.stats}")
        logger.info(f"图谱统计: {db_stats}")

        return {**self.stats, **db_stats}

    def _write_triplet(self, triplet: Dict, source_doc: Optional[Dict] = None):
        """
        将单条三元组写入 Neo4j

        流程：推断实体类型 → 创建/更新节点 → 创建关系
        """
        subj = triplet.get("subject", "").strip()
        obj = triplet.get("object", "").strip()
        rel = triplet.get("relation", "").strip()

        if not subj or not obj or not rel:
            return

        # 推断节点类型
        subj_label, obj_label = self._infer_labels(subj, obj, rel)

        # 准备节点属性
        subj_props = {
            "name": subj,
            "alias": triplet.get("original_subject", ""),
        }
        obj_props = {
            "name": obj,
            "value": triplet.get("object", ""),
        }

        # 添加溯源属性
        if source_doc:
            subj_props["source_doc"] = source_doc.get("name", "")
            obj_props["source_doc"] = source_doc.get("name", "")

        # 添加置信度
        subj_props["confidence"] = triplet.get("subject_link_confidence", 1.0)
        obj_props["confidence"] = triplet.get("object_link_confidence", 1.0)

        # 写入节点
        subj_id = self._generate_id(subj_label, subj)
        obj_id = self._generate_id(obj_label, obj)

        self.neo4j.upsert_node(subj_label, subj_id, subj_props)
        self.stats["nodes_created"] += 1

        self.neo4j.upsert_node(obj_label, obj_id, obj_props)
        self.stats["nodes_created"] += 1

        # 创建关系
        rel_props = {
            "confidence": triplet.get("confidence", 0.5),
            "data_type": triplet.get("data_type", "text"),
            "context": triplet.get("context", "")[:500],
        }
        if source_doc:
            rel_props["source_doc"] = source_doc.get("name", "")

        self.neo4j.create_relationship(
            subject_label=subj_label,
            subject_id=subj_id,
            relation_type=rel,
            object_label=obj_label,
            object_id=obj_id,
            properties=rel_props,
        )
        self.stats["relationships_created"] += 1

    def _infer_labels(
        self,
        subject: str,
        object: str,
        relation: str,
    ) -> tuple:
        """
        推断主体和客体的节点标签

        优先级:
        1. 规则匹配（关系类型 → 已知映射）
        2. 名称模式匹配（如包含"水库"→ Reservoir）
        3. 默认回退
        """
        # 规则匹配
        for rel_pattern, (subj_label, obj_label) in self.ENTITY_TYPE_RULES.items():
            if rel_pattern.lower() in relation.lower() or relation.lower() in rel_pattern.lower():
                return (subj_label, obj_label)

        # 名称模式匹配
        subj_label = self._guess_label_by_name(subject)
        obj_label = self._guess_label_by_name(object)

        return (subj_label, obj_label)

    def _guess_label_by_name(self, name: str) -> str:
        """根据实体名称推测节点类型（优先使用已知实体缓存）"""
        name_str = name.strip()

        # Step 0: 查预构建的已知实体缓存
        if name_str in self._known_entities:
            return self._known_entities[name_str]
        for known_name, label in self._known_entities.items():
            if known_name in name_str or name_str in known_name:
                return label

        # Step 1: 名称关键词匹配
        if any(kw in name_str for kw in ["水库", "水电站", "水利枢纽"]):
            return "Reservoir"
        if any(kw in name_str for kw in ["水文站", "水文"]):
            return "HydrologicalStation"
        if any(kw in name_str for kw in ["省", "自治区", "直辖市"]):
            return "Province"
        if any(kw in name_str for kw in ["河", "江", "水系", "流域"]):
            return "River"
        if any(kw in name_str for kw in ["区", "区间"]):
            return "WaterResourceZone"
        if any(kw in name_str for kw in ["水位", "库容", "径流量", "输沙量", "降水量",
                                           "亿立方米", "米", "mm", "吨", "%"]):
            return "AnnualHydrologyData"
        if any(kw in name_str for kw in ["规则", "规程", "调度", "方案"]):
            return "DispatchRule"

        # 默认
        return "Constraint"

    def _generate_id(self, label: str, name: str) -> str:
        """生成节点唯一 ID"""
        # 简单规则：标签前缀 + 名称哈希
        import hashlib
        hash_suffix = hashlib.md5(f"{label}:{name}".encode()).hexdigest()[:8]
        safe_name = re.sub(r'[^\w一-鿿]', '_', name)[:30]
        return f"{label}_{safe_name}_{hash_suffix}"
