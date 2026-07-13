"""
表格 Schema 映射器 (Layer 4)
将结构化的表格数据映射到 Neo4j 存储模板
支持预定义模板匹配 + LLM 辅助推断
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnMapping:
    """列映射定义"""
    source_column: str       # 表格中的列名
    target_field: str         # Neo4j 属性/节点名
    transform: str = "text"   # text / number / year / unit


@dataclass
class TableSchema:
    """表格 Schema 模板"""
    schema_id: str
    description: str
    table_keywords: List[str]       # 用于匹配的表头关键词
    subject_type: str               # 主体节点类型 (Reservoir/HydrologicalStation/...)
    subject_column: str             # 主体实体所在的列名
    object_type: str                # 数据节点类型 (AnnualHydrologyData/Constraint/...)
    relation_type: str              # Neo4j 关系类型
    indicator: str = ""             # 指标类型（如 实测径流量）
    default_unit: str = ""          # 默认单位
    column_mappings: List[ColumnMapping] = field(default_factory=list)
    year_column: str = ""           # 年份所在列
    value_column: str = ""          # 数值所在列


class SchemaMapper:
    """
    表格 → Neo4j Schema 映射器
    通过表头模式匹配将结构化表格映射到存储模板
    """

    # ---- 预定义的表格 Schema 模板库 ----
    BUILTIN_SCHEMAS: List[TableSchema] = [
        TableSchema(
            schema_id="runoff_station",
            description="水文站实测径流量表",
            table_keywords=["水文站", "径流量", "实测", "亿立方米"],
            subject_type="HydrologicalStation",
            subject_column="水文站",
            object_type="AnnualHydrologyData",
            relation_type="HAS_ANNUAL_DATA",
            indicator="实测径流量",
            default_unit="亿立方米",
            value_column="径流量",
        ),
        TableSchema(
            schema_id="runoff_change",
            description="水文站径流量同比变化表",
            table_keywords=["水文站", "与上年比较", "增幅", "偏多"],
            subject_type="HydrologicalStation",
            subject_column="水文站",
            object_type="AnnualHydrologyData",
            relation_type="HAS_ANNUAL_DATA",
            indicator="径流量同比变化",
            default_unit="%",
            value_column="同比变化",
        ),
        TableSchema(
            schema_id="precipitation_zone",
            description="水资源分区降水量表",
            table_keywords=["水资源", "二级区", "降水量", "mm"],
            subject_type="WaterResourceZone",
            subject_column="水资源二级区",
            object_type="AnnualHydrologyData",
            relation_type="HAS_ANNUAL_DATA",
            indicator="降水量",
            default_unit="mm",
            value_column="降水量",
        ),
        TableSchema(
            schema_id="sediment_station",
            description="水文站输沙量表",
            table_keywords=["水文站", "输沙量", "亿吨", "亿吨"],
            subject_type="HydrologicalStation",
            subject_column="水文站",
            object_type="AnnualHydrologyData",
            relation_type="HAS_ANNUAL_DATA",
            indicator="输沙量",
            default_unit="亿吨",
            value_column="输沙量",
        ),
        TableSchema(
            schema_id="water_supply_zone",
            description="水资源分区供水量表",
            table_keywords=["供水量", "耗水量", "亿立方米", "二级区"],
            subject_type="WaterResourceZone",
            subject_column="水资源二级区",
            object_type="AnnualHydrologyData",
            relation_type="HAS_ANNUAL_DATA",
            indicator="供水量",
            default_unit="亿立方米",
            value_column="供水量",
        ),
        TableSchema(
            schema_id="water_use_structure",
            description="用水结构占比表",
            table_keywords=["农业", "工业", "生活", "用水量", "占比", "%"],
            subject_type="Province",
            subject_column="流域/省区",
            object_type="AnnualHydrologyData",
            relation_type="HAS_ANNUAL_DATA",
            indicator="用水结构",
            default_unit="%",
            value_column="用水量",
        ),
        TableSchema(
            schema_id="reservoir_params",
            description="水库参数表",
            table_keywords=["水库", "正常蓄水位", "死水位", "库容", "汛限"],
            subject_type="Reservoir",
            subject_column="水库",
            object_type="Constraint",
            relation_type="HAS_CONSTRAINT",
            value_column="参数值",
        ),
    ]

    def __init__(
        self,
        schemas_path: Optional[str] = None,
        embed_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        初始化 Schema 映射器

        Args:
            schemas_path: 自定义 Schema 文件路径 (JSON)
            embed_model_name: 用于表头匹配的 embedding 模型
        """
        self.schemas = list(self.BUILTIN_SCHEMAS)

        # 加载自定义 Schema
        if schemas_path:
            self._load_custom_schemas(schemas_path)

        self.embed_model_name = embed_model_name
        self._embed_model = None

        logger.info(f"SchemaMapper 初始化: {len(self.schemas)} 个模板")

    @property
    def embed_model(self):
        """延迟加载 embedding 模型"""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(self.embed_model_name, device="cpu")
        return self._embed_model

    def match_schema(
        self,
        headers: List[str],
        sample_row: Optional[List[str]] = None,
    ) -> Tuple[Optional[TableSchema], float]:
        """
        匹配表格到最合适的 Schema 模板

        Args:
            headers: 表头列表
            sample_row: 可选的第一行数据

        Returns:
            (最佳匹配的 TableSchema, 置信度分数)
        """
        if not headers:
            return (None, 0.0)

        # 方法 1: 关键词精确匹配
        best_match, score = self._keyword_match(headers)
        if score >= 0.8:
            return (best_match, score)

        # 方法 2: Embedding 语义匹配
        best_match, score = self._embedding_match(headers)
        if score >= 0.6:
            return (best_match, score)

        # 方法 3: 数据列类型推断
        best_match, score = self._type_inference(headers, sample_row)
        return (best_match, score)

    def map_to_neo4j(
        self,
        table,
        year: Optional[int] = None,
        source_doc: str = "",
    ) -> List[Dict[str, Any]]:
        """
        将结构化表格映射为 Neo4j 写入指令

        Args:
            table: StructuredTable 实例
            year: 数据年份
            source_doc: 来源文档名

        Returns:
            Neo4j 写入指令列表
            [
                {"node_label": str, "node_id": str, "properties": {...}},
                {"relation": str, "subject_id": str, "object_id": str}
            ]
        """
        schema, score = self.match_schema(table.headers, table.rows[0] if table.rows else None)

        if schema is None or score < 0.5:
            logger.warning(f"表格 {table.table_id} 无法匹配 Schema，采用通用存储")
            return self._generic_map(table, year, source_doc)

        logger.info(
            f"表格 {table.table_id} 匹配到 Schema: {schema.schema_id} "
            f"({schema.description}), score={score:.2f}"
        )
        return self._apply_schema(table, schema, year, source_doc)

    def _keyword_match(self, headers: List[str]) -> Tuple[Optional[TableSchema], float]:
        """关键词匹配"""
        headers_text = " ".join(h for h in headers if h).lower()
        if not headers_text.strip():
            return (None, 0.0)

        best_schema = None
        best_score = 0.0

        for schema in self.schemas:
            hits = 0
            for kw in schema.table_keywords:
                kw_lower = kw.lower()
                if kw_lower in headers_text:
                    hits += 1

            score = hits / max(len(schema.table_keywords), 1)
            if score > best_score:
                best_score = score
                best_schema = schema

        return (best_schema, best_score)

    def _embedding_match(self, headers: List[str]) -> Tuple[Optional[TableSchema], float]:
        """Embedding 语义匹配"""
        try:
            headers_text = " ".join(h for h in headers if h)
            if not headers_text.strip():
                return (None, 0.0)

            header_emb = self.embed_model.encode(headers_text, normalize_embeddings=True)

            best_schema = None
            best_score = 0.0

            for schema in self.schemas:
                schema_text = f"{schema.description} {' '.join(schema.table_keywords)}"
                schema_emb = self.embed_model.encode(schema_text, normalize_embeddings=True)
                score = float(np.dot(header_emb, schema_emb))

                if score > best_score:
                    best_score = score
                    best_schema = schema

            return (best_schema, best_score)

        except Exception as e:
            logger.error(f"Embedding 匹配失败: {e}")
            return (None, 0.0)

    def _type_inference(
        self,
        headers: List[str],
        sample_row: Optional[List[str]],
    ) -> Tuple[Optional[TableSchema], float]:
        """根据数据类型列推断"""
        if not sample_row:
            return (None, 0.0)

        for schema in self.schemas:
            # 检查数值列是否包含数字
            if schema.value_column:
                col_idx = self._find_column_index(headers, schema.value_column)
                if col_idx is not None and col_idx < len(sample_row):
                    val = sample_row[col_idx]
                    if any(c.isdigit() for c in val):
                        return (schema, 0.6)

        return (None, 0.0)

    def _apply_schema(
        self,
        table,
        schema: TableSchema,
        year: Optional[int],
        source_doc: str,
    ) -> List[Dict[str, Any]]:
        """应用 Schema 模板生成 Neo4j 写入指令"""
        instructions = []
        header_idx = {h.lower(): i for i, h in enumerate(table.headers)}

        # 找到主体列和数值列
        subj_col = self._find_column_index(table.headers, schema.subject_column)
        val_col = self._find_column_index(table.headers, schema.value_column)

        for row in table.rows:
            if not row or len(row) < max(subj_col or 0, val_col or 0) + 1:
                continue

            subj_name = row[subj_col].strip() if subj_col is not None else ""
            val_text = row[val_col].strip() if val_col is not None else ""

            if not subj_name or not val_text:
                continue

            # 提取数值和单位
            import re
            num_match = re.search(r'(-?\d+\.?\d*)', val_text)
            value = float(num_match.group(1)) if num_match else None
            unit = val_text.replace(str(value), '').strip() if value else schema.default_unit

            # 生成节点 ID
            subj_id = f"{schema.subject_type}_{subj_name}"
            data_id = f"AnnualData_{subj_name}_{schema.indicator}_{year or 'N/A'}"

            # 主体节点 upsert
            instructions.append({
                "action": "upsert_node",
                "label": schema.subject_type,
                "id": subj_id,
                "properties": {"name": subj_name, "source_doc": source_doc},
            })

            # 数据节点 upsert
            data_props = {
                "name": f"{schema.indicator}: {val_text}",
                "indicator": schema.indicator,
                "value": value,
                "unit": unit or schema.default_unit,
                "year": year or 0,
                "source_doc": source_doc,
            }
            instructions.append({
                "action": "upsert_node",
                "label": schema.object_type,
                "id": data_id,
                "properties": data_props,
            })

            # 关系
            instructions.append({
                "action": "create_relationship",
                "subject_label": schema.subject_type,
                "subject_id": subj_id,
                "relation": schema.relation_type,
                "object_label": schema.object_type,
                "object_id": data_id,
                "properties": {"source_doc": source_doc, "indicator": schema.indicator},
            })

        return instructions

    def _generic_map(
        self,
        table,
        year: Optional[int],
        source_doc: str,
    ) -> List[Dict[str, Any]]:
        """通用映射（无 Schema 匹配时的兜底）"""
        # 将所有数据按行存储为通用数据节点
        instructions = []
        for row_idx, row in enumerate(table.rows):
            for col_idx, cell in enumerate(row):
                if not cell or col_idx >= len(table.headers):
                    continue
                header = table.headers[col_idx] if col_idx < len(table.headers) else f"col_{col_idx}"

                # 尝试数值提取
                import re
                num_match = re.search(r'(-?\d+\.?\d*)', cell)
                if num_match:
                    node_id = f"Data_{table.table_id}_r{row_idx}_c{col_idx}"
                    instructions.append({
                        "action": "upsert_node",
                        "label": "AnnualHydrologyData",
                        "id": node_id,
                        "properties": {
                            "name": f"{header}: {cell}",
                            "indicator": header,
                            "value": float(num_match.group(1)),
                            "year": year or 0,
                            "source_doc": source_doc,
                        },
                    })

        return instructions

    def _find_column_index(self, headers: List[str], target: str) -> Optional[int]:
        """在表头中模糊查找列索引"""
        if not headers or not target:
            return None

        target_lower = target.lower()

        for i, h in enumerate(headers):
            if not h:
                continue
            h_lower = h.lower()
            if target_lower == h_lower or target_lower in h_lower or h_lower in target_lower:
                return i

        return None  # 找不到返回 None，取第一列作为兜底

    def _load_custom_schemas(self, path: str):
        """从 JSON 加载自定义 Schema"""
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"加载自定义 Schema: {len(data)} 个模板")

import numpy as np
