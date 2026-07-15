"""
知识图谱查询封装
常用业务查询的 Cypher 模板
注意事项：关系类型由 LLM 抽取，是动态的中文/英文关系名，
          查询时不硬编码关系类型，而是通过节点标签来定位。
"""
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.knowledge_graph.neo4j_client import Neo4jClient

logger = get_logger(__name__)


class GraphQuery:
    """图谱查询封装，提供常用业务查询接口"""

    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client

    # ==================== 水库查询 ====================

    def get_reservoir(self, reservoir_id: str) -> Optional[Dict]:
        """查询单个水库的完整信息"""
        query = """
        MATCH (r:Reservoir {id: $id})
        OPTIONAL MATCH (r)-[rel]->(n)
        RETURN r, collect({type: type(rel), target: labels(n), props: properties(n)}) AS relations
        """
        return self.neo4j.execute_read_single(query, {"id": reservoir_id})

    def search_reservoirs(self, keyword: str, limit: int = 20) -> List[Dict]:
        """按名称模糊搜索水库"""
        query = """
        MATCH (r:Reservoir)
        WHERE r.name CONTAINS $keyword OR r.alias CONTAINS $keyword
        RETURN r
        LIMIT $limit
        """
        return self.neo4j.execute_read(query, {"keyword": keyword, "limit": limit})

    def get_reservoir_rules(self, reservoir_id: str) -> List[Dict]:
        """查询水库的调度规则（匹配目标为 DispatchRule 的关联）"""
        query = """
        MATCH (r:Reservoir {id: $id})-[rel]->(rule:DispatchRule)
        RETURN rule, type(rel) AS relation, rel.confidence AS confidence, rel.source_doc AS source
        """
        return self.neo4j.execute_read(query, {"id": reservoir_id})

    def get_reservoir_constraints(self, reservoir_id: str) -> List[Dict]:
        """查询水库的约束条件（匹配目标为 Constraint 的关联）"""
        query = """
        MATCH (r:Reservoir {id: $id})-[rel]->(c:Constraint)
        RETURN c, type(rel) AS relation, rel.confidence AS confidence, rel.source_doc AS source
        """
        return self.neo4j.execute_read(query, {"id": reservoir_id})

    # ==================== 水文数据查询 ====================

    def get_stations(self, limit: int = 50) -> List[Dict]:
        """获取水文站列表"""
        query = """
        MATCH (s:HydrologicalStation)
        RETURN s
        ORDER BY s.name
        LIMIT $limit
        """
        return self.neo4j.execute_read(query, {"limit": limit})

    def get_station_data(
        self,
        station_id: str,
        year: Optional[int] = None,
        indicator: Optional[str] = None,
    ) -> List[Dict]:
        """查询水文站的年度数据（匹配目标为 AnnualHydrologyData 的关联）"""
        conditions = ["s.id = $station_id"]
        params: Dict = {"station_id": station_id}

        if year:
            conditions.append("d.year = $year")
            params["year"] = year
        if indicator:
            conditions.append("d.indicator CONTAINS $indicator")
            params["indicator"] = indicator

        where_clause = " AND ".join(conditions)

        query = f"""
        MATCH (s:HydrologicalStation {{id: $station_id}})-[rel]->(d:AnnualHydrologyData)
        WHERE {where_clause}
        RETURN d, type(rel) AS relation, s.name AS station_name
        ORDER BY d.year DESC
        LIMIT 100
        """
        return self.neo4j.execute_read(query, params)

    def get_zone_data(
        self,
        zone_id: Optional[str] = None,
        year: Optional[int] = None,
    ) -> List[Dict]:
        """查询水资源分区的年度数据（匹配目标为 AnnualHydrologyData 的关联）"""
        params: Dict = {}
        zone_filter = ""
        if zone_id:
            zone_filter = "WHERE z.id = $zone_id"
            params["zone_id"] = zone_id

        year_filter = ""
        if year:
            year_filter = "WHERE d.year = $year" if not zone_filter else "AND d.year = $year"
            params["year"] = year

        query = f"""
        MATCH (z:WaterResourceZone)-[rel]->(d:AnnualHydrologyData)
        {zone_filter}
        {year_filter}
        RETURN z.name AS zone_name, d, type(rel) AS relation
        ORDER BY d.year DESC
        LIMIT 100
        """
        return self.neo4j.execute_read(query, params)

    # ==================== 关系与路径查询 ====================

    def find_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 4,
    ) -> List[Dict]:
        """查询两个实体之间的最短路径"""
        query = """
        MATCH path = shortestPath(
            (a {id: $from_id})-[*1..$max_depth]-(b {id: $to_id})
        )
        RETURN [node in nodes(path) | {id: node.id, labels: labels(node)}] AS nodes,
               [rel in relationships(path) | {type: type(rel), props: properties(rel)}] AS relationships,
               length(path) AS depth
        LIMIT 5
        """
        return self.neo4j.execute_read(
            query,
            {"from_id": from_id, "to_id": to_id, "max_depth": max_depth},
        )

    # ==================== 溯源查询 ====================

    def trace_by_document(self, doc_name: str, limit: int = 100) -> List[Dict]:
        """按来源文档查询所有相关知识"""
        query = """
        MATCH (a)-[r]->(b)
        WHERE r.source_doc = $doc_name
        RETURN a.name AS subject, type(r) AS relation, b.name AS object,
               r.confidence AS confidence, r.context AS context,
               r.data_type AS data_type
        LIMIT $limit
        """
        return self.neo4j.execute_read(
            query,
            {"doc_name": doc_name, "limit": limit},
        )

    def search_triplets(
        self,
        keyword: str,
        limit: int = 50,
    ) -> List[Dict]:
        """关键词搜索三元组（在主体/关系/客体名中搜索）"""
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.name CONTAINS $keyword
           OR b.name CONTAINS $keyword
        RETURN a.name AS subject, labels(a) AS subject_type,
               type(r) AS relation,
               b.name AS object, labels(b) AS object_type,
               r.confidence AS confidence,
               r.source_doc AS source_doc,
               r.context AS context
        ORDER BY r.confidence DESC
        LIMIT $limit
        """
        return self.neo4j.execute_read(
            query,
            {"keyword": keyword, "limit": limit},
        )

    # ==================== 统计查询 ====================

    def get_kg_stats(self) -> Dict:
        """获取知识图谱统计概览"""
        return self.neo4j.get_stats()

    def get_reservoir_list(self, limit: int = 50) -> List[Dict]:
        """获取所有水库列表"""
        query = """
        MATCH (r:Reservoir)
        OPTIONAL MATCH (r)-[rel]->(n)
        WHERE n:River OR n:HydrologicalStation
        RETURN r.name AS name, r.id AS id
        ORDER BY r.name
        LIMIT $limit
        """
        return self.neo4j.execute_read(query, {"limit": limit})

    def get_water_resource_zones(self) -> List[Dict]:
        """获取所有水资源分区"""
        query = """
        MATCH (z:WaterResourceZone)
        RETURN z.name AS name, z.id AS id, z.area AS area
        ORDER BY z.name
        """
        return self.neo4j.execute_read(query)

    # ==================== 优化调度数据查询 ====================

    def get_optimization_constraints(self, reservoir_id: str) -> List[Dict]:
        """
        查水库所有约束数据，合并两个来源:
        1. Constraint 节点 (直接约束)
        2. AnnualHydrologyData 中包含约束关键词的 (间接约束，如水位/流量限制)

        返回统一格式的字典列表
        """
        # 查询1: Constraint 节点
        constraint_query = """
        MATCH (r:Reservoir {id: $id})-[rel]->(c:Constraint)
        RETURN c,
               type(rel) AS relation,
               rel.confidence AS confidence,
               rel.source_doc AS source_doc
        """
        constraint_results = self.neo4j.execute_read(constraint_query, {"id": reservoir_id})

        # 查询2: 含约束关键词的 AnnualHydrologyData
        # 关键词: 水位/流量/库容/出力/供水/生态/汛限/死水位/正常蓄水位
        hydrology_query = """
        MATCH (r:Reservoir {id: $id})-[rel]->(d:AnnualHydrologyData)
        WHERE type(rel) CONTAINS '水位'
           OR type(rel) CONTAINS '流量'
           OR type(rel) CONTAINS '库容'
           OR type(rel) CONTAINS '出力'
           OR type(rel) CONTAINS '供水'
           OR type(rel) CONTAINS '生态'
           OR type(rel) CONTAINS '汛限'
           OR type(rel) CONTAINS '死水位'
           OR type(rel) CONTAINS '蓄水位'
           OR type(rel) CONTAINS '防洪'
           OR type(rel) CONTAINS '限制'
           OR type(rel) CONTAINS '约束'
           OR type(rel) CONTAINS '上限'
           OR type(rel) CONTAINS '下限'
           OR type(rel) CONTAINS '不超过'
           OR type(rel) CONTAINS '不低于'
           OR type(rel) CONTAINS '用水'
           OR type(rel) CONTAINS '灌溉'
        RETURN d.indicator AS indicator,
               d.value AS value,
               d.unit AS unit,
               d.year AS year,
               type(rel) AS relation,
               d.name AS name,
               rel.confidence AS confidence,
               rel.source_doc AS source_doc
        """
        hydrology_results = self.neo4j.execute_read(hydrology_query, {"id": reservoir_id})

        # 合并: 将 AnnualHydrologyData 结果转为与 Constraint 兼容的格式
        merged = list(constraint_results)
        for h in hydrology_results:
            merged.append({
                "c": None,
                "relation": h.get("relation", ""),
                "confidence": h.get("confidence"),
                "source_doc": h.get("source_doc"),
                "indicator": h.get("indicator", ""),
                "d": {
                    "indicator": h.get("indicator", ""),
                    "value": h.get("value", ""),
                    "unit": h.get("unit", ""),
                    "year": h.get("year", ""),
                    "name": h.get("name", ""),
                },
            })

        logger.info(
            f"优化约束查询: reservoir={reservoir_id}, "
            f"Constraint={len(constraint_results)}, HydrologyData={len(hydrology_results)}"
        )
        return merged

    def get_reservoir_parameters(self, reservoir_id: str) -> Dict:
        """
        查询水库物理参数，优先级:
        1. Constraint 节点中的参数值 (水位/库容类约束)
        2. Reservoir 节点属性中的值 (total_capacity, flood_control_capacity 等)
        """
        params: Dict[str, Any] = {}

        # 查询1: Reservoir 节点属性
        reservoir_query = """
        MATCH (r:Reservoir {id: $id})
        RETURN r.total_capacity AS total_capacity,
               r.flood_control_capacity AS flood_control_capacity,
               r.normal_storage_level AS normal_storage_level,
               r.dead_storage_level AS dead_storage_level
        """
        res_result = self.neo4j.execute_read_single(reservoir_query, {"id": reservoir_id})
        if res_result:
            for key, val in res_result.items():
                if val is not None:
                    params[key] = val

        # 查询2: 从 Constraint 节点补充水位/库容参数
        # 查找约束关系名中含有关键水位关键词的 Constraint
        param_query = """
        MATCH (r:Reservoir {id: $id})-[rel]->(c:Constraint)
        WHERE type(rel) CONTAINS '死水位'
           OR type(rel) CONTAINS '正常蓄水位'
           OR type(rel) CONTAINS '汛限'
           OR type(rel) CONTAINS '防洪限制'
           OR type(rel) CONTAINS '总库容'
           OR type(rel) CONTAINS '防洪库容'
           OR type(rel) CONTAINS '兴利库容'
        RETURN c.name AS name,
               c.value AS value,
               c.unit AS unit,
               type(rel) AS relation
        """
        constraint_params = self.neo4j.execute_read(param_query, {"id": reservoir_id})

        # 将 Constraint 参数映射到标准 key
        KEYWORD_MAP = {
            "死水位": "dead_storage_level",
            "正常蓄水位": "normal_storage_level",
            "汛限": "flood_control_level",
            "防洪限制": "flood_control_level",
            "总库容": "total_capacity",
            "防洪库容": "flood_control_capacity",
            "兴利库容": "active_capacity",
        }

        for cp in constraint_params:
            relation = cp.get("relation", "")
            value = cp.get("value", "")
            unit = cp.get("unit", "")
            for keyword, key in KEYWORD_MAP.items():
                if keyword in relation:
                    params[key] = {"value": value, "unit": unit, "source_relation": relation}
                    break
            else:
                # 兜底: 存到约束列表
                key = f"constraint_{cp.get('name', 'unknown')}"
                params[key] = {"value": value, "unit": unit, "source_relation": relation}

        return params

    def get_reservoir_hydrology_series(
        self,
        reservoir_id: str,
        indicator_keywords: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        查询水库关联水文站的时间序列数据

        Args:
            reservoir_id: 水库ID
            indicator_keywords: 指标关键词列表，默认 ['径流', '流量', '水位', '降水']
        """
        if indicator_keywords is None:
            indicator_keywords = ["径流", "流量", "入流", "水位", "降水", "来水"]

        # 构建多个 CONTAINS 条件
        conditions = " OR ".join(
            [f"type(rel) CONTAINS '{kw}'" for kw in indicator_keywords]
        )

        # 方案1: 直接查水库关联的年度水文数据
        direct_query = f"""
        MATCH (r:Reservoir {{id: $id}})-[rel]->(d:AnnualHydrologyData)
        WHERE {conditions}
        RETURN d, type(rel) AS relation,
               rel.source_doc AS source_doc,
               rel.confidence AS confidence
        ORDER BY d.year
        LIMIT 200
        """
        results = self.neo4j.execute_read(direct_query, {"id": reservoir_id})

        # 方案2: 如果水库没有直接水文数据，通过同河流的水文站获取
        if not results:
            station_query = f"""
            MATCH (res:Reservoir {{id: $id}})-[:LOCATED_ON]->(river:River)
            MATCH (station:HydrologicalStation)-[:LOCATED_ON]->(river)
            MATCH (station)-[rel]->(d:AnnualHydrologyData)
            WHERE {conditions}
            RETURN d, type(rel) AS relation,
                   station.name AS station_name,
                   rel.source_doc AS source_doc,
                   rel.confidence AS confidence
            ORDER BY d.year
            LIMIT 200
            """
            results = self.neo4j.execute_read(station_query, {"id": reservoir_id})

        logger.info(
            f"水文时间序列: reservoir={reservoir_id}, "
            f"keywords={indicator_keywords}, results={len(results)}"
        )
        return results

    def get_optimization_formulation(self, reservoir_id: str) -> Dict:
        """
        汇总查询：返回完整优化问题所需的所有原始数据
        调用者可以再经过 OptimizationFormatter 格式化
        """
        # 获取水库基本信息
        reservoir = self.neo4j.execute_read_single(
            "MATCH (r:Reservoir {id: $id}) RETURN r",
            {"id": reservoir_id},
        )
        if not reservoir:
            return {}

        reservoir_props = reservoir.get("r", reservoir) if isinstance(reservoir, dict) else {}

        return {
            "reservoir": reservoir_props,
            "constraints": self.get_optimization_constraints(reservoir_id),
            "parameters": self.get_reservoir_parameters(reservoir_id),
            "hydrology_series": self.get_reservoir_hydrology_series(reservoir_id),
            "dispatch_rules": self.get_reservoir_rules(reservoir_id),
        }
