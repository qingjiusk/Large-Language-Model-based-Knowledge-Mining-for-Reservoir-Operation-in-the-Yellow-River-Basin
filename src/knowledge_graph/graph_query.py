"""
知识图谱查询封装
常用业务查询的 Cypher 模板
"""
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.knowledge_graph.neo4j_client import Neo4jClient

logger = get_logger(__name__)


class GraphQuery:
    """图谱查询封装，提供常用业务查询接口"""

    def __init__(self, neo4j_client: Neo4jClient):
        """
        初始化查询封装

        Args:
            neo4j_client: Neo4jClient 实例
        """
        self.neo4j = neo4j_client

    # ==================== 水库查询 ====================

    def get_reservoir(self, reservoir_id: str) -> Optional[Dict]:
        """查询单个水库的完整信息"""
        query = """
        MATCH (r:Reservoir {id: $id})
        OPTIONAL MATCH (r)-[rel]->(n)
        RETURN r, collect({type: type(rel), target: n, props: properties(rel)}) AS relations
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
        """查询水库的调度规则"""
        query = """
        MATCH (r:Reservoir {id: $id})-[rel:HAS_DISPATCH_RULE]->(rule:DispatchRule)
        RETURN rule, rel.confidence AS confidence, rel.source_doc AS source
        """
        return self.neo4j.execute_read(query, {"id": reservoir_id})

    def get_reservoir_constraints(self, reservoir_id: str) -> List[Dict]:
        """查询水库的约束条件"""
        query = """
        MATCH (r:Reservoir {id: $id})-[rel:HAS_CONSTRAINT]->(c:Constraint)
        RETURN c, rel.confidence AS confidence, rel.source_doc AS source
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
        """查询水文站的年度数据"""
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
        MATCH (s:HydrologicalStation {{id: $station_id}})-[:HAS_ANNUAL_DATA]->(d:AnnualHydrologyData)
        WHERE {where_clause}
        RETURN d, s.name AS station_name
        ORDER BY d.year DESC
        LIMIT 100
        """
        return self.neo4j.execute_read(query, params)

    def get_zone_data(
        self,
        zone_id: Optional[str] = None,
        year: Optional[int] = None,
    ) -> List[Dict]:
        """查询水资源分区的年度数据"""
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
        MATCH (z:WaterResourceZone)-[:HAS_ANNUAL_DATA]->(d:AnnualHydrologyData)
        {zone_filter}
        {year_filter}
        RETURN z.name AS zone_name, d
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
        """
        查询两个实体之间的最短路径

        Args:
            from_id: 起始实体 ID
            to_id: 目标实体 ID
            max_depth: 最大跳数

        Returns:
            路径列表
        """
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
        """
        按来源文档查询所有相关知识

        Args:
            doc_name: 文档名称
            limit: 最大返回数

        Returns:
            三元组列表
        """
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
        """
        关键词搜索三元组（在 subject/relation/object 中搜索）

        Args:
            keyword: 搜索关键词
            limit: 最大返回数

        Returns:
            匹配的三元组列表
        """
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.name CONTAINS $keyword
           OR b.name CONTAINS $keyword
           OR type(r) CONTAINS $keyword
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
        RETURN r.name AS name, r.id AS id, r.river AS river
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
