"""
Neo4j 图数据库客户端
封装连接管理、节点/关系 CRUD、索引与约束管理
"""
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import Neo4jError

from src.common.logger import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """Neo4j 图数据库客户端"""

    # 初始化 Cypher 脚本（索引与唯一约束）
    SCHEMA_CYPHER = """
    // 唯一约束 (Neo4j 5.x 兼容语法)
    CREATE CONSTRAINT reservoir_id_unique IF NOT EXISTS FOR (n:Reservoir) REQUIRE (n.id) IS UNIQUE;
    CREATE CONSTRAINT station_id_unique IF NOT EXISTS FOR (n:HydrologicalStation) REQUIRE (n.id) IS UNIQUE;
    CREATE CONSTRAINT zone_id_unique IF NOT EXISTS FOR (n:WaterResourceZone) REQUIRE (n.id) IS UNIQUE;
    CREATE CONSTRAINT province_id_unique IF NOT EXISTS FOR (n:Province) REQUIRE (n.id) IS UNIQUE;
    CREATE CONSTRAINT river_id_unique IF NOT EXISTS FOR (n:River) REQUIRE (n.id) IS UNIQUE;
    CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (n:Document) REQUIRE (n.id) IS UNIQUE;
    CREATE CONSTRAINT rule_id_unique IF NOT EXISTS FOR (n:DispatchRule) REQUIRE (n.id) IS UNIQUE;
    CREATE CONSTRAINT data_id_unique IF NOT EXISTS FOR (n:AnnualHydrologyData) REQUIRE (n.id) IS UNIQUE;
    CREATE CONSTRAINT constraint_id_unique IF NOT EXISTS FOR (n:Constraint) REQUIRE (n.id) IS UNIQUE;
    CREATE CONSTRAINT gwoverdraft_id_unique IF NOT EXISTS FOR (n:GroundwaterOverdraftArea) REQUIRE (n.id) IS UNIQUE;
    CREATE CONSTRAINT gwregion_id_unique IF NOT EXISTS FOR (n:GroundwaterRegion) REQUIRE (n.id) IS UNIQUE;
    CREATE CONSTRAINT statagg_id_unique IF NOT EXISTS FOR (n:StatisticAggregate) REQUIRE (n.id) IS UNIQUE;

    // 检索索引
    CREATE INDEX reservoir_name_idx IF NOT EXISTS FOR (n:Reservoir) ON (n.name);
    CREATE INDEX station_name_idx IF NOT EXISTS FOR (n:HydrologicalStation) ON (n.name);
    CREATE INDEX zone_name_idx IF NOT EXISTS FOR (n:WaterResourceZone) ON (n.name);
    CREATE INDEX province_name_idx IF NOT EXISTS FOR (n:Province) ON (n.name);
    CREATE INDEX data_year_idx IF NOT EXISTS FOR (n:AnnualHydrologyData) ON (n.year);
    CREATE INDEX data_indicator_idx IF NOT EXISTS FOR (n:AnnualHydrologyData) ON (n.indicator);
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        """
        初始化 Neo4j 客户端

        Args:
            uri: Neo4j Bolt 连接地址
            user: 用户名
            password: 密码
        """
        self.uri = uri
        self.user = user
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._verify_connectivity()

    def _verify_connectivity(self):
        """验证连接并初始化索引约束"""
        try:
            self.driver.verify_connectivity()
            logger.info(f"Neo4j 连接成功: {self.uri}")
            self._init_schema()
        except Exception as e:
            logger.error(f"Neo4j 连接失败: {e}")
            raise

    def _init_schema(self):
        """初始化图谱 schema（索引和约束）"""
        try:
            for stmt in self.SCHEMA_CYPHER.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    self.execute_write(stmt)
            logger.info("Neo4j Schema 初始化完成")
        except Neo4jError as e:
            if "EquivalentSchemaRuleAlreadyExists" in str(e) or "already exists" in str(e):
                logger.info("Schema 已存在，跳过")
            else:
                logger.warning(f"Schema 初始化警告: {e}")

    @contextmanager
    def _session(self) -> Session:
        """获取 session 上下文管理器"""
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()

    def execute_write(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        执行写操作

        Args:
            query: Cypher 查询语句
            parameters: 查询参数

        Returns:
            结果记录列表
        """
        with self._session() as session:
            result = session.run(query, parameters or {})
            records = [dict(record) for record in result]
            logger.debug(f"写操作完成: {len(records)} 条记录")
            return records

    def execute_read(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        执行读操作

        Args:
            query: Cypher 查询语句
            parameters: 查询参数

        Returns:
            结果记录列表
        """
        with self._session() as session:
            result = session.run(query, parameters or {})
            records = [dict(record) for record in result]
            return records

    def execute_read_single(self, query: str, parameters: Optional[Dict] = None) -> Optional[Dict]:
        """执行读操作，返回单条结果"""
        records = self.execute_read(query, parameters)
        return records[0] if records else None

    # ==================== 节点操作 ====================

    def upsert_node(
        self,
        label: str,
        node_id: str,
        properties: Dict[str, Any],
    ) -> Dict:
        """
        创建或更新节点（MERGE）

        Args:
            label: 节点标签
            node_id: 节点唯一 ID
            properties: 节点属性

        Returns:
            创建/更新后的节点
        """
        props_str = ", ".join(f"n.{k} = ${k}" for k in properties.keys())
        query = f"""
        MERGE (n:{label} {{id: $node_id}})
        SET {props_str}
        RETURN n
        """
        params = {"node_id": node_id, **properties}
        records = self.execute_write(query, params)
        return records[0] if records else {}

    def upsert_nodes_batch(
        self,
        label: str,
        nodes: List[Dict[str, Any]],
    ) -> int:
        """
        批量创建/更新节点

        Args:
            label: 节点标签
            nodes: 节点列表，每个节点需要 "id" 字段

        Returns:
            成功写入的节点数
        """
        if not nodes:
            return 0

        count = 0
        for node in nodes:
            node_id = node.pop("id", node.get("name", f"auto_{count}"))
            try:
                self.upsert_node(label, node_id, node)
                count += 1
            except Exception as e:
                logger.error(f"节点写入失败: {node_id}, {e}")
            finally:
                node["id"] = node_id  # 恢复

        logger.info(f"批量节点写入: {count}/{len(nodes)} ({label})")
        return count

    # ==================== 关系操作 ====================

    def create_relationship(
        self,
        subject_label: str,
        subject_id: str,
        relation_type: str,
        object_label: str,
        object_id: str,
        properties: Optional[Dict] = None,
    ):
        """
        创建节点间关系

        Args:
            subject_label: 主体节点标签
            subject_id: 主体节点 ID
            relation_type: 关系类型（直接用原名，反引号包裹支持中文/数字开头）
            object_label: 客体节点标签
            object_id: 客体节点 ID
            properties: 关系属性
        """
        # 用反引号包裹关系类型，支持中文和数字开头，无需 R_ 前缀
        safe_rel = f"`{relation_type}`"

        props_str = ""
        if properties:
            props_parts = [f"{k}: ${k}" for k in properties.keys()]
            props_str = "{" + ", ".join(props_parts) + "}"

        query = f"""
        MATCH (a:{subject_label} {{id: $subject_id}})
        MATCH (b:{object_label} {{id: $object_id}})
        MERGE (a)-[r:{safe_rel}]->(b)
        SET r += {props_str or '{}'}
        RETURN r
        """
        params = {"subject_id": subject_id, "object_id": object_id}
        if properties:
            params.update(properties)

        try:
            self.execute_write(query, params)
        except Neo4jError as e:
            logger.error(f"关系创建失败: ({subject_id})-[{rel_type}]->({object_id}): {e}")

    def create_relationships_batch(
        self,
        relationships: List[Dict],
    ) -> int:
        """
        批量创建关系

        Args:
            relationships: 关系列表
                [{"subject_label": ..., "subject_id": ..., "relation": ...,
                  "object_label": ..., "object_id": ..., "properties": {...}}, ...]

        Returns:
            成功创建的关系数
        """
        if not relationships:
            return 0

        count = 0
        for rel in relationships:
            try:
                self.create_relationship(
                    subject_label=rel["subject_label"],
                    subject_id=rel["subject_id"],
                    relation_type=rel["relation"],
                    object_label=rel["object_label"],
                    object_id=rel["object_id"],
                    properties=rel.get("properties"),
                )
                count += 1
            except Exception as e:
                logger.error(f"关系创建失败: {rel.get('subject_id')} - {rel.get('relation')}: {e}")

        logger.info(f"批量关系创建: {count}/{len(relationships)}")
        return count

    # ==================== 工具方法 ====================

    def _sanitize_rel_type(self, name: str) -> str:
        """已废弃：现在用反引号包裹关系类型 (e.g. `2024年实测径流量为`)，无需清洗"""
        return name

    def delete_all(self, confirm: bool = False):
        """清空数据库（危险操作）"""
        if not confirm:
            logger.warning("delete_all 需要 confirm=True")
            return
        self.execute_write("MATCH (n) DETACH DELETE n")
        logger.warning("数据库已清空")

    def get_stats(self) -> Dict[str, int]:
        """获取图谱统计信息"""
        node_counts = {}
        labels = [
            "Reservoir", "HydrologicalStation", "WaterResourceZone",
            "Province", "River", "DispatchRule", "Constraint",
            "AnnualHydrologyData", "Document",
        ]
        for label in labels:
            try:
                result = self.execute_read_single(
                    f"MATCH (n:{label}) RETURN count(n) AS count"
                )
                node_counts[label] = result["count"] if result else 0
            except Exception:
                node_counts[label] = 0

        rel_result = self.execute_read_single(
            "MATCH ()-[r]->() RETURN count(r) AS count"
        )
        rel_count = rel_result["count"] if rel_result else 0

        return {
            "nodes": node_counts,
            "total_nodes": sum(node_counts.values()),
            "relationships": rel_count,
        }

    def close(self):
        """关闭连接"""
        self.driver.close()
        logger.info("Neo4j 连接已关闭")
