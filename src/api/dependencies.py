"""
API 共享依赖 — 避免循环导入
"""
from typing import Any, Dict, Optional

from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.graph_query import GraphQuery

# 全局实例（由 main.py 初始化）
neo4j_client: Optional[Neo4jClient] = None
graph_query: Optional[GraphQuery] = None


def get_graph_query() -> GraphQuery:
    """获取 GraphQuery 实例，未就绪时抛异常"""
    if graph_query is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Neo4j 服务不可用")
    return graph_query


def success_response(data: Any = None, message: str = "ok") -> Dict:
    """统一成功响应"""
    return {"code": 200, "message": message, "data": data}


def error_response(code: int, message: str) -> Dict:
    """统一错误响应"""
    return {"code": code, "message": message, "data": None}
