"""
知识查询接口
三元组检索、文档溯源、实体间路径
"""
from fastapi import APIRouter, HTTPException, Query

from src.api.main import get_graph_query, success_response

router = APIRouter()


@router.get("/search")
def search_triplets(
    q: str = Query(..., min_length=1, description="搜索关键词"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    关键词搜索三元组
    在主体、关系、客体中搜索匹配内容
    """
    gq = get_graph_query()
    data = gq.search_triplets(q, limit=limit)
    return success_response(data, f"找到 {len(data)} 条知识")


@router.get("/trace")
def trace_by_document(
    doc: str = Query(..., min_length=1, description="来源文档名称"),
    limit: int = Query(100, ge=1, le=500),
):
    """
    按来源文档溯源查询
    获取该文档中的所有知识三元组
    """
    gq = get_graph_query()
    data = gq.trace_by_document(doc, limit=limit)
    return success_response(data, f"文档 '{doc}' 共包含 {len(data)} 条知识")


@router.get("/path")
def find_path(
    from_entity: str = Query(..., alias="from", description="起始实体 ID"),
    to_entity: str = Query(..., alias="to", description="目标实体 ID"),
    max_depth: int = Query(4, ge=1, le=6, description="最大路径深度"),
):
    """
    查询两个实体之间的最短路径
    """
    gq = get_graph_query()
    data = gq.find_path(from_entity, to_entity, max_depth=max_depth)
    if not data:
        return success_response([], f"未找到从 '{from_entity}' 到 '{to_entity}' 的路径")
    return success_response(data, f"最短路径: {data[0].get('depth', 'N/A')} 跳")


@router.get("/stats")
def get_stats():
    """获取知识图谱统计信息"""
    gq = get_graph_query()
    data = gq.get_kg_stats()
    return success_response(data)
