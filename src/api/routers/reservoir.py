"""
水库信息查询接口
"""
from fastapi import APIRouter, HTTPException, Query

from src.api.main import get_graph_query, success_response

router = APIRouter()


@router.get("/list")
def list_reservoirs(limit: int = Query(50, ge=1, le=200)):
    """获取水库列表"""
    gq = get_graph_query()
    data = gq.get_reservoir_list(limit=limit)
    return success_response(data, f"共 {len(data)} 座水库")


@router.get("/{reservoir_id}")
def get_reservoir(reservoir_id: str):
    """获取水库详细信息"""
    gq = get_graph_query()
    data = gq.get_reservoir(reservoir_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"水库不存在: {reservoir_id}")
    return success_response(data)


@router.get("/search")
def search_reservoirs(
    name: str = Query(..., min_length=1, description="水库名称关键词"),
    limit: int = Query(20, ge=1, le=100),
):
    """按名称搜索水库"""
    gq = get_graph_query()
    data = gq.search_reservoirs(name, limit=limit)
    return success_response(data, f"找到 {len(data)} 条匹配结果")


@router.get("/{reservoir_id}/rules")
def get_rules(reservoir_id: str):
    """查询水库调度规则"""
    gq = get_graph_query()
    data = gq.get_reservoir_rules(reservoir_id)
    return success_response(data, f"共 {len(data)} 条规则")


@router.get("/{reservoir_id}/constraints")
def get_constraints(reservoir_id: str):
    """查询水库约束条件"""
    gq = get_graph_query()
    data = gq.get_reservoir_constraints(reservoir_id)
    return success_response(data, f"共 {len(data)} 条约束")
