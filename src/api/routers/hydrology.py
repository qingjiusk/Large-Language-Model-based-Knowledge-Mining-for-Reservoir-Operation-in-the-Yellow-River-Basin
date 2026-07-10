"""
水文数据查询接口
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.main import get_graph_query, success_response

router = APIRouter()


@router.get("/stations")
def list_stations(limit: int = Query(50, ge=1, le=200)):
    """获取水文站列表"""
    gq = get_graph_query()
    data = gq.get_stations(limit=limit)
    return success_response(data, f"共 {len(data)} 个水文站")


@router.get("/station/{station_id}/data")
def get_station_data(
    station_id: str,
    year: Optional[int] = Query(None, description="年份筛选"),
    indicator: Optional[str] = Query(None, description="指标类型：径流量/输沙量等"),
):
    """查询水文站年度数据"""
    gq = get_graph_query()
    data = gq.get_station_data(station_id, year=year, indicator=indicator)
    return success_response(data, f"共 {len(data)} 条数据")


@router.get("/zones")
def list_zones():
    """获取水资源二级区列表"""
    gq = get_graph_query()
    data = gq.get_water_resource_zones()
    return success_response(data, f"共 {len(data)} 个分区")


@router.get("/zone/data")
def get_zone_data(
    zone_id: Optional[str] = Query(None, description="分区 ID 筛选"),
    year: Optional[int] = Query(None, description="年份筛选"),
):
    """查询水资源分区年度数据"""
    gq = get_graph_query()
    data = gq.get_zone_data(zone_id=zone_id, year=year)
    return success_response(data, f"共 {len(data)} 条数据")
