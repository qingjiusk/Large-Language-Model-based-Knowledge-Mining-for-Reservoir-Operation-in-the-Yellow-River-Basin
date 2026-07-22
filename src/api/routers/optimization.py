"""
优化调度数据接口
从知识图谱中提取水库调度优化所需的结构化数据
"""
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from src.api.dependencies import get_graph_query, success_response
from src.knowledge_graph.optimization_formatter import OptimizationFormatter

router = APIRouter()

_formatter: Optional[OptimizationFormatter] = None


def get_formatter() -> OptimizationFormatter:
    global _formatter
    if _formatter is None:
        _formatter = OptimizationFormatter()
    return _formatter


# ── 缓存头 helper ──
def cached_response(data, message: str = "ok"):
    """返回带 Cache-Control 头的成功响应"""
    content = success_response(data, message)
    return JSONResponse(
        content=content,
        headers={"Cache-Control": "max-age=3600"},
    )


@router.get("/reservoirs")
def list_reservoirs_for_optimization():
    """列出可用于优化调度的水库"""
    gq = get_graph_query()
    reservoirs = gq.get_reservoir_list(limit=100)
    return cached_response(reservoirs, f"共 {len(reservoirs)} 座水库可用于优化调度")


@router.get("/formulate")
def formulate_optimization_problem(
    reservoir_id: str = Query(..., description="水库ID"),
    include_hydrology: bool = Query(True, description="是否包含水文时间序列"),
    include_rules: bool = Query(True, description="是否包含调度规则原文"),
    year: Optional[int] = Query(None, description="按年份过滤约束和水文数据"),
):
    """生成水库优化调度问题的结构化数据"""
    gq = get_graph_query()
    fmt = get_formatter()

    raw = gq.get_optimization_formulation(reservoir_id)
    if not raw.get("reservoir"):
        raise HTTPException(status_code=404, detail=f"水库不存在: {reservoir_id}")

    reservoir = raw.get("reservoir", {})
    if not reservoir:
        reservoir = {"id": reservoir_id, "name": reservoir_id}

    hydrology = raw.get("hydrology_series", []) if include_hydrology else None
    if hydrology and year is not None:
        hydrology = [h for h in hydrology
                     if str(h.get("d", {}).get("year", "")) == str(year)
                     or str(h.get("year", "")) == str(year)]

    result = fmt.build_formulation(
        reservoir=reservoir,
        constraints_raw=raw.get("constraints", []),
        parameters=raw.get("parameters", {}),
        hydrology_series=hydrology,
        dispatch_rules=raw.get("dispatch_rules", []) if include_rules else None,
    )

    n_constraints = len(result.get("constraints", []))
    n_vars = len(result.get("decision_variables", []))
    n_objectives = len(result.get("objective_candidates", []))
    n_series = result.get("time_series", {}).get("count", 0)
    year_suffix = f", year={year}" if year else ""

    return cached_response(
        result,
        f"{reservoir.get('name', reservoir_id)}: "
        f"{n_vars}变量, {n_constraints}约束, {n_objectives}目标候选, {n_series}时序{year_suffix}",
    )


@router.get("/formulate/batch")
def formulate_batch(
    reservoir_ids: str = Query(
        ..., description="水库ID列表，逗号分隔，最多10个",
        pattern=r"^[\w一-鿿-]+(,[\w一-鿿-]+){0,9}$",
    ),
    include_hydrology: bool = Query(True),
    include_rules: bool = Query(True),
):
    """多库联合优化问题（梯级调度用）"""
    gq = get_graph_query()
    fmt = get_formatter()

    ids = [s.strip() for s in reservoir_ids.split(",") if s.strip()]
    if len(ids) > 10:
        raise HTTPException(status_code=400, detail="最多支持 10 个水库")

    try:
        batch_data = gq.get_formulation_batch(ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量查询失败: {e}")

    result = fmt.build_formulation_batch(batch_data, include_hydrology, include_rules)

    return cached_response(
        result,
        f"共 {len(ids)} 座水库" + (
            f", {len(result.get('relations', []))} 条梯级关系"
            if result.get("relations") else ""
        ),
    )


@router.get("/solver-ready")
def solver_ready(
    reservoir_ids: str = Query(
        ..., description="水库ID列表，逗号分隔，1-10个",
        pattern=r"^[\w一-鿿-]+(,[\w一-鿿-]+){0,9}$",
    ),
    year: Optional[int] = Query(None, description="按年份过滤时间序列"),
):
    """求解器就绪格式 — 紧凑，可直接喂给 cvxpy/PuLP/scipy"""
    gq = get_graph_query()
    fmt = get_formatter()

    ids = [s.strip() for s in reservoir_ids.split(",") if s.strip()]
    if len(ids) > 10:
        raise HTTPException(status_code=400, detail="最多支持 10 个水库")

    try:
        batch_data = gq.get_formulation_batch(ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {e}")

    result = fmt.build_solver_ready(batch_data, year)

    return cached_response(
        result,
        f"{len(ids)} 水库求解器格式: "
        f"{len(result.get('variables', []))}变量, "
        f"{len(result.get('constraints', []))}约束, "
        f"{len(result.get('objective_hints', []))}目标候选",
    )


@router.get("/compare")
def compare_reservoirs(
    reservoir_ids: str = Query(
        ..., description="水库ID列表，逗号分隔",
        pattern=r"^[\w一-鿿-]+(,[\w一-鿿-]+){0,9}$",
    ),
):
    """多库同指标横向对比"""
    gq = get_graph_query()

    ids = [s.strip() for s in reservoir_ids.split(",") if s.strip()]
    if len(ids) > 10:
        raise HTTPException(status_code=400, detail="最多支持 10 个水库")

    # 收集每个水库的参数
    PARAM_KEYS = [
        "dead_storage_level", "normal_storage_level", "flood_control_level",
        "total_capacity", "flood_control_capacity",
    ]
    PARAM_LABELS = {
        "dead_storage_level": "死水位",
        "normal_storage_level": "正常蓄水位",
        "flood_control_level": "防洪限制水位",
        "total_capacity": "总库容",
        "flood_control_capacity": "防洪库容",
    }

    comparison: Dict[str, Dict[str, Optional[Dict]]] = {
        label: {} for label in PARAM_LABELS.values()
    }

    for rid in ids:
        params = gq.get_reservoir_parameters(rid)
        res_name = rid
        try:
            name_result = gq.neo4j.execute_read_single(
                "MATCH (r:Reservoir {id: $id}) RETURN r.name AS name", {"id": rid}
            )
            if name_result:
                res_name = name_result.get("name", rid)
        except Exception:
            pass

        for key, label in PARAM_LABELS.items():
            raw = params.get(key, {})
            if isinstance(raw, dict):
                comparison[label][res_name] = {
                    "value": raw.get("value"),
                    "unit": raw.get("unit", ""),
                }
            elif raw is not None:
                comparison[label][res_name] = {"value": raw, "unit": ""}
            else:
                comparison[label][res_name] = None

    return cached_response(
        {"parameters": comparison},
        f"共 {len(ids)} 座水库, {len(PARAM_LABELS)} 项指标对比",
    )


@router.get("/constraints")
def get_constraints(
    reservoir_id: str = Query(..., description="水库ID"),
    category: Optional[str] = Query(
        None,
        description="按类别筛选: water_level / storage / discharge / power_output / "
                    "power_generation / water_supply / water_use / ecological_flow",
    ),
    year: Optional[int] = Query(None, description="按年份过滤"),
):
    """查询水库的结构化约束条件"""
    gq = get_graph_query()
    fmt = get_formatter()

    raw_constraints = gq.get_optimization_constraints(reservoir_id)
    reservoir = {"id": reservoir_id, "name": reservoir_id}

    res_info = gq.get_reservoir(reservoir_id)
    if res_info:
        r = res_info.get("r", res_info)
        if isinstance(r, dict):
            reservoir = r

    if year is not None:
        raw_constraints = [
            c for c in raw_constraints
            if str(c.get("d", {}).get("year", "")) == str(year)
            or str(c.get("year", "")) == str(year)
        ]

    formatted = fmt._format_constraints(raw_constraints, reservoir)

    if category:
        formatted = [c for c in formatted if c.get("category") == category]

    return cached_response(
        formatted,
        f"共 {len(formatted)} 条约束"
        + (f" (category={category})" if category else "")
        + (f", year={year}" if year else ""),
    )


@router.get("/parameters")
def get_parameters(
    reservoir_id: str = Query(..., description="水库ID"),
):
    """查询水库物理参数（死水位/正常蓄水位/汛限水位/总库容/防洪库容）"""
    gq = get_graph_query()
    fmt = get_formatter()

    raw_params = gq.get_reservoir_parameters(reservoir_id)

    res_info = gq.get_reservoir(reservoir_id)
    reservoir = {"id": reservoir_id, "name": reservoir_id}
    if res_info:
        r = res_info.get("r", res_info)
        if isinstance(r, dict):
            reservoir = r

    formatted = fmt._format_parameters(raw_params, reservoir)
    return cached_response(formatted, f"{reservoir.get('name', reservoir_id)} 物理参数")


@router.get("/hydrology")
def get_hydrology(
    reservoir_id: str = Query(..., description="水库ID"),
    indicator: Optional[str] = Query(None, description="指标关键词，如 '径流' '水位' '降水'"),
    year: Optional[int] = Query(None, description="按年份过滤"),
):
    """查询水库关联的水文时间序列数据"""
    gq = get_graph_query()
    fmt = get_formatter()

    keywords = [indicator] if indicator else None
    raw_series = gq.get_reservoir_hydrology_series(reservoir_id, keywords, year=year)
    time_series = fmt._format_time_series(raw_series)

    year_suffix = f", year={year}" if year else ""
    return cached_response(time_series, f"共 {time_series.get('count', 0)} 条时序数据{year_suffix}")
