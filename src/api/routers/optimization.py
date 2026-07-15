"""
优化调度数据接口
从知识图谱中提取水库调度优化所需的结构化数据:
- 决策变量 (推断)
- 目标函数候选 (推断)
- 约束条件 (结构化: operator + value + unit + category)
- 水库物理参数
- 水文时间序列
- 调度规则原文
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.dependencies import get_graph_query, success_response
from src.knowledge_graph.optimization_formatter import OptimizationFormatter

router = APIRouter()

# 单例
_formatter: Optional[OptimizationFormatter] = None


def get_formatter() -> OptimizationFormatter:
    global _formatter
    if _formatter is None:
        _formatter = OptimizationFormatter()
    return _formatter


@router.get("/reservoirs")
def list_reservoirs_for_optimization():
    """列出可用于优化调度的水库"""
    gq = get_graph_query()
    reservoirs = gq.get_reservoir_list(limit=100)
    # 补充: 返回每个水库的约束数量
    return success_response(reservoirs, f"共 {len(reservoirs)} 座水库可用于优化调度")


@router.get("/formulate")
def formulate_optimization_problem(
    reservoir_id: str = Query(..., description="水库ID"),
    include_hydrology: bool = Query(True, description="是否包含水文时间序列"),
    include_rules: bool = Query(True, description="是否包含调度规则原文"),
):
    """
    生成水库优化调度问题的结构化数据

    返回决策变量、目标函数候选、约束条件、物理参数、时间序列，
    可直接用于 PuLP / OR-Tools / scipy.optimize / 遗传算法等优化框架。
    """
    gq = get_graph_query()
    fmt = get_formatter()

    # 1. 获取原始数据
    raw = gq.get_optimization_formulation(reservoir_id)
    if not raw:
        raise HTTPException(status_code=404, detail=f"水库不存在: {reservoir_id}")

    reservoir = raw.get("reservoir", {})
    if not reservoir:
        reservoir = {"id": reservoir_id, "name": reservoir_id}

    # 2. 格式化
    result = fmt.build_formulation(
        reservoir=reservoir,
        constraints_raw=raw.get("constraints", []),
        parameters=raw.get("parameters", {}),
        hydrology_series=raw.get("hydrology_series", []) if include_hydrology else None,
        dispatch_rules=raw.get("dispatch_rules", []) if include_rules else None,
    )

    # 统计信息
    n_constraints = len(result.get("constraints", []))
    n_vars = len(result.get("decision_variables", []))
    n_objectives = len(result.get("objective_candidates", []))
    n_series = result.get("time_series", {}).get("count", 0)

    return success_response(
        result,
        f"{reservoir.get('name', reservoir_id)}: "
        f"{n_vars}变量, {n_constraints}约束, {n_objectives}目标候选, {n_series}时序数据",
    )


@router.get("/constraints")
def get_constraints(
    reservoir_id: str = Query(..., description="水库ID"),
    category: Optional[str] = Query(
        None,
        description="按类别筛选: water_level / storage / discharge / power_output / "
                    "power_generation / water_supply / water_use / ecological_flow",
    ),
):
    """
    查询水库的结构化约束条件

    返回已分类的约束，包含 operator + value + unit + category
    """
    gq = get_graph_query()
    fmt = get_formatter()

    raw_constraints = gq.get_optimization_constraints(reservoir_id)
    reservoir = {"id": reservoir_id, "name": reservoir_id}

    # 尝试获取水库名称
    res_info = gq.get_reservoir(reservoir_id)
    if res_info:
        r = res_info.get("r", res_info)
        if isinstance(r, dict):
            reservoir = r

    formatted = fmt._format_constraints(raw_constraints, reservoir)

    # 按类别筛选
    if category:
        formatted = [c for c in formatted if c.get("category") == category]

    return success_response(
        formatted,
        f"共 {len(formatted)} 条约束"
        + (f" (category={category})" if category else ""),
    )


@router.get("/parameters")
def get_parameters(
    reservoir_id: str = Query(..., description="水库ID"),
):
    """查询水库物理参数（死水位/正常蓄水位/汛限水位/总库容/防洪库容）"""
    gq = get_graph_query()
    fmt = get_formatter()

    raw_params = gq.get_reservoir_parameters(reservoir_id)

    # 获取水库名称
    res_info = gq.get_reservoir(reservoir_id)
    reservoir = {"id": reservoir_id, "name": reservoir_id}
    if res_info:
        r = res_info.get("r", res_info)
        if isinstance(r, dict):
            reservoir = r

    formatted = fmt._format_parameters(raw_params, reservoir)
    return success_response(formatted, f"{reservoir.get('name', reservoir_id)} 物理参数")


@router.get("/hydrology")
def get_hydrology(
    reservoir_id: str = Query(..., description="水库ID"),
    indicator: Optional[str] = Query(None, description="指标关键词，如 '径流' '水位' '降水'"),
):
    """查询水库关联的水文时间序列数据"""
    gq = get_graph_query()
    fmt = get_formatter()

    keywords = [indicator] if indicator else None
    raw_series = gq.get_reservoir_hydrology_series(reservoir_id, keywords)
    time_series = fmt._format_time_series(raw_series)

    return success_response(time_series, f"共 {time_series.get('count', 0)} 条时序数据")
