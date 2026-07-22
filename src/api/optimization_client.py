"""
HydroBrain 优化调度 Python SDK
供优化算法直接 import 调用，支持单库/多库/求解器格式
"""
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════
# Pydantic 数据模型
# ═══════════════════════════════════════════════════════════

class Bounds(BaseModel):
    lower: Optional[float] = None
    upper: Optional[float] = None


class DecisionVariable(BaseModel):
    symbol: str
    name: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    unit: str = ""


class Constraint(BaseModel):
    expression: str = ""
    type: str = "unknown"
    description: str = ""


class ParameterValue(BaseModel):
    value: Optional[float] = None
    unit: str = ""


class Parameters(BaseModel):
    dead_storage_level: Optional[ParameterValue] = None
    normal_storage_level: Optional[ParameterValue] = None
    flood_control_level: Optional[ParameterValue] = None
    total_capacity: Optional[ParameterValue] = None
    flood_control_capacity: Optional[ParameterValue] = None


class TimeSeriesEntry(BaseModel):
    year: Optional[int] = None
    value: Optional[float] = None
    unit: str = ""
    indicator: str = ""


class TimeSeries(BaseModel):
    entries: List[TimeSeriesEntry] = []
    count: int = 0


class ProblemMeta(BaseModel):
    reservoir_name: str = ""
    reservoir_id: str = ""
    description: str = ""


class FormulationResult(BaseModel):
    """单库优化问题结构"""
    problem_meta: ProblemMeta = ProblemMeta()
    decision_variables: List[DecisionVariable] = []
    objective_candidates: List[Dict] = []
    constraints: List[Dict] = []
    parameters: Dict[str, Any] = {}
    time_series: Dict[str, Any] = {}
    dispatch_rules: List[Dict] = []

    def to_dict(self) -> dict:
        return self.model_dump()


class BatchFormulationResult(BaseModel):
    """多库优化问题结构"""
    reservoirs: Dict[str, FormulationResult] = {}
    relations: List[Dict] = []

    def to_dict(self) -> dict:
        return self.model_dump()


class SolverReadyResult(BaseModel):
    """求解器就绪格式"""
    variables: List[DecisionVariable] = []
    constraints: List[Constraint] = []
    time_series: Dict[str, Any] = {}
    objective_hints: List[str] = []
    meta: Dict[str, Any] = {}

    def to_dict(self) -> dict:
        return self.model_dump()


class CompareResult(BaseModel):
    """多库对比结果"""
    parameters: Dict[str, Dict[str, Optional[Dict]]] = {}

    def to_dict(self) -> dict:
        return self.model_dump()


# ═══════════════════════════════════════════════════════════
# Client
# ═══════════════════════════════════════════════════════════

class OptimizationClient:
    """优化调度数据客户端"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        """发送 GET 请求并返回 data 字段"""
        url = f"{self.base_url}{path}"
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != 200:
            raise RuntimeError(f"API error: {body.get('message', 'unknown')}")
        return body.get("data", {})

    # ── 单库查询 ──

    def get_formulation(
        self,
        reservoir_id: str,
        include_hydrology: bool = True,
        include_rules: bool = True,
        year: Optional[int] = None,
    ) -> FormulationResult:
        """查询单个水库的完整优化问题结构"""
        params = {
            "reservoir_id": reservoir_id,
            "include_hydrology": include_hydrology,
            "include_rules": include_rules,
        }
        if year is not None:
            params["year"] = year
        data = self._get("/api/optimization/formulate", params)
        return FormulationResult(**data)

    # ── 多库查询 ──

    def get_formulation_batch(
        self,
        reservoir_ids: List[str],
        include_hydrology: bool = True,
        include_rules: bool = True,
    ) -> BatchFormulationResult:
        """查询多个水库的优化问题结构（含梯级关系）"""
        params = {
            "reservoir_ids": ",".join(reservoir_ids),
            "include_hydrology": include_hydrology,
            "include_rules": include_rules,
        }
        data = self._get("/api/optimization/formulate/batch", params)
        # 将嵌套 dict 转换为 FormulationResult
        reservoirs = {}
        for rid, raw in data.get("reservoirs", {}).items():
            reservoirs[rid] = FormulationResult(**raw)
        return BatchFormulationResult(
            reservoirs=reservoirs,
            relations=data.get("relations", []),
        )

    # ── 求解器格式 ──

    def get_solver_ready(
        self,
        reservoir_ids: List[str],
        year: Optional[int] = None,
    ) -> SolverReadyResult:
        """获取求解器就绪的紧凑格式"""
        params = {"reservoir_ids": ",".join(reservoir_ids)}
        if year is not None:
            params["year"] = year
        data = self._get("/api/optimization/solver-ready", params)
        return SolverReadyResult(**data)

    # ── 约束查询 ──

    def get_constraints(
        self,
        reservoir_id: str,
        category: Optional[str] = None,
        year: Optional[int] = None,
    ) -> List[Dict]:
        """查询水库约束条件"""
        params = {"reservoir_id": reservoir_id}
        if category:
            params["category"] = category
        if year is not None:
            params["year"] = year
        return self._get("/api/optimization/constraints", params)

    # ── 参数查询 ──

    def get_parameters(self, reservoir_id: str) -> Dict:
        """查询水库物理参数"""
        return self._get("/api/optimization/parameters", {"reservoir_id": reservoir_id})

    # ── 水文查询 ──

    def get_hydrology(
        self,
        reservoir_id: str,
        indicator: Optional[str] = None,
        year: Optional[int] = None,
    ) -> Dict:
        """查询水库水文时间序列"""
        params = {"reservoir_id": reservoir_id}
        if indicator:
            params["indicator"] = indicator
        if year is not None:
            params["year"] = year
        return self._get("/api/optimization/hydrology", params)

    # ── 对比查询 ──

    def compare(self, reservoir_ids: List[str]) -> CompareResult:
        """多库同指标横向对比"""
        params = {"reservoir_ids": ",".join(reservoir_ids)}
        data = self._get("/api/optimization/compare", params)
        return CompareResult(**data)

    # ── 水库列表 ──

    def list_reservoirs(self) -> List[Dict]:
        """列出可用于优化调度的水库"""
        return self._get("/api/optimization/reservoirs")

    def close(self):
        """关闭 HTTP 会话"""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
