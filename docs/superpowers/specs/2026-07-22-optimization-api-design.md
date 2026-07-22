# 优化调度 API 改进

**Date**: 2026-07-22
**Status**: 设计完成

---

## 1. 背景与动机

HydroBrain 项目的主要下游消费者是**水库多目标综合调度优化算法**。这些算法需要从知识图谱中获取限制水位、历史流量、约束条件等结构化数据来构建和求解优化模型。

当前 `/api/optimization/` 路由已有完整框架（6 个端点 + `OptimizationFormatter`），但存在三个核心问题：

1. **查询脆弱**：`graph_query.py` 中 `get_optimization_constraints()` / `get_reservoir_parameters()` / `get_reservoir_hydrology_series()` 全部使用 `type(rel) CONTAINS '中文关键词'` 做 Cypher 匹配，关键词有 17 个之多。本项目刚修复了 `TripletNormalizer` 使 `relation_id` 可精确映射（如 "死水位为" → `DEAD_STORAGE_LEVEL`），应利用这一改进。

2. **无批量查询**：梯级调度（如龙羊峡→李家峡→刘家峡）需多次串行调用 `/formulate`，无 `/formulate/batch` 端点。

3. **无求解器就绪格式**：`OptimizationFormatter` 输出偏"展示"（expression 为人类可读中文），算法侧需要 `Z[t] >= 2530` 这种可直接用于 cvxpy/PuLP/scipy 的紧凑格式。

本设计在不破坏现有 API 契约的前提下，重构查询层、新增批量端点、增加求解器格式和 Python SDK。

---

## 2. 改动范围

### 2.1 修改的文件

| 文件 | 改动 |
|------|------|
| `src/knowledge_graph/graph_query.py` | 优化相关方法的 Cypher 查询从 `CONTAINS` 中文关键词 → `rel.relation_id IN [...]` 精确匹配；新增 `get_reservoir_upstream_relations()` 用于梯级关系推断 |
| `src/knowledge_graph/optimization_formatter.py` | 新增 `build_formulation_batch()`、`build_solver_ready()`；约束分类改为 `relation_id` 优先 |
| `src/api/routers/optimization.py` | 新增 3 个端点；现有端点增加 `year` 参数 |

### 2.2 新增的文件

| 文件 | 用途 |
|------|------|
| `src/api/optimization_client.py` | Python SDK：pydantic 模型 + `OptimizationClient` 类 |

### 2.3 不改动的文件

- `src/api/routers/reservoir.py` / `hydrology.py` / `knowledge.py`
- `src/knowledge_graph/neo4j_client.py`
- `scripts/query_server.py`
- `src/api/main.py`（仅需确认 router 正常挂载，无需结构改动）

---

## 3. 查询层重构

### 3.1 核心原则

Neo4j 中每个关系存储 `relation_id` 属性（由 `rebuild_graph()` 写入，来自 `TripletNormalizer` 输出）。查询从：

```cypher
WHERE type(rel) CONTAINS '死水位'
   OR type(rel) CONTAINS '汛限'
   ...
```

改为：

```cypher
WHERE rel.relation_id IN ['DEAD_STORAGE_LEVEL', 'FLOOD_CONTROL_LEVEL', ...]
```

`relation_id` 在 Neo4j 中已有索引候选，性能优于全表扫描的 `CONTAINS`。

### 3.2 需修改的方法

**`get_optimization_constraints(reservoir_id)`**

将 17 个 `CONTAINS '关键词'` 替换为 `rel.relation_id IN`。CONSTRAINT_RELATION_IDS 分类表：

```python
# 约束类 relation_id → category
CONSTRAINT_RELATION_IDS = {
    "water_level": [
        "DEAD_STORAGE_LEVEL", "FLOOD_CONTROL_LEVEL", "NORMAL_STORAGE_LEVEL",
        "GROUNDWATER_DEPTH", "GROUNDWATER_DEPTH_START", "GROUNDWATER_DEPTH_END",
        "GROUNDWATER_DEPTH_CHANGE",
    ],
    "storage": [
        "TOTAL_CAPACITY", "FLOOD_CONTROL_CAPACITY",
        "RESERVOIR_STORAGE", "RESERVOIR_STORAGE_START",
        "RESERVOIR_STORAGE_END", "RESERVOIR_STORAGE_CHANGE",
    ],
    "discharge": [
        "ANNUAL_RUNOFF", "ANNUAL_RUNOFF_FLOOD_SEASON",
        "ANNUAL_RUNOFF_TO_SEA", "ECOLOGICAL_FLOW",
    ],
    "power": ["POWER_GENERATION"],
    "water_supply": ["WATER_SUPPLY"],
    "water_use": ["WATER_USE"],
    "water_consumption": ["WATER_CONSUMPTION"],
    "sediment": ["ANNUAL_SEDIMENT"],
    "precipitation": ["ANNUAL_PRECIPITATION", "LONG_TERM_AVG_PRECIPITATION"],
    "comparison": ["COMPARE_LAST_YEAR", "COMPARE_LONG_TERM_AVG", "COMPARISON"],
    "area": ["BASIN_AREA", "HAS_AREA", "GW_LEVEL_RISE_AREA", "GW_LEVEL_DECLINE_AREA"],
    "length": ["RIVER_LENGTH"],
}
```

Fallback: 对 `relation_id IS NULL` 的关系，保留 `CONTAINS` 关键词作为兜底（但预期后续重新入库后该比例大幅下降）。

**`get_reservoir_parameters(reservoir_id)`**

同样替换为 `rel.relation_id IN ['DEAD_STORAGE_LEVEL', 'NORMAL_STORAGE_LEVEL', 'FLOOD_CONTROL_LEVEL', 'TOTAL_CAPACITY', 'FLOOD_CONTROL_CAPACITY']`。移除 `KEYWORD_MAP` 字典。

**`get_reservoir_hydrology_series(reservoir_id, indicator_keywords)`**

将 `CONTAINS` 改为 `rel.relation_id IN` 按传入的 indicator 类型映射。保留同河流水文站的 fallback 逻辑（第二阶段查询）。

### 3.3 新增：梯级关系推断

```python
def get_reservoir_upstream_relations(self, reservoir_ids: List[str]) -> List[Dict]:
    """
    对给定水库列表，推断上下游关系。
    规则：
    1. 同一河流上按 "上距XX距离" 等关系判断
    2. 同河流 + 按 constraint 中的距离值排序
    返回 [{"from": "龙羊峡水库", "to": "李家峡水库", "relation": "upstream"}]
    """
```

---

## 4. 新增 API 端点

### 4.1 `GET /api/optimization/formulate/batch`

**Parameters**: `reservoir_ids: str`（逗号分隔，最多 10 个）, `include_hydrology: bool`, `include_rules: bool`

**Response**:
```json
{
  "code": 200,
  "data": {
    "reservoirs": {
      "龙羊峡水库": { /* 完整 formulation */ },
      "李家峡水库": { /* 完整 formulation */ }
    },
    "relations": [
      {"from": "龙羊峡水库", "to": "李家峡水库", "relation": "upstream"}
    ]
  }
}
```

### 4.2 `GET /api/optimization/solver-ready`

**Parameters**: `reservoir_ids: str`（逗号分隔，1-10 个）, `year: int`（可选，过滤历史数据）

**Response**（紧凑求解器格式）:
```json
{
  "code": 200,
  "data": {
    "variables": [
      {"symbol": "Z_1", "name": "龙羊峡水库_水位", "lower": 2530.0, "upper": 2600.0, "unit": "m"},
      {"symbol": "Z_2", "name": "李家峡水库_水位", "lower": 2160.0, "upper": 2180.0, "unit": "m"},
      {"symbol": "Q_out_1", "name": "龙羊峡水库_出库流量", "lower": 50.0, "upper": null, "unit": "m³/s"}
    ],
    "constraints": [
      {"expression": "Z_1 >= 2530", "type": "water_level"},
      {"expression": "Z_1 <= 2594", "type": "water_level"}
    ],
    "time_series": {
      "龙羊峡水库": {
        "inflow": [{"year": 2024, "values": [120.5, 135.2, ...]}],
        "precipitation": [{"year": 2024, "values": [15.0, 20.1, ...]}]
      }
    },
    "objective_hints": ["maximize_power_generation", "minimize_flood_risk"],
    "meta": {
      "reservoir_count": 2,
      "constraint_count": 25,
      "time_series_years": [2024]
    }
  }
}
```

符号命名规则: `{symbol}_{reservoir_index}`，如 `Z_1` 表示第 1 个水库的水位变量。

### 4.3 `GET /api/optimization/compare`

**Parameters**: `reservoir_ids: str`（逗号分隔）

**Response**: 多库同指标横向对比表
```json
{
  "code": 200,
  "data": {
    "parameters": {
      "dead_storage_level": {
        "龙羊峡水库": {"value": 2530, "unit": "m"},
        "李家峡水库": {"value": 2160, "unit": "m"}
      },
      "total_capacity": {
        "龙羊峡水库": {"value": 247, "unit": "亿m³"},
        "李家峡水库": {"value": 16.5, "unit": "亿m³"}
      }
    }
  }
}
```

### 4.4 现有端点改动

所有现有端点（`/formulate`, `/constraints`, `/parameters`, `/hydrology`）新增可选参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `year` | `int` (optional) | 按年份过滤约束和水文数据。不传则返回全部。 |

响应增加 `Cache-Control: max-age=3600` 头（数据更新频率低）。

---

## 5. Python SDK

### 5.1 位置

`src/api/optimization_client.py`

### 5.2 公共 API

```python
class OptimizationClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        ...

    def get_formulation(self, reservoir_id: str, **kwargs) -> FormulationResult:
        """单库查询 → FormulationResult 对象"""
        ...

    def get_formulation_batch(self, reservoir_ids: List[str], **kwargs) -> BatchFormulationResult:
        """多库查询 → BatchFormulationResult 对象"""
        ...

    def get_solver_ready(self, reservoir_ids: List[str], year: int = None) -> SolverReadyResult:
        """求解器就绪格式 → SolverReadyResult 对象"""
        ...

    def compare(self, reservoir_ids: List[str]) -> CompareResult:
        """多库参数对比 → CompareResult 对象"""
        ...
```

### 5.3 返回类型

全部为 pydantic `BaseModel`，IDE 友好（属性补全）。每个结果类型都有 `to_dict()` 方法导出普通 `dict`，供跨语言调用。

```python
@dataclass / BaseModel
class FormulationResult:
    problem_meta: ProblemMeta
    decision_variables: List[DecisionVariable]
    constraints: List[Constraint]
    parameters: Parameters
    time_series: TimeSeries
    dispatch_rules: List[DispatchRule]

    def to_dict(self) -> dict: ...
```

### 5.4 使用示例

```python
from src.api.optimization_client import OptimizationClient

client = OptimizationClient()

# 单库
data = client.get_formulation("龙羊峡水库")
print(data.parameters.dead_storage_level)  # {"value": 2530, "unit": "m"}

# 梯级调度
batch = client.get_formulation_batch(["龙羊峡水库", "李家峡水库", "刘家峡水库"])
for name, formulation in batch.reservoirs.items():
    print(f"{name}: {len(formulation.constraints)} 条约束")

# 求解器格式
solver = client.get_solver_ready(["龙羊峡水库", "李家峡水库"], year=2024)
# → 可直接作为优化算法的 input_data
```

### 5.5 依赖

仅 `requests` + `pydantic`（项目已有），无其他依赖。

---

## 6. 兼容性

- 现有 API 端点路径、参数、响应结构**完全不变**，只新增可选参数 `year`
- 前端 `optimization.html` 无需改动
- `query_server.py` (port 8001) 不改动
- 旧的 `CONTAINS` 查询保留为 fallback（`relation_id IS NULL` 时），确保新增数据入库前的存量数据仍可查询

---

## 7. 验证方案

### 7.1 单元测试

| 测试 | 覆盖 |
|------|------|
| `test_relation_id_query` | `graph_query.py` 中精确查询返回正确约束 |
| `test_batch_formulation` | 多库批量查询返回数量一致 |
| `test_solver_ready_format` | solver-ready 输出结构完整性 |
| `test_compare_format` | compare 横向对比结构正确 |
| `test_client_models` | pydantic 模型序列化/反序列化 |

### 7.2 端到端测试

```bash
# 1. 启动服务
conda activate zagism
uvicorn src.api.main:app --reload

# 2. 现有端点不受影响
curl "http://localhost:8000/api/optimization/formulate?reservoir_id=龙羊峡水库" | jq .

# 3. 批量端点
curl "http://localhost:8000/api/optimization/formulate/batch?reservoir_ids=龙羊峡水库,李家峡水库" | jq .

# 4. 求解器格式
curl "http://localhost:8000/api/optimization/solver-ready?reservoir_ids=龙羊峡水库&year=2024" | jq .

# 5. 对比
curl "http://localhost:8000/api/optimization/compare?reservoir_ids=龙羊峡水库,李家峡水库,刘家峡水库" | jq .

# 6. Python SDK
python -c "
from src.api.optimization_client import OptimizationClient
c = OptimizationClient()
r = c.get_formulation('龙羊峡水库')
print(r.to_dict())
"
```

---

## 8. 风险评估

| 风险 | 缓解 |
|------|------|
| `relation_id` 在存量数据中缺失（老数据入库时未设置） | 保留 `CONTAINS` fallback + `relation_id IS NULL` 分支；新入库数据自动有 `relation_id` |
| 梯级关系推断不准（同河流但非上下游） | 标注 confidence + `relation` 字段说明推断依据，算法侧自行判断 |
| 求解器符号冲突（多库同名变量） | 使用 `{symbol}_{index}` 命名确保唯一性 |
