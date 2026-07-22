# 优化调度 API 改进 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将优化 API 查询层从脆弱的 `CONTAINS '中文'` 改为 `relation_id` 精确匹配，新增批量查询/求解器格式端点，提供 Python SDK。

**Architecture:** 重构 `GraphQuery` 三个优化方法 → 增强 `OptimizationFormatter` 增加 batch + solver-ready → 新增 3 个 API 端点 → 创建 `OptimizationClient` SDK。保留 CONTAINS fallback 确保存量数据兼容。

**Tech Stack:** Python 3.11, FastAPI, Neo4j (bolt), pydantic, requests

## Global Constraints

- 现有 API 端点路径、参数、响应结构不变，只新增可选参数 `year`
- 所有端点响应增加 `Cache-Control: max-age=3600` 头
- relation_id 精确匹配优先，relation_id IS NULL 时 fallback 到 CONTAINS
- Python SDK 仅依赖 `requests` + `pydantic`（项目已有）
- 不新增 Neo4j 节点或关系类型

---

### Task 1: 重构 graph_query.py — relation_id 精确查询

**Files:**
- Modify: `src/knowledge_graph/graph_query.py:218-434`

**Interfaces:**
- Consumes: `Neo4jClient.execute_read()`, `Neo4jClient.execute_read_single()`
- Produces: `get_optimization_constraints()`, `get_reservoir_parameters()`, `get_reservoir_hydrology_series()`, `get_reservoir_upstream_relations()`, `get_formulation_batch()`

- [ ] **Step 1: 在类开头添加 relation_id 映射常量**

在 `GraphQuery` 类中 `def __init__` 之后添加：

```python
# relation_id → 约束类别映射（用于精确查询）
CONSTRAINT_RELATION_IDS: Dict[str, List[str]] = {
    "water_level": ["DEAD_STORAGE_LEVEL", "FLOOD_CONTROL_LEVEL", "NORMAL_STORAGE_LEVEL"],
    "storage": ["TOTAL_CAPACITY", "FLOOD_CONTROL_CAPACITY", "RESERVOIR_STORAGE",
                "RESERVOIR_STORAGE_START", "RESERVOIR_STORAGE_END", "RESERVOIR_STORAGE_CHANGE"],
    "discharge": ["ANNUAL_RUNOFF", "ANNUAL_RUNOFF_FLOOD_SEASON", "ANNUAL_RUNOFF_TO_SEA", "ECOLOGICAL_FLOW"],
    "power": ["POWER_GENERATION"],
    "water_supply": ["WATER_SUPPLY"],
    "water_use": ["WATER_USE", "WATER_CONSUMPTION"],
    "sediment": ["ANNUAL_SEDIMENT"],
    "precipitation": ["ANNUAL_PRECIPITATION", "LONG_TERM_AVG_PRECIPITATION"],
    "comparison": ["COMPARE_LAST_YEAR", "COMPARE_LONG_TERM_AVG", "COMPARISON"],
    "area": ["BASIN_AREA", "HAS_AREA", "GW_LEVEL_RISE_AREA", "GW_LEVEL_DECLINE_AREA"],
    "length": ["RIVER_LENGTH"],
    "water_quality": ["WATER_QUALITY"],
    "ground_depth": ["GROUNDWATER_DEPTH", "GROUNDWATER_DEPTH_START",
                     "GROUNDWATER_DEPTH_END", "GROUNDWATER_DEPTH_CHANGE"],
}

# 水库参数 relation_id 集合
PARAM_RELATION_IDS = [
    "DEAD_STORAGE_LEVEL", "NORMAL_STORAGE_LEVEL", "FLOOD_CONTROL_LEVEL",
    "TOTAL_CAPACITY", "FLOOD_CONTROL_CAPACITY",
]

# relation_id → 参数 key 映射
PARAM_ID_TO_KEY = {
    "DEAD_STORAGE_LEVEL": "dead_storage_level",
    "NORMAL_STORAGE_LEVEL": "normal_storage_level",
    "FLOOD_CONTROL_LEVEL": "flood_control_level",
    "TOTAL_CAPACITY": "total_capacity",
    "FLOOD_CONTROL_CAPACITY": "flood_control_capacity",
}
```

- [ ] **Step 2: 重构 `get_optimization_constraints()`**

替换整个方法（line 218-291）：

```python
def get_optimization_constraints(self, reservoir_id: str) -> List[Dict]:
    """
    查水库所有约束数据，合并两个来源:
    1. Constraint 节点 (直接约束)
    2. AnnualHydrologyData 中包含约束关键词的 (间接约束)

    优先使用 relation_id 精确匹配，fallback 到 CONTAINS 中文关键词。
    """
    # 收集所有约束类 relation_id
    all_constraint_ids = []
    for ids in self.CONSTRAINT_RELATION_IDS.values():
        all_constraint_ids.extend(ids)
    id_list = json.dumps(all_constraint_ids)  # ["DEAD_STORAGE_LEVEL", ...]

    # 查询1: Constraint 节点 — relation_id 精确匹配 + NULL fallback
    constraint_query = f"""
    MATCH (r:Reservoir {{id: $id}})-[rel]->(c:Constraint)
    WHERE rel.relation_id IN {id_list}
       OR (rel.relation_id IS NULL AND (
           type(rel) CONTAINS '死水位' OR type(rel) CONTAINS '正常蓄水位'
           OR type(rel) CONTAINS '汛限' OR type(rel) CONTAINS '防洪限制'
           OR type(rel) CONTAINS '总库容' OR type(rel) CONTAINS '防洪库容'
           OR type(rel) CONTAINS '兴利库容'
       ))
    RETURN c, type(rel) AS relation, rel.relation_id AS relation_id,
           rel.confidence AS confidence, rel.source_doc AS source_doc
    """
    constraint_results = self.neo4j.execute_read(constraint_query, {"id": reservoir_id})

    # 查询2: AnnualHydrologyData — relation_id 精确匹配 + NULL fallback
    hydrology_query = f"""
    MATCH (r:Reservoir {{id: $id}})-[rel]->(d:AnnualHydrologyData)
    WHERE rel.relation_id IN {id_list}
       OR (rel.relation_id IS NULL AND (
           type(rel) CONTAINS '水位' OR type(rel) CONTAINS '流量'
           OR type(rel) CONTAINS '库容' OR type(rel) CONTAINS '出力'
           OR type(rel) CONTAINS '供水' OR type(rel) CONTAINS '生态'
           OR type(rel) CONTAINS '汛限' OR type(rel) CONTAINS '死水位'
           OR type(rel) CONTAINS '蓄水位' OR type(rel) CONTAINS '防洪'
           OR type(rel) CONTAINS '限制' OR type(rel) CONTAINS '约束'
           OR type(rel) CONTAINS '上限' OR type(rel) CONTAINS '下限'
           OR type(rel) CONTAINS '不超过' OR type(rel) CONTAINS '不低于'
           OR type(rel) CONTAINS '用水' OR type(rel) CONTAINS '灌溉'
       ))
    RETURN d.indicator AS indicator, d.value AS value, d.unit AS unit,
           d.year AS year, type(rel) AS relation,
           rel.relation_id AS relation_id, d.name AS name,
           rel.confidence AS confidence, rel.source_doc AS source_doc
    """
    hydrology_results = self.neo4j.execute_read(hydrology_query, {"id": reservoir_id})

    merged = list(constraint_results)
    for h in hydrology_results:
        merged.append({
            "c": None,
            "relation": h.get("relation", ""),
            "relation_id": h.get("relation_id"),
            "confidence": h.get("confidence"),
            "source_doc": h.get("source_doc"),
            "indicator": h.get("indicator", ""),
            "d": {
                "indicator": h.get("indicator", ""),
                "value": h.get("value", ""),
                "unit": h.get("unit", ""),
                "year": h.get("year", ""),
                "name": h.get("name", ""),
            },
        })

    logger.info(
        f"优化约束查询: reservoir={reservoir_id}, "
        f"Constraint={len(constraint_results)}, HydrologyData={len(hydrology_results)}"
    )
    return merged
```

需要在文件顶部添加 `import json`。

- [ ] **Step 3: 添加 import json**

在 `graph_query.py` 第 1 行 import 块添加：

```python
import json
```

- [ ] **Step 4: 重构 `get_reservoir_parameters()`**

替换整个方法（line 293-357）：

```python
def get_reservoir_parameters(self, reservoir_id: str) -> Dict:
    """
    查询水库物理参数，优先级:
    1. relation_id 精确匹配的 Constraint 节点
    2. Reservoir 节点属性中的值
    """
    params: Dict[str, Any] = {}

    # 查询1: Reservoir 节点属性
    reservoir_query = """
    MATCH (r:Reservoir {id: $id})
    RETURN r.total_capacity AS total_capacity,
           r.flood_control_capacity AS flood_control_capacity,
           r.normal_storage_level AS normal_storage_level,
           r.dead_storage_level AS dead_storage_level
    """
    res_result = self.neo4j.execute_read_single(reservoir_query, {"id": reservoir_id})
    if res_result:
        for key, val in res_result.items():
            if val is not None:
                params[key] = val

    # 查询2: 从 Constraint 节点补充 — 使用 relation_id 精确匹配
    id_list = json.dumps(self.PARAM_RELATION_IDS)
    param_query = f"""
    MATCH (r:Reservoir {{id: $id}})-[rel]->(c:Constraint)
    WHERE rel.relation_id IN {id_list}
       OR (rel.relation_id IS NULL AND (
           type(rel) CONTAINS '死水位' OR type(rel) CONTAINS '正常蓄水位'
           OR type(rel) CONTAINS '汛限' OR type(rel) CONTAINS '防洪限制'
           OR type(rel) CONTAINS '总库容' OR type(rel) CONTAINS '防洪库容'
           OR type(rel) CONTAINS '兴利库容'
       ))
    RETURN c.name AS name, c.value AS value, c.unit AS unit,
           type(rel) AS relation, rel.relation_id AS relation_id
    """
    constraint_params = self.neo4j.execute_read(param_query, {"id": reservoir_id})

    for cp in constraint_params:
        relation_id = cp.get("relation_id", "")
        relation = cp.get("relation", "")
        value = cp.get("value", "")
        unit = cp.get("unit", "")

        # 优先用 relation_id 映射
        if relation_id in self.PARAM_ID_TO_KEY:
            key = self.PARAM_ID_TO_KEY[relation_id]
            params[key] = {"value": value, "unit": unit, "source_relation": relation}
        else:
            # relation_id 为空兜底：用旧关键词映射
            KEYWORD_MAP = {
                "死水位": "dead_storage_level",
                "正常蓄水位": "normal_storage_level",
                "汛限": "flood_control_level",
                "防洪限制": "flood_control_level",
                "总库容": "total_capacity",
                "防洪库容": "flood_control_capacity",
                "兴利库容": "active_capacity",
            }
            matched = False
            for keyword, key in KEYWORD_MAP.items():
                if keyword in relation:
                    params[key] = {"value": value, "unit": unit, "source_relation": relation}
                    matched = True
                    break
            if not matched:
                key = f"constraint_{cp.get('name', 'unknown')}"
                params[key] = {"value": value, "unit": unit, "source_relation": relation}

    return params
```

- [ ] **Step 5: 重构 `get_reservoir_hydrology_series()`**

替换整个方法（line 359-411）：

```python
def get_reservoir_hydrology_series(
    self,
    reservoir_id: str,
    indicator_keywords: Optional[List[str]] = None,
    year: Optional[int] = None,
) -> List[Dict]:
    """
    查询水库关联水文站的时间序列数据

    Args:
        reservoir_id: 水库ID
        indicator_keywords: 指标关键词列表 (改用 relation_id 匹配)
        year: 可选，按年份过滤
    """
    # relation_id → keyword fallback 映射
    INDICATOR_RELATION_IDS = {
        "径流": ["ANNUAL_RUNOFF", "ANNUAL_RUNOFF_FLOOD_SEASON", "ANNUAL_RUNOFF_TO_SEA",
                  "LONG_TERM_AVG_RUNOFF"],
        "流量": ["ANNUAL_RUNOFF", "ANNUAL_RUNOFF_FLOOD_SEASON", "ANNUAL_RUNOFF_TO_SEA",
                  "ECOLOGICAL_FLOW", "LONG_TERM_AVG_RUNOFF"],
        "入流": ["ANNUAL_RUNOFF", "ANNUAL_RUNOFF_FLOOD_SEASON"],
        "水位": ["DEAD_STORAGE_LEVEL", "FLOOD_CONTROL_LEVEL", "NORMAL_STORAGE_LEVEL",
                 "GROUNDWATER_DEPTH", "GROUNDWATER_DEPTH_START", "GROUNDWATER_DEPTH_END",
                 "GROUNDWATER_DEPTH_CHANGE"],
        "降水": ["ANNUAL_PRECIPITATION", "LONG_TERM_AVG_PRECIPITATION"],
        "来水": ["ANNUAL_RUNOFF", "ANNUAL_RUNOFF_FLOOD_SEASON"],
    }

    if indicator_keywords:
        # 收集匹配传入关键词的 relation_ids
        target_ids = set()
        fallback_kws = set()
        for kw in indicator_keywords:
            matched = False
            for fk, ids in INDICATOR_RELATION_IDS.items():
                if fk in kw:
                    target_ids.update(ids)
                    matched = True
            if not matched:
                fallback_kws.add(kw)
        id_list = json.dumps(list(target_ids))
    else:
        # 默认所有水文相关
        all_ids = set()
        for ids in INDICATOR_RELATION_IDS.values():
            all_ids.update(ids)
        id_list = json.dumps(list(all_ids))
        fallback_kws = set()

    # 构建 fallback CONTAINS 条件
    if fallback_kws:
        fallback_conditions = " OR ".join(
            [f"type(rel) CONTAINS '{kw}'" for kw in fallback_kws]
        )
    else:
        fallback_conditions = ""

    # 年份过滤
    year_clause = ""
    params: Dict = {"id": reservoir_id}
    if year is not None:
        year_clause = "AND d.year = $year"
        params["year"] = year

    # 构建 WHERE 子句
    where_parts = [f"rel.relation_id IN {id_list}"]
    if fallback_conditions:
        where_parts.append(f"(rel.relation_id IS NULL AND ({fallback_conditions}))")
    where_clause = " AND ".join(f"({p})" for p in where_parts)

    # 方案1: 直接查水库关联的年度水文数据
    direct_query = f"""
    MATCH (r:Reservoir {{id: $id}})-[rel]->(d:AnnualHydrologyData)
    WHERE {where_clause}
    {year_clause}
    RETURN d, type(rel) AS relation, rel.relation_id AS relation_id,
           rel.source_doc AS source_doc, rel.confidence AS confidence
    ORDER BY d.year
    LIMIT 200
    """
    results = self.neo4j.execute_read(direct_query, params)

    # 方案2: fallback 通过同河流水文站获取
    if not results:
        station_query = f"""
        MATCH (res:Reservoir {{id: $id}})-[:LOCATED_ON]->(river:River)
        MATCH (station:HydrologicalStation)-[:LOCATED_ON]->(river)
        MATCH (station)-[rel]->(d:AnnualHydrologyData)
        WHERE {where_clause}
        {year_clause}
        RETURN d, type(rel) AS relation, rel.relation_id AS relation_id,
               station.name AS station_name,
               rel.source_doc AS source_doc, rel.confidence AS confidence
        ORDER BY d.year
        LIMIT 200
        """
        results = self.neo4j.execute_read(station_query, params)

    logger.info(
        f"水文时间序列: reservoir={reservoir_id}, "
        f"results={len(results)}"
    )
    return results
```

- [ ] **Step 6: 新增 `get_reservoir_upstream_relations()`**

在 `get_optimization_formulation()` 之后新增：

```python
def get_reservoir_upstream_relations(self, reservoir_ids: List[str]) -> List[Dict]:
    """
    推断水库间的上下游关系。
    策略: 检查是否存在方向性距离关系（"上距XX距离"/"下距XX距离"等）
    """
    if len(reservoir_ids) < 2:
        return []

    relations = []
    for i, rid_a in enumerate(reservoir_ids):
        for rid_b in enumerate(reservoir_ids[i + 1:], i + 1):
            # 查 A→B 的关系
            query = """
            MATCH (a:Reservoir {id: $id_a})-[rel]->(c:Constraint)
            WHERE (type(rel) CONTAINS '距' AND type(rel) CONTAINS $name_b)
               OR (type(rel) CONTAINS '上距' AND type(rel) CONTAINS $name_b)
               OR (type(rel) CONTAINS '下距' AND type(rel) CONTAINS $name_b)
            RETURN type(rel) AS relation, c.value AS value, c.unit AS unit
            LIMIT 1
            """
            # 先获取水库名称
            name_query = "MATCH (r:Reservoir {id: $id}) RETURN r.name AS name"
            res_a = self.neo4j.execute_read_single(name_query, {"id": rid_a})
            res_b = self.neo4j.execute_read_single(name_query, {"id": reservoir_ids[i + 1]})
            name_a = res_a.get("name", rid_a) if res_a else rid_a
            name_b = res_b.get("name", reservoir_ids[i + 1]) if res_b else reservoir_ids[i + 1]

            # 检查 A 是否在 B 上游
            result_ab = self.neo4j.execute_read(query, {"id_a": rid_a, "name_b": name_b})
            if result_ab:
                relations.append({
                    "from": rid_a, "from_name": name_a,
                    "to": reservoir_ids[i + 1], "to_name": name_b,
                    "relation": "upstream",
                    "source_relation": result_ab[0].get("relation", ""),
                    "distance_value": result_ab[0].get("value", ""),
                    "distance_unit": result_ab[0].get("unit", ""),
                })
                continue

            # 检查 B 是否在 A 上游
            result_ba = self.neo4j.execute_read(
                query, {"id_a": reservoir_ids[i + 1], "name_b": name_a}
            )
            if result_ba:
                relations.append({
                    "from": reservoir_ids[i + 1], "from_name": name_b,
                    "to": rid_a, "to_name": name_a,
                    "relation": "upstream",
                    "source_relation": result_ba[0].get("relation", ""),
                    "distance_value": result_ba[0].get("value", ""),
                    "distance_unit": result_ba[0].get("unit", ""),
                })

    return relations
```

修复 bug: 第二层 for 循环有变量名错误，应改为：

```python
def get_reservoir_upstream_relations(self, reservoir_ids: List[str]) -> List[Dict]:
    """推断水库间的上下游关系"""
    if len(reservoir_ids) < 2:
        return []

    relations = []
    name_query = "MATCH (r:Reservoir {id: $id}) RETURN r.name AS name"

    for i in range(len(reservoir_ids)):
        for j in range(i + 1, len(reservoir_ids)):
            rid_a = reservoir_ids[i]
            rid_b = reservoir_ids[j]

            res_a = self.neo4j.execute_read_single(name_query, {"id": rid_a})
            res_b = self.neo4j.execute_read_single(name_query, {"id": rid_b})
            name_a = res_a.get("name", rid_a) if res_a else rid_a
            name_b = res_b.get("name", rid_b) if res_b else rid_b

            query = """
            MATCH (a:Reservoir {id: $id_a})-[rel]->(c:Constraint)
            WHERE type(rel) CONTAINS $name_b
            RETURN type(rel) AS relation, c.value AS value, c.unit AS unit
            LIMIT 1
            """
            result = self.neo4j.execute_read(query, {"id_a": rid_a, "name_b": name_b})
            if result:
                rel_text = result[0].get("relation", "")
                is_upstream = "上距" in rel_text or "距" in rel_text
                if is_upstream:
                    relations.append({
                        "from": rid_a, "from_name": name_a,
                        "to": rid_b, "to_name": name_b,
                        "relation": "upstream",
                    })
    return relations
```

- [ ] **Step 7: 新增 `get_formulation_batch()`**

```python
def get_formulation_batch(self, reservoir_ids: List[str]) -> Dict:
    """批量获取多个水库的优化数据 + 梯级关系"""
    if len(reservoir_ids) > 10:
        raise ValueError("最多支持 10 个水库的批量查询")

    reservoirs_data = {}
    for rid in reservoir_ids:
        reservoirs_data[rid] = self.get_optimization_formulation(rid)

    relations = self.get_reservoir_upstream_relations(reservoir_ids)

    return {
        "reservoirs": reservoirs_data,
        "relations": relations,
    }
```

- [ ] **Step 8: Commit**

```bash
git add src/knowledge_graph/graph_query.py
git commit -m "refactor: graph_query 优化方法改用 relation_id 精确查询 + 批量+梯级关系"
```

---

### Task 2: 增强 optimization_formatter.py — batch + solver-ready

**Files:**
- Modify: `src/knowledge_graph/optimization_formatter.py`

**Interfaces:**
- Consumes: `GraphQuery.get_formulation_batch()` 返回值，`GraphQuery.get_optimization_formulation()` 返回值
- Produces: `build_formulation_batch()`, `build_solver_ready()`

- [ ] **Step 1: 新增 `build_formulation_batch()`**

在 `build_formulation()` 方法之后添加：

```python
def build_formulation_batch(
    self,
    batch_data: Dict,
    include_hydrology: bool = True,
    include_rules: bool = True,
) -> Dict:
    """
    批量构建多个水库的优化问题结构

    Args:
        batch_data: get_formulation_batch() 的返回值
            {"reservoirs": {id: {reservoir, constraints, parameters, ...}}, "relations": [...]}

    Returns:
        {"reservoirs": {id: formulation, ...}, "relations": [...]}
    """
    formulations = {}
    for rid, raw in batch_data.get("reservoirs", {}).items():
        reservoir = raw.get("reservoir", {})
        if not reservoir:
            reservoir = {"id": rid, "name": rid}

        formulations[rid] = self.build_formulation(
            reservoir=reservoir,
            constraints_raw=raw.get("constraints", []),
            parameters=raw.get("parameters", {}),
            hydrology_series=raw.get("hydrology_series", []) if include_hydrology else None,
            dispatch_rules=raw.get("dispatch_rules", []) if include_rules else None,
        )

    return {
        "reservoirs": formulations,
        "relations": batch_data.get("relations", []),
    }
```

- [ ] **Step 2: 新增 `build_solver_ready()`**

在 `build_formulation_batch()` 之后添加：

```python
def build_solver_ready(
    self,
    batch_data: Dict,
    year: Optional[int] = None,
) -> Dict:
    """
    构建求解器就绪格式 — 紧凑、可直接喂给优化算法

    Args:
        batch_data: get_formulation_batch() 的返回值
        year: 可选，过滤时间序列

    Returns:
        {"variables": [...], "constraints": [...], "time_series": {...}, "objective_hints": [...]}
    """
    reservoir_ids = list(batch_data.get("reservoirs", {}).keys())
    all_variables = []
    all_constraints = []
    all_time_series = {}
    all_objective_hints = set()
    reservoir_map = {}  # index -> reservoir_id

    for idx, rid in enumerate(reservoir_ids):
        reservoir_map[idx] = rid
        raw = batch_data["reservoirs"][rid]
        reservoir = raw.get("reservoir", {}) or {"id": rid, "name": rid}

        fmt = self.build_formulation(
            reservoir=reservoir,
            constraints_raw=raw.get("constraints", []),
            parameters=raw.get("parameters", {}),
            hydrology_series=raw.get("hydrology_series", []) if year else None,
            dispatch_rules=None,  # solver format doesn't include rules
        )

        # 决策变量 → 加 index 后缀
        for var in fmt.get("decision_variables", []):
            all_variables.append({
                "symbol": f"{var['symbol']}_{idx + 1}",
                "name": f"{reservoir.get('name', rid)}_{var['name']}",
                "lower": var.get("bounds", {}).get("lower"),
                "upper": var.get("bounds", {}).get("upper"),
                "unit": var.get("unit", ""),
            })

        # 约束 → 替换变量名为带 index 的符号
        for cons in fmt.get("constraints", []):
            if cons.get("expression"):
                expr = cons["expression"]
                # 替换变量符号: Z -> Z_1, Q_out -> Q_out_2 等
                for var in fmt.get("decision_variables", []):
                    bare_symbol = var["symbol"]
                    indexed_symbol = f"{bare_symbol}_{idx + 1}"
                    expr = expr.replace(bare_symbol, indexed_symbol)
                all_constraints.append({
                    "expression": expr,
                    "type": cons.get("category", "unknown"),
                    "description": cons.get("name", ""),
                })

        # 目标函数提示
        for obj in fmt.get("objective_candidates", []):
            all_objective_hints.add(
                f"{obj['type']}_{obj['description']}"
            )

        # 时间序列
        ts_entries = fmt.get("time_series", {}).get("entries", [])
        if ts_entries:
            series_by_indicator = {}
            for entry in ts_entries:
                indicator = entry.get("indicator", "unknown")
                if indicator not in series_by_indicator:
                    series_by_indicator[indicator] = []
                series_by_indicator[indicator].append({
                    "year": entry.get("year"),
                    "value": entry.get("value"),
                    "unit": entry.get("unit"),
                })
            all_time_series[reservoir.get("name", rid)] = series_by_indicator

        # 参数中的水位约束也作为 constraint 加入
        params = fmt.get("parameters", {})
        for param_key, param_val in params.items():
            if not isinstance(param_val, dict):
                continue
            val = param_val.get("value")
            unit = param_val.get("unit", "")
            if val is None:
                continue

            # 死水位 → >=约束
            if param_key == "dead_storage_level":
                sym = f"Z_{idx + 1}"
                all_constraints.append({
                    "expression": f"{sym} >= {val}",
                    "type": "water_level",
                    "description": f"{reservoir.get('name', rid)} 死水位约束",
                })
            # 防洪限制水位 → <=约束
            elif param_key == "flood_control_level":
                sym = f"Z_{idx + 1}"
                all_constraints.append({
                    "expression": f"{sym} <= {val}",
                    "type": "water_level",
                    "description": f"{reservoir.get('name', rid)} 防洪限制水位约束",
                })

    return {
        "variables": all_variables,
        "constraints": all_constraints,
        "time_series": all_time_series,
        "objective_hints": sorted(all_objective_hints),
        "meta": {
            "reservoir_count": len(reservoir_ids),
            "constraint_count": len(all_constraints),
            "time_series_years": [year] if year else [],
        },
    }
```

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_graph/optimization_formatter.py
git commit -m "feat: optimization_formatter 新增 build_formulation_batch + build_solver_ready"
```

---

### Task 3: 新增 API 端点 + year 参数

**Files:**
- Modify: `src/api/routers/optimization.py`

**Interfaces:**
- Consumes: `get_graph_query()`, `OptimizationFormatter`
- Produces: 3 new endpoints + year parameter on existing endpoints

- [ ] **Step 1: 替换整个 `optimization.py` router**

```python
"""
优化调度数据接口
从知识图谱中提取水库调度优化所需的结构化数据
"""
from typing import Optional

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
```

- [ ] **Step 2: 确认 `main.py` 无需改动**

`main.py` 第 68 行已挂载 `optimization.router`，新增端点自动注册，无需改动。

- [ ] **Step 3: Commit**

```bash
git add src/api/routers/optimization.py
git commit -m "feat: 新增 /formulate/batch, /solver-ready, /compare 端点 + year 参数"
```

---

### Task 4: 创建 Python SDK

**Files:**
- Create: `src/api/optimization_client.py`

**Interfaces:**
- Produces: `OptimizationClient` class, pydantic models (`FormulationResult`, `BatchFormulationResult`, `SolverReadyResult`, `CompareResult`)

- [ ] **Step 1: 创建 `optimization_client.py`**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add src/api/optimization_client.py
git commit -m "feat: 新增 OptimizationClient Python SDK"
```

---

### Task 5: 端到端验证

**Files:**
- Create: (无新文件，手动测试)

- [ ] **Step 1: 启动服务**

```bash
conda activate zagism
cd d:/work/knowlegeextract
D:/tool/Anaconda3/envs/zagism/python.exe -m uvicorn src.api.main:app --reload --port 8000 &
```

- [ ] **Step 2: 测试现有端点不受影响**

```bash
# 等待服务启动
sleep 3

# 现有 formulate 端点
curl -s "http://localhost:8000/api/optimization/formulate?reservoir_id=龙羊峡水库" | python -c "import sys,json; d=json.load(sys.stdin); print('OK' if d['code']==200 else 'FAIL')"

# 带 year 参数
curl -s "http://localhost:8000/api/optimization/formulate?reservoir_id=龙羊峡水库&year=2024" | python -c "import sys,json; d=json.load(sys.stdin); print('OK with year' if d['code']==200 else 'FAIL')"
```

Expected: `OK` 和 `OK with year`

- [ ] **Step 3: 测试新增批量端点**

```bash
curl -s "http://localhost:8000/api/optimization/formulate/batch?reservoir_ids=龙羊峡水库,李家峡水库" | python -c "
import sys, json
d = json.load(sys.stdin)
assert d['code'] == 200
data = d['data']
assert '龙羊峡水库' in data['reservoirs']
assert '李家峡水库' in data['reservoirs']
print('BATCH OK:', len(data['reservoirs']), 'reservoirs,', len(data.get('relations', [])), 'relations')
"
```

- [ ] **Step 4: 测试 solver-ready 端点**

```bash
curl -s "http://localhost:8000/api/optimization/solver-ready?reservoir_ids=龙羊峡水库&year=2024" | python -c "
import sys, json
d = json.load(sys.stdin)
assert d['code'] == 200
data = d['data']
assert len(data['variables']) > 0
assert len(data['constraints']) > 0
print('SOLVER-READY OK:', len(data['variables']), 'vars,', len(data['constraints']), 'constraints')
"
```

- [ ] **Step 5: 测试 compare 端点**

```bash
curl -s "http://localhost:8000/api/optimization/compare?reservoir_ids=龙羊峡水库,李家峡水库" | python -c "
import sys, json
d = json.load(sys.stdin)
assert d['code'] == 200
params = d['data']['parameters']
assert '死水位' in params
print('COMPARE OK:', len(params), 'parameter groups')
"
```

- [ ] **Step 6: 测试 Python SDK**

```bash
cd d:/work/knowlegeextract
D:/tool/Anaconda3/envs/zagism/python.exe -c "
from src.api.optimization_client import OptimizationClient
c = OptimizationClient()

# 单库
r = c.get_formulation('龙羊峡水库')
print('SDK formulate:', r.problem_meta.reservoir_name, '-',
      len(r.decision_variables), 'vars,', len(r.constraints), 'constraints')

# 批量
b = c.get_formulation_batch(['龙羊峡水库', '李家峡水库'])
print('SDK batch:', len(b.reservoirs), 'reservoirs,', len(b.relations), 'relations')

# solver-ready
s = c.get_solver_ready(['龙羊峡水库'], year=2024)
print('SDK solver-ready:', len(s.variables), 'vars,', len(s.constraints), 'constraints')

# compare
cmp = c.compare(['龙羊峡水库', '李家峡水库'])
print('SDK compare:', len(cmp.parameters), 'parameter groups')

# to_dict
d = r.to_dict()
print('SDK to_dict:', type(d).__name__)

c.close()
print('ALL SDK TESTS PASSED')
"
```

Expected: All output lines ending with success and final `ALL SDK TESTS PASSED`

- [ ] **Step 7: 验证 Cache-Control 头**

```bash
curl -sI "http://localhost:8000/api/optimization/parameters?reservoir_id=龙羊峡水库" | grep -i cache-control
```

Expected: `Cache-Control: max-age=3600`

- [ ] **Step 8: 停止服务**

```bash
# 杀掉 uvicorn 进程
pkill -f "uvicorn src.api.main:app" 2>/dev/null || true
```

- [ ] **Step 9: Final commit**

```bash
git add -A
git commit -m "verify: 所有优化 API 端点 + SDK 端到端验证通过"
```

---

### 验证清单

| # | 验证项 | 方法 |
|---|--------|------|
| 1 | 现有 /formulate 不变 | curl + json 断言 |
| 2 | year 参数过滤有效 | curl /formulate?year=2024 |
| 3 | /formulate/batch 多库返回 | curl batch + 断言 2 reservoirs |
| 4 | solver-ready 变量+约束 | curl + 断言 len > 0 |
| 5 | compare 6 参数组 | curl + 断言 "死水位" in |
| 6 | SDK 单库/多库/solver/compare | Python 脚本 |
| 7 | Cache-Control 头 | curl -sI |
| 8 | relation_id 查询执行 | 检查日志无错误 |
