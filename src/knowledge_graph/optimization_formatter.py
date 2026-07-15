"""
优化数据格式化器
将知识图谱中的原始约束/参数/水文数据重组为优化算法可消费的结构化 JSON

核心职责：
1. 中文关系名 → 约束类别 + 数学运算符
2. 从字段/关系名中提取数值和单位
3. 从约束类型推断决策变量及其边界
4. 从关键词推断目标函数候选
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger

logger = get_logger(__name__)


class OptimizationFormatter:
    """将 KG 查询结果格式化为优化问题结构"""

    # ── 约束关键词 → 类别映射 ──
    CONSTRAINT_CATEGORIES: Dict[str, str] = {
        "水位": "water_level",
        "库水位": "water_level",
        "蓄水位": "water_level",
        "汛限水位": "water_level",
        "死水位": "water_level",
        "正常蓄水位": "water_level",
        "防洪限制水位": "water_level",
        "库容": "storage",
        "总库容": "storage",
        "防洪库容": "storage",
        "兴利库容": "storage",
        "流量": "discharge",
        "下泄": "discharge",
        "泄量": "discharge",
        "径流": "discharge",
        "出力": "power_output",
        "发电": "power_generation",
        "供水": "water_supply",
        "用水": "water_use",
        "灌溉": "irrigation",
        "生态": "ecological_flow",
        "生态基流": "ecological_flow",
        "面积": "area",
        "全长": "length",
        "输沙": "sediment",
    }

    # ── 中文运算符 → 数学符号 ──
    OPERATOR_PATTERNS: List[Tuple[str, str]] = [
        (r"(不超过|不大于|≤|小于等于|上限|不高于|不高过|低于|小于)", "<="),
        (r"(不低于|不小于|≥|大于等于|下限|不少于|保证|不低|高于|大于)", ">="),
        (r"(等于|为|维持|保持|控制在?|控制在)", "=="),
        (r"(范围|区间|之间|～|~)", "between"),
    ]

    # ── 决策变量推断 ──
    VARIABLE_INFERENCE: Dict[str, Dict[str, str]] = {
        "water_level": {"name": "库水位", "symbol": "Z", "unit": "m"},
        "storage": {"name": "库容", "symbol": "V", "unit": "亿m³"},
        "discharge": {"name": "出库流量", "symbol": "Q_out", "unit": "m³/s"},
        "power_output": {"name": "出力", "symbol": "P", "unit": "万kW"},
        "power_generation": {"name": "发电量", "symbol": "E", "unit": "亿kWh"},
        "water_supply": {"name": "供水量", "symbol": "W_supply", "unit": "亿m³"},
        "water_use": {"name": "用水量", "symbol": "W_use", "unit": "亿m³"},
    }

    # ── 目标函数候选推断 ──
    OBJECTIVE_CANDIDATES: Dict[str, List[Dict]] = {
        "power_generation": [
            {"description": "最大化发电量", "type": "maximize", "related_keywords": ["发电", "出力"]},
        ],
        "power_output": [
            {"description": "最大化出力", "type": "maximize", "related_keywords": ["出力", "发电"]},
        ],
        "water_supply": [
            {"description": "最大化供水量", "type": "maximize", "related_keywords": ["供水"]},
        ],
        "water_use": [
            {"description": "最大化用水满足率", "type": "maximize", "related_keywords": ["用水"]},
        ],
        "irrigation": [
            {"description": "最大化灌溉保证率", "type": "maximize", "related_keywords": ["灌溉"]},
        ],
        "ecological_flow": [
            {"description": "最小化生态流量破坏", "type": "minimize", "related_keywords": ["生态", "生态基流"]},
        ],
        "water_level": [
            {"description": "最小化最高库水位（防洪）", "type": "minimize", "related_keywords": ["防洪", "水位"]},
        ],
    }

    # ── 数值提取正则 ──
    VALUE_PATTERN = re.compile(
        r"(\d+\.?\d*)\s*(亿m³|亿立方米|万m³|m³/s|m³|m/s|m\b|米|万kW|kW|亿kWh|kWh|mm|毫米|万平方公里|平方公里|km²|公顷|万亩|吨|万t|亿吨|亿t|%)?"
    )

    def __init__(self):
        self._category_map: Dict[str, str] = {}
        # 预计算：把中文关键词映射到英文类别
        for keyword, category in self.CONSTRAINT_CATEGORIES.items():
            self._category_map[keyword] = category

    # ═══════════════════════════════════════════════════════════
    # 公开方法
    # ═══════════════════════════════════════════════════════════

    def build_formulation(
        self,
        reservoir: Dict,
        constraints_raw: List[Dict],
        parameters: Dict,
        hydrology_series: Optional[List[Dict]] = None,
        dispatch_rules: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        构建完整优化问题结构

        Args:
            reservoir: Reservoir 节点属性 (name, id, river, ...)
            constraints_raw: get_optimization_constraints() 的返回值
            parameters: get_reservoir_parameters() 的返回值
            hydrology_series: get_reservoir_hydrology_series() 的返回值 (可选)
            dispatch_rules: get_reservoir_rules() 的返回值 (可选)

        Returns:
            {
                "problem_meta": ...,
                "decision_variables": [...],
                "objective_candidates": [...],
                "constraints": [...],
                "parameters": {...},
                "time_series": {...},
                "dispatch_rules": [...],
            }
        """
        # 1. 格式化约束
        formatted_constraints = self._format_constraints(constraints_raw, reservoir)

        # 2. 格式参数
        formatted_params = self._format_parameters(parameters, reservoir)

        # 3. 推断决策变量
        decision_vars = self._infer_decision_variables(
            formatted_constraints, formatted_params, reservoir
        )

        # 4. 推断目标函数候选
        objective_candidates = self._infer_objective_candidates(
            formatted_constraints, reservoir
        )

        # 5. 组装时间序列
        time_series = self._format_time_series(hydrology_series or [])

        # 6. 格式化调度规则
        rules = self._format_dispatch_rules(dispatch_rules or [])

        reservoir_name = reservoir.get("name", str(reservoir.get("id", "")))
        reservoir_id = reservoir.get("id", "")

        return {
            "problem_meta": {
                "reservoir_name": reservoir_name,
                "reservoir_id": reservoir_id,
                "description": f"{reservoir_name}优化调度问题",
                "data_sources": {
                    "constraints_count": len(formatted_constraints),
                    "time_series_count": len(time_series.get("entries", [])),
                    "has_objective_info": len(objective_candidates) > 0,
                    "inference_note": "决策变量和目标函数由约束关键词自动推断，标注 source='inferred'",
                },
            },
            "decision_variables": decision_vars,
            "objective_candidates": objective_candidates,
            "constraints": formatted_constraints,
            "parameters": formatted_params,
            "time_series": time_series,
            "dispatch_rules": rules,
        }

    def categorize_constraint(self, text: str) -> Optional[str]:
        """给定一段中文文本，返回约束类别 (英文 key)"""
        if not text:
            return None
        for keyword, category in self.CONSTRAINT_CATEGORIES.items():
            if keyword in text:
                return category
        return None

    def extract_operator(self, text: str) -> Optional[str]:
        """从中文文本中提取数学运算符"""
        if not text:
            return None
        for pattern, operator in self.OPERATOR_PATTERNS:
            if re.search(pattern, text):
                return operator
        return None

    def extract_numeric(self, text: str) -> Optional[Tuple[float, Optional[str]]]:
        """从文本中提取数值 + 单位"""
        if not text:
            return None
        # 先尝试直接从文本中提取
        match = self.VALUE_PATTERN.search(str(text))
        if match:
            value = float(match.group(1))
            unit = match.group(2) if match.lastindex and match.lastindex >= 2 else None
            return (value, unit)
        return None

    # ═══════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════

    def _format_constraints(self, raw: List[Dict], reservoir: Dict) -> List[Dict]:
        """
        将原始约束数据格式化为结构化约束

        输入 (GraphQuery 返回):
          - c.name, c.variable, c.operator, c.value, c.unit (Constraint 节点)
          - type(rel) AS relation
          - d.indicator, d.value, d.unit (AnnualHydrologyData 节点)
        """
        formatted = []
        for i, item in enumerate(raw):
            constr = self._parse_single_constraint(item, i, reservoir)
            if constr:
                formatted.append(constr)
        return formatted

    def _parse_single_constraint(self, item: Dict, idx: int, reservoir: Dict) -> Optional[Dict]:
        """解析单条约束记录"""
        # 情况1: Constraint 节点
        c = item.get("c", {}) or item
        relation_text = str(item.get("relation", ""))
        source_doc = item.get("source_doc") or item.get("source") or ""

        # 情况2: AnnualHydrologyData 节点（用 indicator + value）
        indicator = item.get("indicator", "")
        data_value = item.get("d", {})

        # ── 确定变量名 ──
        variable = c.get("variable", "") if isinstance(c, dict) else ""
        if not variable and indicator:
            variable = indicator
        if not variable:
            variable = relation_text

        # ── 确定运算符 ──
        operator = c.get("operator", "") if isinstance(c, dict) else ""
        if not operator:
            # 从关系名推断
            op_from_rel = self.extract_operator(relation_text)
            if op_from_rel:
                operator = op_from_rel
        if not operator and variable:
            operator = self.extract_operator(str(variable))

        # ── 确定数值和单位 ──
        raw_value = c.get("value", "") if isinstance(c, dict) else ""
        unit = c.get("unit", "") if isinstance(c, dict) else ""

        if not raw_value and data_value and isinstance(data_value, dict):
            raw_value = str(data_value.get("value", ""))
            if not unit:
                unit = data_value.get("unit", "")

        numeric = self.extract_numeric(str(raw_value)) if raw_value else None
        if not numeric and variable:
            # 尝试从变量名中提取（如 "汛限水位275米"）
            numeric = self.extract_numeric(str(variable))

        value = numeric[0] if numeric else None
        if not unit and numeric and numeric[1]:
            unit = numeric[1]

        # ── 确定类别 ──
        category = self.categorize_constraint(relation_text)
        if not category:
            category = self.categorize_constraint(variable)
        if not category and indicator:
            category = self.categorize_constraint(indicator)

        # ── 生成表达 ──
        expression = self._build_expression(variable, operator, value, unit)

        # ── 生成名称 ──
        name = c.get("name", "") if isinstance(c, dict) else ""
        if not name:
            name = relation_text or indicator or variable

        constr_id = f"c_{idx + 1:03d}"
        confidence = item.get("confidence", c.get("confidence", None) if isinstance(c, dict) else None)

        return {
            "id": constr_id,
            "name": str(name),
            "expression": expression,
            "category": category or "unknown",
            "variable": variable,
            "operator": operator or "unknown",
            "value": value,
            "unit": unit,
            "source_doc": str(source_doc) if source_doc else "",
            "source_relation": relation_text,
            "confidence": confidence,
        }

    def _build_expression(
        self,
        variable: str,
        operator: str,
        value: Optional[float],
        unit: Optional[str],
    ) -> str:
        """构建人类可读的数学表达式"""
        if not variable:
            return ""
        unit_str = f" {unit}" if unit else ""
        if value is not None:
            return f"{variable} {operator} {value}{unit_str}"
        return f"{variable} (operator: {operator})"

    def _infer_decision_variables(
        self,
        constraints: List[Dict],
        parameters: Dict,
        reservoir: Dict,
    ) -> List[Dict]:
        """
        从约束类别推断决策变量

        规则:
        - 有 water_level 约束 → 推断水位变量
        - 有 discharge 约束 → 推断流量变量
        - 有 storage 约束 → 推断库容变量
        - 有 power 约束 → 推断出力变量

        边界来源优先级: 约束中的值 > 参数中的值 > None
        """
        seen_categories = set()
        for c in constraints:
            cat = c.get("category", "")
            if cat and cat in self.VARIABLE_INFERENCE:
                seen_categories.add(cat)

        variables = []
        for cat in seen_categories:
            var_def = self.VARIABLE_INFERENCE[cat]
            bounds = self._infer_bounds(cat, constraints, parameters)

            variables.append({
                "name": var_def["name"],
                "symbol": var_def["symbol"],
                "unit": var_def["unit"],
                "bounds": bounds,
                "inference_source": f"从 '{cat}' 类型约束推断",
                "source": "inferred",
            })

        return variables

    def _infer_bounds(
        self,
        category: str,
        constraints: List[Dict],
        parameters: Dict,
    ) -> Dict[str, Optional[float]]:
        """推断变量的上下界"""
        lower = None
        upper = None

        category_constraints = [c for c in constraints if c.get("category") == category]

        for c in category_constraints:
            op = c.get("operator", "")
            val = c.get("value")
            if val is None:
                continue
            if op in (">=",):
                lower = val if lower is None else max(lower, val)
            elif op in ("<=",):
                upper = val if upper is None else min(upper, val)
            elif op in ("==",):
                lower = val
                upper = val

        # 从参数中补充边界
        if category == "water_level":
            dead = parameters.get("dead_storage_level", {})
            normal = parameters.get("normal_storage_level", {})
            flood = parameters.get("flood_control_level", {})
            if lower is None and isinstance(dead, dict):
                lower = dead.get("value")
            if upper is None:
                upper = flood.get("value") if isinstance(flood, dict) else None
            if upper is None:
                upper = normal.get("value") if isinstance(normal, dict) else None

        return {"lower": lower, "upper": upper}

    def _infer_objective_candidates(
        self,
        constraints: List[Dict],
        reservoir: Dict,
    ) -> List[Dict]:
        """从约束类别推断目标函数候选"""
        seen_objectives: Dict[str, Dict] = {}

        for c in constraints:
            name = c.get("name", "") + c.get("variable", "") + c.get("source_relation", "")
            for keyword, candidates in self.OBJECTIVE_CANDIDATES.items():
                if keyword in name:
                    for cand in candidates:
                        desc = cand["description"]
                        if desc not in seen_objectives:
                            seen_objectives[desc] = {**cand, "source": "inferred"}

        return list(seen_objectives.values())

    def _format_parameters(self, raw_params: Dict, reservoir: Dict) -> Dict:
        """
        格式化水库物理参数
        输入可能来自 Reservoir 节点属性 或 Constraint 查询结果
        """
        params = {}

        param_keys = [
            "dead_storage_level",
            "normal_storage_level",
            "flood_control_level",
            "total_capacity",
            "flood_control_capacity",
        ]

        for key in param_keys:
            raw = raw_params.get(key, {})
            if isinstance(raw, dict):
                val = raw.get("value", raw.get(key))
                unit = raw.get("unit", "")
            else:
                val = raw
                unit = ""

            # 尝试提取数值
            if isinstance(val, str):
                numeric = self.extract_numeric(val)
                if numeric:
                    val = numeric[0]
                    if not unit:
                        unit = numeric[1] or ""

            params[key] = {
                "value": val if val is not None else None,
                "unit": unit,
            }

        return params

    def _format_time_series(self, raw: List[Dict]) -> Dict:
        """格式化水文时间序列"""
        entries = []
        for item in raw:
            d = item.get("d", {}) if isinstance(item.get("d"), dict) else item
            indicator = d.get("indicator", item.get("indicator", ""))
            value = d.get("value", item.get("value", ""))
            unit = d.get("unit", item.get("unit", ""))
            year = d.get("year", item.get("year", ""))

            entry = {
                "label": f"{year}",
                "year": year,
                "indicator": indicator,
                "value": value,
                "unit": unit,
                "relation": item.get("relation", ""),
                "source_doc": item.get("source_doc", ""),
            }
            entries.append(entry)

        return {"entries": entries, "count": len(entries)}

    def _format_dispatch_rules(self, raw: List[Dict]) -> List[Dict]:
        """格式化调度规则"""
        rules = []
        for item in raw:
            rule = item.get("rule", item.get("r", {}))
            if isinstance(rule, dict):
                rules.append({
                    "name": rule.get("name", ""),
                    "content": rule.get("content", ""),
                    "applicable_period": rule.get("applicable_period", ""),
                    "condition": rule.get("condition", ""),
                    "source_doc": item.get("source", ""),
                    "confidence": item.get("confidence", None),
                })
            else:
                rules.append({
                    "name": str(item.get("name", item.get("relation", ""))),
                    "content": str(rule) if rule else "",
                })
        return rules
