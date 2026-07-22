"""
三元组规范化器
将原始三元组转换为规范 Schema：实体标准化 + 关系对齐 + 对象分类 + 年份提取 + 字段清洗

设计原则:
    1. 规则优先 — 高频关系走精确/关键词规则，零 API 调用
    2. LLM 兜底 — 低频新关系走 RelationCanonicalizer
    3. 先标准化、后清洗 — 不在抽取阶段做标准化，保证管道可回放
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.knowledge_fusion.entity_linking import EntityLinker

logger = get_logger(__name__)

# ============================================================
# 关系标准化规则库
# ============================================================

# 精确匹配规则: relation_text -> (relation_id, is_annual)
RELATION_RULES: Dict[str, Tuple[str, bool]] = {
    # ---- 径流量 ----
    "实测径流量为": ("ANNUAL_RUNOFF", True),
    "当年实测径流量为": ("ANNUAL_RUNOFF", True),
    "2024年实测径流量为": ("ANNUAL_RUNOFF", True),
    "上年实测径流量为": ("ANNUAL_RUNOFF", True),
    "当年天然河川径流量为": ("ANNUAL_RUNOFF", True),
    "天然河川径流量为": ("ANNUAL_RUNOFF", True),
    "当年7~10月实测径流量为": ("ANNUAL_RUNOFF_FLOOD_SEASON", True),
    # ---- 径流量基准值 ----
    "1987~2016年实测径流量均值为": ("LONG_TERM_AVG_RUNOFF", False),
    "1956~2016年实测径流量均值为": ("LONG_TERM_AVG_RUNOFF", False),
    # ---- 输沙量 ----
    "实测输沙量为": ("ANNUAL_SEDIMENT", True),
    "当年实测输沙量为": ("ANNUAL_SEDIMENT", True),
    "2024年实测年输沙量为": ("ANNUAL_SEDIMENT", True),
    "合计实测年输沙量为": ("ANNUAL_SEDIMENT", True),
    # ---- 降水量 ----
    "上年降水量为": ("ANNUAL_PRECIPITATION", True),
    "当年降水量（毫米）为": ("ANNUAL_PRECIPITATION", True),
    "当年降水量（亿立方米）为": ("ANNUAL_PRECIPITATION", True),
    "2024年降水量（亿立方米）为": ("ANNUAL_PRECIPITATION", True),
    "2024年降水量为": ("ANNUAL_PRECIPITATION", True),
    "1987~2016年平均降水量为": ("LONG_TERM_AVG_PRECIPITATION", False),
    "1956~2016年平均降水量为": ("LONG_TERM_AVG_PRECIPITATION", False),
    # ---- 供水 ----
    "供水量流域内为": ("WATER_SUPPLY", True),
    "供水量调出流域为": ("WATER_SUPPLY", True),
    "供水量合计为": ("WATER_SUPPLY", True),
    "黄河地表水供水量为": ("WATER_SUPPLY", True),
    "地表水源流域内供水量为": ("WATER_SUPPLY", True),
    "地表水源调出流域供水量为": ("WATER_SUPPLY", True),
    "地表水源合计供水量为": ("WATER_SUPPLY", True),
    "地下水源供水量为": ("WATER_SUPPLY", True),
    "非常规水源供水量为": ("WATER_SUPPLY", True),
    "合计供水量为": ("WATER_SUPPLY", True),
    "总计供水量为": ("WATER_SUPPLY", True),
    "调入流水量为": ("WATER_SUPPLY", True),
    "2024年供水量为": ("WATER_SUPPLY", True),
    # ---- 用水 ----
    "用水量合计总量为": ("WATER_USE", True),
    "用水量农业总量为": ("WATER_USE", True),
    "2024年用水总量为": ("WATER_USE", True),
    "2024年地下水用水量为": ("WATER_USE", True),
    # ---- 耗水 ----
    "耗水量流域内为": ("WATER_CONSUMPTION", True),
    "耗水量调出流域为": ("WATER_CONSUMPTION", True),
    "耗水量合计为": ("WATER_CONSUMPTION", True),
    "耗水量合计总量为": ("WATER_CONSUMPTION", True),
    "黄河地表水耗水量为": ("WATER_CONSUMPTION", True),
    "2024年耗水总量为": ("WATER_CONSUMPTION", True),
    "2024年地下水耗水量为": ("WATER_CONSUMPTION", True),
    # ---- 水库蓄水 ----
    "年初蓄水量为": ("RESERVOIR_STORAGE_START", True),
    "年末蓄水量为": ("RESERVOIR_STORAGE_END", True),
    "年蓄水变量为": ("RESERVOIR_STORAGE_CHANGE", True),
    "2024年年初蓄水量为": ("RESERVOIR_STORAGE_START", True),
    "2024年年末蓄水量为": ("RESERVOIR_STORAGE_END", True),
    "2024年年蓄水变量为": ("RESERVOIR_STORAGE_CHANGE", True),
    "2023年年末蓄水量为": ("RESERVOIR_STORAGE_END", True),
    "合计蓄水变量为": ("RESERVOIR_STORAGE_CHANGE", True),
    "合计水库年蓄水变量为": ("RESERVOIR_STORAGE_CHANGE", True),
    # ---- 地下水 ----
    "平均地下水埋深年初": ("GROUNDWATER_DEPTH_START", True),
    "平均地下水埋深年末": ("GROUNDWATER_DEPTH_END", True),
    "平均地下水埋深年变幅": ("GROUNDWATER_DEPTH_CHANGE", True),
    "年初平均地下水埋深为": ("GROUNDWATER_DEPTH_START", True),
    "年末平均地下水埋深为": ("GROUNDWATER_DEPTH_END", True),
    "年变幅为": ("GROUNDWATER_DEPTH_CHANGE", True),
    "平均地下水埋深年变幅为": ("GROUNDWATER_DEPTH_CHANGE", True),
    # ---- 固有属性 ----
    "干流全长为": ("RIVER_LENGTH", False),
    "流域总面积为": ("BASIN_AREA", False),
    "控制面积为": ("BASIN_AREA", False),
    "计算面积为": ("BASIN_AREA", False),
    "面积为": ("HAS_AREA", False),
    "2024年面积为": ("HAS_AREA", False),
    "范围为": ("HAS_SCOPE", False),
    "供水区范围为": ("HAS_SCOPE", False),
    # ---- 结构关系 ----
    "所属行政区为": ("BELONGS_TO_PROVINCE", False),
    "所属水资源二级区为": ("LOCATED_IN_ZONE", False),
    "所属平原子": ("LOCATED_IN_PLAIN", False),
    "主要水文站包括": ("HAS_STATION", False),
    "重要支流控制水文站包括": ("HAS_STATION", False),
    "包括": ("INCLUDES", False),
    "属于": ("BELONGS_TO", False),
    "包含超采区序号": ("HAS_OVERDRAFT_ZONE", False),
    # ---- 地下水属性 ----
    "地下水类型为": ("GROUNDWATER_TYPE", False),
    # ---- 比较关系 (与上年/基准比较) ----
    "与上年比较为": ("COMPARE_LAST_YEAR", True),
    "与上年比较变化为": ("COMPARE_LAST_YEAR", True),
    "与1987~2016年均值比较为": ("COMPARE_LONG_TERM_AVG", False),
    "与1956~2016年均值比较为": ("COMPARE_LONG_TERM_AVG", False),
    # ---- 水资源总量 ----
    "水资源总量为": ("WATER_RESOURCE_TOTAL", True),
    "2024年水资源总量为": ("WATER_RESOURCE_TOTAL", True),
    # ---- 引水/入海 ----
    "2024年入海水量为": ("ANNUAL_RUNOFF_TO_SEA", True),
    # ---- 地下水水位变化面积 ----
    "平原区浅层地下水水位上升面积为": ("GW_LEVEL_RISE_AREA", True),
    "浅层地下水水位下降面积为": ("GW_LEVEL_DECLINE_AREA", True),
    "深层承压水水位上升面积为": ("GW_LEVEL_RISE_AREA", True),
    # ---- 水位参数 ----
    "死水位为": ("DEAD_STORAGE_LEVEL", False),
    "防洪限制水位为": ("FLOOD_CONTROL_LEVEL", False),
    "正常蓄水位为": ("NORMAL_STORAGE_LEVEL", False),
    # ---- 水库基本属性 ----
    "总库容为": ("TOTAL_CAPACITY", False),
    "装机容量为": ("POWER_GENERATION", False),
    "控制流域面积为": ("BASIN_AREA", False),
    "坝址以上流域面积为": ("BASIN_AREA", False),
    "所在河流为": ("LOCATED_ON", False),
    "所在河段为": ("LOCATED_ON", False),
    "坝址位于": ("LOCATED_ON", False),
    "所在地区为": ("BELONGS_TO", False),
    # ---- 变更关系 ----
    "自2023年起更名为": ("STATION_RENAMED_TO", False),
}

# 关键词模糊匹配: ([keywords], relation_id, is_annual)
# 注意：精确匹配优先于关键词匹配
KEYWORD_RULES: List[Tuple[List[str], Optional[str], bool]] = [
    (["径流量", "径流"], "ANNUAL_RUNOFF", True),
    (["输沙量", "输沙"], "ANNUAL_SEDIMENT", True),
    (["降水量", "降水"], "ANNUAL_PRECIPITATION", True),
    (["供水量", "供水"], "WATER_SUPPLY", True),
    (["用水量", "用水"], "WATER_USE", True),
    (["耗水量", "耗水"], "WATER_CONSUMPTION", True),
    # ---- 水位/库容规则（必须在通用"蓄水"前，避免"正常蓄水"被"蓄水"吃掉） ----
    (["死水位", "死库容"], "DEAD_STORAGE_LEVEL", False),
    (["防洪限制", "汛限", "防洪限制水位"], "FLOOD_CONTROL_LEVEL", False),
    (["正常蓄水", "正常蓄水位"], "NORMAL_STORAGE_LEVEL", False),
    (["蓄水位", "水位"], None, False),  # 水位兜底（蓄水位必须在"水位"前）
    (["防洪库容"], "FLOOD_CONTROL_CAPACITY", False),
    (["库容", "总库容"], "TOTAL_CAPACITY", False),
    # ---- 通用蓄水/用水规则 ----
    (["蓄水量", "蓄水变量"], "RESERVOIR_STORAGE", True),
    (["地下水埋深", "埋深"], "GROUNDWATER_DEPTH", True),
    (["流域面积", "控制面积", "集水面积"], "BASIN_AREA", False),
    (["面积"], "HAS_AREA", False),
    (["长度", "全长"], "RIVER_LENGTH", False),
    (["水质", "水质类别"], "WATER_QUALITY", False),
    (["发电", "出力", "装机"], "POWER_GENERATION", False),
    (["生态", "环境"], "ECOLOGICAL_FLOW", False),
    (["改为", "更名为", "改名"], "STATION_RENAMED_TO", False),
    (["属于", "所属", "所在地区"], "BELONGS_TO", False),
    (["岸所属行政区", "左岸", "右岸", "南岸", "北岸"], "BELONGS_TO", False),
    (["水文站包括", "水文站为"], "HAS_STATION", False),
    (["包括", "包含", "划分为"], "INCLUDES", False),
    (["比较", "对比", "变化", "增幅", "减少", "偏少", "增加了"], "COMPARISON", True),
]

# 无效 object 值集合
INVALID_OBJECTS = {"", "/", "1", "一", "-", "—", "无", "null", "None", "无数据"}


def extract_year_from_relation(relation: str) -> Optional[int]:
    """从关系文本中提取数据年份（排除基准期范围如 1987~2016年）"""
    # 排除范围年份模式: "1987~2016年" "1956~2016年" 等
    # 先移除范围年份，再匹配单年份
    cleaned = re.sub(r'\d{4}~\d{4}年', '', relation)
    match = re.search(r'(?<!~)(\d{4})年', cleaned)
    if match:
        return int(match.group(1))
    return None


def extract_year_from_source(source_file: str) -> Optional[int]:
    """从文件名中提取年份"""
    match = re.search(r'(\d{4})', source_file)
    if match:
        return int(match.group(1))
    return None


def classify_object(obj: str, entity_linker: Optional[EntityLinker] = None) -> str:
    """
    分类 object 类型:
        - "numerical_value": 纯数值+单位
        - "entity_reference": 已知实体引用
        - "composite": 逗号/顿号分隔的多实体列表
    """
    obj = obj.strip()
    if not obj:
        return "invalid"

    # 复合检测：包含中文顿号或逗号分隔的多个名称
    # "龙羊峡以上、龙羊峡至兰州、兰州至头道拐"
    if re.search(r'[、，,]', obj):
        parts = re.split(r'[、，,]', obj)
        # 如果大部分部分不含数字，则可能是实体列表
        non_numeric_parts = [p for p in parts if not re.search(r'\d', p)]
        if len(non_numeric_parts) >= 2:
            return "composite"

    # 数值检测：包含数字 + 常见单位
    if re.search(r'\d+\.?\d*', obj):
        return "numerical_value"

    # 实体引用检测
    if entity_linker:
        result = entity_linker.link_entity(obj)
        if result["matched"] and result["confidence"] >= 0.9:
            return "entity_reference"

    # 名称模式检测（无 linker 时）
    if any(kw in obj for kw in ["水库", "水文站", "省", "自治区", "市", "县", "河", "流域", "区"]):
        return "entity_reference"

    return "numerical_value"


def has_valid_number(obj: str) -> bool:
    """检查 object 是否包含有效数值"""
    return bool(re.search(r'\d+\.?\d*', obj))


# ============================================================
# 规范化器主类
# ============================================================

class TripletNormalizer:
    """三元组规范化器：实体→关系→对象 逐层标准化"""

    def __init__(
        self,
        entity_linker: Optional[EntityLinker] = None,
        canonicalizer: Optional[Any] = None,  # RelationCanonicalizer, 可选
        use_llm_fallback: bool = False,
    ):
        """
        Args:
            entity_linker: 实体链接器（复用已有实例）
            canonicalizer: 关系标准化器（LLM 兜底，可选）
            use_llm_fallback: 是否启用 LLM 兜底（对未匹配的关系调用 API）
        """
        self.entity_linker = entity_linker
        self.canonicalizer = canonicalizer
        self.use_llm_fallback = use_llm_fallback
        self.stats = {
            "total": 0,
            "entities_normalized": 0,
            "entity_new": 0,
            "relations_matched_exact": 0,
            "relations_matched_keyword": 0,
            "relations_unmatched": 0,
            "objects_classified": {"numerical_value": 0, "entity_reference": 0, "composite": 0, "invalid": 0},
            "triplets_split": 0,
            "triplets_invalid": 0,
        }

    def normalize(self, triplets: List[Dict]) -> List[Dict]:
        """批量规范化三元组"""
        self.stats["total"] = len(triplets)
        normalized = []

        for t in triplets:
            results = self._normalize_one(t)
            normalized.extend(results)

        # 去重：同一主体+同年份+同数值的多条关系，保留年份显式的那个
        before_dedup = len(normalized)
        normalized = self._deduplicate(normalized)
        self.stats["triplets_deduped"] = before_dedup - len(normalized)

        logger.info(
            f"规范化完成: {len(triplets)} -> {len(normalized)} 条 "
            f"(拆分+{self.stats['triplets_split']}, "
            f"无效-{self.stats['triplets_invalid']}, "
            f"去重-{self.stats.get('triplets_deduped', 0)}, "
            f"实体新-{self.stats['entity_new']}, "
            f"关系匹配: 精确{self.stats['relations_matched_exact']}/"
            f"关键词{self.stats['relations_matched_keyword']}/"
            f"未匹配{self.stats['relations_unmatched']})"
        )
        return normalized

    def _deduplicate(self, triplets: List[Dict]) -> List[Dict]:
        """对同一subject+year+object_value的多条关系去重"""
        groups = {}
        for t in triplets:
            key = (t["subject"], str(t["year"]), t["object"])
            if key not in groups:
                groups[key] = []
            groups[key].append(t)

        result = []
        for key, group in groups.items():
            if len(group) == 1:
                result.append(group[0])
            else:
                # 保留最好的那条：优先选 relation 中含年份的，其次选 confidence 高的
                best = max(group, key=lambda t: (
                    1 if re.search(r'\d{4}年', t.get("relation", "")) else 0,
                    t.get("confidence", 0),
                ))
                result.append(best)

        return result

    def _normalize_one(self, triplet: Dict) -> List[Dict]:
        """规范化单条三元组，可能返回多条（拆分后）"""
        # 如果已经规范化过（含 subject_type + relation_id 键），直接返回，
        # 避免二次调用时 confidence 被重复相乘
        if "relation_id" in triplet and "subject_type" in triplet:
            return [triplet]

        # Step 1: 实体标准化
        subj_info = self._normalize_entity(triplet.get("subject", ""))
        obj_info = self._normalize_object_entity(triplet.get("object", ""))

        # Step 2: 关系标准化
        rel_info = self._standardize_relation(triplet.get("relation", ""))

        # Step 3: 年份提取
        year = self._determine_year(triplet, rel_info)

        # Step 4: Object 分类
        obj_type = classify_object(obj_info["standardized"], self.entity_linker)
        self.stats["objects_classified"][obj_type] += 1

        # Step 5: 脏数据清洗
        obj_val = obj_info["standardized"].strip()
        if obj_val in INVALID_OBJECTS:
            self.stats["triplets_invalid"] += 1
            self.stats["objects_classified"]["invalid"] += 1
            return []

        # Step 6: 综合置信度
        confidence = float(triplet.get("confidence", 0.8)) * float(rel_info.get("confidence", 0.8))

        # Step 7: 构建规范三元组
        base = {
            "subject": subj_info["standardized"],
            "subject_type": subj_info["entity_type"],
            "relation": triplet.get("relation", ""),
            "relation_id": rel_info.get("match_result") if rel_info.get("match_result") != "NEW_RELATION" else None,
            "object": obj_val,
            "object_type": obj_type,
            "year": year,
            "context": triplet.get("context", ""),
            "confidence": round(confidence, 3),
            "source_file": triplet.get("source_file", ""),
        }

        # Step 8: 复合对象拆分
        if obj_type == "composite":
            return self._split_composite(base)
        return [base]

    def _normalize_entity(self, name: str) -> Dict:
        """实体标准化：别名匹配 + 关键词兜底"""
        name = name.strip()
        if not name:
            return {"original_name": name, "standardized": name, "entity_type": "Unknown", "is_new": True}

        # 先用 entity_linker
        if self.entity_linker:
            result = self.entity_linker.link_entity(name)
            if result["matched"]:
                self.stats["entities_normalized"] += 1
                # 从 alias_dict 中反查类型
                entity_type = self._guess_entity_type(result["standardized"])
                return {
                    "original_name": name,
                    "standardized": result["standardized"],
                    "entity_type": entity_type,
                    "is_new": False,
                    "confidence": result["confidence"],
                }

        # 关键词兜底
        entity_type = self._guess_entity_type(name)
        is_new = entity_type in ("Unknown", "GroundwaterOverdraftArea", "StatisticAggregate")
        if is_new:
            self.stats["entity_new"] += 1
        else:
            self.stats["entities_normalized"] += 1

        return {
            "original_name": name,
            "standardized": name,
            "entity_type": entity_type,
            "is_new": is_new,
            "confidence": 0.7 if not is_new else 0.5,
        }

    def _normalize_object_entity(self, name: str) -> Dict:
        """Object 的实体标准化（轻量版）"""
        name = (name or "").strip()
        if not name:
            return {"original_name": name, "standardized": name, "entity_type": None}

        # 如果是纯数值，不需要标准化
        if re.search(r'\d+\.?\d*', name) and len(name) < 50:
            return {"original_name": name, "standardized": name, "entity_type": "numerical_value"}

        # 如果可能是实体，复用 entity_linker
        if self.entity_linker:
            result = self.entity_linker.link_entity(name)
            if result["matched"]:
                return {
                    "original_name": name,
                    "standardized": result["standardized"],
                    "entity_type": self._guess_entity_type(result["standardized"]),
                }

        return {"original_name": name, "standardized": name, "entity_type": None}

    def _standardize_relation(self, relation: str) -> Dict:
        """关系标准化：精确匹配 → 关键词匹配 → LLM兜底"""
        relation = relation.strip()
        if not relation:
            return {"match_result": None, "confidence": 0.0, "is_annual": False, "method": "none"}

        # Level 1: 精确匹配
        if relation in RELATION_RULES:
            rel_id, is_annual = RELATION_RULES[relation]
            self.stats["relations_matched_exact"] += 1
            return {
                "match_result": rel_id,
                "confidence": 0.99,
                "is_annual": is_annual,
                "method": "exact",
            }

        # Level 2: 关键词模糊匹配
        for keywords, rel_id, is_annual in KEYWORD_RULES:
            for kw in keywords:
                if kw in relation:
                    self.stats["relations_matched_keyword"] += 1
                    return {
                        "match_result": rel_id,
                        "confidence": 0.85,
                        "is_annual": is_annual,
                        "method": f"keyword:{kw}",
                    }

        # Level 3: LLM 兜底（可选）
        if self.use_llm_fallback and self.canonicalizer:
            try:
                result = self.canonicalizer.canonicalize(relation, relation)
                self.stats["relations_unmatched"] += 1
                return {
                    "match_result": result.get("match_result"),
                    "confidence": result.get("confidence", 0.5),
                    "is_annual": "径流" in relation or "量" in relation,
                    "method": "llm",
                }
            except Exception:
                pass

        # 兜底: 基于启发式判断
        self.stats["relations_unmatched"] += 1
        is_annual = any(kw in relation for kw in ["径流", "输沙", "降水", "供水", "用水", "耗水", "蓄水", "埋深", "比较", "变化"])
        return {
            "match_result": None,
            "confidence": 0.4,
            "is_annual": is_annual,
            "method": "heuristic",
        }

    def _determine_year(self, triplet: Dict, rel_info: Dict) -> Any:
        """
        确定三元组的年份:
        1. 优先从 relation 文本中提取
        2. 兜底从 source_file 文件名提取
        3. 恒定属性填 "constant"
        """
        relation = triplet.get("relation", "")

        # 尝试从 relation 提取年份
        year = extract_year_from_relation(relation)
        if year is not None:
            return year

        # 如果是"上年"模式
        if "上年" in relation:
            source_year = extract_year_from_source(triplet.get("source_file", ""))
            if source_year:
                return source_year - 1

        # 如果明确是年度数据但无具体年份，从文件名取
        if rel_info.get("is_annual"):
            source_year = extract_year_from_source(triplet.get("source_file", ""))
            if source_year:
                return source_year

        # 默认：恒定属性
        return "constant"

    def _guess_entity_type(self, name: str) -> str:
        """根据名称关键词推断实体类型"""
        name = name.strip()

        # 精确匹配已知实体
        if name in ("合计", "总计"):
            return "StatisticAggregate"

        # 复合合计: "小浪底+黑石关+武陟合计" 等
        if "+" in name and any(kw in name for kw in ["合计", "总计"]):
            return "StatisticAggregate"

        # 地下水超采区
        if "超采区" in name:
            return "GroundwaterOverdraftArea"

        # 地下水盆地/平原/台地/川/塬
        if any(kw in name for kw in ["盆地", "平原", "台地", "河谷", "风沙滩", "谷地",
                                       "景泰川", "董志塬"]):
            return "GroundwaterRegion"

        # Reservoir
        if any(kw in name for kw in ["水库", "水电站", "水利枢纽"]):
            return "Reservoir"

        # HydrologicalStation
        if "水文站" in name:
            return "HydrologicalStation"

        # 支流水文站简称（如 "折桥" = 大夏河折桥水文站）
        # 无法从名字推断，但如果有 "站" 字
        if name.endswith("站"):
            return "HydrologicalStation"

        # Province
        if any(kw in name for kw in ["省", "自治区"]):
            return "Province"

        # 省份简称
        if name in ("青海", "四川", "甘肃", "宁夏", "内蒙古", "山西", "陕西", "河南", "山东", "河北"):
            return "Province"

        # River
        if any(kw in name for kw in ["河", "江", "水系"]):
            return "River"

        # WaterResourceZone
        if any(kw in name for kw in ["以上", "以下", "至", "区间", "内流区"]):
            return "WaterResourceZone"

        # 流域
        if "流域" in name:
            return "River"

        # Document / Organization
        if any(kw in name for kw in ["水利部", "黄河水利委员会", "公报", "方案", "规划", "规程"]):
            return "Document"

        # 默认
        return "Unknown"

    def _split_composite(self, triplet: Dict) -> List[Dict]:
        """拆分逗号/顿号分隔的复合 object 为多条三元组"""
        obj = triplet["object"]
        parts = re.split(r'[、，,]', obj)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) <= 1:
            return [triplet]

        results = []
        for part in parts:
            # 对每个拆分后的部分做实体标准化
            obj_info = self._normalize_object_entity(part)
            new_t = dict(triplet)
            new_t["object"] = obj_info["standardized"]
            new_t["object_type"] = classify_object(obj_info["standardized"], self.entity_linker)
            results.append(new_t)

        self.stats["triplets_split"] += len(results) - 1
        return results
