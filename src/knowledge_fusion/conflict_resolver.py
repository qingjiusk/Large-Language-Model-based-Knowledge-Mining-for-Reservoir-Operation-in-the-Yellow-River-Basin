"""
知识冲突检测与处理模块
处理表格与正文之间的数值冲突，支持跨年份数据区分
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger

logger = get_logger(__name__)


class ConflictResolver:
    """知识冲突检测与裁决器"""

    def __init__(
        self,
        table_priority: bool = True,
        enable_year_distinction: bool = True,
        numeric_tolerance: float = 0.05,
    ):
        """
        初始化冲突检测器

        Args:
            table_priority: 当表格与正文冲突时，表格来源默认优先级更高
            enable_year_distinction: 是否区分不同年份的同指标数据
            numeric_tolerance: 数值冲突容忍度（相对误差，0.05 = 5%）
        """
        self.table_priority = table_priority
        self.enable_year_distinction = enable_year_distinction
        self.numeric_tolerance = numeric_tolerance

    def resolve(
        self,
        triplets: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        检测并解决三元组中的知识冲突

        Args:
            triplets: 三元组列表（含 data_type, confidence 等字段）

        Returns:
            (resolved_triplets, conflicts) — 解决后的三元组和冲突报告
        """
        if not triplets:
            return [], []

        # Step 1: 按 (subject, relation) 分组
        groups: Dict[Tuple[str, str], List[Dict]] = {}
        for t in triplets:
            key = self._make_key(t)
            if key not in groups:
                groups[key] = []
            groups[key].append(t)

        # Step 2: 检测每组内的冲突
        resolved = []
        conflicts = []

        for key, group in groups.items():
            if len(group) == 1:
                resolved.append(group[0])
                continue

            # 检查是否涉及数值
            numeric_entries = self._filter_numeric(group)
            if len(numeric_entries) <= 1:
                resolved.extend(group)
                continue

            # 按年份分组
            year_groups = self._group_by_year(numeric_entries)

            if self.enable_year_distinction and len(year_groups) > 1:
                # 不同年份，不冲突
                resolved.extend(group)
            else:
                # 同一年份或无年份信息，检测冲突
                clean, group_conflicts = self._resolve_numeric_conflict(group)
                resolved.extend(clean)
                conflicts.extend(group_conflicts)

        logger.info(
            f"冲突检测完成: {len(triplets)} 条输入, "
            f"{len(conflicts)} 个冲突, "
            f"{len(resolved)} 条输出"
        )
        return resolved, conflicts

    def _make_key(self, triplet: Dict) -> Tuple[str, str]:
        """生成分组 key"""
        subject = triplet.get("subject", "").strip().lower()
        relation = triplet.get("relation", "").strip().lower()
        # 移除关系中的年份前缀（如"2024年实测径流量" → "实测径流量"）
        if self.enable_year_distinction:
            relation = re.sub(r'^\d{4}年', '', relation)
        return (subject, relation)

    def _filter_numeric(self, triplets: List[Dict]) -> List[Dict]:
        """筛选包含数值的三元组"""
        numeric = []
        for t in triplets:
            obj = str(t.get("object", ""))
            if self._has_number(obj):
                numeric.append(t)
        return numeric

    def _has_number(self, text: str) -> bool:
        """判断文本是否包含数值"""
        # 匹配带单位的数值：362.80亿立方米, 275米, 75.5%
        return bool(re.search(r'\d+\.?\d*', text))

    def _extract_number(self, text: str) -> Optional[float]:
        """提取文本中的第一个数值"""
        match = re.search(r'(\d+\.?\d*)', str(text))
        if match:
            return float(match.group(1))
        return None

    def _group_by_year(self, triplets: List[Dict]) -> Dict[int, List[Dict]]:
        """按年份分组"""
        groups = {}
        for t in triplets:
            year = self._extract_year(t)
            if year not in groups:
                groups[year] = []
            groups[year].append(t)
        return groups

    def _extract_year(self, triplet: Dict) -> int:
        """从三元组中提取年份"""
        # 优先从 relation 中提取
        rel = triplet.get("relation", "")
        match = re.search(r'(\d{4})年', rel)
        if match:
            return int(match.group(1))

        # 其次从 context 中提取
        ctx = triplet.get("context", "")
        match = re.search(r'(\d{4})年', ctx)
        if match:
            return int(match.group(1))

        # 默认为 0（未知年份）
        return 0

    def _resolve_numeric_conflict(
        self,
        group: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        解决同组内的数值冲突
        优先级: 表格 > 正文, 置信度高 > 置信度低
        """
        if len(group) == 1:
            return group, []

        # 计算每个条目的数值
        entries_with_num = []
        for t in group:
            num = self._extract_number(t.get("object", ""))
            if num is not None:
                entries_with_num.append((t, num))

        if len(entries_with_num) <= 1:
            return group, []

        # 检查数值是否一致（在容忍度内）
        nums = [n for _, n in entries_with_num]
        if self._all_close(nums):
            # 数值一致，全部保留
            return group, []

        # 冲突：选择最优
        winner = self._pick_winner(entries_with_num)

        conflicts = []
        for t, num in entries_with_num:
            if t is not winner:
                conflicts.append({
                    "type": "numeric_conflict",
                    "subject": t.get("subject"),
                    "relation": t.get("relation"),
                    "winner_value": winner.get("object"),
                    "loser_value": t.get("object"),
                    "winner_source": winner.get("data_type", "unknown"),
                    "loser_source": t.get("data_type", "unknown"),
                    "resolution": "table_priority" if self.table_priority else "confidence_based",
                })

        # 保留 winner + 所有非数值条目（它们不参与冲突）
        winner_obj = str(winner.get("object", ""))
        resolved = [t for t in group if t is winner or not self._has_number(str(t.get("object", "")))]
        logger.debug(
            f"数值冲突已裁决: {winner.get('subject')} - {winner.get('relation')}, "
            f"保留值={winner.get('object')}"
        )

        return resolved, conflicts

    def _all_close(self, nums: List[float]) -> bool:
        """检查所有数值是否在容忍范围内一致"""
        if not nums:
            return True
        mean_val = sum(nums) / len(nums)
        if mean_val == 0:
            return all(n == 0 for n in nums)
        return all(abs(n - mean_val) / abs(mean_val) <= self.numeric_tolerance for n in nums)

    def _pick_winner(self, entries_with_num: List[Tuple[Dict, float]]) -> Dict:
        """选择最优条目"""
        # 按优先级排序: data_type (table > text) > confidence (高 > 低)
        def sort_key(item):
            t, num = item
            type_score = 0 if t.get("data_type") == "tabular" else 1
            conf_score = 1.0 - t.get("confidence", 0.5)
            return (type_score, conf_score)

        entries_with_num.sort(key=sort_key)
        return entries_with_num[0][0]
