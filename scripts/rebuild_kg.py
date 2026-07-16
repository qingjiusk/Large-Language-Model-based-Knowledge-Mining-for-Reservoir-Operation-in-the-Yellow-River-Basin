#!/usr/bin/env python
"""
从规范化三元组重建 Neo4j 知识图谱
- 使用干净 ID（无中文）
- 写回 subject_type 作为节点标签
- 关系存储中文原名 + 标准 relation_id + year
- 支持全量重建（清空旧数据）

用法:
    python scripts/rebuild_kg.py
    python scripts/rebuild_kg.py --input data/processed/黄河水资源2024_triplets_normalized.json
    python scripts/rebuild_kg.py --mode append   # 增量追加
    python scripts/rebuild_kg.py --dry-run       # 仅验证，不写库
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config_loader import ConfigLoader
from src.common.logger import get_logger, setup_logging
from src.knowledge_graph.neo4j_client import Neo4jClient

logger = get_logger(__name__)


# ============================================================
# 从 relation / subject_type 推断 object 的节点标签
# ============================================================
# 从 relation_id 推断 object 标签（仅用于语义确定的映射）
# 注意：不包含 INCLUDES/BELONGS_TO 等宽泛关系 — 它们的对象类型应通过名称推断
OBJECT_LABEL_HINTS = {
    "BELONGS_TO_PROVINCE": "Province",
    "LOCATED_IN_ZONE": "WaterResourceZone",
    "LOCATED_IN_PLAIN": "GroundwaterRegion",
    "STATION_RENAMED_TO": "HydrologicalStation",
    "LOCATED_ON": "River",
    "IS_TRIBUTARY_OF": "River",
    "HAS_OVERDRAFT_ZONE": "GroundwaterOverdraftArea",
    "HAS_STATION": "HydrologicalStation",
    "SOURCE_FROM": "Document",
}


def _guess_name_label(name: str) -> Optional[str]:
    """根据实体名称推断 Neo4j 节点标签。返回 None 表示无法推断。"""
    name = name.strip()
    # Reservoir
    if any(kw in name for kw in ["水库", "水电站", "水利枢纽"]):
        return "Reservoir"
    # HydrologicalStation — 水文站全称
    if "水文站" in name:
        return "HydrologicalStation"
    # Province
    if any(kw in name for kw in ["省", "自治区"]):
        return "Province"
    # 省份简称
    if name in ("青海", "四川", "甘肃", "宁夏", "内蒙古", "山西", "陕西", "河南", "山东", "河北"):
        return "Province"
    # River
    if any(kw in name for kw in ["河", "江"]):
        return "River"
    # WaterResourceZone — 必须是包含 以上/以下/至/区间/内流区
    if any(kw in name for kw in ["以上", "以下"]):
        return "WaterResourceZone"
    if "至" in name and len(name) > 3:
        return "WaterResourceZone"
    if name in ("内流区",):
        return "WaterResourceZone"
    # GroundwaterOverdraftArea
    if "超采区" in name:
        return "GroundwaterOverdraftArea"
    # GroundwaterRegion
    if any(kw in name for kw in ["盆地", "平原", "台地", "河谷", "风沙滩", "谷地", "川", "塬"]):
        return "GroundwaterRegion"
    # StatisticAggregate
    if name in ("合计", "总计"):
        return "StatisticAggregate"
    if "+" in name and any(kw in name for kw in ["合计", "总计"]):
        return "StatisticAggregate"
    # Document
    if any(kw in name for kw in ["水利部", "黄河水利委员会", "公报", "方案", "规划", "规程"]):
        return "Document"
    # 流域 → River
    if "流域" in name:
        return "River"
    # 以"区"结尾但不是"区"单独的 → 可能是 WaterResourceZone
    if name.endswith("区") and len(name) > 1:
        return "WaterResourceZone"
    return None


def _guess_object_label(
    obj: str,
    obj_type: str,
    relation_id: Optional[str],
    year: str,
    relation: str,
) -> Tuple[str, Dict]:
    """
    推断 object 的节点标签和属性。

    Returns:
        (label, properties_dict)
    """
    # 数值型对象
    if obj_type == "numerical_value":
        # 提取数值和单位
        num_match = re.search(r'(\d+\.?\d*)', obj)
        value = float(num_match.group(1)) if num_match else None
        unit_match = re.sub(r'[\d.]+', '', obj).strip()

        if year != "constant":
            # 年度数据 → AnnualHydrologyData
            props = {
                "name": obj,       # 显示名 = 数值+单位
                "value": obj,
                "year": year,
                "indicator": relation,
            }
            if unit_match:
                props["unit"] = unit_match
            return "AnnualHydrologyData", props
        else:
            # 恒定属性 → Constraint
            props = {
                "name": obj,       # 显示名 = 数值+单位
                "value": obj,
                "variable": relation,
            }
            if value is not None:
                props["numeric_value"] = value
            if unit_match:
                props["unit"] = unit_match
            return "Constraint", props

    # 实体引用对象
    if obj_type == "entity_reference":
        # Step 1: 名称关键词推断（最可靠）
        label = _guess_name_label(obj)
        if label:
            return label, {"name": obj}

        # Step 2: relation_id 兜底（仅对语义确定的映射）
        if relation_id and relation_id in OBJECT_LABEL_HINTS:
            return OBJECT_LABEL_HINTS[relation_id], {"name": obj}

        # 兜底
        return "Constraint", {"name": obj, "value": obj}

    # 复合或其他
    return "Constraint", {"name": obj, "value": obj}


# ============================================================
# 主逻辑
# ============================================================

def generate_node_id(label: str, name: str) -> str:
    """生成干净节点 ID: {Label}_{hash}"""
    hash_suffix = hashlib.md5(f"{label}:{name}".encode()).hexdigest()[:8]
    return f"{label}_{hash_suffix}"



def rebuild_graph(
    client: Neo4jClient,
    triplets: List[Dict],
    mode: str = "full",
    dry_run: bool = False,
) -> Dict:
    """从规范化三元组重建图谱"""
    stats = Counter()

    if mode == "full" and not dry_run:
        logger.warning("全量模式：清空现有图谱...")
        client.delete_all(confirm=True)
        logger.info("图谱已清空")

    # 缓存已创建节点: (label, name) → id
    created_nodes: Dict[Tuple[str, str], str] = {}

    for i, t in enumerate(triplets):
        if i % 200 == 0:
            logger.info(f"进度: {i}/{len(triplets)}")

        subject = t.get("subject", "").strip()
        subject_type = t.get("subject_type", "Unknown")
        relation = t.get("relation", "").strip()
        relation_id = t.get("relation_id")
        obj = t.get("object", "").strip()
        obj_type = t.get("object_type", "numerical_value")
        year = t.get("year", "constant")
        confidence = t.get("confidence", 0.5)
        context = t.get("context", "")[:500]
        source_file = t.get("source_file", "")

        if not subject or not relation or not obj:
            stats["skipped_empty"] += 1
            continue

        # --- Subject 节点 ---
        subj_key = (subject_type, subject)
        if subj_key not in created_nodes:
            subj_id = generate_node_id(subject_type, subject)
            subj_props = {"name": subject, "source_file": source_file}
            if not dry_run:
                client.upsert_node(subject_type, subj_id, subj_props)
            created_nodes[subj_key] = subj_id
            stats["nodes_created"] += 1
        else:
            subj_id = created_nodes[subj_key]

        # --- Object 节点 ---
        obj_label, obj_props = _guess_object_label(obj, obj_type, relation_id, str(year), relation)
        obj_props.setdefault("source_file", source_file)

        # 对 named entities，尝试用已知实体缓存去重
        obj_name = obj_props.get("name", obj)
        obj_key = (obj_label, obj_name)
        if obj_key not in created_nodes:
            obj_id = generate_node_id(obj_label, obj_name)
            if not dry_run:
                client.upsert_node(obj_label, obj_id, obj_props)
            created_nodes[obj_key] = obj_id
            stats["nodes_created"] += 1
        else:
            obj_id = created_nodes[obj_key]

        # --- Relationship ---
        rel_props = {
            "confidence": confidence,
            "context": context,
        }
        if relation_id:
            rel_props["relation_id"] = relation_id
        if year and year != "constant":
            rel_props["year"] = year
        if source_file:
            rel_props["source_doc"] = source_file

        if not dry_run:
            try:
                client.create_relationship(
                    subject_label=subject_type,
                    subject_id=subj_id,
                    relation_type=relation,
                    object_label=obj_label,
                    object_id=obj_id,
                    properties=rel_props,
                )
            except Exception as e:
                logger.error(f"关系创建失败: {subject} -[{relation}]-> {obj}: {e}")
                stats["rel_errors"] += 1
                continue

        stats["relationships_created"] += 1

    # 刷新统计
    if not dry_run:
        db_stats = client.get_stats()
    else:
        db_stats = {}

    logger.info(f"重建完成: {stats['nodes_created']} 节点, {stats['relationships_created']} 关系")
    return dict(stats), db_stats


def main():
    parser = argparse.ArgumentParser(description="HydroBrain 从规范化三元组重建 Neo4j")
    parser.add_argument(
        "--input", default="data/processed/黄河水资源2024_triplets_normalized.json",
        help="规范化三元组 JSON"
    )
    parser.add_argument(
        "--mode", choices=["full", "append"], default="full",
        help="full=清空重建, append=增量追加"
    )
    parser.add_argument("--dry-run", action="store_true", help="仅验证，不写入")
    parser.add_argument("--limit", type=int, default=0, help="限制处理条数（调试用）")
    args = parser.parse_args()

    setup_logging(level="INFO")

    # 加载
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"文件不存在: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        triplets = json.load(f)

    if args.limit > 0:
        triplets = triplets[:args.limit]

    logger.info(f"加载 {len(triplets)} 条规范化三元组")

    # Neo4j
    config = ConfigLoader()
    neo4j_cfg = config.get_neo4j_config()
    client = Neo4jClient(
        uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
        user=neo4j_cfg.get("user", "neo4j"),
        password=neo4j_cfg.get("password", "password"),
    )

    # 重建
    stats, db_stats = rebuild_graph(client, triplets, mode=args.mode, dry_run=args.dry_run)

    # 输出统计
    print("\n" + "=" * 60)
    print("  重建统计")
    print("=" * 60)
    print(f"  模式: {'dry-run (仅验证)' if args.dry_run else args.mode}")
    print(f"  节点: {stats['nodes_created']}")
    print(f"  关系: {stats['relationships_created']}")
    if stats.get("skipped_empty"):
        print(f"  跳过(空字段): {stats['skipped_empty']}")
    if stats.get("rel_errors"):
        print(f"  关系错误: {stats['rel_errors']}")

    if db_stats:
        print(f"\n  ── 图谱状态 ──")
        for k, v in db_stats.items():
            print(f"  {k}: {v}")

    client.close()


if __name__ == "__main__":
    main()
