#!/usr/bin/env python
"""
三元组规范化脚本
对已抽取的三元组做后处理标准化：实体对齐、关系映射、对象分类、年份提取、脏数据清洗

用法:
    python scripts/normalize_triplets.py
    python scripts/normalize_triplets.py --input data/processed/黄河水资源2024_triplets.json
    python scripts/normalize_triplets.py --output data/processed/normalized.json
    python scripts/normalize_triplets.py --stats-only   # 只看统计，不输出文件
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logger import get_logger, setup_logging
from src.knowledge_fusion.entity_linking import EntityLinker
from src.knowledge_fusion.triplet_normalizer import TripletNormalizer

logger = get_logger(__name__)


def print_stats(normalized: list, stats: dict):
    """打印规范化统计"""
    print("\n" + "=" * 60)
    print("  规范化统计")
    print("=" * 60)

    print(f"\n  输入三元组: {stats['total']}")
    print(f"  输出三元组: {len(normalized)}")
    print(f"  拆分新增:   {stats['triplets_split']}")
    print(f"  丢弃无效:   {stats['triplets_invalid']}")

    print(f"\n  ── 实体 ──")
    print(f"  已标准化:   {stats['entities_normalized']}")
    print(f"  新实体:     {stats['entity_new']}")

    print(f"\n  ── 关系 ──")
    print(f"  精确匹配:   {stats['relations_matched_exact']}")
    print(f"  关键词匹配: {stats['relations_matched_keyword']}")
    print(f"  未匹配:     {stats['relations_unmatched']}")
    match_rate = (stats['relations_matched_exact'] + stats['relations_matched_keyword']) / max(stats['total'], 1) * 100
    print(f"  总匹配率:   {match_rate:.1f}%")

    print(f"\n  ── Object 分类 ──")
    for k, v in stats['objects_classified'].items():
        print(f"  {k}: {v}")

    # 分布统计
    print(f"\n  ── subject_type 分布 ──")
    type_dist = Counter(t.get("subject_type", "Unknown") for t in normalized)
    for label, count in type_dist.most_common():
        print(f"  {label}: {count}")

    print(f"\n  ── relation_id 分布 (Top-20) ──")
    rel_dist = Counter(t.get("relation_id") for t in normalized)
    for rid, count in rel_dist.most_common(20):
        print(f"  {rid}: {count}")

    print(f"\n  ── year 分布 ──")
    year_dist = Counter(str(t.get("year", "?")) for t in normalized)
    for y, count in year_dist.most_common():
        print(f"  {y}: {count}")

    # 未匹配的关系
    unmatched_relations = Counter(
        t["relation"] for t in normalized if t.get("relation_id") is None
    )
    if unmatched_relations:
        print(f"\n  ── 未匹配的关系 (Top-20) ──")
        for rel, count in unmatched_relations.most_common(20):
            print(f"  [{count}] {rel}")


def main():
    parser = argparse.ArgumentParser(description="HydroBrain 三元组规范化")
    parser.add_argument(
        "--input", default="data/processed/黄河水资源2024_triplets.json",
        help="原始三元组 JSON 路径"
    )
    parser.add_argument(
        "--output", default=None,
        help="规范化输出 JSON 路径 (默认: 同目录下 _normalized.json)"
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="仅输出统计信息，不保存文件"
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="启用 LLM 兜底（对未匹配的关系调用 API）"
    )
    parser.add_argument(
        "--alias-dict", default="data/ontology/alias_dict.json",
        help="别名字典路径"
    )
    parser.add_argument(
        "--entity-types", default="data/ontology/entity_types.json",
        help="实体类型定义路径"
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    # 加载数据
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        triplets = json.load(f)
    logger.info(f"加载 {len(triplets)} 条原始三元组: {input_path}")

    # 初始化
    entity_linker = EntityLinker(
        alias_dict_path=args.alias_dict,
        entity_types_path=args.entity_types,
        min_similarity=0.75,
    )

    normalizer = TripletNormalizer(
        entity_linker=entity_linker,
        use_llm_fallback=args.use_llm,
    )

    # 执行规范化
    normalized = normalizer.normalize(triplets)

    # 打印统计
    print_stats(normalized, normalizer.stats)

    # 保存
    if not args.stats_only:
        output_path = args.output
        if output_path is None:
            stem = input_path.stem
            output_path = input_path.parent / f"{stem}_normalized.json"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        logger.info(f"规范化三元组已保存: {output_path} ({len(normalized)} 条)")


if __name__ == "__main__":
    main()
