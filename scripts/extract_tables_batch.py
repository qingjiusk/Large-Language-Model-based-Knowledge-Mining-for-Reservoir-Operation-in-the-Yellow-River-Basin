#!/usr/bin/env python
"""
批量表格提取脚本
遍历 data/raw/ 下所有 PDF，提取表格并保存为 JSON
"""
import json
import sys
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config_loader import ConfigLoader
from src.common.logger import get_logger, setup_logging
from src.document_processing.table_parser import TableParser

logger = get_logger(__name__)


def main():
    setup_logging("INFO")

    logger.info("=" * 60)
    logger.info("批量表格提取脚本启动")
    logger.info("=" * 60)

    # 加载配置
    config = ConfigLoader()

    raw_dir = Path(config.get("data.raw_dir", "data/raw"))
    output_dir = Path(config.get("data.tables_dir", "data/processed/tables"))
    output_dir.mkdir(parents=True, exist_ok=True)

    min_rows = config.get("table_parser.min_table_rows", 2)
    out_fmt = config.get("table_parser.output_format", "markdown")

    logger.info(f"输入目录: {raw_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"配置: min_table_rows={min_rows}, output_format={out_fmt}")

    if not raw_dir.exists():
        logger.error(f"输入目录不存在: {raw_dir}")
        sys.exit(1)

    # 执行批量提取
    parser = TableParser(
        min_table_rows=min_rows,
        output_format=out_fmt,
    )

    results = parser.extract_tables_batch(
        pdf_dir=str(raw_dir),
        output_dir=str(output_dir),
        save_format="json",
    )

    # 输出统计
    total_tables = sum(len(v) for v in results.values())
    total_files = len(results)

    logger.info("=" * 60)
    logger.info(f"处理完成: {total_files} 个文件, {total_tables} 个表格")
    logger.info("=" * 60)

    # 保存汇总
    summary = {
        "total_files": total_files,
        "total_tables": total_tables,
        "files": {
            fname: {"table_count": len(tables)}
            for fname, tables in results.items()
        },
    }
    summary_path = output_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
