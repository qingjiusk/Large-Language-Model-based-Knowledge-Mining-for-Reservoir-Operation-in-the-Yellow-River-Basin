#!/usr/bin/env python
"""
批量 PDF 解析脚本
遍历 data/raw/ 下所有 PDF，提取文本 + 切片，保存到 data/processed/
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config_loader import ConfigLoader
from src.common.logger import get_logger, setup_logging
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.text_splitter import TextSplitter

logger = get_logger(__name__)


def main():
    setup_logging("INFO")

    logger.info("=" * 60)
    logger.info("批量 PDF 解析 + 文本切片脚本启动")
    logger.info("=" * 60)

    # 加载配置
    config = ConfigLoader()

    raw_dir = Path(config.get("data.raw_dir", "data/raw"))
    texts_dir = Path(config.get("data.texts_dir", "data/processed/texts"))
    chunks_dir = Path(config.get("data.chunks_dir", "data/processed/chunks"))
    texts_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = config.get("text_splitter.chunk_size", 2000)
    chunk_overlap = config.get("text_splitter.chunk_overlap", 200)

    logger.info(f"输入目录: {raw_dir}")
    logger.info(f"文本输出: {texts_dir}")
    logger.info(f"切片输出: {chunks_dir}")
    logger.info(f"切片参数: chunk_size={chunk_size}, overlap={chunk_overlap}")

    if not raw_dir.exists():
        logger.error(f"输入目录不存在: {raw_dir}")
        sys.exit(1)

    # Step 1: PDF → 文本
    logger.info("--- 阶段 1: PDF 文本提取 ---")
    pdf_parser = PDFParser(skip_header_footer=True)
    parse_results = pdf_parser.batch_extract(
        pdf_dir=str(raw_dir),
        output_dir=str(texts_dir),
    )

    if not parse_results:
        logger.error("未解析到任何 PDF 文本，退出")
        sys.exit(1)

    # Step 2: 文本 → 切片
    logger.info("--- 阶段 2: 文本切片 ---")
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    total_chunks = 0
    for result in parse_results:
        chunks = splitter.split_document(result)

        file_name = result["file_name"]
        stem = Path(file_name).stem
        chunks_path = chunks_dir / f"{stem}_chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        total_chunks += len(chunks)
        logger.info(f"  {file_name}: {len(chunks)} chunks")

    # 输出统计
    logger.info("=" * 60)
    logger.info(
        f"处理完成: {len(parse_results)} 个文件, "
        f"{total_chunks} 个 chunks"
    )
    logger.info(f"文本保存在: {texts_dir}")
    logger.info(f"切片保存在: {chunks_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
