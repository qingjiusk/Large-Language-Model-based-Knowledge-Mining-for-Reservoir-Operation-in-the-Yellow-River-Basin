#!/usr/bin/env python
"""
HydroBrain Phase 1: PDF/图片 → OCR → 文本切片 → 保存中间文件

输出: data/processed/{文件名}_intermediate.json
  包含 clean_chunks + tables + 溯源信息

后续 Phase 2 (zagism 环境):
  python scripts/kg_extract.py --mode append

用法:
    python scripts/build_kg_from_dir.py
    python scripts/build_kg_from_dir.py --pdf-dir data/raw
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config_loader import ConfigLoader
from src.common.logger import get_logger, setup_logging
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.text_splitter import TextSplitter
from src.document_processing.image_parser import ImageParser, SUPPORTED_FORMATS

logger = get_logger(__name__)


class OCRPipeline:
    """Phase 1: OCR + 文本切片 + 保存中间文件"""

    def __init__(self, config: ConfigLoader):
        self.config = config
        self.stats = {
            "pdfs_processed": 0,
            "tables_extracted": 0,
            "chunks_created": 0,
        }

    def run(self, pdf_dir: str):
        start_time = time.time()
        logger.info("=" * 70)
        logger.info("HydroBrain Phase 1: OCR + 文本切片")
        logger.info(f"PDF 目录: {pdf_dir}")
        logger.info("=" * 70)

        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            logger.error(f"PDF 目录不存在: {pdf_dir}")
            sys.exit(1)

        # ---- 初始化组件 ----
        pdf_parser = PDFParser(skip_header_footer=True)
        text_splitter = TextSplitter(
            chunk_size=self.config.get("text_splitter.chunk_size", 2000),
            chunk_overlap=self.config.get("text_splitter.chunk_overlap", 200),
        )

        # ---- 收集文件 (PDF + 图片，递归子目录) ----
        pdf_files = list(set(pdf_dir.rglob("*.pdf")))
        image_files = []
        for fmt in SUPPORTED_FORMATS:
            image_files.extend(pdf_dir.rglob(f"*{fmt}"))
            image_files.extend(pdf_dir.rglob(f"*{fmt.upper()}"))
        image_files = list(set(image_files))

        all_input_files = pdf_files + image_files
        logger.info(f"找到 {len(pdf_files)} 个 PDF + {len(image_files)} 个图片")

        if not all_input_files:
            logger.warning(f"目录中无文件: {pdf_dir}")
            return {}

        output_dir = Path("data/processed/text_extracted")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 图片解析器（延迟初始化）
        image_parser = None

        # ---- 处理循环 ----
        for input_file in all_input_files:
            suffix = input_file.suffix.lower()
            logger.info(f"\n{'=' * 50}")
            logger.info(f"处理文件: {input_file.name}")
            logger.info(f"{'=' * 50}")

            try:
                # Step 1: 解析 → 文本 + 表格
                logger.info("[Step 1/2] OCR 解析...")
                if suffix == ".pdf":
                    doc_result = pdf_parser.extract_text_with_meta(str(input_file))
                else:
                    if image_parser is None:
                        image_parser = ImageParser(text_cleaner=pdf_parser.text_cleaner)
                    doc_result = image_parser.extract_text_with_meta(str(input_file))
                tables = doc_result.get("tables", [])
                noise_pages = doc_result.get("noise_pages", [])
                self.stats["pdfs_processed"] += 1
                self.stats["tables_extracted"] += len(tables)
                logger.info(
                    f"  -> {doc_result['page_count']} 页 "
                    f"(噪声: {len(noise_pages)}), "
                    f"{len(tables)} 个结构化表格"
                )

                # Step 2: 文本切片
                logger.info("[Step 2/2] 文本切片 + 保存...")
                chunks = text_splitter.split_document(doc_result)
                self.stats["chunks_created"] += len(chunks)
                logger.info(f"  -> {len(chunks)} 个 chunks")

                # 过滤噪声 chunk
                clean_chunks = []
                noise_skipped = 0
                for chunk in chunks:
                    content = chunk["content"].strip()
                    if len(content) < 20 or pdf_parser.text_cleaner.is_noise(content):
                        noise_skipped += 1
                        continue
                    clean_chunks.append({
                        "chunk_id": chunk.get("chunk_id", ""),
                        "content": content,
                        "page_num": chunk.get("page_num", 1),
                        "source_file": input_file.name,
                    })
                if noise_skipped > 0:
                    logger.info(f"  跳过噪声 chunk: {noise_skipped}")
                logger.info(f"  有效 chunk: {len(clean_chunks)}")

                # 为表格附加上下文
                for tbl in tables:
                    if isinstance(tbl, dict):
                        tbl["context"] = f"来自 {input_file.name} 第{tbl.get('page_num', 0)}页"

                # ---- 保存中间文件 ----
                intermediate = {
                    "source_file": input_file.name,
                    "chunks": clean_chunks,
                    "tables": tables,
                }
                intermediate_path = output_dir / f"{input_file.stem}_intermediate.json"
                with open(intermediate_path, "w", encoding="utf-8") as f:
                    json.dump(intermediate, f, ensure_ascii=False, indent=2)
                logger.info(f"  -> 中间文件已保存: {intermediate_path}")

            except Exception as e:
                logger.error(f"处理失败: {input_file.name}, 错误: {e}", exc_info=True)
                continue

        # 最终统计
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("Phase 1 完成!")
        logger.info(f"总耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")
        logger.info(f"处理文件: {self.stats['pdfs_processed']} 个")
        logger.info(f"提取表格: {self.stats['tables_extracted']} 个")
        logger.info(f"文本切片: {self.stats['chunks_created']} 个")
        logger.info(f"中间文件 → data/processed/")
        logger.info("=" * 70)
        logger.info("下一步: conda activate zagism && python scripts/kg_extract.py --mode append")

        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description="HydroBrain Phase 1: OCR + 文本切片"
    )
    parser.add_argument(
        "--pdf-dir", default=None,
        help="PDF 目录路径 (默认: config.yaml data.raw_dir)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    setup_logging(level=args.log_level)

    config = ConfigLoader()
    pdf_dir = args.pdf_dir or config.get("data.raw_dir", "data/raw")

    pipeline = OCRPipeline(config)
    pipeline.run(pdf_dir=pdf_dir)


if __name__ == "__main__":
    main()
