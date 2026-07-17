#!/usr/bin/env python
"""
全链路知识图谱构建脚本
从 PDF 目录 → 文本/表格提取 → LLM 抽取 → 标准化 → 知识融合 → Neo4j 入库

用法:
    python scripts/build_kg_from_dir.py
    python scripts/build_kg_from_dir.py --mode full      # 全量模式（清空旧数据）
    python scripts/build_kg_from_dir.py --mode append    # 增量追加（默认）
    python scripts/build_kg_from_dir.py --skip-neo4j     # 跳过 Neo4j 写入（仅做抽取）
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
from src.llm_pipeline.uie_client import UIEClient
from src.llm_pipeline.extractor import TripletExtractor
from src.knowledge_fusion.entity_linking import EntityLinker
from src.knowledge_fusion.triplet_normalizer import TripletNormalizer
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.document_processing.image_parser import ImageParser, SUPPORTED_FORMATS
from scripts.rebuild_kg import rebuild_graph

logger = get_logger(__name__)


class KGPipeline:
    """知识图谱构建全链路流水线"""

    def __init__(self, config: ConfigLoader, skip_neo4j: bool = False):
        self.config = config
        self.skip_neo4j = skip_neo4j
        self.stats = {
            "pdfs_processed": 0,
            "tables_extracted": 0,
            "chunks_created": 0,
            "triplets_extracted": 0,
            "triplets_normalized": 0,
            "nodes_written": 0,
            "relationships_written": 0,
        }

    def run(self, pdf_dir: str, mode: str = "append"):
        """
        执行全链路构建

        Args:
            pdf_dir: PDF 文件目录
            mode: "full" 全量 / "append" 增量追加
        """
        start_time = time.time()
        logger.info("=" * 70)
        logger.info("HydroBrain 知识图谱构建流水线启动")
        logger.info(f"PDF 目录: {pdf_dir}, 模式: {mode}")
        logger.info("=" * 70)

        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            logger.error(f"PDF 目录不存在: {pdf_dir}")
            sys.exit(1)

        # ---- 初始化组件 ----

        # PP-UIE 本地模型（替代 DeepSeek API，零 API 调用）
        uie_config = self.config.get("uie", {})
        uie_client = UIEClient(
            model_path=uie_config.get("model_path", "models/PP-UIE-1.5B"),
            device=uie_config.get("device", "gpu"),
            precision=uie_config.get("precision", "float16"),
            max_length=uie_config.get("max_length", 4096),
            temperature=uie_config.get("temperature", 0.0),
        )

        # 文档处理（PaddleOCR 自动检测表格，无需 pdfplumber）
        pdf_parser = PDFParser(skip_header_footer=True)
        text_splitter = TextSplitter(
            chunk_size=self.config.get("text_splitter.chunk_size", 2000),
            chunk_overlap=self.config.get("text_splitter.chunk_overlap", 200),
        )

        # LLM 管线
        extractor = TripletExtractor(uie_client, prompts_dir="prompts")
        embed_device = self.config.get("embedding.device", "cpu")

        # 知识融合 — 实体链接器
        entity_linker = EntityLinker(
            device=embed_device,
            alias_dict_path=self.config.get("ontology.alias_dict", "data/ontology/alias_dict.json"),
            entity_types_path=self.config.get("ontology.entity_types", "data/ontology/entity_types.json"),
        )

        # 三元组规范化器（纯规则，零 API 调用）
        normalizer = TripletNormalizer(entity_linker=entity_linker)

        # Neo4j（直接连接，供 rebuild_graph 使用）
        neo4j_client = None
        if not self.skip_neo4j:
            neo4j_config = self.config.get_neo4j_config()
            try:
                neo4j_client = Neo4jClient(
                    uri=neo4j_config.get("uri", "bolt://localhost:7687"),
                    user=neo4j_config.get("user", "neo4j"),
                    password=neo4j_config.get("password", "password"),
                )
                logger.info("Neo4j 连接就绪")
            except Exception as e:
                logger.error(f"Neo4j 连接失败: {e}")
                logger.warning("将跳过图谱写入，仅做抽取")
                self.skip_neo4j = True

        # ---- 收集文件 (PDF + 图片) ----
        pdf_files = list(set(pdf_dir.glob("*.pdf")))
        image_files = []
        for fmt in SUPPORTED_FORMATS:
            image_files.extend(pdf_dir.glob(f"*{fmt}"))
            image_files.extend(pdf_dir.glob(f"*{fmt.upper()}"))
        image_files = list(set(image_files))

        all_input_files = pdf_files + image_files
        logger.info(f"找到 {len(pdf_files)} 个 PDF + {len(image_files)} 个图片")

        if not all_input_files:
            logger.warning(f"目录中无文件: {pdf_dir}")
            return {}

        output_dir = Path(self.config.get("data.processed_dir", "data/processed"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 图片解析器（延迟初始化）
        image_parser = None

        all_normalized = []

        # ---- 统一处理循环 ----
        for input_file in all_input_files:
            suffix = input_file.suffix.lower()
            logger.info(f"\n{'=' * 50}")
            logger.info(f"处理文件: {input_file.name}")
            logger.info(f"{'=' * 50}")

            try:
                # Step 1: 解析 → 文本 + 表格
                logger.info("[Step 1/5] 解析...")
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
                logger.info("[Step 2/5] 文本切片...")
                chunks = text_splitter.split_document(doc_result)
                self.stats["chunks_created"] += len(chunks)
                logger.info(f"  -> {len(chunks)} 个 chunks")

                # Step 3: LLM 抽取（文本批量 + 表格）
                logger.info("[Step 3/5] LLM 三元组抽取...")
                all_triplets = []
                batch_size = self.config.get("uie.batch_size", 2)

                # 文本抽取 — 过滤噪声后批量处理
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

                if clean_chunks:
                    logger.info(f"  文本批量抽取: {len(clean_chunks)} chunks, batch_size={batch_size}")
                    text_triplets = extractor.extract_from_text_batch(
                        clean_chunks, batch_size=batch_size
                    )
                    # 附加溯源信息
                    for t in text_triplets:
                        t.setdefault("source_file", input_file.name)
                        t.setdefault("page_num", t.get("page_num", 1))
                    all_triplets.extend(text_triplets)
                    logger.info(f"  文本批量完成: {len(text_triplets)} 三元组")

                if noise_skipped > 0:
                    logger.info(f"  跳过噪声 chunk: {noise_skipped}")

                # 表格抽取（PP-StructureV3 已自动检测 + 识别表格为 Markdown，批量处理）
                table_triplets = []
                if tables:
                    # 附加上下文信息
                    for tbl in tables:
                        if isinstance(tbl, dict):
                            tbl["context"] = f"来自 {input_file.name} 第{tbl.get('page_num', 0)}页"
                    table_triplets = extractor.extract_from_table_batch(tables, batch_size=batch_size)
                    for t in table_triplets:
                        t.setdefault("source_file", input_file.name)
                        t.setdefault("page_num", t.get("page_num", 1))
                    logger.info(f"  表格批量完成: {len(table_triplets)} 三元组")

                all_triplets.extend(table_triplets)

                self.stats["triplets_extracted"] += len(all_triplets)
                logger.info(f"  -> {len(all_triplets)} 个三元组 "
                           f"(文本: {len(all_triplets) - sum(1 for t in all_triplets if t.get('data_type') == 'tabular')}, "
                           f"表格: {sum(1 for t in all_triplets if t.get('data_type') == 'tabular')})")

                if not all_triplets:
                    logger.warning(f"  -> 未抽取到任何三元组，跳过后续步骤")
                    continue

                # Step 4: 三元组规范化（纯规则，零 API 调用）
                logger.info("[Step 4/5] 三元组规范化...")
                normalized = normalizer.normalize(all_triplets)
                self.stats["triplets_normalized"] += len(normalized)
                logger.info(
                    f"  -> {len(normalized)} 条 "
                    f"(拆分+{normalizer.stats['triplets_split']}, "
                    f"无效-{normalizer.stats['triplets_invalid']}, "
                    f"去重-{normalizer.stats.get('triplets_deduped', 0)}, "
                    f"匹配: 精确{normalizer.stats['relations_matched_exact']}/"
                    f"关键词{normalizer.stats['relations_matched_keyword']}/"
                    f"未匹配{normalizer.stats['relations_unmatched']})"
                )

                all_normalized.extend(normalized)

                # 保存中间结果
                result_path = output_dir / f"{input_file.stem}_triplets.json"
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(normalized, f, ensure_ascii=False, indent=2)
                logger.info(f"  -> 中间结果已保存: {result_path}")

            except Exception as e:
                logger.error(f"处理失败: {input_file.name}, 错误: {e}", exc_info=True)
                continue

        # Step 5: Neo4j 图谱写入
        if not self.skip_neo4j and neo4j_client and all_normalized:
            logger.info(f"\n[Step 5/5] Neo4j 图谱写入 ({len(all_normalized)} 条规范化三元组, mode={mode})...")

            build_stats, db_stats = rebuild_graph(neo4j_client, all_normalized, mode=mode)
            self.stats["nodes_written"] = build_stats.get("nodes_created", 0)
            self.stats["relationships_written"] = build_stats.get("relationships_created", 0)
        elif self.skip_neo4j:
            logger.info("\n[Step 5/5] 跳过 Neo4j 写入")

        # 最终统计
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("流水线执行完成!")
        logger.info(f"总耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")
        logger.info(f"处理文件: {self.stats['pdfs_processed']} 个")
        logger.info(f"提取表格: {self.stats['tables_extracted']} 个")
        logger.info(f"文本切片: {self.stats['chunks_created']} 个")
        logger.info(f"抽取三元组: {self.stats['triplets_extracted']} 条")
        logger.info(f"规范化三元组: {self.stats['triplets_normalized']} 条")
        logger.info(f"图谱节点: {self.stats['nodes_written']} 个")
        logger.info(f"图谱关系: {self.stats['relationships_written']} 条")
        logger.info("=" * 70)

        if neo4j_client:
            neo4j_client.close()

        return self.stats



def main():
    parser = argparse.ArgumentParser(description="HydroBrain 知识图谱构建流水线")
    parser.add_argument(
        "--mode", choices=["full", "append"], default="append",
        help="构建模式: full=清空重建, append=增量追加 (默认: append)"
    )
    parser.add_argument(
        "--pdf-dir", default=None,
        help="PDF 目录路径 (默认: 使用 config.yaml 中的 data.raw_dir)"
    )
    parser.add_argument(
        "--skip-neo4j", action="store_true",
        help="跳过 Neo4j 写入，仅做抽取和保存中间结果"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level)

    config = ConfigLoader()
    pdf_dir = args.pdf_dir or config.get("data.raw_dir", "data/raw")

    pipeline = KGPipeline(config, skip_neo4j=args.skip_neo4j)
    pipeline.run(pdf_dir=pdf_dir, mode=args.mode)


if __name__ == "__main__":
    main()
