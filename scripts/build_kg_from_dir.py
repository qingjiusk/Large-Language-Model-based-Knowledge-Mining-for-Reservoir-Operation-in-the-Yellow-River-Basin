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
from src.document_processing.table_parser import TableParser
from src.document_processing.text_splitter import TextSplitter
from src.llm_pipeline.llm_client import DeepSeekClient
from src.llm_pipeline.extractor import TripletExtractor
from src.llm_pipeline.definer import SemanticDefiner
from src.llm_pipeline.canonicalizer import RelationCanonicalizer
from src.knowledge_fusion.entity_linking import EntityLinker
from src.knowledge_fusion.conflict_resolver import ConflictResolver
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.graph_builder import GraphBuilder

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
            "relations_defined": 0,
            "relations_canonicalized": 0,
            "conflicts_resolved": 0,
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

        # DeepSeek
        ds_config = self.config.get_deepseek_config()
        ds_client = DeepSeekClient(
            api_key=ds_config.get("api_key", ""),
            base_url=ds_config.get("base_url", "https://api.deepseek.com"),
            model=ds_config.get("model", "deepseek-chat"),
            max_tokens=ds_config.get("max_tokens", 4096),
            temperature=ds_config.get("temperature", 0.0),
        )

        # 文档处理
        pdf_parser = PDFParser(skip_header_footer=True)
        table_parser = TableParser(
            min_table_rows=self.config.get("table_parser.min_table_rows", 2),
            output_format="markdown",
        )
        text_splitter = TextSplitter(
            chunk_size=self.config.get("text_splitter.chunk_size", 2000),
            chunk_overlap=self.config.get("text_splitter.chunk_overlap", 200),
        )

        # LLM 管线
        extractor = TripletExtractor(ds_client, prompts_dir="prompts")
        embed_device = self.config.get("embedding.device", "cpu")

        definer = SemanticDefiner(ds_client, prompts_dir="prompts", device=embed_device)
        canonicalizer = RelationCanonicalizer(
            ds_client,
            standard_relations=self._load_standard_relations(),
            prompts_dir="prompts",
            device=embed_device,
            top_k=self.config.get("canonicalization.top_k_candidates", 3),
            mode=self.config.get("canonicalization.mode", "strict"),
            min_similarity=self.config.get("canonicalization.min_similarity_threshold", 0.7),
        )

        # 知识融合
        entity_linker = EntityLinker(
            device=embed_device,
            alias_dict_path=self.config.get("ontology.alias_dict", "data/ontology/alias_dict.json"),
            entity_types_path=self.config.get("ontology.entity_types", "data/ontology/entity_types.json"),
        )

        # Neo4j
        graph_builder = None
        if not self.skip_neo4j:
            neo4j_config = self.config.get_neo4j_config()
            try:
                neo4j_client = Neo4jClient(
                    uri=neo4j_config.get("uri", "bolt://localhost:7687"),
                    user=neo4j_config.get("user", "neo4j"),
                    password=neo4j_config.get("password", "password"),
                )
                conflict_resolver = ConflictResolver(
                    table_priority=self.config.get("fusion.table_priority_over_text", True),
                    enable_year_distinction=self.config.get("fusion.enable_year_based_distinction", True),
                )
                graph_builder = GraphBuilder(neo4j_client, entity_linker, conflict_resolver)
                logger.info("Neo4j 连接就绪")
            except Exception as e:
                logger.error(f"Neo4j 连接失败: {e}")
                logger.warning("将跳过图谱写入，仅做抽取")
                self.skip_neo4j = True

        # ---- 逐文件处理 ----
        pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
        logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")

        output_dir = Path(self.config.get("data.processed_dir", "data/processed"))
        output_dir.mkdir(parents=True, exist_ok=True)

        all_resolved = []

        for pdf_file in pdf_files:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"处理文件: {pdf_file.name}")
            logger.info(f"{'=' * 50}")

            try:
                # Step 1: PDF → 文本 + 表格
                logger.info("[Step 1/6] PDF 解析...")
                doc_result = pdf_parser.extract_text_with_meta(str(pdf_file))
                tables = table_parser.extract_tables(str(pdf_file))
                self.stats["pdfs_processed"] += 1
                self.stats["tables_extracted"] += len(tables)
                logger.info(f"  -> {doc_result['page_count']} 页文本, {len(tables)} 个表格")

                # Step 2: 文本切片
                logger.info("[Step 2/6] 文本切片...")
                chunks = text_splitter.split_document(doc_result)
                self.stats["chunks_created"] += len(chunks)
                logger.info(f"  -> {len(chunks)} 个 chunks")

                # Step 3: LLM 抽取（文本 + 表格）
                logger.info("[Step 3/6] LLM 三元组抽取...")
                all_triplets = []

                # 文本抽取
                for chunk in chunks:
                    if len(chunk["content"].strip()) < 20:
                        continue
                    text_triplets = extractor.extract_from_text(chunk["content"])
                    # 附加溯源信息
                    for t in text_triplets:
                        t["source_file"] = pdf_file.name
                        t["page_num"] = chunk.get("page_num", 1)
                    all_triplets.extend(text_triplets)

                # 表格抽取
                for table in tables:
                    md = table.get("markdown", "")
                    if not md:
                        continue
                    ctx = f"来自 {pdf_file.name} 第{table['page']}页"
                    table_triplets = extractor.extract_from_table(md, context=ctx)
                    for t in table_triplets:
                        t["source_file"] = pdf_file.name
                        t["page_num"] = table["page"]
                    all_triplets.extend(table_triplets)

                self.stats["triplets_extracted"] += len(all_triplets)
                logger.info(f"  -> {len(all_triplets)} 个三元组 "
                           f"(文本: {len(all_triplets) - sum(1 for t in all_triplets if t.get('data_type') == 'tabular')}, "
                           f"表格: {sum(1 for t in all_triplets if t.get('data_type') == 'tabular')})")

                if not all_triplets:
                    logger.warning(f"  -> 未抽取到任何三元组，跳过后续步骤")
                    continue

                # Step 4: 语义定义
                logger.info("[Step 4/6] 关系语义定义...")
                relations = list(set(t.get("relation", "") for t in all_triplets if t.get("relation")))
                definitions = definer.define_relations(relations, doc_result.get("full_text", "")[:5000])
                self.stats["relations_defined"] += len(definitions)
                logger.info(f"  -> {len(definitions)} 个关系定义")

                # Step 5: 关系标准化
                logger.info("[Step 5/6] 关系标准化...")
                canonicalized = canonicalizer.batch_canonicalize(all_triplets, definitions)
                self.stats["relations_canonicalized"] += len(canonicalized)
                logger.info(f"  -> {len(canonicalized)} 条标准化三元组")

                all_resolved.extend(canonicalized)

                # 保存中间结果
                result_path = output_dir / f"{pdf_file.stem}_triplets.json"
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(canonicalized, f, ensure_ascii=False, indent=2)
                logger.info(f"  -> 中间结果已保存: {result_path}")

            except Exception as e:
                logger.error(f"处理失败: {pdf_file.name}, 错误: {e}", exc_info=True)
                continue

        # Step 6: Neo4j 图谱写入
        if not self.skip_neo4j and graph_builder and all_resolved:
            logger.info(f"\n[Step 6/6] Neo4j 图谱写入 ({len(all_resolved)} 条三元组)...")

            source_doc = {
                "name": pdf_dir.name or "bulk_import",
                "type": "PDF批量导入",
            }

            build_stats = graph_builder.build_from_triplets(
                all_resolved,
                source_doc=source_doc,
                mode=mode,
            )
            self.stats["nodes_written"] = build_stats.get("nodes_created", 0)
            self.stats["relationships_written"] = build_stats.get("relationships_created", 0)
            self.stats["conflicts_resolved"] = build_stats.get("conflicts_resolved", 0)
        elif self.skip_neo4j:
            logger.info("\n[Step 6/6] 跳过 Neo4j 写入")

        # 最终统计
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("流水线执行完成!")
        logger.info(f"总耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")
        logger.info(f"处理文件: {self.stats['pdfs_processed']} 个")
        logger.info(f"提取表格: {self.stats['tables_extracted']} 个")
        logger.info(f"文本切片: {self.stats['chunks_created']} 个")
        logger.info(f"抽取三元组: {self.stats['triplets_extracted']} 条")
        logger.info(f"关系定义: {self.stats['relations_defined']} 个")
        logger.info(f"标准化三元组: {self.stats['relations_canonicalized']} 条")
        logger.info(f"冲突解决: {self.stats['conflicts_resolved']} 个")
        logger.info(f"图谱节点: {self.stats['nodes_written']} 个")
        logger.info(f"图谱关系: {self.stats['relationships_written']} 条")
        logger.info("=" * 70)

        return self.stats

    def _load_standard_relations(self) -> dict:
        """加载标准关系定义"""
        import json
        rel_path = Path(self.config.get("ontology.relation_types", "data/ontology/relation_types.json"))
        if not rel_path.exists():
            logger.warning(f"标准关系文件不存在: {rel_path}")
            return {}

        with open(rel_path, "r", encoding="utf-8") as f:
            rel_data = json.load(f)

        # 转换为 {id: description} 格式
        return {
            rel_id: info.get("description", "")
            for rel_id, info in rel_data.items()
        }


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
