#!/usr/bin/env python
"""
HydroBrain Phase 2: 中间文件 → Gemma-4B 抽取 → 规范化 → Neo4j

用法 (zagism 环境):
    python scripts/kg_extract.py
    python scripts/kg_extract.py --mode full      # 全量重建
    python scripts/kg_extract.py --mode append    # 增量追加（默认）
    python scripts/kg_extract.py --dry-run        # 仅抽取，不写库
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config_loader import ConfigLoader
from src.common.logger import get_logger, setup_logging
from src.knowledge_fusion.entity_linking import EntityLinker
from src.knowledge_fusion.triplet_normalizer import TripletNormalizer
from src.knowledge_graph.neo4j_client import Neo4jClient
from scripts.rebuild_kg import rebuild_graph

logger = get_logger(__name__)

# ============================================================
# 复用验证过的 prompts（与 DeepSeek 版相同）
# ============================================================
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
TEXT_PROMPT_TEMPLATE = open(PROMPTS_DIR / "extract.txt", encoding="utf-8").read()
TABLE_PROMPT_TEMPLATE = open(PROMPTS_DIR / "table_extract.txt", encoding="utf-8").read()

GEMMA_SYSTEM = "你是资深黄河流域水库调度专家。只输出JSON数组，不输出任何解释。"


def build_gemma_prompt(user_content: str) -> str:
    """构建 Gemma chat 格式的 prompt（不加 <bos>，llama-cpp 自动添加）"""
    return (
        f"<start_of_turn>system\n{GEMMA_SYSTEM}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


class KGExtractor:
    """Phase 2: Gemma-4B抽取 → 规范化 → Neo4j"""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_batch: int = 256,
        f16_kv: bool = True,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.f16_kv = f16_kv
        self._model = None

        self.stats = {
            "chunks_processed": 0,
            "triplets_raw": 0,
            "triplets_normalized": 0,
        }

    def _load_model(self):
        """加载 Gemma-4B GGUF 模型"""
        from llama_cpp import Llama

        logger.info(f"加载模型: {self.model_path}")
        t0 = time.time()

        self._model = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=-1,  # 全部层上 GPU
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            f16_kv=self.f16_kv,
            verbose=False,
        )

        elapsed = time.time() - t0
        logger.info(f"模型加载完成，耗时 {elapsed:.1f}s")

    def extract_from_chunk(self, text: str, is_table: bool = False) -> List[Dict]:
        """对单个 chunk 文本执行三元组抽取"""
        if not text or not text.strip():
            return []

        # 使用验证过的 prompt 模板（与 DeepSeek 版相同）
        if is_table:
            # {table_markdown} + {context} — 表格模板需要两个参数
            # 对于纯文本 chunk，context 为空
            user_prompt = TABLE_PROMPT_TEMPLATE.format(
                table_markdown=text, context=""
            )
        else:
            user_prompt = TEXT_PROMPT_TEMPLATE.format(text_chunk=text)

        prompt = build_gemma_prompt(user_prompt)

        output = self._model(
            prompt,
            max_tokens=8192,
            temperature=0,
            stop=["<end_of_turn>", "<start_of_turn>"],
        )

        content = output["choices"][0]["text"].strip()
        logger.debug(f"模型原始输出: {content[:300]}...")

        return self._parse_response(content)

    def _parse_response(self, content: str) -> List[Dict]:
        """从模型输出中提取 JSON 三元组列表（处理 Qwen thinking 模式）"""
        if not content:
            return []

        # 去掉可能的 markdown 代码块
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        # 尝试直接解析
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return self._validate_triplets(result)
            elif isinstance(result, dict):
                triplets = result.get("triplets", result.get("data", []))
                return self._validate_triplets(triplets)
        except json.JSONDecodeError:
            pass

        # 查找 JSON 数组
        idx = content.find("[")
        if idx >= 0:
            try:
                result = json.loads(content[idx:])
                if isinstance(result, list):
                    return self._validate_triplets(result)
            except json.JSONDecodeError as e:
                return self._repair_json_array(content[idx:], e)

        return []

    def _repair_json_array(self, text: str, error: json.JSONDecodeError) -> List[Dict]:
        """尝试修复截断的 JSON 数组"""
        if not hasattr(error, "pos") or error.pos <= 0:
            return []
        truncated = text[:error.pos].rstrip()
        for cut in range(len(truncated), 0, -1):
            attempt = truncated[:cut].rstrip()
            if attempt.endswith((",", ":", "{")):
                continue
            try:
                fixed = attempt
                open_brackets = fixed.count("[") - fixed.count("]")
                open_braces = fixed.count("{") - fixed.count("}")
                fixed += "}" * open_braces + "]" * open_brackets
                result = json.loads(fixed)
                if isinstance(result, list):
                    logger.warning(f"截断 JSON 回收: {len(result)} 条")
                    return self._validate_triplets(result)
            except json.JSONDecodeError:
                continue
        return []

    def _validate_triplets(self, triplets: List) -> List[Dict]:
        """过滤并补全三元组默认字段"""
        valid = []
        for t in triplets:
            if not isinstance(t, dict):
                continue
            subj = (t.get("subject") or "").strip()
            rel = (t.get("relation") or "").strip()
            obj = (t.get("object") or "").strip()
            if not subj or not rel or not obj:
                continue
            t.setdefault("context", "")
            t.setdefault("confidence", 0.85)
            t.setdefault("data_type", "text")
            t.setdefault("year", None)
            valid.append(t)
        return valid

    # ================================================================
    # 表格预处理
    # ================================================================

    @staticmethod
    def _embed_units_in_table(md: str) -> str:
        """将 Markdown 表头括号中的单位内嵌到每行数值中。

        | 水库 | 总库容(亿m³) | 死水位(m) |
        |------|-------------|-----------|
        | 龙羊峡 | 247 | 2530 |
        →
        | 水库 | 总库容 | 死水位 |
        |------|--------|--------|
        | 龙羊峡 | 247亿m³ | 2530m |
        """
        import re
        lines = md.strip().split("\n")
        if len(lines) < 2:
            return md

        # 解析表头行，提取单位
        header_line = lines[0]
        cells = [c.strip() for c in header_line.strip("|").split("|")]
        units = []
        for cell in cells:
            m = re.search(r'\((.+?)\)$', cell)
            units.append(m.group(1) if m else None)

        if not any(units):
            return md  # 无单位直接返回

        # 清理表头（去掉括号部分）
        clean_header = "| " + " | ".join(
            re.sub(r'\s*\(.+?\)$', '', c).strip() for c in cells
        ) + " |"

        # 逐行处理数据行，拼上单位
        new_lines = [clean_header, lines[1]]  # 表头 + 分隔符
        for line in lines[2:]:
            row_cells = [c.strip() for c in line.strip("|").split("|")]
            new_cells = []
            for i, cell in enumerate(row_cells):
                unit = units[i] if i < len(units) else None
                if unit and cell and re.search(r'\d', cell):
                    cell += unit
                new_cells.append(cell)
            new_lines.append("| " + " | ".join(new_cells) + " |")

        return "\n".join(new_lines)

    @staticmethod
    def _split_large_table(md: str, max_rows: int = 8) -> list:
        """将大表格按行拆分为多个小表格（每组保留表头+分隔符）"""
        lines = md.strip().split("\n")
        if len(lines) <= max_rows + 2:
            return [md]
        header = lines[:2]
        data_rows = lines[2:]
        chunks = []
        for i in range(0, len(data_rows), max_rows):
            chunks.append("\n".join(header + data_rows[i:i + max_rows]))
        return chunks

    def run(
        self,
        input_dir: str = "data/processed/text_extracted",
        mode: str = "append",
        dry_run: bool = False,
        limit: int = 0,
    ):
        """主流程：遍历中间文件 → 抽取 → 规范化 → Neo4j"""
        input_dir = Path(input_dir)
        intermediate_files = sorted(input_dir.glob("*_intermediate.json"))

        if not intermediate_files:
            logger.error(f"未找到中间文件: {input_dir}/*_intermediate.json")
            logger.error("请先运行 Phase 1: conda activate zhishi && python scripts/build_kg_from_dir.py")
            sys.exit(1)

        logger.info(f"找到 {len(intermediate_files)} 个中间文件")

        # ---- 初始化组件 ----
        config = ConfigLoader()

        # EntityLinker + TripletNormalizer
        embed_device = config.get("embedding.device", "cpu")
        entity_linker = EntityLinker(
            device=embed_device,
            alias_dict_path=config.get("ontology.alias_dict", "data/ontology/alias_dict.json"),
            entity_types_path=config.get("ontology.entity_types", "data/ontology/entity_types.json"),
        )
        normalizer = TripletNormalizer(entity_linker=entity_linker)

        # Neo4j
        neo4j_client = None
        if not dry_run:
            neo4j_cfg = config.get_neo4j_config()
            try:
                neo4j_client = Neo4jClient(
                    uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
                    user=neo4j_cfg.get("user", "neo4j"),
                    password=neo4j_cfg.get("password", "password"),
                )
                logger.info("Neo4j 连接就绪")
            except Exception as e:
                logger.error(f"Neo4j 连接失败: {e}")
                sys.exit(1)

        # 加载 LLM 模型
        self._load_model()

        # ---- 主循环 ----
        all_normalized = []
        start_time = time.time()

        for int_file in intermediate_files:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"处理中间文件: {int_file.name}")
            logger.info(f"{'=' * 50}")

            with open(int_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            source_file = data.get("source_file", int_file.stem)
            chunks = data.get("chunks", [])
            tables = data.get("tables", [])

            all_triplets = []

            # ---- 文本抽取 ----
            if chunks:
                logger.info(f"文本抽取: {len(chunks)} chunks")
                if limit > 0:
                    chunks = chunks[:limit]
                for i, chunk in enumerate(chunks):
                    if i % 10 == 0:
                        logger.info(f"  进度: {i}/{len(chunks)}")
                    triplets = self.extract_from_chunk(chunk["content"])
                    for t in triplets:
                        t["source_file"] = source_file
                        t["page_num"] = chunk.get("page_num", 1)
                        t["chunk_id"] = chunk.get("chunk_id", "")
                    all_triplets.extend(triplets)
                    self.stats["chunks_processed"] += 1
                logger.info(f"  文本完成: {len(all_triplets)} 条")

            # ---- 表格抽取 ----
            table_count = 0
            if tables:
                logger.info(f"表格抽取: {len(tables)} 个表格")
                for tbl_idx, tbl in enumerate(tables):
                    md = tbl.get("markdown", "")
                    ctx = tbl.get("context", "")
                    if not md.strip():
                        continue
                    # 预处理: 内嵌单位 + 拆分大表格
                    md = self._embed_units_in_table(md)
                    table_chunks = self._split_large_table(md, max_rows=3)
                    logger.info(f"  表格 {tbl_idx+1}/{len(tables)}: {len(table_chunks)} 段")

                    for seg_idx, tmd in enumerate(table_chunks):
                        if len(table_chunks) > 1:
                            logger.info(f"    段 {seg_idx+1}/{len(table_chunks)}...")
                        user_prompt = TABLE_PROMPT_TEMPLATE.format(
                            table_markdown=tmd, context=ctx or ""
                        )
                        prompt = build_gemma_prompt(user_prompt)
                        output = self._model(
                            prompt, max_tokens=4096, temperature=0,
                            stop=["<end_of_turn>", "<start_of_turn>"],
                        )
                        content = output["choices"][0]["text"].strip()
                        triplets = self._parse_response(content)
                        for t in triplets:
                            t["source_file"] = source_file
                            t["page_num"] = tbl.get("page_num", 1)
                            t["table_id"] = tbl.get("table_id", "")
                            t["data_type"] = "tabular"
                        table_count += len(triplets)
                        all_triplets.extend(triplets)
                logger.info(f"  表格完成: {table_count} 条")

            self.stats["triplets_raw"] += len(all_triplets)
            logger.info(f"  总原始三元组: {len(all_triplets)}")

            if not all_triplets:
                logger.warning("  未抽取到三元组，跳过")
                continue

            # ---- 规范化 ----
            logger.info("规范化...")
            normalized = normalizer.normalize(all_triplets)
            self.stats["triplets_normalized"] += len(normalized)
            logger.info(
                f"  -> {len(normalized)} 条 "
                f"(拆分+{normalizer.stats['triplets_split']}, "
                f"无效-{normalizer.stats['triplets_invalid']}, "
                f"去重-{normalizer.stats.get('triplets_deduped', 0)}, "
                f"匹配: {normalizer.stats['relations_matched_exact']}/"
                f"{normalizer.stats['relations_matched_keyword']}/"
                f"{normalizer.stats['relations_unmatched']})"
            )

            # 保存规范化结果
            norm_dir = Path("data/processed/normalize")
            norm_dir.mkdir(parents=True, exist_ok=True)
            result_path = norm_dir / f"{int_file.stem.replace('_intermediate', '')}_triplets.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(normalized, f, ensure_ascii=False, indent=2)
            logger.info(f"  规范化结果已保存: {result_path}")

            all_normalized.extend(normalized)

        # ---- 跨文件去重 ----
        if all_normalized:
            logger.info(f"\n跨文件汇总规范化 ({len(all_normalized)} 条)...")
            all_normalized = normalizer.normalize(all_normalized)
            self.stats["triplets_normalized"] = len(all_normalized)
            logger.info(
                f"  -> {len(all_normalized)} 条 "
                f"(拆分+{normalizer.stats['triplets_split']}, "
                f"无效-{normalizer.stats['triplets_invalid']}, "
                f"去重-{normalizer.stats.get('triplets_deduped', 0)})"
            )

            # 保存汇总规范化结果
            summary_dir = Path("data/processed/all_normalized_triplets")
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summary_dir / "all_triplets_normalized.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(all_normalized, f, ensure_ascii=False, indent=2)
            logger.info(f"  汇总结果已保存: {summary_path}")

        # ---- Neo4j 写入 ----
        if not dry_run and neo4j_client and all_normalized:
            logger.info(f"\nNeo4j 图谱写入 ({len(all_normalized)} 条, mode={mode})...")
            build_stats, db_stats = rebuild_graph(neo4j_client, all_normalized, mode=mode)
            logger.info(f"  -> 节点: {build_stats.get('nodes_created', 0)}, "
                        f"关系: {build_stats.get('relationships_created', 0)}")

        # ---- 最终统计 ----
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("Phase 2 完成!")
        logger.info(f"总耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")
        logger.info(f"处理 chunk: {self.stats['chunks_processed']} 个")
        logger.info(f"原始三元组: {self.stats['triplets_raw']} 条")
        logger.info(f"规范化后: {self.stats['triplets_normalized']} 条")
        logger.info("=" * 70)

        if neo4j_client:
            neo4j_client.close()

        return self.stats


def main():
    parser = argparse.ArgumentParser(description="HydroBrain Phase 2: LLM 抽取 + Neo4j")
    parser.add_argument(
        "--mode", choices=["full", "append"], default="append",
        help="Neo4j 模式: full=清空重建, append=增量追加"
    )
    parser.add_argument(
        "--input-dir", default="data/processed/text_extracted",
        help="中间文件目录 (默认: data/processed)"
    )
    parser.add_argument(
        "--model", default="models/gemma-4-E4B-it-Q5_K_M.gguf",
        help="GGUF 模型路径"
    )
    parser.add_argument("--dry-run", action="store_true", help="仅抽取，不写 Neo4j")
    parser.add_argument("--limit", type=int, default=0, help="限制处理 chunk 数（调试用）")
    parser.add_argument("--n-ctx", type=int, default=8192, help="上下文窗口大小")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    extractor = KGExtractor(
        model_path=args.model,
        n_ctx=args.n_ctx,
    )
    extractor.run(
        input_dir=args.input_dir,
        mode=args.mode,
        dry_run=args.dry_run,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
