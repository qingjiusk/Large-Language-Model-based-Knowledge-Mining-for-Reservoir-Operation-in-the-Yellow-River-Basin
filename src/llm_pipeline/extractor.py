"""
三元组抽取器 (Extract 阶段)
支持正文文本和表格两种输入模式
"""
import json
from pathlib import Path
from typing import Any, Dict, List

from src.common.logger import get_logger
from src.llm_pipeline.llm_client import DeepSeekClient

logger = get_logger(__name__)


class TripletExtractor:
    """基于 DeepSeek LLM 的三元组开放抽取器"""

    def __init__(self, client: DeepSeekClient, prompts_dir: str = "prompts"):
        """
        初始化抽取器

        Args:
            client: DeepSeekClient 实例
            prompts_dir: Prompt 模板目录路径
        """
        self.client = client
        self.prompts_dir = Path(prompts_dir)
        self._load_prompts()

    def _load_prompts(self):
        """从文件加载 prompt 模板"""
        def _read(name):
            p = self.prompts_dir / name
            if not p.exists():
                raise FileNotFoundError(f"Prompt 文件不存在: {p}")
            return open(p, encoding="utf-8").read()
        def _opt(name):
            p = self.prompts_dir / name
            return open(p, encoding="utf-8").read() if p.exists() else None

        self.text_prompt_template = _read("extract.txt")
        self.table_prompt_template = _read("table_extract.txt")
        self.batch_prompt_template = _opt("extract_batch.txt")
        self.batch_table_template = _opt("table_extract_batch.txt")

    def extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        从正文文本中抽取三元组

        Args:
            text: 正文文本块

        Returns:
            三元组列表 [{"subject": ..., "relation": ..., "object": ..., "context": ..., "confidence": ...}]
        """
        prompt = self.text_prompt_template.format(
            text_chunk=self._escape_format(text)
        )
        result = self.client.extract_json(prompt)

        triplets = self._normalize_result(result)
        logger.debug(f"文本抽取完成: {len(triplets)} 个三元组")
        return self._tag_source(triplets, "text")

    def extract_from_text_batch(
        self,
        chunks: List[Dict],
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        批量抽取：一次 API 调用处理多个 chunk

        Args:
            chunks: chunk 列表，每项需含 {"chunk_id": str, "content": str}
            batch_size: 每批处理的最大 chunk 数

        Returns:
            所有三元组列表（自动附加 chunk_id 溯源信息）
        """
        if not self.batch_prompt_template:
            # 无批量模板时回退到逐个抽取
            logger.warning("无批量 Prompt 模板，回退到逐个抽取")
            all_triplets = []
            for chunk in chunks:
                triplets = self.extract_from_text(chunk["content"])
                for t in triplets:
                    t["chunk_id"] = chunk.get("chunk_id", "")
                    t["page_num"] = chunk.get("page_num", 1)
                all_triplets.extend(triplets)
            return all_triplets

        all_triplets = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_triplets = self._extract_batch(batch)
            all_triplets.extend(batch_triplets)
            logger.debug(
                f"批量抽取: 批次 {i // batch_size + 1}, "
                f"{len(batch)} chunks → {len(batch_triplets)} 三元组"
            )

        logger.info(f"批量抽取完成: {len(chunks)} chunks → {len(all_triplets)} 三元组")
        return all_triplets

    def _extract_batch(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """处理单个批次"""
        # 构建批量文本
        batch_parts = []
        for chunk in chunks:
            cid = chunk.get("chunk_id", f"chunk_{chunks.index(chunk):04d}")
            content = chunk["content"].strip()
            batch_parts.append(f"[ID:{cid}]\n{content}")

        batch_text = "\n\n---\n\n".join(batch_parts)

        # 调用 LLM
        prompt = self.batch_prompt_template.format(
            batch_texts=self._escape_format(batch_text)
        )
        result = self.client.extract_json(prompt)

        # 解析批量结果
        return self._parse_batch_result(result, chunks)

    def _parse_batch_result(
        self,
        result: Any,
        chunks: List[Dict],
    ) -> List[Dict[str, Any]]:
        """解析批量抽取的返回结果"""
        # 建立 chunk_id → chunk 元信息映射
        chunk_meta = {}
        for c in chunks:
            cid = c.get("chunk_id", "")
            chunk_meta[cid] = {
                "page_num": c.get("page_num", 1),
                "source_file": c.get("source_file", ""),
            }

        all_triplets = []

        # 格式1: {"results": [{"chunk_id": ..., "triplets": [...]}, ...]}
        if isinstance(result, dict) and "results" in result:
            for item in result["results"]:
                cid = item.get("chunk_id", "")
                triplets = item.get("triplets", [])
                meta = chunk_meta.get(cid, {})
                for t in triplets:
                    t["chunk_id"] = cid
                    t.setdefault("page_num", meta.get("page_num", 1))
                    t.setdefault("source_file", meta.get("source_file", ""))
                all_triplets.extend(triplets)

        # 格式2: 直接返回三元组列表（无 chunk_id 映射）
        elif isinstance(result, list):
            logger.warning("批量抽取返回了无分组的列表，无法溯源 chunk")
            all_triplets = result

        # 格式3: JSON 解析失败的回退 — 逐个抽取
        else:
            logger.warning("批量结果格式异常，回退逐个抽取")
            for chunk in chunks:
                triplets = self.extract_from_text(chunk["content"])
                cid = chunk.get("chunk_id", "")
                for t in triplets:
                    t["chunk_id"] = cid
                    t["page_num"] = chunk.get("page_num", 1)
                all_triplets.extend(triplets)

        return self._tag_source(self._normalize_result(all_triplets), "text")

    def extract_from_table(self, table_markdown: str, context: str = "") -> List[Dict[str, Any]]:
        """从单个表格 Markdown 中抽取三元组"""
        prompt = self.table_prompt_template.format(
            table_markdown=self._escape_format(table_markdown),
            context=self._escape_format(context or "无"),
        )
        triplets = self._normalize_result(self.client.extract_json(prompt))
        return self._tag_source(triplets, "tabular")

    def extract_from_table_batch(
        self, tables: List[Dict], batch_size: int = 6
    ) -> List[Dict[str, Any]]:
        """批量表格抽取：一次 API 调用处理多个表格"""
        if not self.batch_table_template:
            # 无批量模板时回退逐个
            all_t = []
            for tbl in tables:
                all_t.extend(self.extract_from_table(
                    tbl.get("markdown", ""), tbl.get("context", "")
                ))
            return all_t

        all_triplets = []
        for i in range(0, len(tables), batch_size):
            batch = tables[i:i + batch_size]
            batch_parts = []
            for tbl in batch:
                tid = tbl.get("table_id", f"tbl_{i}")
                batch_parts.append(f"[TABLE_ID:{tid}]\n{tbl.get('markdown', '')}")

            prompt = self.batch_table_template.format(
                batch_tables=self._escape_format("\n\n---\n\n".join(batch_parts))
            )
            result = self.client.extract_json(prompt)

            # 解析批量表格结果
            if isinstance(result, dict) and "results" in result:
                for item in result["results"]:
                    for t in item.get("triplets", []):
                        t["table_id"] = item.get("table_id", "")
                    all_triplets.extend(item.get("triplets", []))
            else:
                # 回退逐个
                for tbl in batch:
                    all_triplets.extend(self.extract_from_table(
                        tbl.get("markdown", ""), tbl.get("context", "")
                    ))

        logger.info(f"批量表格完成: {len(tables)} 表 → {len(all_triplets)} 三元组")
        return self._tag_source(self._normalize_result(all_triplets), "tabular")

    def _normalize_result(self, result: Any) -> List[Dict[str, Any]]:
        """
        归一化 LLM 输出，统一返回三元组列表
        """
        if isinstance(result, list):
            triplets = result
        elif isinstance(result, dict):
            # 兼容 {"triplets": [...]} 包裹格式
            triplets = result.get("triplets", result.get("data", []))
        else:
            triplets = []

        # 过滤无效条目
        valid = []
        for t in triplets:
            if not isinstance(t, dict):
                continue
            if not all(k in t for k in ("subject", "relation", "object")):
                continue
            if not t.get("subject") or not t.get("relation") or not t.get("object"):
                continue
            # 补全默认字段
            t.setdefault("context", "")
            t.setdefault("confidence", 0.5)
            t.setdefault("data_type", "text")
            t.setdefault("year", None)  # 新增：年份字段，None 表示 LLM 未提供
            valid.append(t)

        return valid

    def _tag_source(self, triplets: List[Dict], source_type: str) -> List[Dict]:
        """为三元组标记来源类型"""
        for t in triplets:
            t["data_type"] = source_type
        return triplets

    @staticmethod
    def _escape_format(text: str) -> str:
        """转义文本中的 {} 字符，防止 .format() 崩溃"""
        return text.replace("{", "{{").replace("}", "}}")
