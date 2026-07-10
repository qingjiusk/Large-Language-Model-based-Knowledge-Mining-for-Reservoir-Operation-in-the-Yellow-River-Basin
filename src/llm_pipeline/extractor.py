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
        text_prompt_path = self.prompts_dir / "extract.txt"
        table_prompt_path = self.prompts_dir / "table_extract.txt"

        if not text_prompt_path.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {text_prompt_path}")
        if not table_prompt_path.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {table_prompt_path}")

        with open(text_prompt_path, "r", encoding="utf-8") as f:
            self.text_prompt_template = f.read()

        with open(table_prompt_path, "r", encoding="utf-8") as f:
            self.table_prompt_template = f.read()

        logger.info(f"Prompt 模板加载完成: {len(self.text_prompt_template)} / {len(self.table_prompt_template)} chars")

    def extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        从正文文本中抽取三元组

        Args:
            text: 正文文本块

        Returns:
            三元组列表 [{"subject": ..., "relation": ..., "object": ..., "context": ..., "confidence": ...}]
        """
        prompt = self.text_prompt_template.format(text_chunk=text)
        result = self.client.extract_json(prompt)

        triplets = self._normalize_result(result)
        logger.debug(f"文本抽取完成: {len(triplets)} 个三元组")
        return self._tag_source(triplets, "text")

    def extract_from_table(self, table_markdown: str, context: str = "") -> List[Dict[str, Any]]:
        """
        从表格 Markdown 中抽取三元组

        Args:
            table_markdown: Markdown 格式的表格内容
            context: 表格所在文档上下文（如章节标题、说明文字）

        Returns:
            三元组列表
        """
        prompt = self.table_prompt_template.format(
            table_markdown=table_markdown,
            context=context or "无",
        )
        result = self.client.extract_json(prompt)

        triplets = self._normalize_result(result)
        logger.debug(f"表格抽取完成: {len(triplets)} 个三元组")
        return self._tag_source(triplets, "tabular")

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
            valid.append(t)

        return valid

    def _tag_source(self, triplets: List[Dict], source_type: str) -> List[Dict]:
        """为三元组标记来源类型"""
        for t in triplets:
            t["data_type"] = source_type
        return triplets
