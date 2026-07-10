"""
语义定义生成器 (Define 阶段)
为抽取到的实体类型和关系类型生成自然语言定义
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.common.logger import get_logger
from src.llm_pipeline.llm_client import DeepSeekClient

logger = get_logger(__name__)


class SemanticDefiner:
    """基于 LLM 的术语语义定义生成器"""

    def __init__(
        self,
        client: DeepSeekClient,
        prompts_dir: str = "prompts",
        embed_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        初始化定义生成器

        Args:
            client: DeepSeekClient 实例
            prompts_dir: Prompt 模板目录
            embed_model_name: SentenceTransformer 模型名（用于同义聚类）
        """
        self.client = client
        self.prompts_dir = Path(prompts_dir)
        self.embed_model_name = embed_model_name
        self._embed_model = None  # 延迟加载
        self._load_prompt()

    def _load_prompt(self):
        """加载 Define prompt 模板"""
        prompt_path = self.prompts_dir / "define.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        logger.info("Define prompt 模板加载完成")

    @property
    def embed_model(self):
        """延迟加载 SentenceTransformer（避免未安装时报错）"""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载 embedding 模型: {self.embed_model_name}")
            self._embed_model = SentenceTransformer(self.embed_model_name)
        return self._embed_model

    def define_relations(
        self,
        relations: List[str],
        context_text: str = "",
    ) -> Dict[str, str]:
        """
        批量为关系生成语义定义

        Args:
            relations: 关系短语列表
            context_text: 这些关系出现的原始上下文文本

        Returns:
            {关系名: 语义定义} 字典
        """
        unique_relations = list(set(r for r in relations if r))
        if not unique_relations:
            return {}

        relation_list_str = "\n".join(f"- {r}" for r in unique_relations)
        prompt = self.prompt_template.format(
            relation_list=relation_list_str,
            context_text=context_text or "无",
        )

        raw_response = self.client.chat(
            [{"role": "user", "content": prompt}],
        )

        definitions = self._parse_definitions(raw_response)
        logger.info(f"语义定义生成完成: {len(definitions)}/{len(unique_relations)} 个关系")
        return definitions

    def batch_define(
        self,
        triplets_list: List[List[Dict]],
        context_texts: List[str],
    ) -> List[Dict[str, str]]:
        """
        批量处理多段文本的三元组

        Args:
            triplets_list: 每段文本对应的三元组列表
            context_texts: 对应的上下文文本列表

        Returns:
            每段文本对应的 {关系: 定义} 字典列表
        """
        results = []
        for triplets, ctx in zip(triplets_list, context_texts):
            relations = [t.get("relation", "") for t in triplets]
            definitions = self.define_relations(relations, ctx)
            results.append(definitions)
        return results

    def cluster_similar_relations(
        self,
        definitions: Dict[str, str],
        threshold: float = 0.85,
    ) -> List[List[str]]:
        """
        基于 embedding 相似度对同义关系聚类

        Args:
            definitions: {关系: 定义} 字典
            threshold: 相似度阈值（高于此值视为同义）

        Returns:
            聚类结果 [[关系1, 关系2], ...]
        """
        if len(definitions) < 2:
            return [list(definitions.keys())]

        rel_names = list(definitions.keys())
        rel_defs = [definitions[r] for r in rel_names]

        embeddings = self.embed_model.encode(rel_defs, normalize_embeddings=True)
        sim_matrix = np.dot(embeddings, embeddings.T)

        # 简单贪心聚类
        visited = set()
        clusters = []
        for i, name in enumerate(rel_names):
            if name in visited:
                continue
            cluster = [name]
            visited.add(name)
            for j in range(i + 1, len(rel_names)):
                if rel_names[j] not in visited and sim_matrix[i][j] >= threshold:
                    cluster.append(rel_names[j])
                    visited.add(rel_names[j])
            clusters.append(cluster)

        logger.info(f"同义聚类结果: {len(clusters)} 组（阈值={threshold}）")
        return clusters

    def _parse_definitions(self, raw_text: str) -> Dict[str, str]:
        """
        解析 LLM 输出的关系定义文本
        期望格式: "关系名: 定义内容"
        """
        definitions = {}
        for line in raw_text.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            # 跳过非定义行
            if line.startswith("输出") or line.startswith("角色") or line.startswith("任务"):
                continue

            idx = line.index(":")
            rel_name = line[:idx].strip()
            rel_def = line[idx + 1:].strip()

            if rel_name and rel_def and rel_name != "Answer":
                definitions[rel_name] = rel_def

        return definitions
