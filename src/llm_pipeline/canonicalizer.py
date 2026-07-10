"""
关系标准化器 (Canonicalize 阶段)
两阶段对齐：向量召回 Top-K + LLM 语义校验
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.logger import get_logger
from src.llm_pipeline.llm_client import DeepSeekClient

logger = get_logger(__name__)


class RelationCanonicalizer:
    """关系标准化器：将开放抽取的关系对齐到标准本体"""

    def __init__(
        self,
        client: DeepSeekClient,
        standard_relations: Dict[str, str],
        prompts_dir: str = "prompts",
        embed_model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        top_k: int = 3,
        mode: str = "strict",
        min_similarity: float = 0.7,
    ):
        """
        初始化标准化器

        Args:
            client: DeepSeekClient 实例
            standard_relations: 标准关系字典 {relation_id: description}
            prompts_dir: Prompt 模板目录
            embed_model_name: SentenceTransformer 模型名
            device: 设备 ("cpu" / "cuda" / "cuda:0")
            top_k: 向量召回候选数
            mode: "strict" 无匹配则丢弃, "extended" 无匹配标记为候选新关系
            min_similarity: 向量召回最小相似度阈值
        """
        self.client = client
        self.standard_relations = standard_relations
        self.prompts_dir = Path(prompts_dir)
        self.embed_model_name = embed_model_name
        self.device = device
        self.top_k = top_k
        self.mode = mode
        self.min_similarity = min_similarity

        self._embed_model = None
        self._rel_ids: List[str] = []
        self._rel_embeddings: Optional[np.ndarray] = None

        self._load_prompt()
        self._build_index()

    def _load_prompt(self):
        """加载 Canonicalize prompt 模板"""
        prompt_path = self.prompts_dir / "canonicalize.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        logger.info("Canonicalize prompt 模板加载完成")

    @property
    def embed_model(self):
        """延迟加载 SentenceTransformer"""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载 embedding 模型: {self.embed_model_name}, device={self.device}")
            self._embed_model = SentenceTransformer(self.embed_model_name, device=self.device)
        return self._embed_model

    def _build_index(self):
        """构建标准关系的向量索引"""
        if not self.standard_relations:
            logger.warning("标准关系为空，跳过索引构建")
            self._rel_ids = []
            self._rel_embeddings = None
            return

        self._rel_ids = list(self.standard_relations.keys())
        descriptions = [self.standard_relations[rid] for rid in self._rel_ids]

        logger.info(f"构建标准关系向量索引: {len(descriptions)} 个关系")
        self._rel_embeddings = self.embed_model.encode(
            descriptions, normalize_embeddings=True
        )

    def add_relation(self, relation_id: str, description: str):
        """动态添加新标准关系（extended 模式）"""
        if relation_id in self.standard_relations:
            return

        self.standard_relations[relation_id] = description
        new_emb = self.embed_model.encode(description, normalize_embeddings=True)

        if self._rel_embeddings is not None:
            self._rel_embeddings = np.vstack([self._rel_embeddings, new_emb])
        else:
            self._rel_embeddings = new_emb.reshape(1, -1)

        self._rel_ids.append(relation_id)
        logger.info(f"新增标准关系: {relation_id}")

    def retrieve_candidates(self, definition: str) -> List[Tuple[str, str, float]]:
        """
        向量召回 Top-K 候选标准关系

        Args:
            definition: 待对齐关系的语义定义

        Returns:
            [(relation_id, description, similarity_score), ...]
        """
        if self._rel_embeddings is None or len(self._rel_ids) == 0:
            return []

        query_emb = self.embed_model.encode(definition, normalize_embeddings=True)
        scores = np.dot(query_emb, self._rel_embeddings.T)

        # 按相似度降序排列
        top_indices = np.argsort(scores)[::-1]

        candidates = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < self.min_similarity:
                break
            candidates.append((
                self._rel_ids[idx],
                self.standard_relations[self._rel_ids[idx]],
                score,
            ))
            if len(candidates) >= self.top_k:
                break

        logger.debug(f"向量召回: query='{definition[:50]}...' -> {len(candidates)} candidates")
        return candidates

    def canonicalize(
        self,
        source_relation: str,
        source_definition: str,
    ) -> Dict[str, Any]:
        """
        两阶段对齐：向量召回 + LLM 语义校验

        Args:
            source_relation: 待对齐关系名
            source_definition: 待对齐关系的语义定义

        Returns:
            {
                "source_relation": str,
                "match_result": str (标准关系ID / "NEW_RELATION"),
                "confidence": float,
                "reason": str,
                "candidates": [...]
            }
        """
        # 如果关系已存在于标准库中，直接返回
        if source_relation in self.standard_relations:
            return {
                "source_relation": source_relation,
                "match_result": source_relation,
                "confidence": 1.0,
                "reason": "关系已存在于标准本体中",
                "candidates": [],
            }

        # 向量召回候选
        candidates = self.retrieve_candidates(source_definition)

        # 无候选，按模式处理
        if not candidates:
            return self._handle_no_match(source_relation, source_definition)

        # 构建候选列表文本给 LLM
        candidate_text = ""
        for i, (rid, desc, score) in enumerate(candidates):
            candidate_text += f"{i + 1}. {rid}: {desc} (相似度: {score:.2f})\n"

        # LLM 语义校验
        prompt = self.prompt_template.format(
            source_relation=source_relation,
            source_definition=source_definition.replace("{", "{{").replace("}", "}}"),
            candidate_relations=candidate_text,
        )

        try:
            result = self.client.extract_json(prompt)
            result["source_relation"] = source_relation
            result["candidates"] = [
                {"id": rid, "description": desc, "score": score}
                for rid, desc, score in candidates
            ]
        except Exception as e:
            logger.error(f"LLM 校验失败: {e}")
            result = self._fallback_match(source_relation, candidates)

        # Extended 模式：无匹配时自动注册新关系
        if result.get("match_result") == "NEW_RELATION" and self.mode == "extended":
            self.add_relation(source_relation, source_definition)
            logger.info(f"新关系已注册: {source_relation}")

        return result

    def batch_canonicalize(
        self,
        triplets: List[Dict],
        definitions: Dict[str, str],
    ) -> List[Dict]:
        """
        批量标准化三元组中的关系

        Args:
            triplets: 三元组列表
            definitions: {关系名: 语义定义} 字典

        Returns:
            标准化后的三元组列表（relation 字段替换为标准 ID）
        """
        # 先缓存所有唯一关系的标准化结果
        rel_map = {}
        unique_relations = set(t.get("relation", "") for t in triplets)

        for rel in unique_relations:
            if not rel:
                rel_map[rel] = rel
                continue
            definition = definitions.get(rel, rel)
            result = self.canonicalize(rel, definition)
            match = result.get("match_result", "NEW_RELATION")
            rel_map[rel] = rel if match == "NEW_RELATION" else match

        # 替换三元组中的关系
        canonicalized = []
        for t in triplets:
            new_t = dict(t)
            orig_rel = new_t.get("relation", "")
            new_t["relation"] = rel_map.get(orig_rel, orig_rel)
            new_t["original_relation"] = orig_rel
            canonicalized.append(new_t)

        logger.info(f"批量标准化完成: {len(unique_relations)} 个唯一关系")
        return canonicalized

    def _handle_no_match(self, relation: str, definition: str) -> Dict[str, Any]:
        """处理无候选匹配的情况"""
        if self.mode == "extended":
            self.add_relation(relation, definition)
            return {
                "source_relation": relation,
                "match_result": relation,
                "confidence": 1.0,
                "reason": "extended 模式：自动注册为新标准关系",
                "candidates": [],
            }
        else:
            return {
                "source_relation": relation,
                "match_result": "NEW_RELATION",
                "confidence": 0.0,
                "reason": f"向量召回无超过阈值({self.min_similarity})的候选",
                "candidates": [],
            }

    def _fallback_match(
        self,
        source_relation: str,
        candidates: List[Tuple[str, str, float]],
    ) -> Dict[str, Any]:
        """
        LLM 调用失败时的降级策略：选相似度最高的候选
        """
        if candidates:
            best = candidates[0]
            return {
                "source_relation": source_relation,
                "match_result": best[0],
                "confidence": best[2],
                "reason": f"降级匹配：选最高相似度候选 '{best[0]}' ({best[2]:.2f})",
                "candidates": [
                    {"id": rid, "description": desc, "score": score}
                    for rid, desc, score in candidates
                ],
            }
        else:
            return {
                "source_relation": source_relation,
                "match_result": "NEW_RELATION",
                "confidence": 0.0,
                "reason": "降级：无候选，标记为新关系",
                "candidates": [],
            }
