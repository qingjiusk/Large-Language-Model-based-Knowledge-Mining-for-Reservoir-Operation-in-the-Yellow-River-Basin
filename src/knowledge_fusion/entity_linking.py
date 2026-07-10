"""
实体链接与消歧模块
将抽取的实体名称对齐到知识图谱中的标准实体，支持别名匹配和向量模糊匹配
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.logger import get_logger

logger = get_logger(__name__)


class EntityLinker:
    """实体链接器：将文本中的实体提及映射到知识库中的标准实体"""

    def __init__(
        self,
        alias_dict_path: str = "data/ontology/alias_dict.json",
        entity_types_path: str = "data/ontology/entity_types.json",
        embed_model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        min_similarity: float = 0.75,
    ):
        """
        初始化实体链接器

        Args:
            alias_dict_path: 别名字典 JSON 路径
            entity_types_path: 实体类型定义 JSON 路径
            embed_model_name: embedding 模型名
            device: 设备 ("cpu" / "cuda" / "cuda:0")
            min_similarity: 模糊匹配最小相似度
        """
        self.min_similarity = min_similarity
        self.embed_model_name = embed_model_name
        self.device = device
        self._embed_model = None

        # 加载别名字典
        self.alias_dict = self._load_json(alias_dict_path)
        self.entity_types = self._load_json(entity_types_path)

        # 构建别名 → 标准名反向索引
        self._alias_to_standard: Dict[str, str] = {}
        self._entity_type_map: Dict[str, str] = {}  # entity name -> type
        self._build_index()

        # 实体 embedding 缓存
        self._entity_names: List[str] = []
        self._entity_embeddings: Optional[np.ndarray] = None

        logger.info(
            f"EntityLinker 初始化: {len(self._alias_to_standard)} 个别名, "
            f"{len(self._entity_type_map)} 个实体"
        )

    @property
    def embed_model(self):
        """延迟加载 embedding 模型"""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(self.embed_model_name, device=self.device)
        return self._embed_model

    def _load_json(self, path: str) -> Dict:
        """加载 JSON 文件"""
        p = Path(path)
        if not p.exists():
            logger.warning(f"文件不存在: {p}")
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_index(self):
        """构建别名反向索引"""
        # 处理各种实体类型的别名
        for category in ["reservoirs", "hydrological_stations", "water_resource_zones",
                          "provinces", "rivers"]:
            category_dict = self.alias_dict.get(category, {})
            for standard_name, aliases in category_dict.items():
                self._alias_to_standard[standard_name] = standard_name
                for alias in aliases:
                    alias_lower = alias.strip().lower()
                    if alias_lower not in self._alias_to_standard:
                        self._alias_to_standard[alias_lower] = standard_name

    def link_entity(
        self,
        entity_name: str,
        entity_type_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        链接单个实体

        Args:
            entity_name: 原始实体名称
            entity_type_hint: 可选的实体类型提示

        Returns:
            {
                "original": str,
                "standardized": str,
                "matched": bool,
                "method": "exact" | "alias" | "fuzzy" | "none",
                "confidence": float
            }
        """
        original = entity_name.strip()
        if not original:
            return {
                "original": entity_name,
                "standardized": None,
                "matched": False,
                "method": "none",
                "confidence": 0.0,
            }

        # Step 1: 精确全名匹配（标准名直接命中）
        if original in self._alias_to_standard:
            std_name = self._alias_to_standard[original]
            if original == std_name:
                return {
                    "original": entity_name,
                    "standardized": std_name,
                    "matched": True,
                    "method": "exact",
                    "confidence": 1.0,
                }
            else:
                # 别名通过原始大小写命中
                return {
                    "original": entity_name,
                    "standardized": std_name,
                    "matched": True,
                    "method": "alias",
                    "confidence": 0.95,
                }

        # Step 2: 别名匹配（大小写不敏感）
        lower_name = original.lower()
        if lower_name in self._alias_to_standard:
            std_name = self._alias_to_standard[lower_name]
            return {
                "original": entity_name,
                "standardized": std_name,
                "matched": True,
                "method": "alias",
                "confidence": 0.95,
            }

        # Step 3: 模糊向量匹配
        if self._entity_embeddings is not None and len(self._entity_names) > 0:
            fuzzy_result = self._fuzzy_match(original)
            if fuzzy_result:
                return {
                    "original": entity_name,
                    "standardized": fuzzy_result[0],
                    "matched": True,
                    "method": "fuzzy",
                    "confidence": float(fuzzy_result[1]),
                }

        # 未匹配
        return {
            "original": entity_name,
            "standardized": original,
            "matched": False,
            "method": "none",
            "confidence": 0.0,
        }

    def link_entities_batch(
        self,
        entity_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        批量链接实体

        Args:
            entity_names: 实体名称列表

        Returns:
            链接结果列表
        """
        # 延迟构建 embedding 索引
        if self._entity_embeddings is None:
            self._build_embedding_index()

        results = [self.link_entity(name) for name in entity_names]

        matched = sum(1 for r in results if r["matched"])
        logger.info(f"批量实体链接: {len(entity_names)} 个实体, {matched} 个匹配")
        return results

    def link_triplets(
        self,
        triplets: List[Dict],
    ) -> List[Dict]:
        """
        对三元组列表中的 subject 和 object 做实体链接

        Args:
            triplets: 三元组列表

        Returns:
            subject 和 object 已标准化的三元组列表（附加 original_subject/original_object）
        """
        if not triplets:
            return []

        # 收集所有需要链接的实体
        subjects = list(set(t.get("subject", "") for t in triplets if t.get("subject")))
        objects = list(set(t.get("object", "") for t in triplets if t.get("object")))

        # 批量链接
        if self._entity_embeddings is None:
            self._build_embedding_index()

        subject_map = {s: self.link_entity(s) for s in subjects}
        object_map = {o: self.link_entity(o) for o in objects}

        # 替换
        linked = []
        for t in triplets:
            new_t = dict(t)
            subj = t.get("subject", "")
            obj = t.get("object", "")

            subj_result = subject_map.get(subj, {})
            obj_result = object_map.get(obj, {})

            new_t["original_subject"] = subj
            new_t["original_object"] = obj
            new_t["subject"] = subj_result.get("standardized", subj)
            new_t["object"] = obj_result.get("standardized", obj)
            new_t["subject_link_confidence"] = subj_result.get("confidence", 1.0)
            new_t["object_link_confidence"] = obj_result.get("confidence", 1.0)

            linked.append(new_t)

        logger.info(f"三元组实体链接完成: {len(triplets)} 条")
        return linked

    def _build_embedding_index(self):
        """构建所有标准实体的 embedding 索引"""
        all_names = set(self._alias_to_standard.values())
        self._entity_names = list(all_names)

        if not self._entity_names:
            logger.warning("无标准实体，跳过 embedding 索引构建")
            return

        logger.info(f"构建实体 embedding 索引: {len(self._entity_names)} 个实体")
        self._entity_embeddings = self.embed_model.encode(
            self._entity_names, normalize_embeddings=True
        )

    def _fuzzy_match(self, name: str) -> Optional[Tuple[str, float]]:
        """基于向量相似度的模糊匹配"""
        if self._entity_embeddings is None or len(self._entity_names) == 0:
            return None

        query_emb = self.embed_model.encode(name, normalize_embeddings=True)
        scores = np.dot(query_emb, self._entity_embeddings.T)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.min_similarity:
            return (self._entity_names[best_idx], best_score)
        return None

    def add_alias(self, standard_name: str, alias: str):
        """动态添加别名"""
        alias_lower = alias.strip().lower()
        self._alias_to_standard[alias_lower] = standard_name
        logger.debug(f"新增别名: '{alias}' -> '{standard_name}'")
