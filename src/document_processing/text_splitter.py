"""
文本智能切片器
将长文档按语义边界切分为适合 LLM 上下文窗口的文本块
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from src.common.logger import get_logger

logger = get_logger(__name__)


class TextSplitter:
    """基于语义边界的文档切片器"""

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """
        初始化切片器

        Args:
            chunk_size: 每个 chunk 的目标大小（字符数）
            chunk_overlap: 相邻 chunk 重叠的字符数
            separators: 切分优先级分隔符列表（靠前的优先）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", "；", ";", " "]

    def split_text(
        self,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        将单段文本切分为语义块

        Args:
            text: 输入文本
            metadata: 附加元信息（来源文档、页码、章节等）

        Returns:
            切片列表，每项: {"content": str, "chunk_id": str, "chunk_index": int, **metadata}
        """
        if not text or not text.strip():
            return []

        chunks = self._recursive_split(text)
        logger.debug(f"文本切片完成: {len(chunks)} chunks (原文本 {len(text)} 字符)")

        result = []
        for i, chunk_content in enumerate(chunks):
            chunk = {
                "content": chunk_content,
                "chunk_id": f"chunk_{i:04d}",
                "chunk_index": i,
                "char_count": len(chunk_content),
            }
            if metadata:
                chunk.update(metadata)
            result.append(chunk)

        return result

    def split_document(
        self,
        doc_result: Dict,
    ) -> List[Dict]:
        """
        将 PDFParser.extract_text_with_meta() 的解析结果切片

        Args:
            doc_result: PDFParser 返回的带元信息解析结果

        Returns:
            切片列表，每项包含来源页码
        """
        all_chunks = []
        file_name = doc_result.get("file_name", "unknown")

        for page in doc_result.get("pages", []):
            page_num = page["page_num"]
            page_text = page["text"]

            if not page_text.strip():
                continue

            page_meta = {
                "source_file": file_name,
                "source_path": doc_result.get("file_path", ""),
                "page_num": page_num,
            }

            page_chunks = self.split_text(page_text, metadata=page_meta)
            all_chunks.extend(page_chunks)

        # 重新编号 chunk_index
        for i, chunk in enumerate(all_chunks):
            chunk["chunk_index"] = i
            chunk["chunk_id"] = f"chunk_{i:04d}"

        logger.info(
            f"文档切片完成: {file_name}, "
            f"{doc_result.get('page_count', 0)} 页 -> {len(all_chunks)} chunks"
        )
        return all_chunks

    def split_batch(
        self,
        texts_dir: str,
        output_dir: str,
    ) -> List[Dict]:
        """
        批量处理目录下的所有 .txt 文件

        Args:
            texts_dir: 包含 .txt 文件的目录
            output_dir: 切片输出目录

        Returns:
            所有切片列表
        """
        texts_dir = Path(texts_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_files = list(texts_dir.glob("*.txt"))
        if not txt_files:
            logger.warning(f"目录中无 .txt 文件: {texts_dir}")
            return []

        all_chunks = []
        for txt_file in txt_files:
            logger.info(f"切片处理: {txt_file.name}")
            text = txt_file.read_text(encoding="utf-8")
            chunks = self.split_text(text, metadata={"source_file": txt_file.name})

            # 保存
            out_path = output_dir / f"{txt_file.stem}_chunks.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            all_chunks.extend(chunks)
            logger.info(f"保存: {out_path}, {len(chunks)} chunks")

        logger.info(f"批量切片完成: {len(txt_files)} 文件, {len(all_chunks)} chunks")
        return all_chunks

    def _recursive_split(self, text: str) -> List[str]:
        """递归按分隔符优先级切分，保证每个 chunk 不超过 chunk_size"""
        # 如果文本足够短，直接返回
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # 尝试用当前最优分隔符切分
        chunks = self._split_by_separator(text)
        return self._merge_chunks(chunks)

    def _split_by_separator(self, text: str) -> List[str]:
        """按分隔符切分文本"""
        for sep in self.separators:
            if sep in text:
                parts = text.split(sep)
                # 保留分隔符（加回非空段落）
                result = []
                for i, part in enumerate(parts):
                    if part.strip():
                        result.append(part.strip())
                if len(result) > 1:
                    return result

        # 所有分隔符都找不到，强制按长度切
        return self._force_split(text)

    def _force_split(self, text: str) -> List[str]:
        """强制按 chunk_size 切分"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i: i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def _merge_chunks(self, pieces: List[str]) -> List[str]:
        """将切分后的片段合并为不超过 chunk_size 的块"""
        if not pieces:
            return []

        chunks = []
        current = pieces[0]

        for piece in pieces[1:]:
            if len(current) + len(piece) + 1 <= self.chunk_size:
                current += "\n" + piece
            else:
                if current.strip():
                    chunks.append(current)
                # 处理单个 piece 就超过 chunk_size 的情况
                if len(piece) > self.chunk_size:
                    sub_chunks = self._force_split(piece)
                    # 将最后一小段设置为 current 以便合并到下一轮
                    if sub_chunks:
                        chunks.extend(sub_chunks[:-1])
                        current = sub_chunks[-1]
                    else:
                        current = ""
                else:
                    current = piece

        if current.strip():
            chunks.append(current)

        return chunks
