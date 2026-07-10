"""
PDF 表格解析器
基于 pdfplumber，从电子版 PDF 中提取表格并转换为 Markdown
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pdfplumber

from src.common.logger import get_logger

logger = get_logger(__name__)


class TableParser:
    """从 PDF 中提取表格，支持 Markdown 和 DataFrame 双模式输出"""

    def __init__(
        self,
        min_table_rows: int = 2,
        output_format: str = "markdown",
        clean_cells: bool = True,
    ):
        """
        初始化表格解析器

        Args:
            min_table_rows: 最少行数（低于此行数不视为有效表格）
            output_format: 输出格式 "markdown" / "dataframe" / "both"
            clean_cells: 是否清洗单元格（去换行符、首尾空格）
        """
        self.min_table_rows = min_table_rows
        self.output_format = output_format
        self.clean_cells = clean_cells

    def extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        提取 PDF 中所有表格

        Args:
            pdf_path: PDF 文件路径

        Returns:
            表格信息列表，每项包含:
            - page: 页码
            - table_id: 表格唯一标识
            - source_file: 来源文件名
            - row_count / col_count: 行列数
            - markdown: Markdown 格式表格 (output_format=markdown)
            - dataframe: DataFrame (output_format=dataframe)
            - raw_data: 二维数组
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        all_tables = []

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()

                    if not tables:
                        continue

                    for idx, table in enumerate(tables):
                        if not table or len(table) < self.min_table_rows:
                            continue

                        # 清洗
                        cleaned = self._clean_table(table)
                        if not cleaned or len(cleaned) < self.min_table_rows:
                            continue

                        # 补齐列数不一致的行
                        cleaned = self._normalize_columns(cleaned)

                        table_info = {
                            "page": page_num,
                            "table_id": f"tbl_p{page_num}_{idx}",
                            "source_file": pdf_path.name,
                            "row_count": len(cleaned),
                            "col_count": len(cleaned[0]) if cleaned else 0,
                            "raw_data": cleaned,
                        }

                        if self.output_format in ("markdown", "both"):
                            table_info["markdown"] = self._to_markdown(cleaned)

                        if self.output_format in ("dataframe", "both"):
                            try:
                                df = pd.DataFrame(cleaned[1:], columns=cleaned[0])
                                table_info["dataframe"] = df
                            except Exception as e:
                                logger.warning(f"DataFrame 转换失败: {e}")

                        all_tables.append(table_info)

            logger.info(
                f"表格提取完成: {pdf_path.name}, "
                f"共 {len(all_tables)} 个表格"
            )
            return all_tables

        except Exception as e:
            logger.error(f"表格提取失败: {pdf_path.name}, 错误: {e}")
            raise

    def extract_tables_batch(
        self,
        pdf_dir: str,
        output_dir: str,
        save_format: str = "json",
    ) -> Dict[str, List[Dict]]:
        """
        批量提取目录下所有 PDF 的表格

        Args:
            pdf_dir: 包含 PDF 文件的目录
            output_dir: JSON/CSV 输出目录
            save_format: 保存格式 "json" / "csv"

        Returns:
            {文件名: [表格列表]}
        """
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
        all_results = {}

        for pdf_file in pdf_files:
            logger.info(f"批量表格提取: {pdf_file.name}")
            try:
                tables = self.extract_tables(str(pdf_file))
                all_results[pdf_file.name] = tables

                # 保存
                if save_format == "json":
                    # 移除 DataFrame 对象（不可序列化）
                    serializable = []
                    for t in tables:
                        t_copy = {k: v for k, v in t.items() if k != "dataframe"}
                        serializable.append(t_copy)

                    out_path = output_dir / f"{pdf_file.stem}_tables.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(serializable, f, ensure_ascii=False, indent=2)

                elif save_format == "csv":
                    for t in tables:
                        out_path = output_dir / f"{pdf_file.stem}_{t['table_id']}.csv"
                        t.get("dataframe", pd.DataFrame()).to_csv(
                            out_path, index=False, encoding="utf-8-sig"
                        )

                logger.info(f"保存完成: {pdf_file.stem}, {len(tables)} 个表格")
            except Exception as e:
                logger.error(f"表格提取失败: {pdf_file.name}, 错误: {e}")
                all_results[pdf_file.name] = []

        total = sum(len(v) for v in all_results.values())
        logger.info(f"批量表格提取完成: {len(pdf_files)} 个文件, {total} 个表格")
        return all_results

    def _clean_table(self, table: List[List]) -> List[List]:
        """清洗表格数据"""
        cleaned = []
        for row in table:
            if not row:
                continue
            new_row = []
            for cell in row:
                if cell is None:
                    cell = ""
                if self.clean_cells:
                    cell = str(cell).strip().replace("\n", " ").replace("\r", "")
                new_row.append(cell)
            # 跳过全空行
            if any(c for c in new_row):
                cleaned.append(new_row)
        return cleaned

    def _normalize_columns(self, table: List[List]) -> List[List]:
        """确保所有行列数一致（以第一行为准）"""
        if not table:
            return table
        max_cols = max(len(row) for row in table)
        normalized = []
        for row in table:
            if len(row) < max_cols:
                row = row + [""] * (max_cols - len(row))
            normalized.append(row)
        return normalized

    def _to_markdown(self, table: List[List]) -> str:
        """二维数组转 Markdown 表格"""
        if not table:
            return ""

        header = [str(c) if c else "" for c in table[0]]
        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]

        for row in table[1:]:
            cells = [str(c) if c else "" for c in row]
            # 补齐列数
            while len(cells) < len(header):
                cells.append("")
            md_lines.append("| " + " | ".join(cells[:len(header)]) + " |")

        return "\n".join(md_lines)
