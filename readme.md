项目名称：基于大语言模型的黄河水库调度知识的挖掘

虚拟环境 conda activate zhishi


本次更新完全保留原有 EDC 核心架构与开发节奏，仅新增「矢量 PDF 表格解析→结构化文本→三元组抽取」纯文本链路，全程不依赖多模态大模型，适配《黄河水资源公报》这类官方电子版 PDF。核心思路是：表格优先工具解析，转成 Markdown 文本后复用原有抽取、标准化、融合、入库全链路。
一、更新后 GitHub Issue 开发任务清单
Milestone 0：项目初始化（更新）
Issue #001 工程基础环境搭建（更新）
目标：完成 Python 项目标准化初始化，打通开发环境
更新任务：
新增 pdfplumber 依赖，锁定版本
配置文件新增表格解析参数开关
补充表格处理模块目录
验收标准：环境一键安装，表格解析依赖可用
更新文件：requirements.txt、config/config.yaml
requirements.txt 更新（新增依赖）
txt
langchain
langchain-openai
neo4j
fastapi
uvicorn
pymupdf
pdfplumber      # 新增：高精度表格解析
pandas
numpy
pydantic
sentence-transformers
faiss-cpu
python-dotenv
tqdm
openpyxl
pyyaml
Milestone 1：水库调度文档处理系统（更新）
Issue #101 PDF 文档解析模块（不变）
目标：实现 PDF → 纯文本的稳定转换，保留段落结构
任务清单：
实现 PDFParser 类，支持单文件 / 批量文件解析
增加页眉页脚、目录、图表标注过滤逻辑
保留章节层级信息（可选，基于标题字号）
补充异常处理（文件损坏、加密 PDF 报错提示）
验收标准：单篇调度规程 PDF 解析准确率 ≥95%，无乱码、缺页
关联文件：src/document_processing/pdf_parser.py
Issue #102 文本智能切片模块（不变）
目标：长文档切分为语义完整的文本块，适配 LLM 上下文窗口
任务清单：
基于递归字符分割实现基础切片，支持自定义块大小与重叠
增加语义边界优化（优先在段落、章节处切分）
为每个 chunk 附加元信息（来源文档、页码、章节）
支持批量处理与结果持久化
验收标准：10 万字文档切片无断点，chunk 语义完整度达标
关联文件：src/document_processing/text_splitter.py
Issue #103 表格解析与结构化模块（新增）
目标：从电子版 PDF 中批量提取表格，转换为标准 Markdown 文本，供 LLM 抽取
任务清单：
实现 TableParser 类，基于 pdfplumber 提取单页 / 跨页表格
支持表格输出为 Markdown、二维数组两种格式
为每个表格附加元信息：页码、表格 ID、所属文档
增加简单表头修复、空行过滤、合并单元格兼容逻辑
支持批量目录处理，结果持久化到本地
验收标准：公报类标准表格提取准确率 ≥95%，数值、单位无丢失
关联文件：src/document_processing/table_parser.py
Milestone 2：LLM 知识抽取系统（更新）
Issue #201 开放信息抽取（Extract 阶段）（更新）
新增任务：
新增表格专用抽取 Prompt，强化数值、单位、指标的准确性
抽取器支持传入「文本块 / 表格内容」两种输入，自动适配 Prompt
表格三元组额外标记 data_type: "tabular"，便于后续冲突优先级判定
验收标准：标准参数表抽取准确率 ≥98%，数值单位完整
Issue #202 语义定义生成（Define 阶段）（不变）
目标：为抽取到的实体类型、关系类型生成自然语言定义，消除歧义
任务清单：
编写定义生成 Prompt，要求结合上下文给出精准领域语义
实现 SemanticDefiner 类，批量生成实体 / 关系的语义描述
对同义不同名的关系 / 实体做初步聚类（基于嵌入向量相似度）
输出「术语 - 定义 - 上下文」映射表
验收标准：相同语义关系的定义相似度 ≥0.85，多义词可区分
关联文件：src/llm_pipeline/definer.py、prompts/define.txt
Issue #203 关系标准化（Canonicalize 阶段）（不变）
目标：将开放抽取的零散关系，对齐到标准本体，统一术语
任务清单：
构建标准关系的向量索引库（基于 sentence-transformers）
实现「向量召回 Top3 + LLM 语义校验」两阶段对齐逻辑
支持两种模式：严格对齐（无匹配则丢弃）、扩展模式（无匹配则标记为候选新关系）
输出标准化后的三元组数据集
验收标准：关系对齐准确率 ≥90%，无过度合并错误
关联文件：src/llm_pipeline/canonicalizer.py、prompts/canonicalize.txt
Milestone 3：领域本体构建（更新）
Issue #301 黄河水库调度领域基础本体（更新）
新增实体类型：水文站、水资源二级区、省级行政区、年度水文数据
新增关系类型：隶属关系、水文数据关联、空间分区关系
验收标准：覆盖公报中所有核心实体类型，支持降水量、径流量、输沙量等指标存储
Milestone 4：知识融合（更新）
Issue #401 实体链接与消歧模块（更新）
新增任务：
补充水文站别名字典（全称 / 简称 / 俗称）
补充水资源二级区简称映射（如「龙库以上」→「龙羊峡以上」）
支持省级行政区全称 / 简称归一化
Issue #402 知识冲突检测与处理（更新）
新增任务：
新增「表格 vs 正文」数值冲突检测，表格来源默认优先级更高
支持同指标不同年份数据的区分，避免误判冲突
冲突标记附带模态来源（text/table），便于人工复核
Milestone 5~7：图谱、API、MVP（基本不变）
Issue #501 Neo4j 客户端与图谱操作封装
目标：封装 Neo4j 连接、增删改查基础能力
任务清单：
实现 Neo4jClient 类，支持连接池、关闭、事务执行
封装节点批量创建、关系批量创建方法
实现实体去重写入（存在则更新，不存在则创建）
增加索引与唯一约束创建脚本
验收标准：千条三元组批量写入无报错，无重复节点
关联文件：src/knowledge_graph/neo4j_client.py
Issue #502 图谱构建流水线
目标：打通「标准化三元组 → Neo4j 入库」全流程
任务清单：
实现 GraphBuilder 类，串联实体链接、冲突检测、图谱写入
支持全量构建与增量文档追加两种模式
写入时保留知识溯源（来源文档、页码、抽取置信度）
提供构建日志与质量统计（节点数、关系数、冲突数）
验收标准：单篇文档可一键完成解析→抽取→入库全流程
关联文件：src/knowledge_graph/graph_builder.py
Issue #503 图谱查询封装
目标：封装常用业务查询 Cypher，供上层 API 调用
任务清单：
水库基础信息查询
水库调度规则 / 约束 / 目标查询
两实体间路径查询
按文档来源查询知识
验收标准：常用查询响应时间 <200ms
关联文件：src/knowledge_graph/graph_query.py
Milestone 6：API 服务与 MVP 验证（优先级：中）
Issue #601 FastAPI 知识服务接口
目标：对外提供 RESTful 知识查询接口
任务清单：
搭建 FastAPI 工程，拆分路由模块
实现水库信息查询、约束查询、规则查询接口
实现三元组检索、文档溯源查询接口
接入统一响应格式与异常处理
自动生成接口文档
验收标准：接口可正常调用，返回格式规范，文档可访问
关联文件：src/api/main.py、src/api/routers/
Issue #602 MVP 端到端验证
目标：用真实调度文档跑通全链路，验证系统可用性
任务清单：
选取 1-2 篇典型水库调度规程作为测试数据
执行全量构建流水线，生成知识图谱
验证核心查询接口的准确性与完整性
输出 MVP 测试报告，记录问题与优化点
验收标准：核心知识可正确查询，全链路无阻塞性 bug
Neo4j 客户端与构建流水线：扩展节点 / 关系类型，兼容新增实体
API 服务：新增水文站查询、分区数据查询接口
MVP 验证：以《黄河水资源公报 2024》为测试集，跑通表格→抽取→入库→查询全链路
二、更新后项目目录结构
plaintext
HydroBrain/
├── config/
│   └── config.yaml                 # 新增表格解析配置项
├── data/
│   ├── raw/                        # 原始PDF文档
│   ├── processed/
│   │   ├── texts/                  # 解析后的纯文本
│   │   ├── chunks/                 # 文本切片结果
│   │   └── tables/                 # 提取后的表格Markdown/CSV
│   └── ontology/
│       ├── entity_types.json       # 扩补水文站、分区等实体
│       ├── relation_types.json     # 扩补水文数据关系
│       └── alias_dict.json         # 新增：别名字典
├── prompts/
│   ├── extract.txt                 # 正文通用抽取
│   ├── table_extract.txt           # 新增：表格专用抽取
│   ├── define.txt
│   └── canonicalize.txt
├── scripts/
│   ├── build_kg_from_dir.py
│   └── extract_tables_batch.py     # 新增：批量表格提取脚本
├── src/
│   ├── __init__.py
│   ├── common/                     # 通用工具（不变）
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   └── vector_utils.py
│   ├── document_processing/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py           # 原有：纯文本提取
│   │   ├── text_splitter.py        # 原有：文本切片
│   │   └── table_parser.py         # 新增：表格解析
│   ├── llm_pipeline/
│   │   ├── __init__.py
│   │   ├── extractor.py            # 更新：支持表格抽取模式
│   │   ├── definer.py
│   │   └── canonicalizer.py
│   ├── knowledge_fusion/
│   │   ├── __init__.py
│   │   ├── entity_linking.py       # 更新：支持水文站/分区别名
│   │   └── conflict_resolver.py    # 更新：跨来源数值冲突裁决
│   ├── knowledge_graph/
│   │   ├── __init__.py
│   │   ├── neo4j_client.py
│   │   ├── graph_builder.py        # 更新：支持表格三元组批量入库
│   │   └── graph_query.py          # 更新：新增水文数据查询
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       └── routers/
│           ├── reservoir.py
│           ├── hydrology.py        # 新增：水文数据接口
│           └── knowledge.py
├── tests/
│   └── test_table_parser.py        # 新增：表格解析单元测试
├── .env.example
├── requirements.txt
└── README.md
三、更新后 Neo4j 知识图谱 Schema
在原有水库、规则、约束等核心实体基础上，扩展水资源公报场景专属实体与关系，实现「调度规则 + 水文数据」一体化知识图谱。
1. 新增核心节点类型
表格
节点标签	核心属性	说明
HydrologicalStation（水文站）	id、name、alias、river、control_area（控制面积，km²）、station_type（干流 / 支流）	径流量、输沙量观测节点
WaterResourceZone（水资源分区）	id、name、alias、level（二级区）、area（计算面积，万 km²）	降水、用水统计单元
Province（省级行政区）	id、name、alias	供水、用水统计单元
AnnualHydrologyData（年度水文数据）	id、year、indicator（指标：降水量 / 径流量 / 输沙量等）、value、unit、compare_last_year、compare_baseline	量化指标节点
2. 新增核心关系类型
表格
关系类型	起点 → 终点	语义说明
LOCATED_IN_ZONE	Reservoir/HydrologicalStation → WaterResourceZone	水库 / 水文站属于某水资源分区
BELONGS_TO_PROVINCE	Reservoir/HydrologicalStation → Province	位于某行政区
HAS_ANNUAL_DATA	水库 / 水文站 / 分区 → AnnualHydrologyData	某实体对应某年度指标数据
IS_TRIBUTARY_OF	River → River	支流隶属关系
CONTROLS	HydrologicalStation → River	水文站控制某河段
3. 索引与约束更新
cypher
// 唯一约束
CREATE CONSTRAINT FOR (s:HydrologicalStation) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT FOR (z:WaterResourceZone) REQUIRE z.id IS UNIQUE;

// 检索索引
CREATE INDEX FOR (s:HydrologicalStation) ON (s.name);
CREATE INDEX FOR (d:AnnualHydrologyData) ON (d.year, d.indicator);
四、更新后 LLM Prompt 模板体系
新增表格专用抽取 Prompt，保证数值、单位、指标的精准性，其余 Define/Canonicalize 模板完全复用。
新增：表格抽取专用 Prompt（prompts/table_extract.txt）
plaintext
角色：你是资深黄河水文水资源专家，擅长从统计表格中精准提取知识三元组。
任务：从给定的表格内容中，抽取所有明确的实体、关系与数值，严格基于表格原文，不得推演。

输出要求：
1. 三元组格式：{"subject": "主体实体", "relation": "关系/指标", "object": "客体/数值"}
2. 数值必须附带完整单位，百分比明确标注是同比/距平
3. 保留时间维度（如2024年）、空间维度（如兰州水文站）
4. 同步输出 context（对应行列描述）、confidence（置信度0-1）
5. 最终输出严格 JSON 数组，无多余文字、无解释

表格内容（Markdown格式）：
{table_markdown}

文档上下文：{context}

输出示例：
[
  {{
    "subject": "兰州水文站",
    "relation": "2024年实测径流量",
    "object": "362.80亿立方米",
    "context": "黄河干流主要水文站实测径流量表",
    "confidence": 0.99
  }}
]
3. 索引与约束设计
cypher
// 唯一约束
CREATE CONSTRAINT FOR (r:Reservoir) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT FOR (r:Rule) REQUIRE r.id IS UNIQUE;
CREATE CONSTRAINT FOR (d:Document) REQUIRE d.id IS UNIQUE;

// 检索索引
CREATE INDEX FOR (r:Reservoir) ON (r.name);
CREATE INDEX FOR (c:Constraint) ON (c.variable);
四、LLM Prompt 模板体系（对齐 EDC 三阶段）
1. 抽取阶段 Prompt（extract.txt）
plaintext
角色：你是资深黄河流域水库调度专家，精通水文调度规程、工程运行规范。
任务：从给定文本中抽取所有明确表述的知识三元组，严格基于原文，不得编造、推演。

输出要求：
1. 三元组格式：{"subject": "主体", "relation": "关系", "object": "客体"}
2. 仅抽取原文明确提到的事实，不补充背景知识，不做推断
3. 关系尽量使用动词短语，贴合调度领域表述习惯
4. 同步输出该三元组对应的原文上下文片段、置信度(0-1)
5. 最终输出严格 JSON 数组格式，无多余文字

待处理文本：
{text_chunk}

输出示例：
[
  {{
    "subject": "小浪底水库",
    "relation": "汛期限制水位",
    "object": "275米",
    "context": "小浪底水库汛期防洪限制水位为275米",
    "confidence": 0.98
  }}
]
2. 定义阶段 Prompt（define.txt）
plaintext
角色：你是水利工程术语专家，负责为调度领域的关系术语生成精准语义定义。
任务：根据给定的关系短语及其出现的上下文，生成一句严谨的自然语言定义，明确该关系的主体类型、客体类型、语义内涵。

输入：
关系短语：{relation_phrase}
上下文示例：{context_examples}

输出要求：
1. 定义格式："[主体类型] + [关系语义] + [客体类型]"
2. 区分近义关系的细微差别（如"限制水位"vs"警戒水位"）
3. 输出纯文本定义，无多余格式
3. 标准化阶段 Prompt（canonicalize.txt）
plaintext
角色：你是知识图谱本体对齐专家，负责将抽取到的关系映射到标准关系体系。
任务：对比待对齐关系与候选标准关系的语义定义，选择最匹配的一项；若无匹配项则选择"新增候选关系"。

输入：
待对齐关系：{source_relation}
待对齐关系定义：{source_definition}

候选标准关系列表：
{candidate_relations}

输出要求：
1. 严格输出 JSON 格式，包含字段：
   - match_result: 标准关系ID / "NEW_RELATION"
   - confidence: 匹配置信度(0-1)
   - reason: 匹配/不匹配的简短理由
2. 语义必须高度一致才能匹配，避免过度泛化
3. 不得修改候选标准关系的含义
五、更新后 Python 代码工程骨架
1. 新增：表格解析模块
python
运行
# src/document_processing/table_parser.py
import pdfplumber
import pandas as pd
from pathlib import Path
from typing import List, Dict
from src.common.logger import get_logger

logger = get_logger(__name__)

class TableParser:
    def __init__(self, min_table_rows: int = 2, output_format: str = "markdown"):
        self.min_table_rows = min_table_rows
        self.output_format = output_format  # markdown / dataframe

    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """提取PDF中所有表格，返回带元信息的列表"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        all_tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    for idx, table in enumerate(tables):
                        if not table or len(table) < self.min_table_rows:
                            continue
                        
                        # 清洗空行空列
                        cleaned = self._clean_table(table)
                        if not cleaned:
                            continue

                        table_info = {
                            "page": page_num,
                            "table_id": f"tbl_{page_num}_{idx}",
                            "source_file": Path(pdf_path).name,
                            "row_count": len(cleaned),
                            "content": cleaned
                        }

                        if self.output_format == "markdown":
                            table_info["markdown"] = self._to_markdown(cleaned)
                        else:
                            table_info["dataframe"] = pd.DataFrame(cleaned[1:], columns=cleaned[0])

                        all_tables.append(table_info)

            logger.info(f"表格提取完成：{pdf_path}，共提取{len(all_tables)}个表格")
            return all_tables

        except Exception as e:
            logger.error(f"表格提取失败: {pdf_path}, 错误: {str(e)}")
            raise

    def _clean_table(self, table: List[List]) -> List[List]:
        """清洗表格：去除全空行、合并单元格换行符"""
        cleaned = []
        for row in table:
            row = [str(cell).strip().replace("\n", "") if cell else "" for cell in row]
            if any(cell for cell in row):
                cleaned.append(row)
        return cleaned

    def _to_markdown(self, table: List[List]) -> str:
        """二维列表转Markdown表格"""
        if not table:
            return ""
        header = table[0]
        md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        for row in table[1:]:
            md_lines.append("| " + " | ".join(row) + " |")
        return "\n".join(md_lines)
2. 更新：三元组抽取器（支持表格模式）
python
运行
# src/llm_pipeline/extractor.py
from openai import OpenAI
import json
from src.common.logger import get_logger

logger = get_logger(__name__)

class TripletExtractor:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self._load_prompts()

    def _load_prompts(self):
        with open("prompts/extract.txt", "r", encoding="utf-8") as f:
            self.text_prompt = f.read()
        with open("prompts/table_extract.txt", "r", encoding="utf-8") as f:
            self.table_prompt = f.read()

    def extract_from_text(self, text: str) -> list:
        """从正文文本中抽取三元组"""
        return self._extract(self.text_prompt, text)

    def extract_from_table(self, table_markdown: str, context: str = "") -> list:
        """从表格Markdown中抽取三元组"""
        prompt = self.table_prompt.format(table_markdown=table_markdown, context=context)
        return self._extract(prompt, table_markdown)

    def _extract(self, prompt_template: str, content: str) -> list:
        """通用LLM抽取逻辑"""
        prompt = prompt_template.format(text_chunk=content)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            # 兼容不同输出结构，统一取三元组数组
            triplets = result.get("triplets", result) if isinstance(result, dict) else result
            return triplets
        except Exception as e:
            logger.error(f"三元组抽取失败: {str(e)}")
            return []
3. 新增：批量表格抽取脚本
python
运行
# scripts/extract_tables_batch.py
import os
import json
from pathlib import Path
from src.document_processing.table_parser import TableParser
from src.common.config_loader import ConfigLoader

def main():
    config = ConfigLoader()
    raw_dir = Path(config.get("data.raw_dir", "data/raw"))
    output_dir = Path(config.get("data.processed_dir", "data/processed/tables"))
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = TableParser(output_format="markdown")

    for pdf_file in raw_dir.glob("*.pdf"):
        print(f"处理: {pdf_file.name}")
        tables = parser.extract_tables(str(pdf_file))
        
        out_path = output_dir / f"{pdf_file.stem}_tables.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tables, f, ensure_ascii=False, indent=2)
        print(f"保存结果到: {out_path}")

if __name__ == "__main__":
    main()

    五、Python 代码工程骨架（核心模块完整版）
1. 通用工具：配置加载
python
运行
# src/common/config_loader.py
import yaml
from pathlib import Path
from typing import Dict

class ConfigLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default=None):
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
2. 文档处理：PDF 解析
python
运行
# src/document_processing/pdf_parser.py
import fitz
from pathlib import Path
from src.common.logger import get_logger

logger = get_logger(__name__)

class PDFParser:
    def __init__(self, skip_header_footer: bool = True):
        self.skip_header_footer = skip_header_footer

    def extract_text(self, pdf_path: str) -> str:
        """解析单PDF为纯文本"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
            page_texts = []
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if self.skip_header_footer:
                    lines = text.strip().split("\n")
                    # 简单过滤首尾行页眉页脚，可根据业务优化
                    if len(lines) > 3:
                        text = "\n".join(lines[1:-1])
                page_texts.append(text)
            doc.close()
            logger.info(f"解析PDF完成: {pdf_path}, 共{len(page_texts)}页")
            return "\n".join(page_texts)
        except Exception as e:
            logger.error(f"解析PDF失败: {pdf_path}, 错误: {str(e)}")
            raise
3. EDC 核心：关系标准化
python
运行
# src/llm_pipeline/canonicalizer.py
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
from src.common.vector_utils import cosine_similarity

class RelationCanonicalizer:
    def __init__(self, standard_relations: dict, embed_model: str = "all-MiniLM-L6-v2"):
        self.standard_relations = standard_relations  # {relation_id: {"name": "", "description": ""}}
        self.embed_model = SentenceTransformer(embed_model)
        self.llm_client = OpenAI()
        self._build_standard_index()

    def _build_standard_index(self):
        """构建标准关系的向量索引"""
        self.rel_ids = list(self.standard_relations.keys())
        descriptions = [self.standard_relations[rid]["description"] for rid in self.rel_ids]
        self.rel_embeddings = self.embed_model.encode(descriptions, normalize_embeddings=True)

    def _retrieve_top_k(self, source_def: str, k: int = 3):
        """向量召回TopK候选标准关系"""
        source_emb = self.embed_model.encode(source_def, normalize_embeddings=True)
        scores = cosine_similarity(source_emb, self.rel_embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.rel_ids[i], scores[i]) for i in top_idx]

    def canonicalize(self, source_relation: str, source_def: str) -> dict:
        """两阶段对齐：向量召回 + LLM语义校验"""
        candidates = self._retrieve_top_k(source_def)
        candidate_text = "\n".join([
            f"- {rid}: {self.standard_relations[rid]['name']} - {self.standard_relations[rid]['description']}"
            for rid, _ in candidates
        ])

        prompt = open("prompts/canonicalize.txt", encoding="utf-8").read().format(
            source_relation=source_relation,
            source_definition=source_def,
            candidate_relations=candidate_text
        )

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
4. 知识图谱：Neo4j 客户端
python
运行
# src/knowledge_graph/neo4j_client.py
from neo4j import GraphDatabase, Driver, Session
from src.common.logger import get_logger

logger = get_logger(__name__)

class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._verify_connectivity()

    def _verify_connectivity(self):
        try:
            self.driver.verify_connectivity()
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {str(e)}")
            raise

    def execute_query(self, query: str, parameters: dict = None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return list(result)

    def upsert_reservoir(self, reservoir: dict):
        """水库节点 upsert 写入"""
        query = """
        MERGE (r:Reservoir {id: $id})
        SET r.name = $name,
            r.alias = $alias,
            r.river = $river,
            r.location = $location
        RETURN r
        """
        return self.execute_query(query, reservoir)

    def close(self):
        self.driver.close()
5. API 入口
python
运行
# src/api/main.py
from fastapi import FastAPI
from src.api.routers import reservoir, knowledge

app = FastAPI(
    title="HydroBrain 黄河水库调度知识图谱服务",
    version="0.1.0",
    description="基于LLM与知识图谱的水库调度知识挖掘系统"
)

# 注册路由
app.include_router(reservoir.router, prefix="/api/reservoir", tags=["水库信息"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["知识查询"])

@app.get("/")
def health_check():
    return {"system": "HydroBrain", "status": "running", "version": "0.1.0"}