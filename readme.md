# HydroBrain — 基于大语言模型的黄河水库调度知识挖掘

基于 EDC（Extract-Define-Canonicalize）框架，使用 **DeepSeek API** + **PaddleOCR (PP-StructureV3)** 从《黄河水资源公报》等官方 PDF 中自动提取水文知识，构建 Neo4j 知识图谱，提供 RESTful API 查询。

```
PDF → PaddleOCR (PP-StructureV3) → 文本清洗 → LLM批量抽取 → 语义定义 → 关系标准化 → 知识融合 → Neo4j
```

---

## 技术栈

| 层 | 技术 |
|---|------|
| LLM | DeepSeek API (`deepseek-v4-flash`) |
| OCR | **PaddleOCR PP-StructureV3** (PP-OCRv5, SLANet, RT-DETR) |
| OCR 加速 | NVIDIA GPU (CUDA 12.9, RTX 5070) |
| 向量嵌入 | sentence-transformers (`all-MiniLM-L6-v2`) |
| 知识图谱 | Neo4j 5.x |
| API 服务 | FastAPI + Uvicorn |
| 环境管理 | Conda (`zhishi`) |

---

## 快速开始

```bash
# 1. 激活环境
conda activate zhishi

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 PaddleOCR GPU 版 (Blackwell 架构)
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
pip install paddlex paddleocr beautifulsoup4 python-docx einops ftfy sentencepiece tiktoken

# 4. 配置 API Key + Neo4j
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxx

# 5. 启动 Neo4j，放入 PDF 到 data/raw/

# 6. 运行全链路 (~8 min)
python scripts/build_kg_from_dir.py

# 7. 启动 API
uvicorn src.api.main:app --reload
# 访问 http://127.0.0.1:8000/docs
```

---

## 目录结构

```
HydroBrain/
├── config/config.yaml              # DeepSeek/Neo4j/OCR/切片参数
├── .env                            # API Key 等环境变量
├── requirements.txt
│
├── data/
│   ├── raw/                        # 原始 PDF
│   ├── processed/                  # 解析结果（文本/chunks/表格/三元组）
│   └── ontology/                   # 领域本体（实体/关系/别名字典）
│
├── prompts/
│   ├── extract.txt                 # 单 chunk 三元组抽取
│   ├── extract_batch.txt           # 批量 chunk 三元组抽取
│   ├── table_extract.txt           # 单表格三元组抽取
│   ├── table_extract_batch.txt     # 批量表格三元组抽取
│   ├── define.txt                  # 关系语义定义
│   └── canonicalize.txt            # 关系标准化
│
├── scripts/
│   ├── build_kg_from_dir.py        # 全链路构建脚本
│   ├── parse_pdfs_batch.py         # 批量 PDF 解析
│   └── extract_tables_batch.py     # 文本型 PDF 表格提取 (pdfplumber)
│
├── src/
│   ├── common/                     # ConfigLoader, Logger
│   ├── document_processing/        # PDF解析 + PaddleOCR + Tesseract兜底 + 文本清洗 + 表格重建 + Schema映射
│   ├── llm_pipeline/               # DeepSeek Client + Extract/Define/Canonicalize
│   ├── knowledge_fusion/           # 实体链接 + 冲突检测
│   ├── knowledge_graph/            # Neo4j 客户端 + 图谱构建 + 查询
│   └── api/                        # FastAPI 服务 + 路由
│
└── tests/                          # 单元测试 (27例全部通过)
```

---

## MVP 验证结果

以《黄河水资源公报 2024》（92MB 扫描版 PDF，46 页）为测试集：

| 指标 | 数值 |
|------|------|
| OCR 引擎 | **PaddleOCR PP-StructureV3** (PP-OCRv5_server) |
| OCR 耗时 | ~75s (GPU) |
| 有效页 | 45/46 (仅1页噪声) |
| 识别表格 | 20 个结构化表格 |
| LLM 抽取 | 批量模式 (batch_size=4, max_tokens=32768) |
| 文本三元组 | **360** |
| 表格三元组 | **489** |
| 总三元组 | **849** (+57% vs Tesseract) |
| 冲突解决 | 89 |
| LLM API 调用 | ~17 次 (vs 50+ 次逐个抽取) |

### Neo4j 知识图谱

| 节点类型 | 数量 |
|---------|------|
| AnnualHydrologyData | 493 |
| Constraint | 111 |
| Province | 55 |
| River | 32 |
| WaterResourceZone | 27 |
| HydrologicalStation | 25 |
| Reservoir | 22 |
| Document | 1 |
| **总计** | **766 节点, 739 关系** |

抽取示例：
```
[兰州水文站] --[2024年实测径流量为]--> [362.80亿立方米]
[黄河流域] --[划分为]--> [8个水资源二级区]
[黄河流域] --[总面积]--> [79.58万平方公里]
[黄河干流] --[全长]--> [5464公里]
```

---

## OCR 引擎

### PaddleOCR PP-StructureV3 (默认)
- **15+ 模型串联**: 版面分析 + 文字检测 + 文字识别 + 表格结构 + 公式识别
- **GPU 加速**: 显存占用 ~2.2GB，单页 ~1.6s
- **自动表格**: 检测→结构识别→Markdown 输出，无需后处理
- **中文精度**: 远高于 Tesseract，乱码率 <5%

### Tesseract (兜底)
- PaddleOCR 不可用时自动回退
- 语言包: `chi_sim+eng`

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 |
| GET | `/api/reservoir/list` | 水库列表 |
| GET | `/api/reservoir/{id}` | 水库详情 |
| GET | `/api/reservoir/search?name=xxx` | 按名称搜索 |
| GET | `/api/reservoir/{id}/rules` | 调度规则 |
| GET | `/api/reservoir/{id}/constraints` | 约束条件 |
| GET | `/api/hydrology/stations` | 水文站列表 |
| GET | `/api/hydrology/station/{id}/data?year=2024` | 年度水文数据 |
| GET | `/api/hydrology/zones` | 水资源分区 |
| GET | `/api/knowledge/search?q=xxx` | 三元组检索 |
| GET | `/api/knowledge/trace?doc=xxx` | 文档溯源 |
| GET | `/api/knowledge/path?from=xx&to=xx` | 实体间路径 |
| GET | `/api/knowledge/stats` | 图谱统计 |

---

## 配置优化参考

`config/config.yaml` 中的关键参数：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `deepseek.max_tokens` | 32768 | 表格批量 JSON 大，需要充分输出空间 |
| `deepseek.batch_size` | 4 | 批次太大 JSON 截断，太小 API 调用多 |
| `deepseek.model` | deepseek-v4-flash | 速度快，适合批量抽取 |
| `embedding.device` | cpu/cuda | 80MB 模型 CPU 足够 |
| `text_splitter.chunk_size` | 2000 | 适合中文 OCR 文本 |

---

## 运行单元测试

```bash
conda activate zhishi
python -m pytest tests/ -v   # 27 passed
```
