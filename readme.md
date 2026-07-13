# HydroBrain — 基于大语言模型的黄河水库调度知识挖掘

基于 EDC（Extract-Define-Canonicalize）框架，使用 **DeepSeek API** 从《黄河水资源公报》等官方 PDF 中自动提取水文知识，构建 Neo4j 知识图谱，提供 RESTful API 查询。

```
PDF → OCR → 页面分类 → 文本清洗 → LLM抽取 → 语义定义 → 关系标准化 → 知识融合 → Neo4j
```

---

## 技术栈

| 层 | 技术 |
|---|------|
| LLM | DeepSeek API (`deepseek-v4-flash`) |
| PDF 解析 | PyMuPDF + Tesseract OCR (chi_sim) |
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

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxx

# 4. 启动 Neo4j，放入 PDF 到 data/raw/

# 5. 运行全链路
python scripts/build_kg_from_dir.py

# 6. 启动 API
uvicorn src.api.main:app --reload
# 访问 http://127.0.0.1:8000/docs
```

---

## 目录结构

```
HydroBrain/
├── config/config.yaml              # DeepSeek/Neo4j/文本处理配置
├── .env                            # API Key 等环境变量
├── requirements.txt
│
├── data/
│   ├── raw/                        # 原始 PDF
│   ├── processed/                  # 解析结果（文本/chunks/表格）
│   └── ontology/                   # 领域本体（实体类型/关系/别名字典）
│
├── prompts/                        # LLM Prompt 模板
│   ├── extract.txt                 # 正文三元组抽取
│   ├── table_extract.txt           # 表格三元组抽取
│   ├── define.txt                  # 关系语义定义
│   └── canonicalize.txt            # 关系标准化
│
├── scripts/
│   ├── build_kg_from_dir.py        # 全链路构建脚本
│   ├── parse_pdfs_batch.py         # 批量 PDF 解析
│   └── extract_tables_batch.py     # 批量表格提取
│
├── src/
│   ├── common/                     # ConfigLoader, Logger
│   ├── document_processing/        # PDF 解析 + OCR + 页面分类 + 文本清洗 + 表格重建
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
| 页面分类 | TEXT 8 / TABLE 16 / MIXED 21 / CHART 2 |
| 噪声跳过 | 18 页（纯图表/乱码） |
| 有效文本 | ~19,000 字符 |
| 抽取三元组 | **540**（文本 343 + 表格 197） |
| 冲突解决 | 50 |
| 入库三元组 | **490** |

### Neo4j 知识图谱

| 节点类型 | 数量 |
|---------|------|
| AnnualHydrologyData | 209 |
| Constraint | 76 |
| WaterResourceZone | 44 |
| River | 35 |
| HydrologicalStation | 18 |
| Province | 17 |
| Reservoir | 9 |
| Document | 1 |
| **总计** | **409 节点, 465 关系** |

抽取示例：
```
[兰州水文站] --[2024年实测径流量为]--> [362.80亿立方米]
[黄河流域] --[划分为]--> [8个水资源二级区]
[黄河流域] --[总面积]--> [79.58万平方公里]
```

---

## 四层处理管道

### Layer 0: 页面分类器
PyMuPDF 渲染 + OpenCV 分析 → 检测页面类型: TEXT/TABLE/CHART/MIXED

### Layer 1: 分类 OCR
- 文本页: `chi_sim+eng` 标准 OCR
- 表格页: `--psm 6` + 位置数据提取
- 图表页: 自动跳过

### Layer 2: 文本清洗器
- 50+ 常见 OCR 错误字典修复
- 有效中文字符占比过滤
- 乱码行/空行清理

### Layer 3: 表格重建器
Tesseract `image_to_data` → 列聚类 → 行识别 → 结构化表格

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

## 运行单元测试

```bash
conda activate zhishi
python -m pytest tests/ -v   # 27 passed
```
