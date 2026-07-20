# HydroBrain — 基于本地大模型的黄河水库调度知识挖掘

从《黄河水资源公报》等官方 PDF 和图片中，**零 API 调用**，全本地提取水文知识，构建 Neo4j 知识图谱，提供 RESTful API 和自然语言查询。

```
PDF/PNG → PaddleOCR (PP-StructureV3) → 文本切片 → 保存中间文件 (Phase 1, zhishi)
                                                         ↓
                                            Gemma 4 4B GGUF 抽取 (Phase 2, zagism)
                                                         ↓
                                            三元组规范化 (规则引擎, 98.1%匹配率)
                                                         ↓
                                            Neo4j (中文关系, 干净ID)
                                                         ↓
                        用户 ← NL2Cypher ← Gemma 3 4B (llama-cpp)
                        用户 ← 优化接口 ← OptimizationFormatter
```

---

## 技术栈

| 层 | 技术 |
|---|------|
| LLM 抽取 | **Gemma 4 E4B** Q5_K_M GGUF (llama-cpp-python, GPU) |
| LLM 查询 | **Gemma 3 4B Text2Cypher** (llama-cpp-python, GPU) |
| OCR | **PaddleOCR PP-StructureV3** (PP-OCRv5, SLANet, RT-DETR) |
| 向量嵌入 | sentence-transformers (`all-MiniLM-L6-v2`, 80MB) |
| 知识图谱 | Neo4j 5.x |
| API 服务 | FastAPI + Uvicorn |
| GPU | NVIDIA RTX 5070 12GB (CUDA 12.9, Blackwell) |
| 环境管理 | Conda (`zhishi` OCR, `zagism` LLM+查询) |

---

## 快速开始

### 前置条件

1. 启动 Neo4j（bolt://localhost:7687）
2. 放入 PDF/图片到 `data/raw/`

### Phase 1 — OCR + 文本切片 (zhishi)

```bash
conda activate zhishi

# 安装依赖
pip install -r requirements.txt
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
pip install paddlex paddleocr beautifulsoup4 python-docx einops ftfy sentencepiece tiktoken

# 运行 OCR + 切片 (输出: data/processed/*_intermediate.json)
python scripts/build_kg_from_dir.py
```

### Phase 2 — LLM 抽取 + Neo4j 入库 (zagism)

```bash
conda activate zagism

# 确保已安装 llama-cpp-python (CUDA 12.8)
# pip install D:\work\knowlegeextract\llama_cpp_python-0.3.42+cu128-cp311-cp311-win_amd64.whl
pip install python-dotenv neo4j  # Phase 2 额外依赖

# 下载模型到 models/ 目录:
# gemma-4-E4B-it-Q5_K_M.gguf (~5.5GB)

# 运行抽取 + 规范化 + Neo4j
python scripts/kg_extract.py --mode append

# 可选: 仅测试不加库
python scripts/kg_extract.py --dry-run --limit 5
```

### 启动 API 服务

```bash
# 主 API (:8000) — 知识图谱查询 + 优化调度
conda activate zhishi
uvicorn src.api.main:app --reload
# → http://127.0.0.1:8000/docs

# 查询 API (:8001) — 自然语言问答 + 前端
conda activate zagism
python scripts/query_server.py
# → http://127.0.0.1:8001
```

---

## 双环境架构

| 环境 | 用途 | 核心依赖 |
|------|------|---------|
| `zhishi` | Phase 1: OCR + 文本切片 | PaddlePaddle 3.2.0 GPU, PyMuPDF, sentence-transformers |
| `zagism` | Phase 2: LLM抽取 + NL2Cypher查询 | llama-cpp-python 0.3.42+cu128, Neo4j, FastAPI |

**为什么分开？** PaddlePaddle 和 llama-cpp-python 的 CUDA 依赖可能冲突。两个环境各自独立。

---

## 目录结构

```
HydroBrain/
├── config/config.yaml              # 配置参数
├── .env                            # 环境变量
├── requirements.txt                # zhishi 环境依赖
├── readme.md
│
├── data/
│   ├── raw/                        # 原始 PDF + 图片
│   ├── processed/                  # 中间文件 + 三元组 + 规范化结果
│   └── ontology/                   # 领域本体 (实体/关系/别名字典)
│
├── models/                         # 本地 GGUF 模型
│   ├── gemma-4-E4B-it-Q5_K_M.gguf         # Phase 2 抽取
│   └── text-to-cypher-Gemma-3-4B-*.gguf   # NL2Cypher 查询
│
├── prompts/
│   ├── extract.txt                 # 文本三元组抽取
│   └── table_extract.txt           # 表格三元组抽取
│
├── fronted/                        # 前端页面 (纯静态)
│   ├── index_hydra.html            # 问答系统首页
│   └── optimization.html           # 优化调度数据页面
│
├── scripts/
│   ├── build_kg_from_dir.py        # Phase 1: OCR + 切片
│   ├── kg_extract.py               # Phase 2: LLM抽取 + 规范化 + Neo4j
│   ├── normalize_triplets.py       # 独立规范化工具
│   ├── rebuild_kg.py               # Neo4j 重建 (含 rebuild_graph 函数)
│   └── query_server.py             # NL2Cypher 查询服务 (:8001)
│
├── src/
│   ├── common/                     # ConfigLoader, Logger
│   ├── document_processing/        # PDF/图片解析 + PaddleOCR + 文本切片
│   ├── knowledge_fusion/           # 实体链接 + 三元组规范化 (规则引擎)
│   ├── knowledge_graph/            # Neo4j 客户端 + 图谱查询 + 优化格式化
│   └── api/                        # FastAPI 服务 + 路由
│
└── tests/
    └── test_entity_linking.py      # 实体链接单元测试
```

---

## 管道流程

### Phase 1: OCR + 切片

```
PDF/PNG → PDFParser / ImageParser
       → PaddleOCR PP-StructureV3 (15+模型串联)
       → TextSplitter (chunk_size=2000, overlap=200)
       → 噪声过滤 → 保存 *_intermediate.json
```

### Phase 2: 抽取 + 规范化 + Neo4j

```
加载 *_intermediate.json
       → Gemma 4 4B GGUF 逐 chunk 抽取 JSON 三元组
       → TripletNormalizer 文件级规范化 (实体对齐 + 关系映射 + 年份 + 去重)
       → TripletNormalizer 跨文件汇总规范化 (全局去重)
       → rebuild_graph → Neo4j
```

### 三元组规范化管道

全规则驱动，零 API 调用：

| 步骤 | 说明 |
|------|------|
| 实体标准化 | EntityLinker 别名匹配 (241条) + 关键词兜底推断 11 种实体类型 |
| 关系标准化 | 精确规则 (95条) → 关键词模糊匹配 (18组) → 启发式兜底 |
| 年份提取 | 从 relation 提取 `\d{4}年` + 文件名兜底 |
| Object 分类 | 数值 / 实体引用 / 复合 (顿号拆分) |
| 去重 + 清洗 | 同 subject+year+value 去重，丢弃无效值 |

**核心指标**:

| 指标 | 数值 |
|------|------|
| 关系精确匹配规则 | 95 条 |
| 关键词模糊规则 | 18 组 |
| 标准关系类型 | 48 个 |
| 实体别名 | 241 条 |
| 关系匹配率 | **98.1%** (规则驱动) |

---

## OCR 引擎

### PaddleOCR PP-StructureV3

- **15+ 模型串联**: 版面分析 + 文字检测 + 文字识别 + 表格结构 + 公式识别
- **GPU 加速**: 显存 ~2.2GB, 单页 ~1.6s
- **表格**: 自动检测 → 结构识别 → Markdown 输出
- **中文精度**: 远高于 Tesseract, 乱码率 <5%
- **图片**: PNG / JPG / TIFF / BMP / WEBP

### Tesseract (兜底)

PaddleOCR 不可用时自动回退。语言包: `chi_sim+eng`

---

## API 接口

### 主 API (:8000) — 知识图谱查询

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 |
| GET | `/api/reservoir/list` | 水库列表 |
| GET | `/api/reservoir/{id}` | 水库详情 |
| GET | `/api/reservoir/search?name=xxx` | 按名称搜索 |
| GET | `/api/hydrology/stations` | 水文站列表 |
| GET | `/api/hydrology/station/{id}/data?year=2024` | 年度水文数据 |
| GET | `/api/knowledge/search?q=xxx` | 三元组检索 |
| GET | `/api/knowledge/stats` | 图谱统计 |
| GET | `/api/optimization/formulate?reservoir_id=xxx` | 优化问题结构化数据 |
| GET | `/api/optimization/constraints?reservoir_id=xxx&category=` | 约束条件 |
| GET | `/api/optimization/parameters?reservoir_id=xxx` | 水库物理参数 |
| GET | `/api/optimization/hydrology?reservoir_id=xxx` | 水文时间序列 |

### 查询 API (:8001) — 自然语言问答

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 问答系统前端 |
| POST | `/api/chat` | 自然语言问答 (answer + cypher + timing) |
| GET | `/optimization.html` | 优化调度数据页面 |

---

## 自然语言查询

```
用户问题 → FastAPI (:8001) → Gemma 3 4B Text2Cypher → Cypher → Neo4j → 结果
```

- 只允许只读 MATCH 查询，自动拦截写操作
- 关系语法自动修复：`:中文` → `type(r) CONTAINS '中文'`
- 年份过滤：`d.year = '2024'`

---

## 独立工具

```bash
# 对存量三元组做规范化
python scripts/normalize_triplets.py
python scripts/normalize_triplets.py --stats-only   # 仅看统计

# 从规范化三元组重建 Neo4j
python scripts/rebuild_kg.py --mode full             # 全量重建
python scripts/rebuild_kg.py --mode append           # 增量追加
python scripts/rebuild_kg.py --dry-run               # 仅验证
```

---

## 运行测试

```bash
conda activate zhishi
python -m pytest tests/ -v
```
