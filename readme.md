# HydroBrain — 基于本地大模型的黄河水库调度知识挖掘

从《黄河水资源公报》等官方 PDF 和图片中，**零 API 调用全本地**提取水文知识，构建 Neo4j 知识图谱，提供 RESTful API 和自然语言查询。

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

| 层 | 技术 | 显存 |
|---|------|------|
| LLM 抽取 | **Gemma 4 E4B** Q5_K_M GGUF (llama-cpp-python, GPU) | ~2.7 GB |
| LLM 查询 | **Gemma 3 4B Text2Cypher** (llama-cpp-python, GPU) | ~2.4 GB |
| OCR | **PaddleOCR PP-StructureV3** (15+ 模型串联) | ~2.2 GB |
| 向量嵌入 | sentence-transformers (`all-MiniLM-L6-v2`, 80MB) | CPU |
| 知识图谱 | Neo4j 5.x | — |
| API 服务 | FastAPI + Uvicorn | — |
| GPU | NVIDIA RTX 5070 12GB (CUDA 12.9, Blackwell) | — |
| 环境管理 | Conda (`zhishi` OCR, `zagism` LLM+查询) | — |

**VRAM**: Phase 1 (OCR ~2.2GB) 和 Phase 2 (Gemma 4 ~3.5GB) 分时运行，互不冲突。

---

## 快速开始

### 前置条件

1. 启动 Neo4j（bolt://localhost:7687, 密码 `neo4jneo4j`）
2. PDF/图片放入 `data/raw/`（支持递归子目录，如 `data/raw/picture/`）

### Phase 1 — OCR + 文本切片 (zhishi)

```bash
conda activate zhishi

# 安装依赖
pip install -r requirements.txt
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
pip install paddlex paddleocr beautifulsoup4 python-docx einops ftfy sentencepiece tiktoken

# 运行 (输出: data/processed/text_extracted/*_intermediate.json)
python scripts/build_kg_from_dir.py
```

### Phase 2 — LLM 抽取 + Neo4j 入库 (zagism)

```bash
conda activate zagism

# 确保已安装 llama-cpp-python (CUDA 12.8)
# pip install D:\work\knowlegeextract\llama_cpp_python-0.3.42+cu128-cp311-cp311-win_amd64.whl
pip install python-dotenv neo4j

# 运行 (先 dry-run 测试)
python scripts/kg_extract.py --dry-run --limit 5
# 正式运行
python scripts/kg_extract.py --mode append
```

### 启动服务

```bash
# 主 API (:8000)
conda activate zhishi
uvicorn src.api.main:app --reload

# 查询 API + 前端 (:8001)
conda activate zagism
python scripts/query_server.py
```

---

## 双环境架构

| 环境 | 用途 | 核心依赖 |
|------|------|---------|
| `zhishi` | Phase 1: OCR + 文本切片 | PaddlePaddle 3.2.0 GPU, PyMuPDF, sentence-transformers |
| `zagism` | Phase 2: LLM抽取 + NL2Cypher查询 | llama-cpp-python 0.3.42+cu128, Neo4j, FastAPI |

**为什么分开？** PaddlePaddle 和 llama-cpp-python 的 CUDA 依赖可能冲突。

---

## 数据目录结构

```
data/
├── raw/                              # 原始 PDF + 图片 (支持子目录递归)
│   ├── *.pdf
│   └── picture/*.png
├── processed/
│   ├── text_extracted/               # Phase 1 输出
│   │   └── {文件名}_intermediate.json  # chunks + tables + 溯源
│   ├── normalize/                    # Phase 2 单文件规范化结果
│   │   └── {文件名}_triplets.json
│   └── all_normalized_triplets/      # Phase 2 跨文件汇总去重
│       └── all_triplets_normalized.json
└── ontology/                         # 领域本体
    ├── entity_types.json             # 11 种实体类型定义
    ├── relation_types.json           # 48 种标准关系类型
    └── alias_dict.json               # 241 条实体别名
```

---

## 脚本清单

| 脚本 | 环境 | 用途 |
|------|------|------|
| `build_kg_from_dir.py` | zhishi | Phase 1: PDF/图片 → PaddleOCR → 文本切片 → 保存 `_intermediate.json` |
| `kg_extract.py` | zagism | Phase 2: 加载中间文件 → Gemma 4 逐 chunk 抽取 → 规范化 → Neo4j |
| `rebuild_kg.py` | 通用 | `rebuild_graph()` 函数 (被 kg_extract 导入) + 独立 Neo4j 重建工具 |
| `normalize_triplets.py` | 通用 | 对存量三元组单独做规范化 + 统计 |
| `query_server.py` | zagism | NL2Cypher 查询服务 + 前端托管 (:8001) |

---

## 管道实现细节

### Phase 1: 表格处理

PP-StructureV3 自动检测表格结构并输出 Markdown。Phase 1 不做任何表格预处理，原样保存到 `_intermediate.json`。

### Phase 2: 关键实现细节

**表格预处理 (解决两个问题)**:
1. **单位嵌入** (`_embed_units_in_table`): 解析 Markdown 表头括号中的单位 (如 `径流量(亿m³)`)，自动拼接到每行数值 → `| 362.80 |` 变成 `| 362.80亿m³ |`
2. **表格拆分** (`_split_large_table`): 密集表格拆分为小段 (每段 3 行+表头)，避免 JSON 输出超过模型 token 上限

**推理参数 (经实测调优)**:
| 参数 | 文本 chunk | 表格段 | 说明 |
|------|-----------|--------|------|
| `max_tokens` | 4096 | 4096 | 表格 3 行约需 850-1339 tok |
| `n_ctx` | 8192 | 8192 | KV cache ~0.8GB |
| `max_rows` | — | 3 | 每段最多 3 行数据 |
| 速度 | — | ~78 tok/s | 每段约 10-17s |

**Gemma chat 格式**:
```
<start_of_turn>system
你是资深黄河流域水库调度专家。只输出JSON数组，不输出任何解释。<end_of_turn>
<start_of_turn>user
{prompt from prompts/*.txt}<end_of_turn>
<start_of_turn>model
```

**注意**: 不加 `<bos>` 前缀 (llama-cpp 自动添加，重复会导致警告)。

**规范化流程**: 文件级 → 汇总级 (跨文件去重)，TripletNormalizer 纯规则驱动零 API 调用。

---

## Prompt 模板

| 文件 | 用途 | 使用方 |
|------|------|--------|
| `prompts/extract.txt` | 文本三元组抽取 | `kg_extract.py` |
| `prompts/table_extract.txt` | 表格三元组抽取 | `kg_extract.py` |

---

## 已知问题与修复记录

1. **aistudio-sdk 兼容性**: paddlenlp 3.0.0b3 需要 `aistudio_sdk.hub.download` 但 0.3.8 版本没有 → monkey-patch 解决 (仅影响 PP-UIE，已弃用但 paddlenlp 保留)
2. **PaddlePaddle/torch DLL 冲突**: 导入 paddlenlp 前必须先 `import torch` 避免 `shm.dll` 加载失败
3. **Qwen3.5-9B 思考循环**: DeepSeek-V4-Flash 微调版会无限思考不输出 JSON → 不可用
4. **Gemma 4 截断问题**: 密集表格超过 max_tokens 导致 JSON 截断 → 通过表格拆分 (max_rows=3) + max_tokens=4096 解决
5. **表格单位丢失**: Markdown 表头含单位但单元格无单位 → `_embed_units_in_table` 预处理嵌入

---

## API 接口

### 主 API (:8000)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/reservoir/list` | 水库列表 |
| GET | `/api/hydrology/stations` | 水文站列表 |
| GET | `/api/hydrology/station/{id}/data?year=2024` | 年度水文数据 |
| GET | `/api/knowledge/search?q=xxx` | 三元组检索 |
| GET | `/api/optimization/formulate?reservoir_id=xxx` | 优化问题结构化数据 |

### 查询 API (:8001)

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 问答系统前端 |
| POST | `/api/chat` | 自然语言问答 |

---

## 独立工具

```bash
# 存量三元组规范化 + 统计
python scripts/normalize_triplets.py
python scripts/normalize_triplets.py --stats-only

# Neo4j 重建
python scripts/rebuild_kg.py --mode full     # 全量
python scripts/rebuild_kg.py --mode append   # 增量
python scripts/rebuild_kg.py --dry-run       # 仅验证
```

## 运行测试

```bash
conda activate zhishi
python -m pytest tests/ -v
```
