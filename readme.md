# HydroBrain — 基于大语言模型的黄河水库调度知识挖掘

基于 EDC（Extract-Define-Canonicalize）框架，使用 **DeepSeek API** + **PaddleOCR (PP-StructureV3)** 从《黄河水资源公报》等官方 PDF 和图片中自动提取水文知识，构建 Neo4j 知识图谱，提供 **RESTful API** 和 **自然语言查询**。

```
PDF/PNG → PaddleOCR (PP-StructureV3) → 文本清洗 → LLM 批量抽取 (含year)
    ↓
三元组规范化 (实体对齐 + 关系映射 + 对象分类 + 年份补全 + 复合拆分 + 去重)
    ↓
Neo4j (干净ID, 中文关系类型, 标准relation_id, year, 无R_前缀)
    ↓
用户 ← 自然语言查询 ← Gemma 3 4B Text2Cypher (llama-cpp, GPU)
用户 ← 优化数据接口 ← OptimizationFormatter
```

---

## 技术栈

| 层 | 技术 |
|---|------|
| LLM 抽取 | DeepSeek API (`deepseek-v4-flash`) |
| LLM 查询 | **Gemma 3 4B Text2Cypher** (llama-cpp-python, GPU 推理) |
| OCR | **PaddleOCR PP-StructureV3** (PP-OCRv5, SLANet, RT-DETR) |
| OCR 加速 | NVIDIA GPU (CUDA 12.9, RTX 5070, Blackwell) |
| 向量嵌入 | sentence-transformers (`all-MiniLM-L6-v2`) |
| 知识图谱 | Neo4j 5.x |
| API 服务 | FastAPI + Uvicorn (双服务) |
| 环境管理 | Conda (`zhishi` 数据导入, `zagism` 自然语言查询) |

---

## 快速开始

### 数据导入（zhishi 环境）

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

# 5. 启动 Neo4j，放入 PDF/图片到 data/raw/

# 6. 运行全链路 (~8 min)
python scripts/build_kg_from_dir.py

# 7. 规范化三元组（实体对齐 + 关系映射 + 年份补全 + 脏数据清洗）
python scripts/normalize_triplets.py

# 8. 重建知识图谱（干净 ID + 完整元数据）
python scripts/rebuild_kg.py --mode full

# 9. 启动主 API
uvicorn src.api.main:app --reload
# 访问 http://127.0.0.1:8000/docs
```

### 自然语言查询（zagism 环境）

```bash
# 1. 激活查询环境
conda activate zagism

# 2. 安装 llama-cpp-python (CUDA 12.8)
pip install D:\work\knowlegeextract\llama_cpp_python-0.3.42+cu128-cp311-cp311-win_amd64.whl

# 3. 下载模型到 models/ 目录
# Gemma-3-4B-Text2Cypher-Q4_K_M.gguf (~2.4GB)
# https://huggingface.co/bartowski/gemma-3-4b-it-GGUF

# 4. 启动查询服务
python scripts/query_server.py
# 访问 http://127.0.0.1:8001/docs
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
│   ├── raw/                        # 原始 PDF + 图片
│   ├── processed/                  # 解析结果（文本/chunks/表格/三元组）
│   └── ontology/                   # 领域本体（实体/关系/别名字典）
│
├── models/                         # 本地模型
│   └── text-to-cypher-Gemma-3-4B-Instruct-2025.04.0.Q4_K_M.gguf
│
├── prompts/
│   ├── extract.txt                 # 单 chunk 三元组抽取
│   ├── extract_batch.txt           # 批量 chunk 三元组抽取 (含 year 字段)
│   ├── table_extract.txt           # 单表格三元组抽取
│   ├── table_extract_batch.txt     # 批量表格三元组抽取 (含 year 字段)
│   ├── define.txt                  # 关系语义定义
│   └── canonicalize.txt            # 关系标准化
│
├── scripts/
│   ├── build_kg_from_dir.py        # 全链路构建脚本（PDF + 图片）
│   ├── normalize_triplets.py       # 三元组规范化（实体对齐+关系映射+年份+清洗）
│   ├── rebuild_kg.py               # 从规范化三元组重建 Neo4j 图谱
│   ├── query_server.py             # 自然语言查询服务 (:8001)
│   ├── parse_pdfs_batch.py         # 批量 PDF 解析
│   └── extract_tables_batch.py     # 文本型 PDF 表格提取 (pdfplumber)
│
├── src/
│   ├── common/                     # ConfigLoader, Logger
│   ├── document_processing/        # PDF解析 + 图片解析 + PaddleOCR + Tesseract兜底
│   ├── llm_pipeline/               # DeepSeek Client + Extract/Define/Canonicalize
│   ├── knowledge_fusion/           # 实体链接 + 冲突检测 + 三元组规范化
│   ├── knowledge_graph/            # Neo4j 客户端 + 图谱构建 + 查询 + 优化格式化
│   └── api/                        # FastAPI 服务 + 路由
│
└── tests/                          # 单元测试 (27例全部通过)
```

---

## 双环境架构

| 环境 | 用途 | 核心依赖 |
|------|------|---------|
| `zhishi` | 数据导入（OCR + LLM 抽取 + 图谱构建） | PaddlePaddle 3.2.0 GPU, DeepSeek SDK, Neo4j |
| `zagism` | 自然语言查询（NL2Cypher） | llama-cpp-python 0.3.42+cu128, Neo4j |

**为什么分开？** PaddlePaddle 和 llama-cpp-python 的 CUDA 依赖可能冲突。数据导入和查询分别使用独立环境，互不干扰。

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
| 总三元组 | **1,257** (含二期增量数据) |
| 冲突解决 | 89 |
| LLM API 调用 | ~17 次 (vs 50+ 次逐个抽取) |

### 三元组规范化结果

经过 `normalize_triplets.py` 处理后的统计：

| 指标 | 数值 |
|------|------|
| 规范化输出 | **1,034** 条 (去重 228 条冗余) |
| 关系匹配率 | **98.1%** (精确 1,080 + 关键词 153 / 1,257) |
| 实体标准化 | 949 条匹配标准别名库 |
| 新实体类型 | 308 条（地下水超采区、平原盆地等） |
| 年份分布 | 2024=812, constant=345, 2023=105 |
| 复合拆分 | +34 条 (split comma-separated entities) |
| 脏数据丢弃 | -29 条 |
| **去重** | **-228 条** (同subject+同year+同value的不同relation合并，保留含年份版本) |

### Neo4j 知识图谱（重建后）

| 节点类型 | 数量 |
|---------|------|
| AnnualHydrologyData | 605 |
| Constraint | 265 |
| GroundwaterOverdraftArea | 61 |
| HydrologicalStation | 36 |
| GroundwaterRegion | 25 |
| Province | 22 |
| WaterResourceZone | 9 |
| Reservoir | 17 |
| River | 13 |
| StatisticAggregate | 3 |
| Document | 1 |
| **总计** | **993 节点, 1,034 关系** |

节点 ID 示例：`Reservoir_d45931c6`（干净、无中文）
关系类型：保留中文原名，使用反引号包裹（如 `` `2024年实测径流量为` ``），无需 R_ 前缀

抽取示例：
```
[兰州水文站] --[2024年实测径流量为]--> [362.80亿立方米]
[黄河流域] --[划分为]--> [8个水资源二级区]
[黄河流域] --[总面积]--> [79.58万平方公里]
[黄河干流] --[全长]--> [5464公里]
```

### 规范三元组 Schema

每条三元组经规范化后包含 10 个字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `subject` | string | 标准实体全名（经别名匹配后） |
| `subject_type` | enum | 10 种之一：Reservoir / HydrologicalStation / WaterResourceZone / Province / River / AnnualHydrologyData / Constraint / DispatchRule / GroundwaterOverdraftArea / GroundwaterRegion / StatisticAggregate |
| `relation` | string | 保留的中文关系原文 |
| `relation_id` | string | 标准关系 ID（48 种，如 `ANNUAL_RUNOFF`），无匹配则为 null |
| `object` | string | 数值+单位 或 标准实体名 |
| `object_type` | enum | `numerical_value` / `entity_reference` / `composite`（多值拆分） |
| `year` | int / "constant" | 数据年份；恒定属性为 "constant" |
| `context` | string | 原文片段（溯源） |
| `confidence` | float | 综合置信度（抽取 × 标准化） |
| `source_file` | string | 来源文件名 |

已移除的冗余字段：`chunk_id`, `page_num`, `data_type`, `original_relation`

### 三元组质量保障

| 规则 | 说明 |
|------|------|
| 实体分离 | subject 只含独立实体名，指标/年份归入 relation |
| 关系保留中文 | Canonicalizer 保留原始中文关系，`relation_id` 字段存标准 ID |
| 禁止泛指词 | 禁止使用"为""是""有"等无意义关系词 |
| 数值完整 | object 保留完整数值+单位（如 `362.80亿立方米`） |

**效果**: 中文关系保留率 100%，subject 含指标比例 < 0.5%。

### 数据质量保障

| 规则 | 说明 |
|------|------|
| 实体去重 | subject/object 经 EntityLinker 别名匹配统一为全称（如"花园口"→"花园口水文站"） |
| 关系去重 | 同 subject+year+value 的多条关系合并为一条，保留含显式年份的版本（-228 条冗余） |
| 类型校验 | object 标签优先按名称关键词推断（"水文站"→HydrologicalStation），不再因宽泛关系类型误分类 |
| 反引号关系 | Neo4j 关系类型用 `` ` `` 包裹保留中文原名，无 `R_` 前缀 |

---

## 三元组规范化管道

将 LLM 抽取的原始三元组转换为可直接入库的标准格式，全规则驱动（零额外 API 调用）。

### 管道流程

```
原始三元组 → EntityLinker (别名匹配 + 类型标注)
           → RelationStandardizer (精确规则 → 关键词匹配 → LLM兜底)
           → YearExtractor (从relation提取年份, 排除基准期范围)
           → ObjectClassifier (数值 / 实体引用 / 复合)
           → SchemaMapper (映射到规范字段, 剔除冗余)
           → MultiValueSplitter (拆分顿号/逗号分隔的复合object)
           → 规范三元组
```

### 核心指标

| 指标 | 数值 |
|------|------|
| 关系精确匹配规则 | 95 条（覆盖径流/输沙/降水/供水/用水/耗水/蓄水/地下水/固有属性） |
| 关键词模糊规则 | 18 组（兜底未精确匹配的关系） |
| 标准关系类型 | 48 个（地理属性/空间关系/水文数据/用水数据/地下水数据/比较数据/溯源） |
| 实体别名 | 241 条（水库 18 + 水文站 24 + 水资源分区 8 + 省级 10 + 河流 16） |
| 关系匹配率 | **98.1%**（规则驱动，余下为低频专有术语） |

### 用法

```bash
# 对存量三元组做规范化
python scripts/normalize_triplets.py

# 仅看统计，不输出文件
python scripts/normalize_triplets.py --stats-only

# 从规范化三元组重建 Neo4j 图谱
python scripts/rebuild_kg.py --mode full     # 全量重建
python scripts/rebuild_kg.py --mode append   # 增量追加
python scripts/rebuild_kg.py --dry-run       # 仅验证，不写库
```

---

## OCR 引擎

### PaddleOCR PP-StructureV3 (默认)
- **15+ 模型串联**: 版面分析 + 文字检测 + 文字识别 + 表格结构 + 公式识别
- **GPU 加速**: 显存占用 ~2.2GB，单页 ~1.6s
- **自动表格**: 检测→结构识别→Markdown 输出，无需后处理
- **中文精度**: 远高于 Tesseract，乱码率 <5%
- **图片支持**: PNG / JPG / TIFF / BMP / WEBP

### Tesseract (兜底)
- PaddleOCR 不可用时自动回退
- 语言包: `chi_sim+eng`

---

## 自然语言查询

### 架构

```
用户问题 → FastAPI (:8001) → Gemma 3 4B Text2Cypher → Cypher → Neo4j → 结果
```

### 测试结果

| 问题 | 结果 | 说明 |
|------|:---:|------|
| 兰州水文站径流量 | ✅ 8 条 | 正确生成 CONTAINS 模糊匹配 Cypher |
| 龙羊峡水库水位 | ✅ 1 条 | 找到水库基本信息 |
| 黄河流域用水总量 | ❌ | 4B 模型对复杂多条件查询不稳定 |
| 利津水文站输沙量 | ❌ | 模型未生成有效 Cypher |

**成功率 ~50%**，符合 4B 模型的预期。简单单实体+单指标查询表现好，多条件组合查询需要更大模型。

### 安全机制

- 只允许只读 MATCH 查询
- 自动拦截 CREATE / DELETE / DROP / SET / REMOVE / MERGE
- 关系语法自动修复：`:中文` → `type(r) CONTAINS '中文'`

### 后续优化方向

- 补充更多 few-shot 示例覆盖失败查询模式
- 换用 Gemma 3 12B 或 Qwen2.5-7B 提升指令遵循
- DeepSeek API 兜底复杂查询（混合模式）
- 论文中可评估 P/R/F1 指标

---

## API 接口

### 主 API (:8000) — 知识图谱查询

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
| GET | `/api/optimization/formulate?reservoir_id=xxx` | **优化问题结构化数据** |
| GET | `/api/optimization/constraints?reservoir_id=xxx&category=` | 约束条件（结构化） |
| GET | `/api/optimization/parameters?reservoir_id=xxx` | 水库物理参数 |
| GET | `/api/optimization/hydrology?reservoir_id=xxx` | 水文时间序列 |
| GET | `/api/optimization/reservoirs` | 可优化水库列表 |

### 查询 API (:8001) — 自然语言查询

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 |
| POST | `/query` | JSON 格式自然语言查询 |
| GET | `/query?q=xxx` | URL 参数自然语言查询 |

查询请求示例：
```json
{
  "question": "兰州水文站2024年径流量是多少"
}
```

查询响应示例：
```json
{
  "question": "兰州水文站2024年径流量是多少",
  "cypher": "MATCH (s)-[r]->(d) WHERE s.name CONTAINS '兰州' AND type(r) CONTAINS '径流' RETURN s.name, type(r), d.value, d.unit LIMIT 20",
  "explanation": "查询兰州水文站径流量数据",
  "results": [...],
  "result_count": 8
}
```

---

## 优化调度数据接口

### 设计思路

将知识图谱中的原始数据（Constraint 节点 / AnnualHydrologyData / Reservoir 参数）重组为优化算法可直接消费的结构化 JSON。不做优化计算，只做**知识格式化**。

### 数据流

```
优化算法 (PuLP/OR-Tools/scipy/GA)
    ↓ HTTP
GET /api/optimization/formulate?reservoir_id=xxx
    ↓
OptimizationFormatter  ← 中文关系→数学表达 + 约束分类 + 变量推断
    ↓
GraphQuery (Cypher)   ← 多源合并: Constraint + HydrologyData + Reservoir
    ↓
Neo4j
```

### 核心能力

| 能力 | 实现 |
|------|------|
| 约束分类 | 中文关系名→`water_level`/`discharge`/`storage`/`power_output` 等 9 类 |
| 运算符提取 | `不超过/上限→<=`, `不低于/保证→>=`, `控制在→==` |
| 数值+单位抽取 | 正则 `(\d+\.?\d*)\s*(亿m³\|m³/s\|m\|...)` 从中文字段提取 |
| 决策变量推断 | 有水位约束→推断变量 `Z`(m)，边界从死水位/汛限水位取 |
| 目标候选推断 | 有发电/出力关键词→`最大化发电量`，标注 `source: "inferred"` |

### 请求示例

```bash
# 获取可优化水库列表
curl "http://127.0.0.1:8000/api/optimization/reservoirs"

# 构建龙羊峡水库优化问题
curl "http://127.0.0.1:8000/api/optimization/formulate?reservoir_id=res_longyangxia"

# 只看水位类约束
curl "http://127.0.0.1:8000/api/optimization/constraints?reservoir_id=res_longyangxia&category=water_level"
```

### 响应示例

```json
{
  "code": 200,
  "message": "龙羊峡水库: 3变量, 12约束, 2目标候选, 5时序数据",
  "data": {
    "problem_meta": {
      "reservoir_name": "龙羊峡水库",
      "description": "龙羊峡水库优化调度问题"
    },
    "decision_variables": [
      {
        "name": "库水位",
        "symbol": "Z",
        "unit": "m",
        "bounds": {"lower": 2530.0, "upper": 2594.0},
        "source": "inferred"
      }
    ],
    "objective_candidates": [
      {
        "description": "最大化发电量",
        "type": "maximize",
        "related_keywords": ["发电", "出力"],
        "source": "inferred"
      }
    ],
    "constraints": [
      {
        "id": "c_001",
        "name": "汛限水位",
        "expression": "水位 <= 2594m",
        "category": "water_level",
        "operator": "<=",
        "value": 2594.0,
        "unit": "m",
        "source_doc": "黄河水资源公报2024"
      }
    ],
    "parameters": {
      "dead_storage_level": {"value": 2530, "unit": "m"},
      "normal_storage_level": {"value": 2600, "unit": "m"},
      "total_capacity": {"value": 247.0, "unit": "亿m³"}
    },
    "time_series": {
      "entries": [{"label": "2024", "value": 650, "unit": "m³/s", "indicator": "径流量"}],
      "count": 5
    }
  }
}
```

---

## 配置优化参考

`config/config.yaml` 中的关键参数：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `deepseek.max_tokens` | 32768 | 表格批量 JSON 大，需要充分输出空间 |
| `deepseek.batch_size` | 4 | 批次太大 JSON 截断，太小 API 调用多 |
| `deepseek.model` | deepseek-v4-flash | 速度快，适合批量抽取 |
| `embedding.device` | cpu | 80MB 模型 CPU 足够 |
| `text_splitter.chunk_size` | 2000 | 适合中文 OCR 文本 |

---

## 运行单元测试

```bash
conda activate zhishi
python -m pytest tests/ -v   # 27 passed
```
