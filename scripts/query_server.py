#!/usr/bin/env python
"""
HydroBrain 自然语言查询服务
Gemma 3 4B Text2Cypher (llama-cpp-python) + Neo4j
同时托管前端静态页面 (fronted/)

启动: conda activate zagism && python scripts/query_server.py
访问: http://127.0.0.1:8001
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 前端目录
ROOT_DIR = Path(__file__).parent.parent
FRONTED_DIR = ROOT_DIR / "fronted"

# ============================================================
# Neo4j Schema Context (注入 Prompt)
# ============================================================
SCHEMA_CONTEXT = """
## 知识图谱 Schema (黄河水资源2024公报)

### 节点类型
- Reservoir (水库): id, name, alias, river, location
- HydrologicalStation (水文站): id, name, alias, river
- WaterResourceZone (水资源二级区): id, name, alias
- Province (省级行政区): id, name
- River (河流): id, name
- AnnualHydrologyData (年度水文数据): id, name, indicator, value, unit, year
- Constraint (约束条件): id, name, value, variable, unit
- Document (文档): id, name, year, type

### 节点数量
Reservoir 18个, HydrologicalStation 38个, WaterResourceZone 21个, Province 22个, River 15个, AnnualHydrologyData 626个, Constraint 267个

### 关系
所有关系类型都是 LLM 抽取的中文短语，如 "2024年实测径流量为" "天然河川径流量为" "用水总量为" "流域总面积为"
查询时使用 type(r) CONTAINS '关键词' 模糊匹配

### 年份过滤（重要！）
- AnnualHydrologyData 节点有 d.year 属性（如 "2024", "2023"）
- 关系也有 r.year 属性
- 查询特定年份数据时，用 d.year 或 r.year 过滤比用 type(r) CONTAINS '2024' 更准确
- 示例: WHERE d.year = '2024' AND type(r) CONTAINS '径流'

### 示例问题 → Cypher
Q: 兰州2024年径流量
A: MATCH (s)-[r]->(d) WHERE s.name CONTAINS '兰州' AND type(r) CONTAINS '径流' AND d.year = '2024' RETURN s.name AS 实体, type(r) AS 指标, d.value AS 数值 LIMIT 20

Q: 龙羊峡2024年和2023年的蓄水量
A: MATCH (s)-[r]->(d:AnnualHydrologyData) WHERE s.name CONTAINS '龙羊峡' AND type(r) CONTAINS '蓄水' AND d.year IN ['2024', '2023'] RETURN s.name AS 实体, type(r) AS 指标, d.value AS 数值, d.year AS 年份 ORDER BY d.year DESC LIMIT 20

Q: 兰州水文站各年径流量
A: MATCH (s)-[r]->(d:AnnualHydrologyData) WHERE s.name CONTAINS '兰州' AND type(r) CONTAINS '径流' RETURN s.name AS 实体, type(r) AS 指标, d.value AS 数值, d.year AS 年份 ORDER BY d.year DESC LIMIT 20

Q: 黄河流域有哪些水库
A: MATCH (r:Reservoir) RETURN r.name, r.river, r.location LIMIT 20

Q: 各分区2024年降水量
A: MATCH (z:WaterResourceZone)-[r]->(d) WHERE type(r) CONTAINS '降水' AND d.year = '2024' RETURN z.name, type(r), d.value, d.unit LIMIT 20

Q: 花园口以下用水量
A: MATCH (s)-[r]->(d) WHERE s.name CONTAINS '花园口以下' AND type(r) CONTAINS '用水' RETURN s.name, type(r), d.value, d.unit LIMIT 20
"""

CYPHER_PROMPT = """<bos><start_of_turn>user
你是 Neo4j Cypher 查询专家。图谱中节点是中文标签(Reservoir/HydrologicalStation等)，但所有关系都使用动态中文短语，没有预定义的关系类型！因此永远不要写 :RELATION_TYPE 这样的语法，必须用 -[r]-> 然后 WHERE type(r) CONTAINS '关键词'。

{schema}

## 铁律
1. 永远不要使用 :RelationshipType 语法！用 -[r]-> 替代
2. 过滤关系用 WHERE type(r) CONTAINS '中文关键词'
3. 用户指定年份时，用 d.year = '2024' 过滤（而不是 type(r) CONTAINS '2024'）
4. 查询所有年份数据时，加 d.year AS 年份 并 ORDER BY d.year DESC
5. 只生成 MATCH ... RETURN
6. LIMIT 20
7. 输出 JSON

## 必看示例
Q: 兰州水文站径流量
A: {{"cypher":"MATCH (s)-[r]->(d:AnnualHydrologyData) WHERE s.name CONTAINS '兰州' AND type(r) CONTAINS '径流' RETURN s.name AS 实体, type(r) AS 指标, d.value AS 数值, d.year AS 年份 ORDER BY d.year DESC LIMIT 20","explanation":"查兰州历年的径流量"}}

Q: 兰州水文站2024年径流量
A: {{"cypher":"MATCH (s)-[r]->(d:AnnualHydrologyData) WHERE s.name CONTAINS '兰州' AND type(r) CONTAINS '径流' AND d.year = '2024' RETURN s.name AS 实体, type(r) AS 指标, d.value AS 数值 LIMIT 20","explanation":"查兰州2024年径流量"}}

Q: 黄河流域用水总量
A: {{"cypher":"MATCH (s)-[r]->(d) WHERE s.name CONTAINS '黄河' AND type(r) CONTAINS '用水' AND type(r) CONTAINS '总量' RETURN s.name AS 实体, type(r) AS 指标, d.value AS 数值 LIMIT 20","explanation":"查用水总量"}}

Q: 龙羊峡水库水位
A: {{"cypher":"MATCH (s)-[r]->(d) WHERE s.name CONTAINS '龙羊峡' AND type(r) CONTAINS '水位' RETURN s.name AS 实体, type(r) AS 指标, d.value AS 数值 LIMIT 20","explanation":"查水位"}}

Q: {question}
<end_of_turn>
<start_of_turn>model
"""


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    cypher: str
    explanation: str
    results: list
    result_count: int

# ============================================================
# 全局 Model
# ============================================================
_model = None
_neo4j = None


def load_model():
    global _model
    if _model is None:
        from llama_cpp import Llama
        model_path = Path(__file__).parent.parent / "models" / "text-to-cypher-Gemma-3-4B-Instruct-2025.04.0.Q4_K_M.gguf"
        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")
        print(f"Loading: {model_path}")
        _model = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,
            n_ctx=2048,
            n_batch=128,
            verbose=False,
        )
        print("Model loaded (GPU)")

def load_neo4j():
    global _neo4j
    if _neo4j is None:
        from neo4j import GraphDatabase
        _neo4j = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "zhishiku"))
        _neo4j.verify_connectivity()
        print("Neo4j connected")


def generate_cypher(question: str) -> dict:
    prompt = CYPHER_PROMPT.format(schema=SCHEMA_CONTEXT, question=question)
    output = _model(prompt, max_tokens=256, temperature=0, stop=["<end_of_turn>", "用户问题:"])
    text = output["choices"][0]["text"].strip()

    import json, re
    match = re.search(r'\{[^{}]*"cypher"[^{}]*\}', text, re.DOTALL)
    result = {"cypher": "", "explanation": ""}
    if match:
        try:
            result = json.loads(match.group())
        except json.JSONDecodeError:
            pass

    if not result.get("cypher"):
        result["explanation"] = f"模型未生成有效Cypher: {text[:200]}"
        return result

    # 后处理：修复不正确的 :关系语法 → type(r) CONTAINS
    cypher = result["cypher"]
    cypher = _fix_relation_syntax(cypher)
    result["cypher"] = cypher
    return result


def _fix_relation_syntax(cypher: str) -> str:
    """修复 Cypher: 将 :中文关系 转为 -[r]-> WHERE type(r) CONTAINS '中文'"""
    import re
    # 模式: -[:关系名]-> 或 -[r:关系名]->
    # 匹配中文字符开头的 :关系
    pattern = re.compile(r'\[(\w*):([一-鿿\w]+)\]')
    matches = pattern.findall(cypher)
    for var, rel_name in matches:
        var_name = var if var else 'r'
        # 替换为无类型的 -[r]->
        old = f'[{var}:{rel_name}]' if var else f'[:{rel_name}]'
        new = f'[{var_name}]'
        cypher = cypher.replace(old, new)

        # 如果还没有 WHERE 子句，添加
        if 'WHERE' not in cypher:
            cypher += f'\nWHERE type({var_name}) CONTAINS \'{rel_name}\''
        elif f'type({var_name}) CONTAINS' not in cypher:
            cypher = cypher.replace('WHERE ', f'WHERE type({var_name}) CONTAINS \'{rel_name}\' AND ')

    return cypher


def execute_cypher(cypher: str) -> list:
    if not cypher or not cypher.strip():
        return []
    # 安全检查：拒绝写操作
    upper = cypher.upper().strip()
    forbidden = ["CREATE", "DELETE", "DROP", "SET ", "REMOVE", "MERGE"]
    for kw in forbidden:
        if upper.startswith(kw) or f" {kw} " in upper:
            return [{"error": f"禁止执行写操作: {kw}"}]

    with _neo4j.session() as session:
        records = session.run(cypher)
        return [dict(r) for r in records]


def format_answer(results: list, question: str) -> str:
    """将查询结果转为自然语言回答"""
    if not results:
        return f"未找到与「{question}」相关的数据，请尝试换个问法。"

    # 检查是否包含错误
    if len(results) == 1 and "error" in results[0]:
        return f"查询执行受限: {results[0]['error']}"

    lines = []
    for r in results[:8]:
        # 尝试多种可能的字段名
        entity = (
            r.get("实体")
            or r.get("s.name")
            or r.get("r.name")
            or r.get("z.name")
            or r.get("name")
            or ""
        )
        indicator = r.get("指标") or r.get("type(r)") or ""
        value = (
            r.get("数值")
            or r.get("d.value")
            or r.get("c.value")
            or r.get("value")
            or ""
        )
        unit = r.get("d.unit") or r.get("unit") or ""
        year = r.get("年份") or r.get("d.year") or r.get("year") or ""

        parts = []
        if entity:
            parts.append(str(entity))
        if year and str(year) != "constant":
            # 在实体后显示年份
            if parts:
                parts[0] = f"{parts[0]}({year}年)"
        if indicator:
            parts.append(str(indicator))
        if value:
            val_str = str(value)
            if unit:
                val_str += str(unit)
            parts.append(val_str)

        if parts:
            lines.append("• " + " | ".join(parts))

    if not lines:
        # 兜底：直接展示原始字段
        for r in results[:5]:
            kv = ", ".join(f"{k}: {v}" for k, v in r.items())
            lines.append(f"• {kv}")

    if len(results) > 8:
        lines.append(f"... 共 {len(results)} 条结果")

    return "\n".join(lines)


# ============================================================
# FastAPI
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    load_neo4j()
    yield
    if _neo4j:
        _neo4j.close()

app = FastAPI(title="HydroBrain 查询服务", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── 静态文件 ──
if FRONTED_DIR.exists():
    for subdir in ["css", "js", "images"]:
        sd = FRONTED_DIR / subdir
        if sd.exists():
            app.mount(f"/{subdir}", StaticFiles(directory=str(sd)), name=subdir)


# ── 页面路由 ──
@app.get("/")
def index():
    """返回前端首页"""
    index_path = FRONTED_DIR / "index_hydra.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"status": "running", "model": "Gemma-3-4B-Text2Cypher", "neo4j": _neo4j is not None}


@app.get("/health")
def root_health():
    """兼容旧版"""
    return {"status": "running", "model": "Gemma-3-4B-Text2Cypher", "neo4j": _neo4j is not None}


# ── 前端 API ──
@app.get("/api/health")
def api_health():
    """健康检查 — 前端期望格式"""
    examples = [
        "兰州水文站径流量",
        "黄河流域用水总量",
        "龙羊峡水库水位",
        "黄河流域有哪些水库",
    ]
    return {
        "neo4j": _neo4j is not None,
        "rag_size": len(examples),
        "model": "Gemma-3-4B-Text2Cypher",
    }


@app.post("/api/chat")
def api_chat(req: QueryRequest):
    """自然语言问答 — 前端期望格式"""
    import time
    t0 = time.time()
    parsed = generate_cypher(req.question)
    t1 = time.time()
    cypher = parsed.get("cypher", "")
    results = execute_cypher(cypher) if cypher else []
    t2 = time.time()

    answer = format_answer(results, req.question)

    return {
        "answer": answer,
        "cypher": cypher,
        "timing": {
            "cypher_gen": round(t1 - t0, 2),
            "neo4j": round(t2 - t1, 2),
            "total": round(t2 - t0, 2),
        },
    }


# ── 旧版 API (保留兼容) ──
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    import time
    t0 = time.time()
    parsed = generate_cypher(req.question)
    cypher = parsed.get("cypher", "")
    explanation = parsed.get("explanation", "")

    results = execute_cypher(cypher) if cypher else []
    elapsed = time.time() - t0

    return QueryResponse(
        question=req.question,
        cypher=cypher,
        explanation=explanation,
        results=results,
        result_count=len(results),
    )


@app.get("/query")
def query_get(q: str = Query(..., description="自然语言问题")):
    return query(QueryRequest(question=q))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
