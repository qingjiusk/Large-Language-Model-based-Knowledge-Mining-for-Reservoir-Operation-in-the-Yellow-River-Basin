#!/usr/bin/env python
"""
HydroBrain 自然语言查询服务
Gemma 3 4B Text2Cypher (llama-cpp-python) + Neo4j

启动: conda activate zagism && python scripts/query_server.py
访问: http://127.0.0.1:8001/docs
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
Reservoir 13个, HydrologicalStation 14个, WaterResourceZone 15个, Province 70个, River 20个, AnnualHydrologyData 710个, Constraint 155个

### 关系
所有关系类型都是 LLM 抽取的中文短语，如 "2024年实测径流量为" "天然河川径流量为" "用水总量为" "流域总面积为"
查询时使用 type(r) CONTAINS '关键词' 模糊匹配

### 示例问题 → Cypher
Q: 兰州2024年径流量
A: MATCH (s)-[r]->(d) WHERE s.name CONTAINS '兰州' AND type(r) CONTAINS '径流量' RETURN s.name, type(r), d.value, d.unit LIMIT 20

Q: 黄河流域有哪些水库
A: MATCH (r:Reservoir) RETURN r.name, r.river, r.location LIMIT 20

Q: 各分区降水量
A: MATCH (z:WaterResourceZone)-[r]->(d) WHERE type(r) CONTAINS '降水' RETURN z.name, type(r), d.value, d.unit, d.year LIMIT 20

Q: 花园口以下用水量
A: MATCH (s)-[r]->(d) WHERE s.name CONTAINS '花园口以下' AND type(r) CONTAINS '用水' RETURN s.name, type(r), d.value, d.unit LIMIT 20
"""

CYPHER_PROMPT = """<bos><start_of_turn>user
你是 Neo4j Cypher 查询专家。图谱中节点是中文标签(Reservoir/HydrologicalStation等)，但所有关系都使用动态中文短语，没有预定义的关系类型！因此永远不要写 :RELATION_TYPE 这样的语法，必须用 -[r]-> 然后 WHERE type(r) CONTAINS '关键词'。

{schema}

## 铁律
1. 永远不要使用 :RelationshipType 语法！用 -[r]-> 替代
2. 过滤关系用 WHERE type(r) CONTAINS '中文关键词'
3. 只生成 MATCH ... RETURN
4. LIMIT 20
5. 输出 JSON

## 必看示例
Q: 兰州水文站径流量
A: {{"cypher":"MATCH (s)-[r]->(d) WHERE s.name CONTAINS '兰州' AND type(r) CONTAINS '径流' RETURN s.name AS 实体, type(r) AS 指标, d.value AS 数值 LIMIT 20","explanation":"查径流量"}}

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

app = FastAPI(title="HydroBrain 查询服务", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def health():
    return {"status": "running", "model": "Gemma-3-4B-Text2Cypher", "neo4j": _neo4j is not None}


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
