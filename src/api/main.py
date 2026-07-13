"""
HydroBrain FastAPI 知识服务入口
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import neo4j_client, graph_query, get_graph_query, success_response
from src.api.routers import reservoir, hydrology, knowledge
from src.common.config_loader import ConfigLoader
from src.common.logger import get_logger
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.graph_query import GraphQuery

logger = get_logger(__name__)

# 用模块引用更新 dependencies 中的全局变量
import src.api.dependencies as deps

app_config: ConfigLoader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_config

    logger.info("HydroBrain API 启动中...")
    app_config = ConfigLoader()

    try:
        neo4j_cfg = app_config.get_neo4j_config()
        deps.neo4j_client = Neo4jClient(
            uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
            user=neo4j_cfg.get("user", "neo4j"),
            password=neo4j_cfg.get("password", "password"),
        )
        deps.graph_query = GraphQuery(deps.neo4j_client)
        logger.info("Neo4j 连接就绪")
    except Exception as e:
        logger.warning(f"Neo4j 不可用: {e}")

    yield

    if deps.neo4j_client:
        deps.neo4j_client.close()
    logger.info("HydroBrain API 已关闭")


app = FastAPI(
    title="HydroBrain - 黄河水库调度知识图谱服务",
    version="0.1.0",
    description="基于 DeepSeek LLM 与 Neo4j 知识图谱的水库调度知识挖掘系统",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reservoir.router, prefix="/api/reservoir", tags=["水库信息"])
app.include_router(hydrology.router, prefix="/api/hydrology", tags=["水文数据"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["知识查询"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"未捕获异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": f"服务器内部错误: {str(exc)}", "data": None},
    )


@app.get("/")
def health_check():
    neo4j_status = "connected" if deps.neo4j_client else "unavailable"
    return {
        "system": "HydroBrain",
        "version": "0.1.0",
        "status": "running",
        "neo4j": neo4j_status,
        "deepseek_model": app_config.get("deepseek.model", "N/A") if app_config else "N/A",
    }
