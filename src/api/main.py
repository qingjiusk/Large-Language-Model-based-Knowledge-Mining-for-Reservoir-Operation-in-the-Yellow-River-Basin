"""
HydroBrain FastAPI 知识服务入口
"""
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers import reservoir, hydrology, knowledge
from src.common.config_loader import ConfigLoader
from src.common.logger import get_logger
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.graph_query import GraphQuery

logger = get_logger(__name__)

# 全局实例（延迟初始化）
neo4j_client: Neo4jClient = None
graph_query: GraphQuery = None
app_config: ConfigLoader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global neo4j_client, graph_query, app_config

    logger.info("HydroBrain API 启动中...")

    # 加载配置
    app_config = ConfigLoader()

    # 初始化 Neo4j（可选，如果不可用则跳过）
    try:
        neo4j_config = app_config.get_neo4j_config()
        neo4j_client = Neo4jClient(
            uri=neo4j_config.get("uri", "bolt://localhost:7687"),
            user=neo4j_config.get("user", "neo4j"),
            password=neo4j_config.get("password", "password"),
        )
        graph_query = GraphQuery(neo4j_client)
        logger.info("Neo4j 连接就绪")
    except Exception as e:
        logger.warning(f"Neo4j 不可用，图谱查询接口将返回 503: {e}")
        neo4j_client = None
        graph_query = None

    yield

    # 关闭
    if neo4j_client:
        neo4j_client.close()
    logger.info("HydroBrain API 已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="HydroBrain - 黄河水库调度知识图谱服务",
    version="0.1.0",
    description="基于 DeepSeek LLM 与 Neo4j 知识图谱的水库调度知识挖掘系统",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(reservoir.router, prefix="/api/reservoir", tags=["水库信息"])
app.include_router(hydrology.router, prefix="/api/hydrology", tags=["水文数据"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["知识查询"])


# ==================== 全局异常处理 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"未捕获异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": f"服务器内部错误: {str(exc)}", "data": None},
    )


# ==================== 统一响应模型 ====================

def success_response(data: Any = None, message: str = "ok") -> Dict:
    """统一成功响应"""
    return {"code": 200, "message": message, "data": data}


def error_response(code: int, message: str) -> Dict:
    """统一错误响应"""
    return {"code": code, "message": message, "data": None}


# ==================== 健康检查 ====================

@app.get("/")
def health_check():
    """系统健康检查"""
    neo4j_status = "connected" if neo4j_client else "unavailable"
    return {
        "system": "HydroBrain",
        "version": "0.1.0",
        "status": "running",
        "neo4j": neo4j_status,
        "deepseek_model": app_config.get("deepseek.model", "N/A") if app_config else "N/A",
    }


def get_graph_query():
    """获取 GraphQuery 实例，未就绪时抛异常"""
    if graph_query is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Neo4j 服务不可用")
    return graph_query
