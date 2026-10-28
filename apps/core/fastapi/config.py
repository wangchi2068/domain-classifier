import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps import BS_PROFILE_NAME
from apps.config.config import get_app_config, init_config
from apps.controllers import controller_router
from apps.services.ten_domain_service import get_ten_domain_service
from apps.services.new_domain_service import get_new_domain_service


@asynccontextmanager
async def lifespan(application: FastAPI):
    print("=" * 65)
    print("[INFO] 正在预热加载投资领域智能分类双 DNN 深度学习模型...")

    # 1. 预热十大领域模型
    ten_service = get_ten_domain_service()
    cats_ten = ten_service.get_categories()
    print(
        f"[INFO] 1. 重大项目十大领域模型预热就绪: 一级领域 {len(cats_ten.一级领域列表)} 类，二级领域 {len(cats_ten.二级领域列表)} 类"
    )

    # 2. 预热新领域案例模型
    new_service = get_new_domain_service()
    cats_new = new_service.get_categories()
    print(
        f"[INFO] 2. 专项债新领域案例模型预热就绪: 一级领域 {len(cats_new.一级领域列表)} 类，二级领域 {len(cats_new.二级领域列表)} 类"
    )

    print("=" * 65)
    yield
    print("[INFO] 投资领域智能分类微服务已安全关闭。")


def create_application() -> FastAPI:
    profile = os.getenv(BS_PROFILE_NAME, "dev")
    if get_app_config() is None:
        init_config(profile)

    server_config = get_app_config().get("server", {})
    enable_openapi = server_config.get("openapi", True)
    context_path = server_config.get("context_path", "/domain-api")

    if enable_openapi:
        application = FastAPI(
            title="投资领域智能分类微服务 API (双标准 DNN 引擎)",
            description="集成【浙江省重大项目十大领域】与【专项债/预算内新领域案例】双标准的深度神经网络分类微服务",
            version="2.0.0",
            lifespan=lifespan,
        )
    else:
        application = FastAPI(
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
            lifespan=lifespan,
        )

    # 跨域配置
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册统一路由
    application.include_router(controller_router, prefix=context_path)
    # 同时在根路径注册一份，方便直接访问 /domain/ten/predict 等
    application.include_router(controller_router, prefix="")

    return application


app = create_application()
