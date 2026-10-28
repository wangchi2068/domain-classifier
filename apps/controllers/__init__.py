from fastapi import APIRouter
from apps.controllers import ten_domain_controller, new_domain_controller, health_controller

controller_router = APIRouter()

# 1. 浙江省重大项目十大领域分类 (46 个二级领域，10 个一级领域)
controller_router.include_router(ten_domain_controller.router, prefix="/domain/ten", tags=["重大项目-十大领域分类模块"])

# 2. 专项债/预算内新领域案例分类 (20 个二级领域，4 个一级领域)
controller_router.include_router(new_domain_controller.router, prefix="/domain/new", tags=["专项债-新领域案例分类模块"])

# 3. 根路径兼容 alias (默认指向新领域案例预测)
controller_router.include_router(new_domain_controller.router, prefix="/domain", tags=["领域分类预测通用接口"])

# 4. 健康检查
controller_router.include_router(health_controller.router, prefix="", tags=["健康检查"])
