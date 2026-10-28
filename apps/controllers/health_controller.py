from fastapi import APIRouter
from apps.models.CommonModels import ApiResult
from apps.services.ten_domain_service import get_ten_domain_service
from apps.services.new_domain_service import get_new_domain_service

router = APIRouter()


@router.get("/health", response_model=ApiResult[dict], summary="服务健康检查")
async def health():
    """检测服务是否正常运行，返回双模型加载状态"""
    try:
        ten_service = get_ten_domain_service()
        new_service = get_new_domain_service()
        data = {
            "status": "healthy",
            "models": {
                "重大项目十大领域DNN": {
                    "loaded": ten_service.is_loaded,
                    "一级领域数量": len(ten_service.get_categories().一级领域列表),
                    "二级领域数量": len(ten_service.get_categories().二级领域列表),
                },
                "专项债新领域案例DNN": {
                    "loaded": new_service.is_loaded,
                    "一级领域数量": len(new_service.classes_l1) if new_service.classes_l1 else 4,
                    "二级领域数量": len(new_service.classes_l2) if new_service.classes_l2 else 20,
                },
            },
        }
        return ApiResult.success(data)
    except Exception as e:
        return ApiResult.fail(code=500, msg=str(e))
