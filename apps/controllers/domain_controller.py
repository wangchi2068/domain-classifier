from fastapi import APIRouter

from apps.models.CommonModels import ApiResult
from apps.models.DomainModels import (
    ProjectItem,
    BatchProjectRequest,
    PredictionResult,
    BatchPredictionResponse,
    CategoryMappingResponse,
)
from apps.services.new_domain_service import get_new_domain_service

router = APIRouter()


@router.post("/predict", response_model=ApiResult[PredictionResult], summary="领域分类识别 - 单项目即时预测 (通用默认路径)")
async def predict_single(item: ProjectItem):
    name = (item.name or "").strip()
    content = (item.content or "").strip()
    if not name and not content:
        return ApiResult.fail(code=400, msg="项目名称和建设内容不能同时为空！")

    try:
        service = get_new_domain_service()
        res = service.predict_one(name, content)
        return ApiResult.success(res)
    except Exception as e:
        return ApiResult.fail(code=500, msg=f"模型推断过程出现异常: {str(e)}")


@router.post("/predict/batch", response_model=ApiResult[BatchPredictionResponse], summary="领域分类识别 - 批量项目并发预测")
async def predict_batch(req: BatchProjectRequest):
    try:
        service = get_new_domain_service()
        res = service.predict_batch(req.projects)
        return ApiResult.success(res)
    except Exception as e:
        return ApiResult.fail(code=500, msg=f"批量推断出现异常: {str(e)}")


@router.get("/categories", response_model=ApiResult[CategoryMappingResponse], summary="领域分类识别 - 获取类别字典")
async def get_categories():
    try:
        service = get_new_domain_service()
        res = service.get_categories()
        return ApiResult.success(res)
    except Exception as e:
        return ApiResult.fail(code=500, msg=f"获取类别字典异常: {str(e)}")
