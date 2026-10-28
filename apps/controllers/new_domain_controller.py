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


@router.post("/predict", response_model=ApiResult[PredictionResult], summary="专项债新领域案例 - 单项目即时预测")
async def predict_single_new_domain(item: ProjectItem):
    """
    输入单项目的【项目名称】与【主要建设内容】，
    基于 DNN 深度学习模型，预测其所属的 20 个二级领域并自动锁定一级领域。
    """
    name = (item.name or "").strip()
    content = (item.content or "").strip()
    if not name and not content:
        return ApiResult.fail(code=400, msg="项目名称和建设内容不能同时为空！")

    try:
        service = get_new_domain_service()
        res = service.predict_one(name, content)
        return ApiResult.success(res)
    except Exception as e:
        return ApiResult.fail(code=500, msg=f"新领域案例模型推断异常: {str(e)}")


@router.post("/predict/batch", response_model=ApiResult[BatchPredictionResponse], summary="专项债新领域案例 - 批量并发预测")
async def predict_batch_new_domain(req: BatchProjectRequest):
    """批量传入多个项目，执行新领域案例快速前向推断，返回预测结果列表"""
    try:
        service = get_new_domain_service()
        res = service.predict_batch(req.projects)
        return ApiResult.success(res)
    except Exception as e:
        return ApiResult.fail(code=500, msg=f"新领域案例批量推断异常: {str(e)}")


@router.get("/categories", response_model=ApiResult[CategoryMappingResponse], summary="专项债新领域案例 - 类别与层级映射字典")
async def get_new_domain_categories():
    """获取所有 20 个二级领域及其唯一绑定的 4 个一级领域映射字典"""
    try:
        service = get_new_domain_service()
        res = service.get_categories()
        return ApiResult.success(res)
    except Exception as e:
        return ApiResult.fail(code=500, msg=f"获取新领域案例类别字典异常: {str(e)}")
