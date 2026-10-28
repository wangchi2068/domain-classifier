from fastapi import APIRouter, HTTPException

from apps.models.CommonModels import ApiResult
from apps.models.DomainModels import (
    ProjectItem,
    BatchProjectRequest,
    PredictionResult,
    BatchPredictionResponse,
    CategoryMappingResponse,
)
from apps.services.domain_service import get_domain_service

router = APIRouter()


@router.post("/predict", response_model=ApiResult[PredictionResult], summary="单项目即时领域预测")
async def predict_single(item: ProjectItem):
    """
    输入单项目的【项目名称】与【主要建设内容】，
    返回其最精确的二级领域、对应一级领域、置信度及 Top 3 候选概率分布。
    """
    if not item.name.strip() and not item.content.strip():
        raise HTTPException(status_code=400, detail="项目名称和建设内容不能同时为空！")

    try:
        service = get_domain_service()
        res = service.predict_one(item.name, item.content)
        return ApiResult.success(res)
    except Exception as e:
        return ApiResult.fail(code=500, msg=f"模型推断过程出现异常: {str(e)}")


@router.post("/predict/batch", response_model=ApiResult[BatchPredictionResponse], summary="批量项目并发预测")
async def predict_batch(req: BatchProjectRequest):
    """批量传入多个项目列表，执行快速前向推断，统一返回预测结果列表"""
    try:
        service = get_domain_service()
        res = service.predict_batch(req.projects)
        return ApiResult.success(res)
    except Exception as e:
        return ApiResult.fail(code=500, msg=f"批量推断出现异常: {str(e)}")


@router.get("/categories", response_model=ApiResult[CategoryMappingResponse], summary="获取所有领域类别与层级映射")
async def get_categories():
    """获取所有 20 个二级领域及其唯一绑定的 4 个一级领域父子字典"""
    try:
        service = get_domain_service()
        res = service.get_categories()
        return ApiResult.success(res)
    except Exception as e:
        return ApiResult.fail(code=500, msg=f"获取类别字典异常: {str(e)}")
