from fastapi import APIRouter, HTTPException

from apps.models.CommonModels import ApiResult
from apps.models.DomainModels import (
    ProjectItem,
    BatchProjectRequest,
    PredictionResult,
    BatchPredictionResponse,
    CategoryMappingResponse,
)
from apps.services.ten_domain_service import get_ten_domain_service

router = APIRouter()


@router.post("/predict", response_model=ApiResult[PredictionResult], summary="重大项目十大领域 - 单项目即时预测")
async def predict_single_ten_domain(item: ProjectItem):
    """
    输入单项目的【项目名称】与【建设规模和内容】，
    基于 DNN 深度学习模型，预测其所属的浙江省重大项目十大领域与二级细分类别。
    """
    if not item.name.strip() and not item.content.strip():
        raise HTTPException(status_code=400, detail="项目名称和建设内容不能同时为空！")

    try:
        service = get_ten_domain_service()
        res = service.predict_one(item.name, item.content)
        return ApiResult.success(res)
    except Exception as e:
        return ApiResult.fail(code=500, msg=f"十大领域模型推断异常: {str(e)}")


@router.post("/predict/batch", response_model=ApiResult[BatchPredictionResponse], summary="重大项目十大领域 - 批量并发预测")
async def predict_batch_ten_domain(req: BatchProjectRequest):
    """批量传入多个项目，执行十大领域快速前向推断，返回预测结果列表"""
    try:
        service = get_ten_domain_service()
        res = service.predict_batch(req.projects)
        return ApiResult.success(res)
    except Exception as e:
        return ApiResult.fail(code=500, msg=f"十大领域批量推断异常: {str(e)}")


@router.get("/categories", response_model=ApiResult[CategoryMappingResponse], summary="重大项目十大领域 - 类别与层级映射字典")
async def get_ten_domain_categories():
    """获取所有标准二级领域及其所属的十大一级领域完整映射表"""
    try:
        service = get_ten_domain_service()
        res = service.get_categories()
        return ApiResult.success(res)
    except Exception as e:
        return ApiResult.fail(code=500, msg=f"获取十大领域类别字典异常: {str(e)}")
