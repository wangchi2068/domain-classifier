from typing import List, Optional
from pydantic import BaseModel, Field


class ProjectItem(BaseModel):
    name: str = Field(..., description="项目名称", examples=["某市主城区雨污水管网提质增效与综合管廊工程"])
    content: Optional[str] = Field("", description="主要建设内容（可选）", examples=["对老旧供排水管网进行全面改造，新建地下综合管廊3.2公里。"])


class BatchProjectRequest(BaseModel):
    projects: List[ProjectItem] = Field(..., description="待预测项目列表")


class CandidateItem(BaseModel):
    二级领域: str
    一级领域: str
    概率: float


class PredictionResult(BaseModel):
    项目名称: str
    一级领域: str = Field(..., description="自动连带锁定的所属一级领域")
    二级领域: str = Field(..., description="模型预测的二级领域细分")
    置信度: float = Field(..., description="模型预测置信度 (0.0~1.0)")
    Top3候选: List[CandidateItem] = Field(..., description="概率最高的 Top 3 候选类别分布")


class BatchPredictionResponse(BaseModel):
    total: int
    results: List[PredictionResult]


class CategoryMappingResponse(BaseModel):
    一级领域列表: List[str]
    二级领域列表: List[str]
    二级到一级归属映射: dict
