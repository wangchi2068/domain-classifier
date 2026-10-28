import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from apps import singleton
from apps.config.config import get_app_config
from apps.models.DomainModels import PredictionResult, CandidateItem, BatchPredictionResponse, CategoryMappingResponse
from apps.services.strict_pipeline import StrictFeaturePipeline


class HierarchicalNet(nn.Module):
    def __init__(self, in_dim, n_l1, n_l2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head_l1 = nn.Linear(64, n_l1)
        self.head_l2 = nn.Linear(64, n_l2)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head_l1(feat), self.head_l2(feat)


@singleton
class NewDomainService:
    """专项债与预算内新领域案例分类服务 (DNN 567维层次网络)"""

    def __init__(self):
        self.is_loaded = False
        self.bge = None
        self.pipeline = None
        self.classes_l2 = None
        self.classes_l1 = None
        self.l2_to_l1 = None
        self.l2_parent_idx = None
        self.dnn_models = []

    def load_model(self):
        if self.is_loaded:
            return

        app_config = get_app_config()
        base_model_dir = app_config.get("model_dir", "./model_dir")
        new_dir = os.path.join(base_model_dir, "new_domain")
        bge_path = app_config.get(
            "bge_model_path",
            r"D:/trae/实体识别/embedding_cache_pkg/models--BAAI--bge-small-zh/snapshots/1d2363c5de6ce9ba9c890c8e23a4c72dce540ca8",
        )

        # 1. 类别元数据
        classes_json_path = os.path.join(new_dir, "classes.json")
        with open(classes_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.classes_l2 = meta["classes_l2"]
        self.classes_l1 = meta["classes_l1"]
        self.l2_to_l1 = meta["l2_to_l1"]
        self.l2_parent_idx = np.array(meta["l2_parent_idx"], dtype=np.int64)

        # 2. 特征工程管道 (注册别名以兼容 pickle 协议)
        import sys
        from apps.services import strict_pipeline

        sys.modules["strict_pipeline"] = strict_pipeline

        pipeline_path = os.path.join(new_dir, "pipeline.pkl")
        with open(pipeline_path, "rb") as f:
            self.pipeline = pickle.load(f)

        # 3. DNN 模型权重
        in_dim = self.pipeline.scaler.mean_.shape[0]
        ens_dnn_path = os.path.join(new_dir, "dnn_ensemble_weights.pth")
        single_dnn_path = os.path.join(new_dir, "dnn_model.pth")

        self.dnn_models = []
        if os.path.exists(ens_dnn_path):
            weights_dict = torch.load(ens_dnn_path, map_location="cpu")
            for k in ["net_seed42", "net_seed100", "net_seed2024"]:
                if k in weights_dict:
                    net = HierarchicalNet(in_dim, len(self.classes_l1), len(self.classes_l2))
                    net.load_state_dict(weights_dict[k])
                    net.eval()
                    self.dnn_models.append(net)
        elif os.path.exists(single_dnn_path):
            net = HierarchicalNet(in_dim, len(self.classes_l1), len(self.classes_l2))
            net.load_state_dict(torch.load(single_dnn_path, map_location="cpu"))
            net.eval()
            self.dnn_models.append(net)
        else:
            raise FileNotFoundError(f"未找到新领域 DNN 权重文件: {single_dnn_path}")

        # 4. BGE 模型
        torch.set_num_threads(4)
        self.bge = SentenceTransformer(bge_path)

        # 预热一次
        _ = self._raw_predict("预热项目", "预热内容")
        self.is_loaded = True

    def _raw_predict(self, name: str, content: str = "") -> dict:
        full_text = f"{name}: {content}".strip(": ")
        if not full_text:
            return {
                "一级领域": "未知",
                "二级领域": "未知",
                "置信度": 0.0,
                "Top3候选": [],
            }

        x_emb = self.bge.encode([full_text], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        feat_norm = self.pipeline.transform([full_text], x_emb)

        Xt = torch.tensor(feat_norm, dtype=torch.float32)
        dnn_p1_list = []
        dnn_p2_list = []
        with torch.no_grad():
            for m in self.dnn_models:
                o1, o2 = m(Xt)
                dnn_p1_list.append(torch.softmax(o1, dim=1).numpy())
                dnn_p2_list.append(torch.softmax(o2, dim=1).numpy())

        avg_p1 = np.mean(dnn_p1_list, axis=0)[0]
        avg_p2 = np.mean(dnn_p2_list, axis=0)[0]

        # 贝叶斯层级概率门控融合 (Beta=0.3)
        fused_p2 = avg_p2 * (avg_p1[self.l2_parent_idx] ** 0.3)
        final_probs = fused_p2 / fused_p2.sum()

        pred_idx = int(final_probs.argmax())
        confidence = float(final_probs[pred_idx])
        pred_l2 = self.classes_l2[pred_idx]
        pred_l1 = self.l2_to_l1.get(pred_l2, "")

        top3_idx = final_probs.argsort()[-3:][::-1]
        top3 = [
            CandidateItem(
                二级领域=self.classes_l2[i],
                一级领域=self.l2_to_l1.get(self.classes_l2[i], ""),
                概率=round(float(final_probs[i]), 4),
            )
            for i in top3_idx
        ]

        return {
            "一级领域": pred_l1,
            "二级领域": pred_l2,
            "置信度": round(confidence, 4),
            "Top3候选": top3,
        }

    def predict_one(self, name: str, content: str = "") -> PredictionResult:
        res = self._raw_predict(name, content)
        return PredictionResult(
            项目名称=name,
            一级领域=res["一级领域"],
            二级领域=res["二级领域"],
            置信度=res["置信度"],
            Top3候选=res["Top3候选"],
        )

    def predict_batch(self, projects) -> BatchPredictionResponse:
        results = []
        for p in projects:
            res = self.predict_one(p.name, p.content)
            results.append(res)
        return BatchPredictionResponse(total=len(results), results=results)

    def get_categories(self) -> CategoryMappingResponse:
        return CategoryMappingResponse(
            一级领域列表=self.classes_l1,
            二级领域列表=self.classes_l2,
            二级到一级归属映射=self.l2_to_l1,
        )


def get_new_domain_service() -> NewDomainService:
    service = NewDomainService()
    if not service.is_loaded:
        service.load_model()
    return service
