import os
import re
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from apps import singleton
from apps.config.config import get_app_config
from apps.models.DomainModels import PredictionResult, CandidateItem, BatchPredictionResponse, CategoryMappingResponse


class TenDomainHierarchicalNet(nn.Module):
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
class TenDomainService:
    """浙江省重大投资项目十大领域分类服务 (DNN 614维层次网络)"""

    def __init__(self):
        self.is_loaded = False
        self.bge = None
        self.scaler = None
        self.proto_means = None
        self.classes_l2 = None
        self.classes_l1 = None
        self.l2_to_l1 = None
        self.l2_parent_idx = None
        self.bank = None
        self.kw_idx = None
        self.entity_patterns = None
        self.dnn_models = []

    def load_model(self):
        if self.is_loaded:
            return

        app_config = get_app_config()
        base_model_dir = app_config.get("model_dir", "./model_dir")
        ten_dir = os.path.join(base_model_dir, "ten_domain")
        bge_path = app_config.get(
            "bge_model_path",
            r"D:/trae/实体识别/embedding_cache_pkg/models--BAAI--bge-small-zh/snapshots/1d2363c5de6ce9ba9c890c8e23a4c72dce540ca8",
        )

        # 1. 类别元数据与词典
        classes_path = os.path.join(ten_dir, "classes.json")
        with open(classes_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.classes_l2 = meta["classes"]
        self.classes_l1 = meta.get("classes_l1", [])
        self.l2_to_l1 = meta["l2_to_l1"]
        self.l2_parent_idx = np.array(meta.get("l2_parent_idx", []), dtype=np.int64)
        self.bank = meta["bank"]
        self.kw_idx = {self.classes_l2[i]: i for i in range(len(self.classes_l2))}
        self.entity_patterns = meta.get(
            "entity_patterns",
            [
                r"有限公司|股份有限公司|实业|制造|机械|电气|重工|科技公司|集团",
                r"大学|学院|中科院|中国科学院|国家实验室|大科学装置|研究院所",
                r"医院|卫生院|疾控中心|妇幼保健|医学中心",
                r"中学|小学|幼儿园|校区|校舍|职教",
                r"高速公路|省道|国道|公路|立交|高架|城市道路",
                r"铁路|城际|轨道交通|地铁|轻轨|站前广场|枢纽",
                r"港口|码头|泊位|航道|船闸|防波堤",
                r"变电站|输变电|特高压|风电|光伏|电站|储能|抽水蓄能",
                r"水库|海塘|堤防|排涝|灌区|引水|供水|闸站",
                r"未来社区|老旧小区|三化九场景|社区服务|安置房|城中村",
            ],
        )

        # 2. 标准化器与类原型
        with open(os.path.join(ten_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        self.proto_means = np.load(os.path.join(ten_dir, "proto_means.npy")).astype(np.float32)

        # 3. DNN 模型权重 (优先 3-Seed 集成)
        in_dim = self.scaler.mean_.shape[0]  # 614维
        n_l1 = len(self.classes_l1)
        n_l2 = len(self.classes_l2)

        ens_path = os.path.join(ten_dir, "ensemble_weights.pth")
        single_path = os.path.join(ten_dir, "model_weights.pth")

        self.dnn_models = []
        if os.path.exists(ens_path):
            weights_dict = torch.load(ens_path, map_location="cpu")
            for k in ["net_seed42", "net_seed100", "net_seed2024"]:
                if k in weights_dict:
                    net = TenDomainHierarchicalNet(in_dim, n_l1, n_l2)
                    net.load_state_dict(weights_dict[k])
                    net.eval()
                    self.dnn_models.append(net)
        elif os.path.exists(single_path):
            net = TenDomainHierarchicalNet(in_dim, n_l1, n_l2)
            net.load_state_dict(torch.load(single_path, map_location="cpu"))
            net.eval()
            self.dnn_models.append(net)
        else:
            raise FileNotFoundError(f"未找到十大领域 DNN 权重文件: {single_path}")

        # 4. BGE 语义模型
        torch.set_num_threads(4)
        self.bge = SentenceTransformer(bge_path)

        # 预热一次
        _ = self._raw_predict("预热项目", "主要建设工程预热")
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

        # 1. BGE 512维
        x_emb = self.bge.encode([full_text], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

        # 2. 46 维关键词词频
        kw_vec = np.zeros((1, len(self.classes_l2)), dtype=np.float32)
        for l2, terms in self.bank.items():
            idx = self.kw_idx.get(l2)
            if idx is not None:
                c = sum(1 for term in terms if term in full_text)
                if c:
                    kw_vec[0, idx] = np.log1p(c)

        # 3. 46 维类原型几何相似度
        x_norm = x_emb / (np.linalg.norm(x_emb, axis=1, keepdims=True) + 1e-12)
        proto_vec = (x_norm @ self.proto_means.T).astype(np.float32)

        # 4. 10 维实体先验
        ent_vec = np.zeros((1, len(self.entity_patterns)), dtype=np.float32)
        for j, pat in enumerate(self.entity_patterns):
            if re.search(pat, full_text):
                ent_vec[0, j] = 1.0

        # 5. 特征拼接与标准化 (614维)
        feat = np.hstack([x_emb, kw_vec, proto_vec, ent_vec]).astype(np.float32)
        feat_norm = self.scaler.transform(feat).astype(np.float32)

        # 6. 神经网络前向推理
        Xt = torch.tensor(feat_norm, dtype=torch.float32)
        p1_list = []
        p2_list = []
        with torch.no_grad():
            for m in self.dnn_models:
                o1, o2 = m(Xt)
                p1_list.append(torch.softmax(o1, dim=1).numpy())
                p2_list.append(torch.softmax(o2, dim=1).numpy())

        avg_p1 = np.mean(p1_list, axis=0)[0]
        avg_p2 = np.mean(p2_list, axis=0)[0]

        # 7. 贝叶斯层级门控融合 (Beta=0.3)
        if len(self.l2_parent_idx) == len(self.classes_l2):
            fused_p2 = avg_p2 * (avg_p1[self.l2_parent_idx] ** 0.3)
        else:
            fused_p2 = avg_p2

        # 过滤表头脏类别 '二级领域'
        for i, c in enumerate(self.classes_l2):
            if c == "二级领域":
                fused_p2[i] = -1.0

        pred_idx = int(fused_p2.argmax())
        confidence = float(avg_p2[pred_idx])
        pred_l2 = self.classes_l2[pred_idx]
        pred_l1 = self.l2_to_l1.get(pred_l2, "")

        top3_indices = [i for i in fused_p2.argsort()[::-1] if self.classes_l2[i] != "二级领域"][:3]
        top3 = [
            CandidateItem(
                二级领域=self.classes_l2[i],
                一级领域=self.l2_to_l1.get(self.classes_l2[i], ""),
                概率=round(float(avg_p2[i]), 4),
            )
            for i in top3_indices
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
        # 清洗返回列表，剔除表头误入项
        classes_l2_clean = [c for c in self.classes_l2 if c != "二级领域"]
        classes_l1_clean = [c for c in self.classes_l1 if c != "一级领域"]
        l2_to_l1_clean = {k: v for k, v in self.l2_to_l1.items() if k != "二级领域"}
        return CategoryMappingResponse(
            一级领域列表=classes_l1_clean,
            二级领域列表=classes_l2_clean,
            二级到一级归属映射=l2_to_l1_clean,
        )


def get_ten_domain_service() -> TenDomainService:
    service = TenDomainService()
    if not service.is_loaded:
        service.load_model()
    return service
