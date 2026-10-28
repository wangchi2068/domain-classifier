# -*- coding: utf-8 -*-
"""
纯粹的特征变换管道模块 (严格防数据泄露)
在任何交叉验证或全量训练中：
- 变换器的 fit 仅使用当次给定的训练折数据
- transform 用于对应的验证折或测试集
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

STOP_WORDS = set("项目 建设 生产 年产 面积 厂房 设备 形成 能力 约 平方米 总投资 新建 扩建 主要内容 包括 以及 等 生产设施 辅助设施 先进 主要 相关 及其他 等配套".split())

ENTITY_PATTERNS = [
    r"有限公司|股份有限公司|实业|制造|机械|电气|重工|科技公司|集团|厂",  # 0: 工业企业
    r"大学|学院|中科院|中国科学院|国家实验室|大科学装置|研究院|试验中心",  # 1: 科研教育
    r"医院|卫生院|疾控中心|妇幼保健|医学中心|门诊",  # 2: 医疗机构
    r"养老院|敬老院|托育|托儿所|老年公寓|福利院",  # 3: 养老托育
    r"高速公路|省道|国道|公路|立交|高架|城市道路|桥梁",  # 4: 道路公路
    r"铁路|城际|轨道交通|地铁|轻轨|站前广场|高铁站",  # 5: 铁路轨道
    r"港口|码头|泊位|航道|船闸|防波堤|水运",  # 6: 水运港口
    r"变电站|输变电|特高压|风电|光伏|电站|储能|抽水蓄能|核电|生物质发电",  # 7: 清洁能源
    r"水库|海塘|堤防|排涝|灌区|引水|供水|闸站|水利",  # 8: 水利工程
    r"地下管网|管廊|污水管网|雨水管网|供水管网|综合管廊|供热管网",  # 9: 地下管网管廊
    r"老旧小区|老旧厂区|老旧街区|棚户区|城中村|城市更新|未来社区",  # 10: 城市更新改造
    r"农贸市场|批发市场|仓储配送|冷链物流|商贸综合体|商场",  # 11: 商贸市场
    r"体育馆|体育中心|健身步道|足球场|运动公园",  # 12: 体育设施
    r"景区|旅游度假区|文旅综合体|文化街区|历史文化|公园",  # 13: 文旅设施
    r"数字化改造|数字化车间|智能制造|工业互联网|MES|PLC|物联网",  # 14: 数字化转型
]


class StrictFeaturePipeline:
    def __init__(self, classes_l2, classes_l1, l2_to_l1):
        self.classes_l2 = classes_l2
        self.classes_l1 = classes_l1
        self.l2_to_l1 = l2_to_l1
        self.n_cls_l2 = len(classes_l2)
        self.n_cls_l1 = len(classes_l1)
        self.kw_bank = {}
        self.proto_means = None
        self.scaler = None

    def fit(self, texts_train, emb_train, y_train_l2, df_train=None):
        """仅由给定的训练切片数据拟合所有统计特征"""
        # 1. 拟合关键词库 (仅从当期训练样本的二级领域及投向提取)
        self.kw_bank = {}
        for l2_c in self.classes_l2:
            terms = [l2_c]
            for sub in re.split(r"[、，,（）()]+", l2_c):
                if len(sub.strip()) >= 2:
                    terms.append(sub.strip())
            if df_train is not None and "所属投向领域" in df_train.columns:
                sub_df = df_train[df_train["二级领域"] == l2_c]
                for t_val in sub_df["所属投向领域"].dropna().unique():
                    for sub in re.split(r"[、，,（）()]+", str(t_val)):
                        if len(sub.strip()) >= 2 and sub.strip() not in STOP_WORDS:
                            terms.append(sub.strip())
            seen = set()
            self.kw_bank[l2_c] = [t for t in terms if not (t in seen or seen.add(t))]

        # 2. 拟合类原型几何重心 (仅使用当前训练切片的 BGE 向量)
        means = np.zeros((self.n_cls_l2, emb_train.shape[1]), dtype=np.float32)
        for c in np.unique(y_train_l2):
            mask = y_train_l2 == int(c)
            if mask.sum() > 0:
                means[int(c)] = emb_train[mask].mean(0)
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        self.proto_means = np.where(norms > 0, means / np.maximum(norms, 1e-12), 0.0)

        # 3. 拟合特征矩阵与 StandardScaler
        F_tr_raw = self._build_raw_features(texts_train, emb_train)
        self.scaler = StandardScaler().fit(F_tr_raw)
        return self

    def transform(self, texts, emb):
        """对任意验证集或测试集做确定性投影变换"""
        F_raw = self._build_raw_features(texts, emb)
        return self.scaler.transform(F_raw).astype(np.float32)

    def fit_transform(self, texts_train, emb_train, y_train_l2, df_train=None):
        self.fit(texts_train, emb_train, y_train_l2, df_train)
        return self.transform(texts_train, emb_train)

    def _build_raw_features(self, texts, emb):
        n_samples = len(texts)
        # A. 关键词特征 (20 维)
        kw_feats = np.zeros((n_samples, self.n_cls_l2), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, l2_c in enumerate(self.classes_l2):
                terms = self.kw_bank.get(l2_c, [l2_c])
                c = sum(1 for term in terms if term in t)
                if c:
                    kw_feats[i, j] = np.log1p(c)

        # B. 类原型几何相似度 (20 维)
        x_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        proto_feats = (x_norm @ self.proto_means.T).astype(np.float32)

        # C. 实体先验特征 (15 维)
        ent_feats = np.zeros((n_samples, len(ENTITY_PATTERNS)), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, pat in enumerate(ENTITY_PATTERNS):
                if re.search(pat, t):
                    ent_feats[i, j] = 1.0

        return np.hstack([emb, kw_feats, proto_feats, ent_feats]).astype(np.float32)
