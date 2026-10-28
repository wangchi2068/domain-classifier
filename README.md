# 投资领域智能分类微服务 (FastAPI 双标准 DNN 引擎)

本项目基于 **Hierarchical Multi-Task DNN 深度神经网络** 架构开发，统一集成了两套核心投资领域的智能分类标准：
1. **浙江省“千项万亿”重大投资项目【十大领域分类标准】**（10 个一级领域，45 个细分二级领域）；
2. **中央预算内/地方政府专项债【新领域案例分类标准】**（4 个一级领域，20 个细分二级领域）。

输入【项目名称】与【建设规模和内容】，模型秒级输出最精准的二级领域并自动锁定对应的一级领域。工程内**仅包含核心代码、FastAPI 接口与必要模型权重**，完全脱离离线训练集与测试代码，结构干净纯粹。

---

## 1. 目录架构说明

```text
D:/python/domain_classifier/
├── main.py                     # 主启动入口文件 (支持 --profile dev/prod, --port 等参数)
├── requirements.txt            # 项目 Python 核心依赖清单
├── README.md                   # 接口文档与调用说明
│
├── apps/                       # 核心业务微服务架构
│   ├── __init__.py             # 全局单例装饰器与环境常量
│   │
│   ├── config/                 # 配置中心
│   │   ├── config.py           # 配置读取与初始化
│   │   ├── application-dev.yml # 开发环境配置 (端口、模型目录、文档开关)
│   │   └── application-prod.yml# 生产环境配置
│   │
│   ├── models/                 # 数据模型层 (Pydantic Models)
│   │   ├── CommonModels.py     # ApiResult 统一响应结构 (code, msg, data)
│   │   └── DomainModels.py     # 请求与响应实体模型
│   │
│   ├── controllers/            # 路由控制层 (APIRouter)
│   │   ├── __init__.py         # 统一注册并导出路由
│   │   ├── ten_domain_controller.py # 重大项目十大领域预测与字典接口 (/domain/ten/...)
│   │   ├── new_domain_controller.py # 专项债新领域案例预测与字典接口 (/domain/new/...)
│   │   └── health_controller.py# 双模型健康检查接口 (/health)
│   │
│   ├── services/               # 推理服务层 (DNN 614维/567维多任务网络)
│   │   ├── ten_domain_service.py # 十大领域推理服务 (614维复合特征 + 3种子集成)
│   │   ├── new_domain_service.py # 新领域案例推理服务 (567维复合特征 + 3种子集成)
│   │   └── strict_pipeline.py  # 严格特征工程管道 (词频、类原型余弦、标准化)
│   │
│   └── core/                   # 核心底层组件
│       └── fastapi/
│           └── config.py       # FastAPI 实例创建、lifespan 双模型预热加载、CORS
│
└── model_dir/                  # 生产模型权重与必要资产目录 (严格隔离)
    ├── ten_domain/             # 十大领域专用模型资产
    │   ├── classes.json        # 45 个二级领域与 10 个一级领域映射字典
    │   ├── ensemble_weights.pth# 3-Seed 集成 DNN 神经网络权重
    │   ├── model_weights.pth   # 单模型权重
    │   ├── proto_means.npy     # 45 类别原型几何重心矩阵
    │   └── scaler.pkl          # 614 维特征标准化器
    │
    └── new_domain/             # 新领域案例专用模型资产
        ├── classes.json        # 20 个二级领域与 4 个一级领域映射字典
        ├── dnn_ensemble_weights.pth # 3-Seed 集成 DNN 神经网络权重
        ├── dnn_model.pth       # 单模型权重
        ├── pipeline.pkl        # 特征工程变换管道
        ├── proto_means.npy     # 20 类别原型几何重心矩阵
        └── scaler.pkl          # 567 维特征标准化器
```

---

## 2. 服务启动与访问

### 2.1 启动方式
```bash
cd /d D:\python\domain_classifier
python main.py --profile dev --port 8000
```
启动后后台会自动完成两个 DNN 模型与 BGE 向量模型的预热加载。

### 2.2 在线交互式文档
服务启动后，在浏览器直接打开 Swagger UI：  
**`http://127.0.0.1:8000/docs`**

---

## 3. 核心 API 接口与规范

所有接口均返回标准统一包装格式：
```json
{
  "code": 0,
  "msg": "success",
  "data": { ... }
}
```

### 3.1 浙江省重大投资项目【十大领域分类】接口

#### 端点 A1：单项目即时预测
* **URL**: `POST http://127.0.0.1:8000/domain/ten/predict`
* **Content-Type**: `application/json`
* **请求体 (Request Body)**:
  ```json
  {
    "name": "中科富海气体能源高端装备制造基地项目",
    "content": "主要建设深低温制冷装备制造厂房及辅助设施，年产低温气体装备300套。"
  }
  ```
* **响应结果 (Response)**:
  ```json
  {
    "code": 0,
    "msg": "success",
    "data": {
      "项目名称": "中科富海气体能源高端装备制造基地项目",
      "一级领域": "先进制造业基地领域",
      "二级领域": "高端装备",
      "置信度": 0.6683,
      "Top3候选": [
        { "二级领域": "高端装备", "一级领域": "先进制造业基地领域", "概率": 0.6683 },
        { "二级领域": "天然气管线", "一级领域": "清洁能源保供领域", "概率": 0.2632 },
        { "二级领域": "新一代信息技术", "一级领域": "先进制造业基地领域", "概率": 0.0537 }
      ]
    }
  }
  ```

#### 端点 A2：批量项目并发预测
* **URL**: `POST http://127.0.0.1:8000/domain/ten/predict/batch`
* **请求体**: `{"projects": [{"name": "项目A", "content": "..."}, {"name": "项目B", "content": "..."}]}`

#### 端点 A3：获取十大领域类别字典
* **URL**: `GET http://127.0.0.1:8000/domain/ten/categories`
* **响应**: 返回 10 个一级领域与 45 个细分二级领域的父子树结构。

---

### 3.2 专项债/预算内【新领域案例分类】接口

#### 端点 B1：单项目即时预测
* **URL**: `POST http://127.0.0.1:8000/domain/new/predict` (或通用路径 `POST /domain/predict`)
* **Content-Type**: `application/json`
* **请求体 (Request Body)**:
  ```json
  {
    "name": "某市主城区雨污水管网提质增效与综合管廊工程",
    "content": "对主城区老旧供排水管网进行全面改造，新建地下综合管廊3.2公里，配套雨污分流管道。"
  }
  ```
* **响应结果 (Response)**:
  ```json
  {
    "code": 0,
    "msg": "success",
    "data": {
      "项目名称": "某市主城区雨污水管网提质增效与综合管廊工程",
      "一级领域": "其他",
      "二级领域": "城市地下管网管廊建设改造、其他城市更新",
      "置信度": 0.9999,
      "Top3候选": [
        { "二级领域": "城市地下管网管廊建设改造、其他城市更新", "一级领域": "其他", "概率": 0.9999 },
        { "二级领域": "清洁能源基础设施", "一级领域": "其他", "概率": 0.0001 },
        { "二级领域": "水利基础设施", "一级领域": "其他", "概率": 0.0 }
      ]
    }
  }
  ```

#### 端点 B2：批量项目并发预测
* **URL**: `POST http://127.0.0.1:8000/domain/new/predict/batch` (或通用路径 `POST /domain/predict/batch`)
* **请求体**: `{"projects": [{"name": "项目A", "content": "..."}, {"name": "项目B", "content": "..."}]}`

#### 端点 B3：获取新领域案例类别字典
* **URL**: `GET http://127.0.0.1:8000/domain/new/categories`
* **响应**: 返回 4 个一级领域与 20 个二级领域的映射字典。

---

### 3.3 服务健康检查
* **URL**: `GET http://127.0.0.1:8000/health`
* **响应结果**:
  ```json
  {
    "code": 0,
    "msg": "success",
    "data": {
      "status": "healthy",
      "models": {
        "重大项目十大领域DNN": {
          "loaded": true,
          "一级领域数量": 10,
          "二级领域数量": 45
        },
        "专项债新领域案例DNN": {
          "loaded": true,
          "一级领域数量": 4,
          "二级领域数量": 20
        }
      }
    }
  }
  ```
