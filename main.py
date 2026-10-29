import argparse
import os
import sys
import uvicorn

# 强制 UTF-8 控制台
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 默认启用国内高速镜像源下载开源模型，防止网络超时
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from apps import BS_PROFILE_NAME
from apps.config.config import init_config, get_app_config


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="新投资领域分类系统微服务")
    parser.add_argument("--start", action="store_true", default=True, help="启动应用服务")
    parser.add_argument("--profile", type=str, default="dev", help="指定运行环境配置 profile (dev/prod)")
    parser.add_argument("--host", type=str, default=None, help="覆盖监听主机地址 (默认读取配置)")
    parser.add_argument("--port", type=int, default=None, help="覆盖监听端口号 (默认读取配置)")

    args = parser.parse_args()

    profile = args.profile
    init_config(profile)

    server_config = get_app_config().get("server", {})
    host = args.host or server_config.get("host", "0.0.0.0")
    port = args.port or server_config.get("port", 8000)

    # 设置环境变量，供 FastAPI 应用初始化读取
    os.environ[BS_PROFILE_NAME] = profile

    print("=" * 60)
    print(f"[INFO] 正在启动投资领域智能分类微服务 (Profile: {profile}) ...")
    print(f"[INFO] 监听地址: http://{host}:{port}")
    print(f"[INFO] Swagger 文档: http://127.0.0.1:{port}/docs")
    print(f"[INFO] API 根路径: http://127.0.0.1:{port}/domain-api")
    print("=" * 60)

    uvicorn.run("apps.core.fastapi.config:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
