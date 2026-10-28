import functools
import os
import threading

BS_PROFILE_NAME = "BS_PROFILE"
BS_PROFILE = os.getenv(BS_PROFILE_NAME, "dev")


def singleton(cls):
    """线程安全的单例模式装饰器"""
    instances = {}
    _lock = threading.Lock()

    @functools.wraps(cls)
    def _singleton_wrapper(*cls_args, **cls_kwargs):
        key = str(cls) + str(os.getpid())
        if key not in instances:
            with _lock:
                if key not in instances:
                    instances[key] = cls(*cls_args, **cls_kwargs)
        return instances[key]

    return _singleton_wrapper


__all__ = [BS_PROFILE_NAME, BS_PROFILE, singleton]
