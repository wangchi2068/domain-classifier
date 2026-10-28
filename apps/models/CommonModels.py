from typing import Generic, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar("T")


class ApiResult(BaseModel, Generic[T]):
    code: int = 0
    msg: str = "success"
    data: Optional[T] = None

    @classmethod
    def success(cls, data: T = None):
        return cls(code=0, msg="success", data=data)

    @classmethod
    def fail(cls, code: int = 500, msg: str = "fail", data: T = None):
        return cls(code=code, msg=msg, data=data)
