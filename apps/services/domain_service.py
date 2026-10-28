# 兼容性包装：默认 domain_service 指向 new_domain_service
from apps.services.new_domain_service import (
    NewDomainService as DomainClassifierService,
    get_new_domain_service as get_domain_service,
)

__all__ = ["DomainClassifierService", "get_domain_service"]
