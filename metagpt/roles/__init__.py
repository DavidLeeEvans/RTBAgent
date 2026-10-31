#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : __init__.py
"""

from metagpt.roles.role import Role

__all__ = ["Role"]

for _mod_name, _cls_name in [
    ("metagpt.roles.architect", "Architect"),
    ("metagpt.roles.project_manager", "ProjectManager"),
    ("metagpt.roles.product_manager", "ProductManager"),
    ("metagpt.roles.engineer", "Engineer"),
    ("metagpt.roles.qa_engineer", "QaEngineer"),
    ("metagpt.roles.searcher", "Searcher"),
    ("metagpt.roles.sales", "Sales"),
    ("metagpt.roles.customer_service", "CustomerService"),
]:
    try:
        _mod = __import__(_mod_name, fromlist=[_cls_name])
        globals()[_cls_name] = getattr(_mod, _cls_name)
        __all__.append(_cls_name)
    except ImportError:
        pass
