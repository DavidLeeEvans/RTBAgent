#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 22:59
@Author  : alexanderwu
@File    : __init__.py
"""

from metagpt.provider.human_provider import HumanProvider
from metagpt.provider.openai_api import OpenAILLM
from metagpt.provider.ollama_api import OllamaLLM
from metagpt.provider.azure_openai_api import AzureOpenAILLM
from metagpt.provider.metagpt_api import MetaGPTLLM

__all__ = [
    "OpenAILLM",
    "AzureOpenAILLM",
    "MetaGPTLLM",
    "OllamaLLM",
    "HumanProvider",
]

for _mod_name, _cls_name in [
    ("metagpt.provider.google_gemini_api", "GeminiLLM"),
    ("metagpt.provider.zhipuai_api", "ZhiPuAILLM"),
    ("metagpt.provider.spark_api", "SparkLLM"),
    ("metagpt.provider.qianfan_api", "QianFanLLM"),
    ("metagpt.provider.dashscope_api", "DashScopeLLM"),
    ("metagpt.provider.anthropic_api", "AnthropicLLM"),
]:
    try:
        _mod = __import__(_mod_name, fromlist=[_cls_name])
        globals()[_cls_name] = getattr(_mod, _cls_name)
        __all__.append(_cls_name)
    except ImportError:
        pass
