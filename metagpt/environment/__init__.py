#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   :

from metagpt.environment.base_env import Environment

__all__ = ["Environment"]

for _mod_name, _cls_name in [
    ("metagpt.environment.android_env.android_env", "AndroidEnv"),
    ("metagpt.environment.mincraft_env.mincraft_env", "MincraftExtEnv"),
    ("metagpt.environment.werewolf_env.werewolf_env", "WerewolfEnv"),
    ("metagpt.environment.stanford_town_env.stanford_town_env", "StanfordTownEnv"),
    ("metagpt.environment.software_env.software_env", "SoftwareEnv"),
]:
    try:
        _mod = __import__(_mod_name, fromlist=[_cls_name])
        globals()[_cls_name] = getattr(_mod, _cls_name)
        __all__.append(_cls_name)
    except ImportError:
        pass
