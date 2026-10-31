#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:44
@Author  : alexanderwu
@File    : __init__.py
"""
from enum import Enum

from metagpt.actions.action import Action
from metagpt.actions.action_output import ActionOutput
from metagpt.actions.add_requirement import UserRequirement

_OPTIONAL_ACTIONS = {}
for _enum_key, _mod_name, _cls_name in [
    ("WRITE_PRD", "metagpt.actions.write_prd", "WritePRD"),
    ("WRITE_PRD_REVIEW", "metagpt.actions.write_prd_review", "WritePRDReview"),
    ("WRITE_DESIGN", "metagpt.actions.design_api", "WriteDesign"),
    ("DESIGN_REVIEW", "metagpt.actions.design_api_review", "DesignReview"),
    ("WRTIE_CODE", "metagpt.actions.write_code", "WriteCode"),
    ("WRITE_CODE_REVIEW", "metagpt.actions.write_code_review", "WriteCodeReview"),
    ("WRITE_TEST", "metagpt.actions.write_test", "WriteTest"),
    ("RUN_CODE", "metagpt.actions.run_code", "RunCode"),
    ("DEBUG_ERROR", "metagpt.actions.debug_error", "DebugError"),
    ("WRITE_TASKS", "metagpt.actions.project_management", "WriteTasks"),
    ("SEARCH_AND_SUMMARIZE", "metagpt.actions.search_and_summarize", "SearchAndSummarize"),
    ("COLLECT_LINKS", "metagpt.actions.research", "CollectLinks"),
    ("WEB_BROWSE_AND_SUMMARIZE", "metagpt.actions.research", "WebBrowseAndSummarize"),
    ("CONDUCT_RESEARCH", "metagpt.actions.research", "ConductResearch"),
    ("EXECUTE_NB_CODE", "metagpt.actions.di.execute_nb_code", "ExecuteNbCode"),
    ("WRITE_ANALYSIS_CODE", "metagpt.actions.di.write_analysis_code", "WriteAnalysisCode"),
    ("WRITE_PLAN", "metagpt.actions.di.write_plan", "WritePlan"),
]:
    try:
        _mod = __import__(_mod_name, fromlist=[_cls_name])
        _cls = getattr(_mod, _cls_name)
        globals()[_cls_name] = _cls
        _OPTIONAL_ACTIONS[_enum_key] = _cls
    except ImportError:
        pass

ActionType = Enum("ActionType", {"ADD_REQUIREMENT": UserRequirement, **_OPTIONAL_ACTIONS})


__all__ = [
    "ActionType",
    "Action",
    "ActionOutput",
]
