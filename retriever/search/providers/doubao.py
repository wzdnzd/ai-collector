#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Doubao provider implementation.
"""

from typing import List

from models import CheckResult, Condition, ErrorReason

from utils import trim

from .openai_like import OpenAILikeProvider


class DoubaoProvider(OpenAILikeProvider):
    """Doubao provider implementation."""

    def __init__(self, conditions: List[Condition], default_model: str = "", skip_search: bool = False):
        default_model = trim(default_model) or "doubao-pro-32k"
        base_url = "https://ark.cn-beijing.volces.com"

        super().__init__(
            name="doubao",
            base_url=base_url,
            default_model=default_model,
            conditions=conditions,
            completion_path="/api/v3/chat/completions",
            model_path="/api/v3/models",
            model_pattern=r"ep-[0-9]{14}-[a-z0-9]{5}",
            skip_search=skip_search,
        )

    def _judge(self, code: int, message: str) -> CheckResult:
        """Judge Doubao API response."""
        if code == 404:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super()._judge(code, message)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        """Check Doubao token validity."""
        model = trim(model)
        if not model:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super().check(token=token, address=address, endpoint=endpoint, model=model)
