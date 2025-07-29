#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QianFan provider implementation.
"""

import urllib.parse
from typing import List

from models import CheckResult, Condition, ErrorReason
from utils import trim

from ..client import chat
from .openai_like import OpenAILikeProvider


class QianFanProvider(OpenAILikeProvider):
    """QianFan provider implementation."""

    def __init__(self, conditions: List[Condition], default_model: str = "", skip_search: bool = False):
        default_model = trim(default_model) or "ernie-4.0-8k-latest"
        base_url = "https://qianfan.baidubce.com"

        super().__init__(
            name="qianfan",
            base_url=base_url,
            default_model=default_model,
            conditions=conditions,
            completion_path="/v2/chat/completions",
            model_path="/v2/models",
            endpoint_pattern=r"[a-z0-9]{8}(?:-[a-z0-9]{4}){3}-[a-z0-9]{12}",
            skip_search=skip_search,
        )

    def _judge(self, code: int, message: str) -> CheckResult:
        """Judge QianFan API response."""
        if code == 404:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super()._judge(code, message)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        """Check QianFan token validity."""
        headers = self._get_headers(token=token)
        if not headers:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        endpoint = trim(endpoint)
        if endpoint:
            headers["appid"] = endpoint

        model = trim(model) or self.default_model
        url = urllib.parse.urljoin(self.base_url, self.completion_path)

        code, message = chat(url=url, headers=headers, model=model)
        return self._judge(code=code, message=message)

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available QianFan models."""
        headers = self._get_headers(token=token)
        if not headers:
            return []

        endpoint = trim(endpoint)
        if endpoint:
            headers["appid"] = endpoint

        url = urllib.parse.urljoin(self.base_url, self.model_path)
        return self._fetch_models(url=url, headers=headers)
