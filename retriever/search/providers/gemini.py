#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Google Gemini provider implementation.
"""

import json
import os
import re
import sys
import urllib.parse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from logger import get_provider_logger

# Get Gemini logger
logger = get_provider_logger()
from typing import Dict, List, Optional

from constants import DEFAULT_QUESTION
from models import CheckResult, Condition, ErrorReason
from utils import trim

from ..client import chat, http_get
from .base import Provider


class GeminiProvider(Provider):
    """Google Gemini provider implementation."""

    def __init__(self, conditions: List[Condition], default_model: str = "", skip_search: bool = False):
        default_model = trim(default_model) or "gemini-2.5-pro"
        base_url = "https://generativelanguage.googleapis.com"
        sub_path = "/v1beta/models"

        super().__init__("gemini", base_url, sub_path, sub_path, default_model, conditions, skip_search=skip_search)

    def _get_headers(self, token: str, additional: Optional[Dict] = None) -> Optional[Dict]:
        """Get headers for Gemini API requests."""
        return {"accept": "application/json", "content-type": "application/json"}

    def _judge(self, code: int, message: str) -> CheckResult:
        """Judge Gemini API response."""
        if code == 200:
            return CheckResult.ok()

        message = trim(message)
        if code == 400:
            if re.findall(r"API_KEY_INVALID", message, flags=re.I):
                return CheckResult.fail(ErrorReason.INVALID_KEY)
            elif re.findall(r"FAILED_PRECONDITION", message, flags=re.I):
                return CheckResult.fail(ErrorReason.NO_ACCESS)

        return super()._judge(code, message)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        """Check Gemini token validity."""
        token = trim(token)
        if not token:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        model = trim(model) or self.default_model
        url = f"{urllib.parse.urljoin(self.base_url, self.completion_path)}/{model}:generateContent?key={token}"

        params = {"contents": [{"role": "user", "parts": [{"text": DEFAULT_QUESTION}]}]}
        code, message = chat(url=url, headers=self._get_headers(token=token), params=params)
        return self._judge(code=code, message=message)

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available Gemini models."""
        token = trim(token)
        if not token:
            return []

        url = urllib.parse.urljoin(self.base_url, self.model_path) + f"?key={token}"
        content = http_get(url=url, headers=self._get_headers(token=token), interval=1)
        if not content:
            return []

        try:
            data = json.loads(content)
            models = data.get("models", [])
            return [x.get("name", "").removeprefix("models/") for x in models]
        except:
            logger.error(f"Failed to parse models from response: {content}")
            return []
