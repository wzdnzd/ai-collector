#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenAI-like provider base class.
"""

import json
import os
import re
import sys
import urllib.parse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from logger import get_provider_logger

# Get OpenAI-like logger
logger = get_provider_logger()
from typing import Dict, List, Optional

from constants import DEFAULT_COMPLETION_PATH, DEFAULT_MODEL_PATH, USER_AGENT
from models import CheckResult, Condition, ErrorReason

from utils import trim

from ..client import http_get
from .base import Provider


class OpenAILikeProvider(Provider):
    """Base class for OpenAI-compatible providers."""

    def __init__(
        self,
        name: str,
        base_url: str,
        default_model: str,
        conditions: List[Condition],
        completion_path: str = "",
        model_path: str = "",
        skip_search: bool = False,
        **kwargs,
    ):
        completion_path = trim(completion_path) or DEFAULT_COMPLETION_PATH
        model_path = trim(model_path) or DEFAULT_MODEL_PATH

        super().__init__(
            name, base_url, completion_path, model_path, default_model, conditions, skip_search=skip_search, **kwargs
        )

    def _get_headers(self, token: str, additional: Optional[Dict] = None) -> Optional[Dict]:
        """Get headers for OpenAI-like API requests."""
        token = trim(token)
        if not token:
            return None

        if not isinstance(additional, dict):
            additional = {}

        auth_key = (trim(self.extras.get("auth_key", None)) if isinstance(self.extras, dict) else "") or "authorization"
        auth_value = f"Bearer {token}" if auth_key.lower() == "authorization" else token

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            auth_key: auth_value,
            "user-agent": USER_AGENT,
        }
        headers.update(additional)

        return headers

    def _judge(self, code: int, message: str) -> CheckResult:
        """Judge OpenAI-like API response."""
        if code == 200:
            try:
                data = json.loads(trim(message))
                if data and isinstance(data, dict):
                    error = data.get("error", None)
                    if error and isinstance(error, dict):
                        error_type = trim(error.get("type", ""))
                        error_reason = trim(error.get("message", "")).lower()

                        if error_type or "authorization" in error_reason:
                            return CheckResult.fail(ErrorReason.INVALID_KEY)
            except:
                logger.error(f"Failed to parse response, domain: {self.base_url}, message: {message}")
                return CheckResult.fail(ErrorReason.UNKNOWN)

            return CheckResult.ok()

        message = trim(message)
        if message:
            if code == 403:
                if re.findall(r"model_not_found", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.NO_MODEL)
                elif re.findall(r"unauthorized|已被封禁", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.INVALID_KEY)
                elif re.findall(r"unsupported_country_region_territory|该令牌无权访问模型", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.NO_ACCESS)
                elif re.findall(
                    r"exceeded_current_quota_error|insufficient_user_quota|(额度|余额)(不足|过低)", message, flags=re.I
                ):
                    return CheckResult.fail(ErrorReason.NO_QUOTA)
            elif code == 429:
                if re.findall(r"insufficient_quota|billing_not_active|欠费|请充值|recharge", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.NO_QUOTA)
                elif re.findall(r"rate_limit_exceeded", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.RATE_LIMITED)
            elif code == 503 and re.findall(r"无可用渠道", message, flags=re.I):
                return CheckResult.fail(ErrorReason.NO_MODEL)

        return super()._judge(code, message)

    def _fetch_models(self, url: str, headers: Dict) -> List[str]:
        """Fetch models from API endpoint."""
        url = trim(url)
        if not url:
            return []

        content = http_get(url=url, headers=headers, interval=1)
        if not content:
            return []

        try:
            result = json.loads(content)
            return [trim(x.get("id", "")) for x in result.get("data", [])]
        except:
            return []

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available models from OpenAI-like API."""
        headers = self._get_headers(token=token)
        if not headers or not self.base_url or not self.model_path:
            return []

        url = urllib.parse.urljoin(self.base_url, self.model_path)
        return self._fetch_models(url=url, headers=headers)
