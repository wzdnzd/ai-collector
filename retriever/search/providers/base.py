#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base provider class for AI service providers.
"""

import os
import re
import sys
import urllib.parse
from typing import Dict, List, Optional, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from logger import get_provider_logger
from models import CheckResult, Condition, ErrorReason
from utils import trim

from ..client import chat

# Get provider logger
logger = get_provider_logger()


class Provider(object):
    """Base class for AI service providers."""

    def __init__(
        self,
        name: str,
        base_url: str,
        completion_path: str,
        model_path: str,
        default_model: str,
        conditions: Union[Condition, List[Condition]],
        skip_search: bool = False,
        **kwargs,
    ):
        name = str(name)
        if not name:
            raise ValueError("provider name cannot be empty")

        default_model = trim(default_model)
        if not default_model:
            raise ValueError("default_model cannot be empty")

        base_url = trim(base_url)

        # see: https://stackoverflow.com/questions/10893374/python-confusions-with-urljoin
        if base_url and not base_url.endswith("/"):
            base_url += "/"

        # provider name
        self.name = name

        # directory
        self.directory = re.sub(r"[^a-zA-Z0-9_\-]", "-", name, flags=re.I).lower()

        # filename for valid keys
        self.keys_filename = "valid-keys.txt"

        # filename for no quota keys
        self.no_quota_filename = "no-quota-keys.txt"

        # filename for need check again keys
        self.wait_check_filename = "wait-check-keys.txt"

        # filename for invalid keys
        self.invalid_keys_filename = "invalid-keys.txt"

        # filename for extract keys
        self.material_filename = f"material.txt"

        # filename for summary
        self.summary_filename = f"summary.json"

        # filename for links included keys
        self.links_filename = f"links.txt"

        # base url for llm service api
        self.base_url = base_url

        # path for completion api
        self.completion_path = trim(completion_path).removeprefix("/")

        # path for model list api
        self.model_path = trim(model_path).removeprefix("/")

        # default model for completion api used to verify token
        self.default_model = default_model

        conditions = (
            [conditions]
            if isinstance(conditions, Condition)
            else ([] if not isinstance(conditions, list) else conditions)
        )

        items = set()
        for condition in conditions:
            if not isinstance(condition, Condition) or not condition.regex:
                logger.warning(f"Invalid condition: {condition}, skipping it")
                continue

            items.add(condition)

        # search and extract keys conditions
        self.conditions = list(items)

        # skip search and collect stages
        self.skip_search = skip_search

        # additional parameters for provider
        self.extras = kwargs

    def _get_headers(self, token: str, additional: Optional[Dict] = None) -> Optional[Dict]:
        """Get headers for API requests. Must be implemented by subclasses."""
        raise NotImplementedError

    def _judge(self, code: int, message: str) -> CheckResult:
        """Judge API response and return check result."""
        message = trim(message)

        if code == 200 and message:
            return CheckResult.ok()
        elif code == 400:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)
        elif code == 401 or re.findall(r"invalid_api_key", message, flags=re.I):
            return CheckResult.fail(ErrorReason.INVALID_KEY)
        elif code == 402 or re.findall(r"insufficient", message, flags=re.I):
            return CheckResult.fail(ErrorReason.NO_QUOTA)
        elif code == 403 or code == 404:
            return CheckResult.fail(ErrorReason.NO_ACCESS)
        elif code == 418 or code == 429:
            return CheckResult.fail(ErrorReason.RATE_LIMITED)
        elif code >= 500:
            return CheckResult.fail(ErrorReason.SERVER_ERROR)

        return CheckResult.fail(ErrorReason.UNKNOWN)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        """Check if token is valid."""
        url, regex = trim(address), r"^https?://([\w\-_]+\.[\w\-_]+)+"
        if not url and re.match(regex, self.base_url, flags=re.I):
            url = urllib.parse.urljoin(self.base_url, self.completion_path)

        if not re.match(regex, url, flags=re.I):
            logger.error(f"Invalid URL: {url}, skipping check")
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        headers = self._get_headers(token=token)
        if not headers:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        model = trim(model) or self.default_model
        code, message = chat(url=url, headers=headers, model=model)
        return self._judge(code=code, message=message)

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available models. Must be implemented by subclasses."""
        raise NotImplementedError
