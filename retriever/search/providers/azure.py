#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Azure OpenAI provider implementation.
"""

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from logger import get_provider_logger

# Get Azure logger
logger = get_provider_logger()
from typing import Dict, List, Optional

from constants import USER_AGENT
from models import CheckResult, Condition, ErrorReason
from utils import trim

from .openai_like import OpenAILikeProvider


class AzureOpenAIProvider(OpenAILikeProvider):
    """Azure OpenAI provider implementation."""

    def __init__(self, conditions: List[Condition], default_model: str = "", skip_search: bool = False):
        default_model = trim(default_model) or "gpt-4o"
        super().__init__(
            name="azure",
            base_url="",
            completion_path="/chat/completions",
            model_path="/models",
            default_model=default_model,
            conditions=conditions,
            skip_search=skip_search,
            address_pattern=r"https://[a-zA-Z0-9_\-\.]+.openai.azure.com/openai/",
        )

        self.api_version = "2024-10-21"

    def _get_headers(self, token: str, additional: Optional[Dict] = None) -> Optional[Dict]:
        """Get headers for Azure OpenAI API requests."""
        token = trim(token)
        if not token:
            return None

        return {
            "accept": "application/json",
            "api-key": token,
            "content-type": "application/json",
            "user-agent": USER_AGENT,
        }

    def _judge(self, code: int, message: str) -> CheckResult:
        """Judge Azure OpenAI API response."""
        if code == 404:
            message = trim(message)
            if re.finditer(r"The API deployment for this resource does not exist", message, flags=re.I):
                return CheckResult.fail(ErrorReason.NO_MODEL)

            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super()._judge(code, message)

    def __generate_address(self, address: str = "", endpoint: str = "", model: str = "") -> str:
        """Generate Azure OpenAI API address."""
        address = trim(address).removesuffix("/")
        if not re.match(r"^https?://([\w\-_]+\.[\w\-_]+)+", address, flags=re.I):
            return ""

        if re.findall(
            r"(xxx|YOUR_RESOURCE_NAME|your_service|YOUR_AZURE_OPENAI_NAME|YOUR-INSTANCE|YOUR_ENDPOINT_NAME|RESOURCE_NAME|YOURAOAIINSTANCE|yourname|YOUR_NAME|YOUR_AOAI_SERVICE|COMPANY|your-deployment-name|YOUR_AOI_SERVICE_NAME|YOUR_AI_ENDPOINT_NAME|YOUR-APP|YOUR-RESOURCE-NAME).openai.azure.com",
            address,
            flags=re.I,
        ):
            return ""

        model = trim(model) or self.default_model
        return f"{address}/deployments/{model}/{self.completion_path}?api-version={self.api_version}"

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        """Check Azure OpenAI token validity."""
        token = trim(token)
        if not token:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        url = self.__generate_address(address=address, endpoint=endpoint, model=model)
        if not url:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super().check(token=token, address=url, endpoint=endpoint, model=model)

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available Azure OpenAI models."""
        domain = trim(address).removesuffix("/")
        if not re.match(r"^https?://([\w\-_]+\.[\w\-_]+)+", domain, flags=re.I):
            logger.error(f"Invalid domain: {domain}, skipping model listing")
            return []

        headers = self._get_headers(token=token)
        if not headers or not self.model_path:
            return []

        url = f"{domain}/{self.model_path}?api-version={self.api_version}"
        return self._fetch_models(url=url, headers=headers)
