#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StabilityAI provider implementation.
"""

import codecs
import os
import sys
import urllib.parse
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from logger import get_provider_logger

# Get StabilityAI logger
logger = get_provider_logger()
import uuid
from typing import Dict, List, Optional, Tuple

from constants import CTX, NO_RETRY_ERROR_CODES, USER_AGENT
from models import CheckResult, Condition, ErrorReason

from utils import trim

from .base import Provider


class StabilityAIProvider(Provider):
    """StabilityAI provider implementation."""

    def __init__(self, conditions: List[Condition], default_model: str = "", skip_search: bool = False):
        default_model = trim(default_model) or "core"
        base_url = "https://api.stability.ai"
        sub_path = "/v2beta/stable-image/generate"

        super().__init__("stabilityai", base_url, sub_path, "", default_model, conditions, skip_search=skip_search)

    def _get_headers(self, token: str, additional: Optional[Dict] = None) -> Optional[Dict]:
        """Get headers for StabilityAI API requests."""
        key = trim(token)
        if not key:
            return None

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "multipart/form-data",
            "Accept": "application/json",
        }
        if additional and isinstance(additional, dict):
            headers.update(additional)

        return headers

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        """Check StabilityAI token validity."""

        def post_multipart(
            url: str, token: str, fields: Optional[Dict] = None, files: Optional[Dict] = None, retries: int = 3
        ) -> Tuple[int, str]:
            url, token = trim(url), trim(token)
            if not url or not token:
                return 401, ""

            boundary, contents = str(uuid.uuid4()), []
            if not isinstance(fields, dict):
                fields = dict()
            if not isinstance(files, dict):
                files = dict()

            # add common form fields
            for k, v in fields.items():
                contents.append(f"--{boundary}")
                contents.append(f'Content-Disposition: form-data; name="{k}"')
                contents.append("Content-Type: text/plain")
                contents.append("")
                contents.append(v)
                contents.append("")

            # add files
            for k, v in files.items():
                filename, data = v
                contents.append(f"--{boundary}")
                contents.append(f'Content-Disposition: form-data; name="{k}"; filename="{filename}"')
                contents.append("Content-Type: application/octet-stream")
                contents.append("")
                contents.append(data)
                contents.append("")

            # add end flag
            contents.append(f"--{boundary}--")
            contents.append("")

            # encode content
            payload = b"\r\n".join(codecs.encode(x, encoding="utf8") for x in contents)

            req = urllib.request.Request(url, data=payload, method="POST")

            # set request headers
            req.add_header("Accept", "application/json")
            req.add_header("Authorization", f"Bearer {token}")
            req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
            req.add_header("User-Agent", USER_AGENT)

            # send request with retry
            code, message, attempt, retries = 401, "", 0, max(1, retries)
            while attempt < retries:
                try:
                    with urllib.request.urlopen(req, timeout=15, context=CTX) as response:
                        code = 200
                        message = response.read().decode("utf8")
                        break
                except urllib.error.HTTPError as e:
                    code = e.code
                    if code != 401:
                        try:
                            message = e.read().decode("utf8")
                            if not message.startswith("{") or not message.endswith("}"):
                                message = e.reason
                        except:
                            message = e.reason

                        logger.error(
                            f"[Chat] Failed to request URL: {url}, token: {token}, status code: {code}, message: {message}"
                        )

                    if code in NO_RETRY_ERROR_CODES:
                        break
                except Exception:
                    pass

                attempt += 1
                import time

                time.sleep(1)

            return code, message

        token = trim(token)
        if not token:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        model = trim(model) or self.default_model
        url = f"{urllib.parse.urljoin(self.base_url, self.completion_path)}/{model}"
        fields = {"prompt": "Lighthouse on a cliff overlooking the ocean", "aspect_ratio": "3:2"}

        code, message = post_multipart(url=url, token=token, fields=fields)
        return self._judge(code=code, message=message)

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available StabilityAI models."""
        return []
