# -*- coding: utf-8 -*-

# @Author  : wzdnzd
# @Time    : 2024-12-09

import argparse
import codecs
import gzip
import itertools
import json
import logging
import math
import os
import queue
import random
import re
import socket
import ssl
import threading
import time
import traceback
import typing
import urllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from concurrent import futures
from dataclasses import dataclass, field
from enum import Enum, unique
from functools import partial
from threading import Lock
from typing import Callable


@unique
class TimeInterval(Enum):
    """
    Time interval enumeration for search refinement.
    Each interval has a type name and corresponding days.
    """

    DAILY = ("daily", 1)
    WEEKLY = ("weekly", 7)
    MONTHLY = ("monthly", 30)
    QUARTERLY = ("quarterly", 90)
    YEARLY = ("yearly", 365)

    def __init__(self, category: str, interval: int):
        self.category = category
        self.interval = interval

    def __str__(self) -> str:
        return f"{self.category} ({self.interval} days)"

    @classmethod
    def from_days(cls, days: int) -> "TimeInterval":
        """Get the most appropriate TimeInterval for given days."""
        if days <= 1:
            return cls.DAILY
        elif days <= 7:
            return cls.WEEKLY
        elif days <= 30:
            return cls.MONTHLY
        elif days <= 90:
            return cls.QUARTERLY
        else:
            return cls.YEARLY


CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
}


DEFAULT_QUESTION = "Hello"


DEFAULT_COMPLETION_PATH = "/v1/chat/completions"


DEFAULT_MODEL_PATH = "/v1/models"

# error http status code that do not need to retry
NO_RETRY_ERROR_CODES = {400, 401, 402, 404, 422}


FILE_LOCK = Lock()


PATH = os.path.abspath(os.path.dirname(__file__))


# Language popularity tiers for LLM/API key domain with time step multipliers
LANGUAGE_TIERS = {
    # Tier 1: Extremely popular in LLM/API domain (multiplier: 1 - finest granularity)
    "tier1": {
        "languages": ["python", "javascript", "typescript"],
        "multiplier": 1,
        "description": "Dominant in LLM/API development",
    },
    # Tier 2: Very popular (multiplier: 4 - finer granularity)
    "tier2": {
        "languages": ["json", "yaml", "markdown", "text", "ini", "java", "go"],
        "multiplier": 4,
        "description": "Very popular in enterprise/web development",
    },
    # Tier 3: Very popular (multiplier: 8 - finer granularity)
    "tier3": {
        "languages": ["csharp", "rust", "shell", "c", "cpp"],
        "multiplier": 8,
        "description": "Common in enterprise/web development",
    },
    # Tier 4: Moderately popular (multiplier: 12 - finer granularity)
    "tier4": {
        "languages": [
            "scala",
            "dart",
            "r",
            "php",
            "lua",
            "perl",
            "sql",
            "xml",
            "toml",
            "dockerfile",
            "bash",
            "ruby",
            "kotlin",
            "swift",
            "html",
            "bat",
            "powershell",
        ],
        "multiplier": 12,
        "description": "Specialized domains",
    },
    # Tier 5: Less common (multiplier: 16 - much coarser granularity)
    "tier5": {
        "languages": [
            "coffeescript",
            "handlebars",
            "clojure",
            "fsharp",
            "scheme",
            "mysql",
            "pgsql",
            "graphql",
            "redis",
            "apex",
            "pascal",
            "tcl",
            "vb",
        ],
        "multiplier": 16,
        "description": "Niche/specialized languages",
    },
}

# Create language to tier mapping for quick lookup
LANGUAGE_TO_TIER = {}
for name, data in LANGUAGE_TIERS.items():
    for lang in data["languages"]:
        LANGUAGE_TO_TIER[lang] = name

# Flatten all languages for backward compatibility
POPULAR_LANGUAGES = []
for data in LANGUAGE_TIERS.values():
    POPULAR_LANGUAGES.extend(data["languages"])

# Language to extension mapping to avoid invalid combinations
LANGUAGE_EXTENSIONS = {
    "python": ["py", "pyw", "pyi"],
    "javascript": ["js", "mjs", "jsx"],
    "typescript": ["ts", "tsx"],
    "java": ["java"],
    "cpp": ["cpp", "cc", "cxx", "c++", "hpp", "h++"],
    "c": ["c", "h"],
    "csharp": ["cs"],
    "go": ["go"],
    "rust": ["rs"],
    "php": ["php", "phtml"],
    "ruby": ["rb", "rake"],
    "swift": ["swift"],
    "kotlin": ["kt", "kts"],
    "scala": ["scala", "sc"],
    "dart": ["dart"],
    "r": ["r", "R"],
    "lua": ["lua"],
    "perl": ["pl", "pm"],
    "html": ["html", "htm"],
    "coffeescript": ["coffee"],
    "handlebars": ["hbs", "handlebars"],
    "clojure": ["clj", "cljs", "cljc"],
    "fsharp": ["fs", "fsi", "fsx"],
    "scheme": ["scm", "ss"],
    "sql": ["sql"],
    "mysql": ["sql"],
    "pgsql": ["sql"],
    "graphql": ["graphql", "gql"],
    "redis": ["redis"],
    "markdown": ["md", "markdown"],
    "json": ["json"],
    "yaml": ["yaml", "yml"],
    "xml": ["xml"],
    "ini": ["ini", "cfg", "conf"],
    "dockerfile": ["dockerfile", "Dockerfile"],
    "shell": ["sh", "bash", "zsh", "fish"],
    "powershell": ["ps1", "psm1"],
    "bat": ["bat", "cmd"],
    "plaintext": ["txt", "text"],
    "apex": ["cls", "trigger"],
    "pascal": ["pas", "pp"],
    "tcl": ["tcl"],
    "vb": ["vb", "vbs"],
    "toml": ["toml"],
}


# File size ranges for refinement
SIZE_RANGES = [
    "<1000",  # < 1KB
    "1000..5000",  # 1-5KB
    "5000..20000",  # 5-20KB
    "20000..100000",  # 20-100KB
    ">100000",  # > 100KB
]


# Maximum pages for API search
API_MAX_PAGES = 10


# Maximum pages for web search
WEB_MAX_PAGES = 5


# Results per page for API search
API_RESULTS_PER_PAGE = 100


# REST API limit (10 pages * 100 results)
API_LIMIT = API_MAX_PAGES * API_RESULTS_PER_PAGE


# Results per page for web search
WEB_RESULTS_PER_PAGE = 20


# Web search limit (5 pages * 20 results)
WEB_LIMIT = WEB_MAX_PAGES * WEB_RESULTS_PER_PAGE


# Start date for search
SEARCH_START_DATE = "2022-10-01"


logging.basicConfig(
    format="%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(os.path.join(PATH, "search.log")), logging.StreamHandler()],
)


@dataclass
class KeyDetail(object):
    # token
    key: str

    # available
    available: bool = False

    # models that the key can access
    models: list[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        if not isinstance(other, KeyDetail):
            return False

        return self.key == other.key


@dataclass
class Service(object):
    # server address
    address: str = ""

    # application name or id
    endpoint: str = ""

    # api token
    key: str = ""

    # model name
    model: str = ""

    def __hash__(self):
        return hash((self.address, self.endpoint, self.key, self.model))

    def __eq__(self, other):
        if not isinstance(other, Service):
            return False

        return (
            self.address == other.address
            and self.endpoint == other.endpoint
            and self.key == other.key
            and self.model == other.model
        )

    def serialize(self) -> str:
        if not self.address and not self.endpoint and not self.model:
            return self.key

        data = dict()
        if self.address:
            data["address"] = self.address
        if self.endpoint:
            data["endpoint"] = self.endpoint
        if self.key:
            data["key"] = self.key
        if self.model:
            data["model"] = self.model

        return "" if not data else json.dumps(data)

    @classmethod
    def deserialize(cls, text: str) -> "Service":
        if not text:
            return None

        try:
            item = json.loads(text)
            return cls(
                address=item.get("address", ""),
                endpoint=item.get("endpoint", ""),
                key=item.get("key", ""),
                model=item.get("model", ""),
            )
        except:
            return cls(key=text)


@unique
class ErrorReason(Enum):
    # no error
    NONE = 1

    # insufficient_quota
    NO_QUOTA = 2

    # rate_limit_exceeded
    RATE_LIMITED = 3

    # model_not_found
    NO_MODEL = 4

    # account_deactivated
    EXPIRED_KEY = 5

    # invalid_api_key
    INVALID_KEY = 6

    # unsupported_country_region_territory
    NO_ACCESS = 7

    # server_error
    SERVER_ERROR = 8

    # bad request
    BAD_REQUEST = 9

    # unknown error
    UNKNOWN = 10


@dataclass
class CheckResult(object):
    # whether the key can be used now
    available: bool = False

    # error message if the key cannot be used
    reason: ErrorReason = ErrorReason.UNKNOWN

    def ok():
        return CheckResult(available=True, reason=ErrorReason.NONE)

    def fail(reason: ErrorReason):
        return CheckResult(available=False, reason=reason)


@dataclass
class Condition(object):
    # pattern for extract key from code
    regex: str

    # search keyword or pattern
    query: str = ""

    def __hash__(self):
        return hash((self.query, self.regex))

    def __eq__(self, other):
        if not isinstance(other, Condition):
            return False

        return self.query == other.query and self.regex == other.regex


class Provider(object):
    def __init__(
        self,
        name: str,
        base_url: str,
        completion_path: str,
        model_path: str,
        default_model: str,
        conditions: Condition | list[Condition],
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
                logging.warning(f"invalid condition: {condition}, skip it")
                continue

            items.add(condition)

        # search and extract keys conditions
        self.conditions = list(items)

        # additional parameters for provider
        self.extras = kwargs

    def _get_headers(self, token: str, additional: dict = None) -> dict:
        raise NotImplementedError

    def _judge(self, code: int, message: str) -> CheckResult:
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
        url, regex = trim(address), r"^https?://([\w\-_]+\.[\w\-_]+)+"
        if not url and re.match(regex, self.base_url, flags=re.I):
            url = urllib.parse.urljoin(self.base_url, self.completion_path)

        if not re.match(regex, url, flags=re.I):
            logging.error(f"invalid url: {url}, skip to check")
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        headers = self._get_headers(token=token)
        if not headers:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        model = trim(model) or self.default_model
        code, message = chat(url=url, headers=headers, model=model)
        return self._judge(code=code, message=message)

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> list[str]:
        raise NotImplementedError


class OpenAILikeProvider(Provider):
    def __init__(
        self,
        name: str,
        base_url: str,
        default_model: str,
        conditions: list[Condition],
        completion_path: str = "",
        model_path: str = "",
        **kwargs,
    ):
        completion_path = trim(completion_path) or DEFAULT_COMPLETION_PATH
        model_path = trim(model_path) or DEFAULT_MODEL_PATH

        super().__init__(name, base_url, completion_path, model_path, default_model, conditions, **kwargs)

    def _get_headers(self, token: str, additional: dict = None) -> dict:
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
                logging.error(f"failed to parse response, domain: {self.base_url}, message: {message}")
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

    def _fetch_models(self, url: str, headers: dict) -> list[str]:
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

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> list[str]:
        headers = self._get_headers(token=token)
        if not headers or not self.base_url or not self.model_path:
            return []

        url = urllib.parse.urljoin(self.base_url, self.model_path)
        return self._fetch_models(url=url, headers=headers)


class OpenAIProvider(OpenAILikeProvider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "gpt-4o-mini"
        base_url = "https://api.openai.com"

        super().__init__("openai", base_url, default_model, conditions)


class DoubaoProvider(OpenAILikeProvider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
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
        )

    def _judge(self, code: int, message: str) -> CheckResult:
        if code == 404:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super()._judge(code, message)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        model = trim(model)
        if not model:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super().check(token=token, address=address, endpoint=endpoint, model=model)


class QianFanProvider(OpenAILikeProvider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
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
        )

    def _judge(self, code: int, message: str) -> CheckResult:
        if code == 404:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super()._judge(code, message)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
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

    def list_models(self, token, address="", endpoint=""):
        headers = self._get_headers(token=token)
        if not headers:
            return []

        endpoint = trim(endpoint)
        if endpoint:
            headers["appid"] = endpoint

        url = urllib.parse.urljoin(self.base_url, self.model_path)
        return self._fetch_models(url=url, headers=headers)


class AnthropicProvider(Provider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "claude-3-5-sonnet-latest"
        super().__init__("anthropic", "https://api.anthropic.com", "/v1/messages", "", default_model, conditions)

    def _get_headers(self, token: str, additional: dict = None) -> dict:
        token = trim(token)
        if not token:
            return None

        return {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
        }

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        token = trim(token)
        if token.startswith("sk-ant-sid01-"):
            url = "https://api.claude.ai/api/organizations"
            headers = {
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "accept-language": "en-US,en;q=0.9",
                "cache-control": "max-age=0",
                "cookie": f"sessionKey={token}",
                "user-agent": USER_AGENT,
                "sec-fetch-mode": "navigate",
                "sec-fetch-site": "none",
            }

            # content = http_get(url=url, headers=headers, interval=1)
            content, success = "", False
            attempt, retries, timeout = 0, 3, 10

            req = urllib.request.Request(url, headers=headers, method="GET")
            while attempt < retries:
                try:
                    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as response:
                        content = response.read().decode("utf8")
                        success = True
                        break
                except urllib.error.HTTPError as e:
                    if e.code == 401:
                        return CheckResult.fail(ErrorReason.INVALID_KEY)
                    else:
                        try:
                            content = e.read().decode("utf8")
                            if not content.startswith("{") or not content.endswith("}"):
                                content = e.reason
                        except:
                            content = e.reason

                        if e.code == 403:
                            message = ""
                            try:
                                data = json.loads(content)
                                message = data.get("error", {}).get("message", "")
                            except:
                                message = content

                            if re.findall(r"Invalid authorization", message, flags=re.I):
                                return CheckResult.fail(ErrorReason.INVALID_KEY)

                        if e.code in NO_RETRY_ERROR_CODES:
                            break
                except Exception as e:
                    if not isinstance(e, urllib.error.URLError) or not isinstance(e.reason, socket.timeout):
                        logging.error(f"check claude session error, key: {token}, message: {traceback.format_exc()}")

                attempt += 1
                time.sleep(1)

            if not content or re.findall(r"Invalid authorization", content, flags=re.I):
                return CheckResult.fail(ErrorReason.INVALID_KEY)
            elif not success:
                logging.error(f"check claude session error, key: {token}, message: {content}")
                return CheckResult.fail(ErrorReason.UNKNOWN)

            try:
                data = json.loads(content)
                valid = False
                if data and isinstance(data, list):
                    valid = trim(data[0].get("name", None)) != ""

                    capabilities = data[0].get("capabilities", [])
                    if capabilities and isinstance(capabilities, list) and "claude_pro" in capabilities:
                        logging.info(f"found claude pro key: {token}")

                if not valid:
                    logging.warning(f"check error, anthropic session key: {token}, message: {content}")

                return CheckResult.ok() if valid else CheckResult.fail(ErrorReason.INVALID_KEY)
            except:
                return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super().check(token=token, address=address, endpoint=endpoint, model=model)

    def _judge(self, code: int, message: str) -> CheckResult:
        message = trim(message)
        if re.findall(r"credit balance is too low|Billing|purchase", message, flags=re.I):
            return CheckResult.fail(ErrorReason.NO_QUOTA)
        elif code == 404 and re.findall(r"not_found_error", trim(message), flags=re.I):
            return CheckResult.fail(ErrorReason.NO_MODEL)

        return super()._judge(code, message)

    def list_models(self, token, address: str = "", endpoint: str = "") -> list[str]:
        token = trim(token)
        if not token:
            return []

        # see: https://docs.anthropic.com/en/docs/about-claude/models
        return [
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]


class AzureOpenAIProvider(OpenAILikeProvider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "gpt-4o"
        super().__init__(
            name="azure",
            base_url="",
            completion_path="/chat/completions",
            model_path="/models",
            default_model=default_model,
            conditions=conditions,
            address_pattern=r"https://[a-zA-Z0-9_\-\.]+.openai.azure.com/openai/",
        )

        self.api_version = "2024-10-21"

    def _get_headers(self, token: str, additional: dict = None) -> dict:
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
        if code == 404:
            message = trim(message)
            if re.finditer(r"The API deployment for this resource does not exist", message, flags=re.I):
                return CheckResult.fail(ErrorReason.NO_MODEL)

            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super()._judge(code, message)

    def __generate_address(self, address: str = "", endpoint: str = "", model: str = "") -> str:
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
        token = trim(token)
        if not token:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        url = self.__generate_address(address=address, endpoint=endpoint, model=model)
        if not url:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super().check(token=token, address=url, endpoint=endpoint, model=model)

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> list[str]:
        domain = trim(address).removesuffix("/")
        if not re.match(r"^https?://([\w\-_]+\.[\w\-_]+)+", domain, flags=re.I):
            logging.error(f"invalid domain: {domain}, skip to list models")
            return []

        headers = self._get_headers(token=token)
        if not headers or not self.model_path:
            return []

        url = f"{domain}/{self.model_path}?api-version={self.api_version}"
        return self._fetch_models(url=url, headers=headers)


class GeminiProvider(Provider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "gemini-exp-1206"
        base_url = "https://generativelanguage.googleapis.com"
        sub_path = "/v1beta/models"

        super().__init__("gemini", base_url, sub_path, sub_path, default_model, conditions)

    def _get_headers(self, token: str, additional: dict = None) -> dict:
        return {"accept": "application/json", "content-type": "application/json"}

    def _judge(self, code: int, message: str) -> CheckResult:
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
        token = trim(token)
        if not token:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        model = trim(model) or self.default_model
        url = f"{urllib.parse.urljoin(self.base_url, self.completion_path)}/{model}:generateContent?key={token}"

        params = {"contents": [{"role": "user", "parts": [{"text": DEFAULT_QUESTION}]}]}
        code, message = chat(url=url, headers=self._get_headers(token=token), params=params)
        return self._judge(code=code, message=message)

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> list[str]:
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
            logging.error(f"failed to parse models from response: {content}")
            return []


class GooeyAIProvider(Provider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "gpt_4_o_mini"
        base_url = "https://api.gooey.ai"
        sub_path = "/v2/google-gpt"

        super().__init__("gooeyai", base_url, sub_path, "", default_model, conditions)

    def _get_headers(self, token: str, additional: dict = None) -> dict:
        token = trim(token)

        return (
            {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {token}",
                "user-agent": USER_AGENT,
            }
            if token
            else None
        )

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        headers = self._get_headers(token=token)
        if not headers:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        model = trim(model) or self.default_model
        url = urllib.parse.urljoin(self.base_url, self.completion_path)

        params = {
            "search_query": "I'm looking for 4 stats that have a negative spin and create FOMO/urgency. and 4 stats that have a positive spin.\n\nI only want stats that focus on how Al can help people, teams and companies be better.\n\nSearch the web for reports created this year. Only cite actual stats from those reports. BE CAREFUL. Give a link to each source after each stat. Preferably use reports from companies like Microsoft, Linkedin, Gartner, PWC, Deloitte, Accenture, BCG, McKinsey.",
            "site_filter": "",
            "selected_model": model,
            "max_search_urls": 3,
            "max_references": 3,
            "embedding_model": "openai_3_large",
            "avoid_repetition": True,
            "max_tokens": 2000,
            "sampling_temperature": 0,
            "response_format_type": "json_object",
        }

        code, message = chat(url=url, headers=headers, params=params)
        return self._judge(code=code, message=message)

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> list[str]:
        # see: https://api.gooey.ai/docs#tag/Web-Search-+-GPT3/operation/google-gpt
        return [
            "o1_preview",
            "o1_mini",
            "gpt_4_o",
            "gpt_4_o_mini",
            "chatgpt_4_o",
            "gpt_4_turbo_vision",
            "gpt_4_vision",
            "gpt_4_turbo",
            "gpt_4",
            "gpt_4_32k",
            "gpt_3_5_turbo",
            "gpt_3_5_turbo_16k",
            "gpt_3_5_turbo_instruct",
            "llama3_3_70b",
            "llama3_2_90b_vision",
            "llama3_2_11b_vision",
            "llama3_2_3b",
            "llama3_2_1b",
            "llama3_1_70b",
            "llama3_1_8b",
            "llama3_70b",
            "llama3_8b",
            "mixtral_8x7b_instruct_0_1",
            "gemma_2_9b_it",
            "gemma_7b_it",
            "gemini_1_5_flash",
            "gemini_1_5_pro",
            "gemini_1_pro_vision",
            "gemini_1_pro",
            "palm2_chat",
            "palm2_text",
            "claude_3_5_sonnet",
            "claude_3_opus",
            "claude_3_sonnet",
            "claude_3_haiku",
            "afrollama_v1",
            "llama3_8b_cpt_sea_lion_v2_1_instruct",
            "sarvam_2b",
            "llama_3_groq_70b_tool_use",
            "llama_3_groq_8b_tool_use",
            "llama2_70b_chat",
            "sea_lion_7b_instruct",
            "llama3_8b_cpt_sea_lion_v2_instruct",
            "text_davinci_003",
            "text_davinci_002",
            "code_davinci_002",
            "text_curie_001",
            "text_babbage_001",
            "text_ada_001",
        ]


class StabilityAIProvider(Provider):
    def __init__(self, conditions: list[Condition], default_model: str = ""):
        default_model = trim(default_model) or "core"
        base_url = "https://api.stability.ai"
        sub_path = "/v2beta/stable-image/generate"

        super().__init__("stabilityai", base_url, sub_path, "", default_model, conditions)

    def _get_headers(self, token: str, additional: dict = None) -> dict:
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
        def post_multipart(
            url: str, token: str, fields: dict = None, files: dict = None, retries: int = 3
        ) -> tuple[int, str]:
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

                        logging.error(
                            f"[Chat] failed to request url: {url}, token: {token}, status code: {code}, message: {message}"
                        )

                    if code in NO_RETRY_ERROR_CODES:
                        break
                except Exception:
                    pass

                attempt += 1
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

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> list[str]:
        return []


def search_github_web(query: str, session: str, page: int) -> str:
    """use github web search instead of rest api due to it not support regex syntax"""

    if page <= 0 or isblank(session) or isblank(query):
        return ""

    url = f"https://github.com/search?o=desc&p={page}&type=code&q={query}"
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Referer": "https://github.com",
        "User-Agent": USER_AGENT,
        "Cookie": f"user_session={session}",
    }

    content = http_get(url=url, headers=headers)
    if re.search(r"<h1>Sign in to GitHub</h1>", content, flags=re.I):
        logging.error("[GithubCrawl] session has expired, please provide a valid session and try again")
        return ""

    return content


def search_web_with_count(
    query: str,
    session: str,
    page: int = 1,
    callback: Callable[[list[str], str], None] = None,
) -> tuple[list[str], int]:
    """
    Search GitHub web and return both results and total count.
    Returns: (results_list, total_count)
    """
    if page <= 0 or isblank(session) or isblank(query):
        return [], 0

    # Get results from web search
    content = search_github_web(query, session, page)
    if isblank(content):
        return [], 0

    # Extract links from content
    try:
        regex = r'href="(/[^\s"]+/blob/(?:[^"]+)?)#L\d+"'
        groups = re.findall(regex, content, flags=re.I)
        uris = list(set(groups)) if groups else []
        links = set()

        for uri in uris:
            links.add(f"https://github.com{uri}")

        results = list(links)
    except:
        results = []

    # Call extract callback if provided
    if callback and isinstance(callback, Callable) and results:
        try:
            callback(results, content)
        except Exception as e:
            logging.error(f"[Search] callback failed: {e}")

    # Get total count (only for first page to avoid redundant calls)
    if page == 1:
        total = estimate_web_total(query, session, content)
    else:
        # For non-first pages, we don't need total count, use 0 as placeholder
        total = 0

    return results, total


def search_github_api(query: str, token: str, page: int = 1, peer_page: int = API_RESULTS_PER_PAGE) -> list[str]:
    """rate limit: 10RPM"""
    if isblank(token) or isblank(query):
        return []

    peer_page, page = min(max(peer_page, 1), API_RESULTS_PER_PAGE), max(1, page)
    url = f"https://api.github.com/search/code?q={query}&sort=indexed&order=desc&per_page={peer_page}&page={page}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    content = http_get(url=url, headers=headers, interval=2, timeout=30)
    if isblank(content):
        return []
    try:
        items = json.loads(content).get("items", [])
        links = set()

        for item in items:
            if not item or type(item) != dict:
                continue

            link = item.get("html_url", "")
            if isblank(link):
                continue
            links.add(link)

        return list(links)
    except:
        return []


def search_api_with_count(
    query: str, token: str, page: int = 1, peer_page: int = API_RESULTS_PER_PAGE
) -> tuple[list[str], int]:
    """
    Search GitHub API and return both results and total count.
    Returns: (results_list, total_count)
    """
    if isblank(token) or isblank(query):
        return [], 0

    peer_page, page = min(max(peer_page, 1), API_RESULTS_PER_PAGE), max(1, page)
    url = f"https://api.github.com/search/code?q={query}&sort=indexed&order=desc&per_page={peer_page}&page={page}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    content = http_get(url=url, headers=headers, interval=2, timeout=30)
    if isblank(content):
        return [], 0

    try:
        data = json.loads(content)
        items = data.get("items", [])
        total = data.get("total_count", 0)

        links = set()
        for item in items:
            if not item or type(item) != dict:
                continue

            link = item.get("html_url", "")
            if isblank(link):
                continue
            links.add(link)

        return list(links), total
    except:
        return [], 0


def search_with_count(
    query: str,
    session: str,
    page: int,
    with_api: bool,
    peer_page: int,
    callback: Callable[[list[str], str], None] = None,
) -> tuple[list[str], int]:
    """
    Unified search interface that returns both results and total count.
    Returns: (results_list, total_count)
    """
    if with_api:
        return search_api_with_count(query, session, page, peer_page)
    else:
        return search_web_with_count(query, session, page, callback)


def can_refine_regex(query: str, partitions: int) -> bool:
    """
    Check if regex query can be further refined based on mathematical analysis.

    Args:
        query: The query string containing regex pattern
        partitions: Number of partitions needed

    Returns:
        True if regex can be refined to handle the partitions, False otherwise
    """
    if not has_regex_pattern(query):
        return False

    # Extract regex pattern
    match = re.search(r"/([^/]+)/", query)
    if not match:
        return False

    pattern = match.group(1)

    # Parse pattern to find dynamic parts
    parts = re.match(r"^([^[]*)\[([^\]]+)\]\{(\d+(?:,\d+)?)\}", pattern)
    if not parts:
        return False

    classes = parts.group(2)
    lengths = parts.group(3)

    # Parse length specification
    if "," in lengths:
        _, max_len = map(int, lengths.split(","))
    else:
        max_len = int(lengths)

    # Calculate character possibilities (GitHub case-insensitive)
    chars = parse_chars(classes)
    count = len(chars)

    # Calculate maximum combinations with current length
    combinations = count**max_len

    # Check if current regex can handle required partitions
    result = combinations >= partitions

    logging.info(
        f"[Search] regex analysis: chars={count}, max_len={max_len}, partitions={partitions}, can_refine={result}"
    )

    return result


def progressive_search(
    queries: list[str],
    session: str,
    with_api: bool,
    thread_num: int = None,
    fast: bool = False,
    callback: Callable[[list[str], str], None] = None,
) -> list[str]:
    """
    Progressive search using task queue with dynamic refinement support.
    """
    # Initialize task queue and results
    jobs = deque()
    results = set()
    limit = API_RESULTS_PER_PAGE if with_api else WEB_RESULTS_PER_PAGE

    # Thread-safe operations
    queue_lock = threading.Lock()
    results_lock = threading.Lock()

    def record_task_addition(func):
        """Decorator to record task addition statistics in thread-safe manner."""

        def wrapper(*args, **kwargs):
            with queue_lock:
                before = len(jobs)

                # Call the original function
                func(*args, **kwargs)

                after = len(jobs)
                count = after - before

                logging.info(f"[Search] {func.__name__}: before={before}, after={after}, added={count}")

        return wrapper

    @record_task_addition
    def add_pages(query: str, start: int, end: int):
        """Add page tasks to queue in thread-safe manner."""
        for page in range(start, end + 1):
            jobs.append((query, page))

    @record_task_addition
    def add_queries(conditions: list[str]):
        """Add query tasks to queue in thread-safe manner."""
        for query in conditions:
            jobs.append((query, 1))

    def handle_first(query: str, total: int, partitions: int):
        """Handle first page processing and determine next actions."""
        if partitions <= 1:
            return

        # For web search with too many partitions, try regex refinement first
        if not with_api and partitions > WEB_MAX_PAGES and can_refine_regex(query, partitions):
            # Generate refined queries
            conditions = generate_regex_queries(query, total)
            add_queries(conditions)

            logging.info(f"[Search] refined query '{query}' into {len(conditions)} sub-queries")
        else:
            # Add remaining pages (with limit for web search)
            if not with_api and partitions > WEB_MAX_PAGES:
                # Web search with limit
                add_pages(query, 2, WEB_MAX_PAGES)

                logging.info(f"[Search] cannot refine '{query}', added {WEB_MAX_PAGES - 1} pages (limited)")
            else:
                # API search or web search within limit
                add_pages(query, 2, partitions)

                logging.info(f"[Search] query '{query}' has {total} results, added {partitions - 1} pages")

    # Initialize queue with all queries
    add_queries(queries)
    logging.info(f"[Search] starting progressive search with {len(jobs)} initial tasks")

    def process_task(item):
        """Process a single task from the queue."""
        query, page = item
        encoded = urllib.parse.quote_plus(query)

        if page == 1:
            # First page - get results and total count
            data, total = search_with_count(encoded, session, page, with_api, limit, callback)

            with results_lock:
                if data:
                    results.update(data)

            # Calculate partitions needed and handle next actions
            partitions = math.ceil(total / limit)
            handle_first(query, total, partitions)
        else:
            # Subsequent pages - just get results
            data = search_code(encoded, session, page, with_api, limit, callback)

            with results_lock:
                if data:
                    results.update(data)

    # Process tasks
    if fast and thread_num and thread_num > 1:
        # Concurrent processing using producer-consumer pattern
        tasks = queue.Queue()

        # Add initial tasks to queue
        with queue_lock:
            while jobs:
                tasks.put(jobs.popleft())

        def worker():
            """Worker thread function for processing tasks."""
            while True:
                try:
                    # Get task with timeout to avoid infinite waiting
                    item = tasks.get(timeout=30)
                    if item is None:  # Poison pill - shutdown signal
                        break

                    # Process the task
                    process_task(item)

                    # Check for new tasks added by process_task
                    with queue_lock:
                        while jobs:
                            tasks.put(jobs.popleft())

                    # Mark task as done
                    tasks.task_done()

                    # Rate limiting for API calls
                    if with_api:
                        time.sleep(random.randint(6, 12))
                    else:
                        time.sleep(random.randint(1, 3))

                except queue.Empty:
                    # Timeout occurred, check if all tasks are done
                    if tasks.unfinished_tasks == 0:
                        break
                except Exception as e:
                    logging.error(f"[Search] task failed: {e}")
                    tasks.task_done()

        # Start worker threads
        workers = list()
        for i in range(thread_num):
            thread = threading.Thread(target=worker, name=f"SearchWorker-{i+1}")
            thread.start()
            workers.append(thread)

        # Wait for all tasks to complete
        tasks.join()

        # Send shutdown signal to all workers
        for _ in range(thread_num):
            tasks.put(None)

        # Wait for all worker threads to finish
        for thread in workers:
            thread.join()

        logging.info(f"[Search] concurrent processing completed with {thread_num} workers")
    else:
        # Sequential processing
        while jobs:
            task = jobs.popleft()
            try:
                process_task(task)

                # Rate limiting
                if with_api:
                    time.sleep(random.randint(6, 12))
                else:
                    time.sleep(random.randint(2, 5))

            except Exception as e:
                logging.error(f"[Search] task {task} failed: {e}")

    final = list(results)
    logging.info(f"[Search] progressive search completed, found {len(final)} unique links")
    return final


def get_total_num(query: str, token: str) -> int:
    if isblank(token) or isblank(query):
        return 0

    url = f"https://api.github.com/search/code?q={query}&sort=indexed&order=desc&per_page=20&page=1"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    content = http_get(url=url, headers=headers, interval=1)
    try:
        data = json.loads(content)
        return data.get("total_count", 0)
    except:
        logging.error(f"[GithubCrawl] failed to get total number of items with query: {query}")
        return 0


def estimate_web_total(query: str, session: str, content: str = None) -> int:
    """
    Get total count for web search using GitHub's blackbird_count API.
    Performs a single search and then queries the count API.
    """
    if isblank(session) or isblank(query):
        return 0

    try:
        message = urllib.parse.unquote_plus(query)
    except:
        message = query

    try:
        if content is None:
            # Perform initial search to trigger count calculation and get content for fallback
            content = search_github_web(query=query, session=session, page=1)

        content = trim(content)
        if not content:
            logging.warning(f"[Search] initial search failed for query: {message}, using conservative estimate")

            # Conservative estimate
            return WEB_RESULTS_PER_PAGE

        # Check if query is already encoded to avoid double encoding
        if "%" in query and any(c in query for c in ["%2F", "%5B", "%5D", "%7B", "%7D"]):
            encoded = query.replace(" ", "+")
        else:
            encoded = urllib.parse.quote_plus(query)

        # Query the blackbird_count API
        url = f"https://github.com/search/blackbird_count?saved_searches=^&q={encoded}"
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"https://github.com/search?q={encoded}^&type=code",
            "X-Requested-With": "XMLHttpRequest",
            "Cookie": f"user_session={session}",
        }

        # Random delay to ensure count is calculated
        time.sleep(random.random() * 2)

        response = http_get(url=url, headers=headers, interval=1)
        if response:
            data = json.loads(response)
            if not data.get("failed", True):
                count = data.get("count", 0)
                mode = data.get("mode", "unknown")
                logging.info(f"[Search] got {count} results, mode: {mode}, query: {message}")

                # Return count if valid, otherwise try page extraction
                return count if count > 0 else extract_count_from_page(content, query)

        # Fallback: extract count from search page
        return extract_count_from_page(content, query)

    except Exception as e:
        logging.error(f"[Search] estimation failed for query: {message}, error: {e}, using conservative estimate")

        # Conservative estimate
        return WEB_RESULTS_PER_PAGE


def extract_count_from_page(content: str, query: str) -> int:
    """
    Extract result count from GitHub search page content.
    """
    if isblank(content):
        return WEB_RESULTS_PER_PAGE

    try:
        message = urllib.parse.unquote_plus(query)

        # Try different patterns GitHub uses to show result counts
        patterns = [
            r"We\'ve found ([\d,]+) code results",
            r"([\d,]+) code results",
            r'data-total-count="([\d,]+)"',
            r'"total_count":(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.I)
            if match:
                text = match.group(1).replace(",", "")
                count = int(text)
                logging.info(f"[Search] extracted {count} results from page for query: {message}")
                return count

        # If no count found, use conservative estimate
        logging.warning(f"[Search] could not extract count from page for query: {message}")
        return WEB_RESULTS_PER_PAGE

    except Exception as e:
        logging.error(f"[Search] failed to extract count from page: {e}")
        return WEB_RESULTS_PER_PAGE


def refined_search(
    session: str,
    query: str,
    with_api: bool,
    total: int,
    thread_num: int = None,
    fast: bool = False,
    callback: Callable[[list[str], str], None] = None,
) -> list[str]:
    """
    Execute refined search using progressive search strategy.
    Uses progressive search to reduce redundant queries.
    """
    # Generate refined queries
    if with_api:
        queries = generate_api_refined_queries(query, total)
        logging.info(f"[Search] generated {len(queries)} API queries")
    else:
        queries = generate_web_refined_queries(query, total, max_queries=-1)
        logging.info(f"[Search] generated {len(queries)} web queries")

    if not queries:
        logging.warning(f"[Search] no refined queries generated for: {query}")
        return []

    # Use progressive search strategy
    return progressive_search(queries, session, with_api, thread_num, fast, callback)


def generate_api_refined_queries(query: str, total: int = 1000) -> list[str]:
    """
    Generate refined queries for REST API search.
    Uses adaptive refinement levels based on expected result count to avoid query overlap.
    """
    queries = set()  # Use set for automatic deduplication
    base = query.strip()

    # Calculate weighted number of languages
    num_languages = get_language_weighted_num()

    # Determine refinement level based on total results
    if total <= API_LIMIT:
        # Level 0: Base query only
        level = "base-only"
        queries.add(base)

    elif total <= API_LIMIT * num_languages:
        # Level 1: Language-based refinement only
        level = "language"
        for lang in POPULAR_LANGUAGES:
            queries.add(f"{base} language:{lang}")

    elif total <= API_LIMIT * num_languages * len(SIZE_RANGES):
        # Level 2: Language + Size refinement
        level = "language+size"
        for lang in POPULAR_LANGUAGES:
            for size in SIZE_RANGES:
                queries.add(f"{base} language:{lang} size:{size}")

    else:
        # Level 3: Maximum refinement (Language + Size + Extension)
        level = "language+size+extension"
        for lang in POPULAR_LANGUAGES:
            if lang in LANGUAGE_EXTENSIONS:
                for size in SIZE_RANGES:
                    for ext in LANGUAGE_EXTENSIONS[lang]:
                        queries.add(f"{base} language:{lang} size:{size} extension:{ext}")
            else:
                # For languages without extensions, use language+size
                for size in SIZE_RANGES:
                    queries.add(f"{base} language:{lang} size:{size}")

    logging.info(f"[API] using {level} refinement for {len(queries)} queries")
    return list(queries)


def has_regex_pattern(query: str) -> bool:
    """
    Check if query contains regex patterns that could benefit from prefix enumeration.
    Supports patterns like:
    - /AIzaSy[a-zA-Z0-9_\\-]{33}/
    - /sk-[a-zA-Z0-9]{32,38}/
    """
    if not query:
        return False

    # Look for regex patterns in GitHub search format: /pattern/
    match = re.search(r"/([^/]+)/", query)
    if not match:
        return False

    pattern = match.group(1)

    # Check for patterns with optional fixed prefix + character class + length
    # Patterns like: AIzaSy[a-zA-Z0-9_\-]{33} or sk-[a-zA-Z0-9]{32,38} or [a-z0-9]{8}
    parts = re.match(r"^([^[]*)\[([^\]]+)\]\{(\d+(?:,\d+)?)\}", pattern)
    if parts:
        return True

    return False


def parse_chars(classes: str) -> set[str]:
    """
    Parse character class like 'a-zA-Z0-9_\\-' and return set of characters.
    Note: GitHub search is case-insensitive, so [a-zA-Z] only gives 26 possibilities.
    """
    chars = set()
    i = 0
    while i < len(classes):
        if i + 2 < len(classes) and classes[i + 1] == "-":
            # Handle ranges like a-z, A-Z, 0-9
            start = classes[i]
            end = classes[i + 2]

            # For GitHub case-insensitive search, treat A-Z same as a-z
            if start.isupper() and end.isupper():
                start = start.lower()
                end = end.lower()
            elif start.islower() and end.isupper():
                # Mixed case range, convert to lowercase
                end = end.lower()
            elif start.isupper() and end.islower():
                start = start.lower()

            for c in range(ord(start), ord(end) + 1):
                chars.add(chr(c))
            i += 3
        else:
            # Single character or escaped character
            char = classes[i]
            if char == "\\" and i + 1 < len(classes):
                # Handle escaped characters like \-
                escaped = classes[i + 1]
                # Convert uppercase to lowercase for GitHub case-insensitive search
                if escaped.isupper():
                    escaped = escaped.lower()
                chars.add(escaped)
                i += 2
            else:
                # Convert uppercase to lowercase for GitHub case-insensitive search
                if char.isupper():
                    char = char.lower()
                chars.add(char)
                i += 1

    return chars


def calculate_depth(total: int, chars: int, limit: int) -> int:
    """
    Calculate how many prefix characters need to be enumerated.

    Args:
        total: Total number of expected results
        chars: Number of possible characters per position
        limit: Results per page (GitHub limit)

    Returns:
        Number of prefix characters to enumerate (0 means no enumeration needed)
    """
    if limit <= 0:
        raise ValueError("[Search] limit must be greater than 0")

    if total <= limit:
        return 0

    needed = math.ceil(total / limit)

    # Find minimum depth where chars^depth >= needed
    for depth in range(1, 10):  # Try depths 1-9
        queries = chars**depth
        if queries >= needed:
            logging.info(f"[Search] selected depth {depth}: {queries} queries for {needed} needed")
            return depth

    # If no suitable depth found, use maximum
    logging.warning(f"[Search] using maximum depth 9 for {needed} needed queries")
    return 9


def generate_regex_queries(query: str, total: int = 0) -> list[str]:
    """
    Generate queries by enumerating prefixes of complex regex patterns.
    Supports patterns like:
    - /AIzaSy[a-zA-Z0-9_\\-]{33}/
    - /sk-[a-zA-Z0-9]{32,38}/
    """
    results = set()

    # Find regex pattern in query
    match = re.search(r"/([^/]+)/", query)
    if not match:
        return [query]

    pattern = match.group(1)
    base = query.replace(f"/{pattern}/", "").strip()

    # Parse complex pattern: optional_prefix[char_class]{length}
    parts = re.match(r"^([^[]*)\[([^\]]+)\]\{(\d+(?:,\d+)?)\}", pattern)
    if not parts:
        return [query]  # Fallback to original query

    prefix = parts.group(1)
    classes = parts.group(2)
    lengths = parts.group(3)

    # Parse length specification
    if "," in lengths:
        min_len, max_len = map(int, lengths.split(","))
    else:
        min_len = max_len = int(lengths)

    # Generate character set (considering GitHub case-insensitivity)
    chars = parse_chars(classes)
    chars = sorted(list(chars))

    logging.info(
        f"[Search] parsed regex: prefix='{prefix}', classes='{classes}', "
        f"length={min_len}-{max_len}, chars={len(chars)}"
    )

    # Calculate enumeration depth
    depth = calculate_depth(total, len(chars), WEB_LIMIT)
    if depth == 0:
        logging.info(f"[Search] enumeration not needed, using original query")
        return [query]

    logging.info(f"[Search] enumerating {depth} prefix chars, expecting ~{len(chars)**depth} queries")

    # Generate all possible prefix combinations
    def combinations(chars: list[str], depth: int):
        if depth == 0:
            yield ""
        else:
            for char in chars:
                for suffix in combinations(chars, depth - 1):
                    yield char + suffix

    # Generate queries for each prefix combination
    for combo in combinations(chars, depth):
        # Construct new pattern
        full_prefix = prefix + combo
        min_remain = max(0, min_len - depth)
        max_remain = max(0, max_len - depth)

        if min_remain == max_remain:
            if min_remain > 0:
                length_part = f"{{{min_remain}}}"
                pattern = f"{full_prefix}[{classes}]{length_part}"
            else:
                # No remaining characters to match
                pattern = full_prefix
        else:
            if max_remain > 0:
                length_part = f"{{{min_remain},{max_remain}}}"
                pattern = f"{full_prefix}[{classes}]{length_part}"
            else:
                pattern = full_prefix

        # Add to query set
        if base:
            results.add(f"/{pattern}/ {base}")
        else:
            results.add(f"/{pattern}/")

    logging.info(f"[Search] generated {len(results)} regex queries")
    return list(results)


def generate_web_refined_queries(query: str, total: int, max_queries: int = -1) -> list[str]:
    """
    Generate refined queries for web search using regex or language enumeration.
    Uses regex or language prefix enumeration to split large result sets into manageable chunks.
    Falls back to original query if enumeration is not applicable.
    """
    base = trim(query)
    if not base or total <= 0:
        logging.error(f"[Search] invalid parameters for web query generation: {query}")
        return []

    logging.info(f"[Search] generating refined queries for {total} results")

    # Check if query contains regex patterns that we can handle, or if it's too large to handle
    if total > WEB_LIMIT:
        if has_regex_pattern(base):
            logging.info(f"[Search] detected regex pattern, attempting enumeration")
            queries = generate_regex_queries(base, total)
        else:
            # No regex pattern detected, use language-based splitting
            logging.info(f"[Search] using language-based splitting for {total} results")

            queries = list()
            for lang in POPULAR_LANGUAGES:
                queries.append(f"{base} language:{lang}")

            logging.info(f"[Search] generated {len(queries)} language-based queries")

        if len(queries) > 1:
            # Successfully generated queries with regex or language-based splitting
            logging.info(f"[Search] using enumeration with {len(queries)} queries")

            if max_queries > 0 and len(queries) > max_queries:
                queries = queries[:max_queries]
                logging.info(f"[Search] limited to {max_queries} queries due to max_queries limit")

            return queries
        else:
            logging.info(f"[Search] enumeration not suitable, using original query")
            return [base]

    # Fallback to original query since GitHub doesn't support created/stars splitting
    logging.info(f"[Search] using original query as fallback")
    return [base]


def get_language_weighted_num() -> int:
    """
    Calculate weighted sum of programming languages across all tiers.

    Formula: sum(language_count * (1 / multiplier)) for each tier, then ceil.

    Returns:
        int: Ceiling of the weighted sum
    """
    total = 0.0

    # 1. Traverse LANGUAGE_TIERS to get language count and multiplier for each tier
    for tier in LANGUAGE_TIERS.values():
        count = len(tier.get("languages", []))
        multiplier = tier.get("multiplier", 1.0)

        # 2. Accumulate: language_count * (1 / multiplier)
        total += count * (1.0 / multiplier)

    # 3. Round up and return
    return math.ceil(total)


def search_code(
    query: str,
    session: str,
    page: int,
    with_api: bool,
    peer_page: int,
    callback: Callable[[list[str], str], None] = None,
) -> list[str]:
    keyword = trim(query)
    if not keyword:
        return []

    if with_api:
        return search_github_api(query=keyword, token=session, page=page, peer_page=peer_page)

    content = search_github_web(query=keyword, session=session, page=page)
    if isblank(content):
        return []

    try:
        regex = r'href="(/[^\s"]+/blob/(?:[^"]+)?)#L\d+"'
        groups = re.findall(regex, content, flags=re.I)
        uris = list(set(groups)) if groups else []
        links = set()

        for uri in uris:
            links.add(f"https://github.com{uri}")

        results = list(links)

        # Call extract callback if provided
        if callback and isinstance(callback, Callable) and results:
            try:
                callback(results, content)
            except Exception as e:
                logging.error(f"[Search] callback failed: {e}")

        return results
    except:
        return []


def batch_search_code(
    session: str,
    query: str,
    with_api: bool = False,
    thread_num: int = None,
    fast: bool = False,
    callback: Callable[[list[str], str], None] = None,
) -> list[str]:
    session, query = trim(session), trim(query)
    if not query or not session:
        logging.error(f"[Search] skip to search due to query or session is empty")
        return []

    keyword = urllib.parse.quote_plus(query)

    if with_api:
        total = get_total_num(query=keyword, token=session)
        logging.info(f"[Search] found {total} items with query: {query}")

        # Check if refinement is needed for API search
        if total > API_LIMIT:
            logging.info(f"[Search] total count {total} exceeds API limit {API_LIMIT}, using refined queries")
            return refined_search(
                session,
                query,
                with_api=True,
                total=total,
                thread_num=thread_num,
                fast=fast,
                callback=callback,
            )
    else:
        # For web search, estimate total count
        total = estimate_web_total(query=keyword, session=session)
        logging.info(f"[Search] estimated {total} items for web query: {query}")

        # Check if refinement is needed for web search
        if total > WEB_LIMIT:
            logging.info(f"[Search] total count {total} exceeds web limit {WEB_LIMIT}, using refined queries")
            return refined_search(
                session,
                query,
                with_api=False,
                total=total,
                thread_num=thread_num,
                fast=fast,
                callback=callback,
            )

    # Use progressive search for better handling
    links = progressive_search(
        queries=[query],
        session=session,
        with_api=with_api,
        thread_num=thread_num,
        fast=fast,
        callback=callback,
    )

    if not links:
        logging.warning(f"[Search] cannot found any link with query: {query}")

    return links


def collect(
    url: str,
    key_pattern: str,
    retries: int = 3,
    address_pattern: str = "",
    endpoint_pattern: str = "",
    model_pattern: str = "",
    text: str = None,
) -> list[Service]:
    if (not isinstance(url, str) and not isinstance(text, str)) or not isinstance(key_pattern, str):
        return []

    if text:
        content = text
    else:
        content = http_get(url=url, retries=retries, interval=1)

    if not content:
        return []

    # extract keys from content
    key_pattern = trim(key_pattern)
    keys = extract(text=content, regex=key_pattern)
    if not keys:
        return []

    # extract api addresses from content
    address_pattern = trim(address_pattern)
    addresses = extract(text=content, regex=address_pattern)
    if address_pattern and not addresses:
        return []
    if not addresses:
        addresses.append("")

    # extract api endpoints from content
    endpoint_pattern = trim(endpoint_pattern)
    endpoints = extract(text=content, regex=endpoint_pattern)
    if endpoint_pattern and not endpoints:
        return []
    if not endpoints:
        endpoints.append("")

    # extract models from content
    model_pattern = trim(model_pattern)
    models = extract(text=content, regex=model_pattern)
    if model_pattern and not models:
        return []
    if not models:
        models.append("")

    candidates = list()

    # combine keys, addresses and endpoints TODO: optimize this part
    for key, address, endpoint, model in itertools.product(keys, addresses, endpoints, models):
        candidates.append(Service(address=address, endpoint=endpoint, key=key, model=model))

    return candidates


def extract(text: str, regex: str) -> list[str]:
    content, pattern = trim(text), trim(regex)
    if not content or not pattern:
        return []

    items = set()
    try:
        groups = re.findall(pattern, content)
        for x in groups:
            words = list()
            if isinstance(x, str):
                words.append(x)
            elif isinstance(x, (tuple, list)):
                words.extend(list(x))
            else:
                logging.error(f"unknown type: {type(x)}, value: {x}. please optimize your regex")
                continue

            for word in words:
                key = trim(word)
                if key:
                    items.add(key)
    except:
        logging.error(traceback.format_exc())

    return list(items)


def persist_keys(services: list[Service], filepath: str, overwrite: bool = True) -> None:
    """
    Persist keys to file

    Args:
        services: list of Service
        filepath: path to save keys
        overwrite: whether to overwrite existing file

    Returns: None
    """
    filepath = trim(filepath)
    if not filepath or not services:
        logging.error("services or filepath cannot be empty")
        return

    lines = [x.serialize() for x in services if x]
    if not write_file(directory=filepath, lines=lines, overwrite=overwrite):
        logging.error(f"[Scan] failed to save keys to file: {filepath}, keys: {lines}")
    else:
        logging.info(f"[Scan] saved {len(lines)} keys to file: {filepath}")


def scan(
    session: str,
    provider: Provider,
    with_api: bool = False,
    thread_num: int = None,
    fast: bool = False,
    skip: bool = False,
    workspace: str = "",
) -> None:
    if not isinstance(provider, Provider):
        return

    keys_filename = trim(provider.keys_filename)
    if not keys_filename:
        logging.error(f"[Scan] {provider.name}: keys filename cannot be empty")
        return

    workspace = trim(workspace)
    directory = os.path.join(os.path.abspath(workspace) if workspace else PATH, provider.directory)

    valid_keys_file = os.path.join(directory, keys_filename)
    material_keys_file = os.path.join(directory, provider.material_filename)
    links_file = os.path.join(directory, provider.links_filename)

    records: set[Service] = set()
    if os.path.exists(valid_keys_file) and os.path.isfile(valid_keys_file):
        # load exists valid keys
        records.update(load_exist_keys(filepath=valid_keys_file))
        logging.info(f"[Scan] {provider.name}: loaded {len(records)} exists keys from file {valid_keys_file}")

        # backup up exists file with current time
        words = keys_filename.rsplit(".", maxsplit=1)
        keys_filename = f"{words[0]}-{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        if len(words) > 1:
            keys_filename += f".{words[1]}"

        os.rename(valid_keys_file, os.path.join(directory, keys_filename))

    if os.path.exists(material_keys_file) and os.path.isfile(material_keys_file):
        # load potential keys from material file
        records.update(load_exist_keys(filepath=material_keys_file))

    if not skip and provider.conditions:
        candidates: set[Service] = set()
        start_time = time.time()
        for condition in provider.conditions:
            if not isinstance(condition, Condition):
                continue

            query, regex = condition.query, condition.regex
            logging.info(
                f"[Scan] {provider.name}: start to search new keys with query: {query or regex}, regex: {regex}"
            )

            # Prepare temporary files for extract callback
            extra_params = provider.extras.copy() if provider.extras else dict()
            extra_params["temp_keys_file"] = os.path.join(directory, "temp-keys.txt")
            extra_params["temp_links_file"] = os.path.join(directory, "temp-links.txt")

            # Search new keys with conditions and links file
            parts = recall(
                regex=regex,
                session=session,
                query=query,
                with_api=with_api,
                thread_num=thread_num,
                fast=fast,
                links_file=links_file,
                extra_params=extra_params,
            )

            if parts:
                candidates.update(parts)

        # merge new keys with exists keys
        records.update(candidates)

        cost = time.time() - start_time
        count, total = len(provider.conditions), len(candidates)
        logging.info(f"[Scan] {provider.name}: cost {cost:.2f}s to search {count} conditions, got {total} new keys")

    if not records:
        logging.warning(f"[Scan] {provider.name}: cannot extract any candidate with conditions: {provider.conditions}")
        return

    items, caches = [], []
    for record in records:
        if not isinstance(record, Service) or not record.key:
            continue

        items.append([record.key, record.address, record.endpoint, record.model])
        caches.append(record)

    logging.info(f"[Scan] {provider.name}: start to verify {len(items)} potential keys")
    masks: list[CheckResult] = multi_thread_run(func=provider.check, tasks=items, thread_num=thread_num)

    # remove invalid keys and ave all potential keys to material file
    materials: list[Service] = [caches[i] for i in range(len(masks)) if masks[i].reason != ErrorReason.INVALID_KEY]
    if materials:
        persist_keys(services=materials, filepath=material_keys_file)

    # save candidates and avaiable status
    statistics = dict()

    # can be used keys
    valid_services: list[Service] = [caches[i] for i in range(len(masks)) if masks[i].available]
    if not valid_services:
        logging.warning(f"[Scan] {provider.name}: cannot found any key with conditions: {provider.conditions}")
    else:
        persist_keys(services=valid_services, filepath=valid_keys_file)

        statistics.update({s: True for s in valid_services})

    # no quota keys
    no_quota_services: list[Service] = [
        caches[i] for i in range(len(masks)) if not masks[i].available and masks[i].reason == ErrorReason.NO_QUOTA
    ]
    if no_quota_services:
        statistics.update({s: False for s in no_quota_services})

        # save no quota keys to file
        no_quota_keys_file = os.path.join(directory, provider.no_quota_filename)
        persist_keys(services=no_quota_services, filepath=no_quota_keys_file)

    # not expired keys but wait to check again keys
    wait_check_services: list[Service] = [
        caches[i]
        for i in range(len(masks))
        if not masks[i].available
        and masks[i].reason
        in [ErrorReason.RATE_LIMITED, ErrorReason.NO_MODEL, ErrorReason.NO_ACCESS, ErrorReason.UNKNOWN]
    ]
    if wait_check_services:
        statistics.update({s: False for s in wait_check_services})

        # save wait check keys to file
        wait_check_keys_file = os.path.join(directory, provider.wait_check_filename)
        persist_keys(services=wait_check_services, filepath=wait_check_keys_file)

    # list supported models for each key
    services, tasks = [], []
    for service in statistics.keys():
        tasks.append([service.key, service.address, service.endpoint])
        services.append(service)

    if not tasks:
        logging.error(f"[Scan] {provider.name}: no keys to list models")
        return

    data = dict()
    models = multi_thread_run(func=provider.list_models, tasks=tasks, thread_num=thread_num)

    for i in range(len(tasks)):
        service: Service = services[i]
        item = {
            "available": statistics.get(service),
            "models": (models[i] if models else []) or [],
        }

        if service.address:
            item["address"] = service.address
        if service.endpoint:
            item["endpoint"] = service.endpoint

        data[service.key] = item

    summary_path = os.path.join(directory, provider.summary_filename)
    if write_file(directory=summary_path, lines=json.dumps(data, ensure_ascii=False, indent=4)):
        logging.info(f"[Scan] {provider.name}: saved {len(services)} keys summary to file: {summary_path}")
    else:
        logging.error(f"[Scan] {provider.name}: failed to save keys summary to file: {summary_path}, data: {data}")


def load_exist_keys(filepath: str) -> set[Service]:
    lines = read_file(filepath=filepath)
    if not lines:
        return set()

    return set([Service.deserialize(text=line) for line in lines])


def recall(
    regex: str,
    session: str,
    query: str = "",
    with_api: bool = False,
    thread_num: int = None,
    fast: bool = False,
    links_file: str = "",
    extra_params: dict = None,
) -> list[Service]:
    regex = trim(regex)
    if not regex:
        logging.error(f"[Recall] skip to recall due to regex is empty")
        return []

    links = set()
    links_file = os.path.abspath(trim(links_file))
    if os.path.exists(links_file) and os.path.isfile(links_file):
        # load exists links from persisted file
        lines = read_file(filepath=links_file)
        for text in lines:
            if not re.match(r"^https?://", text, flags=re.I):
                text = f"http://{text}"

            links.add(text)

    session = trim(session)
    query = trim(query)
    if not query:
        text = regex.replace("/", "\\/")
        query = f"/{text}/"

    if not isinstance(extra_params, dict):
        extra_params = dict()

    address_pattern = extra_params.get("address_pattern", "")
    endpoint_pattern = extra_params.get("endpoint_pattern", "")
    model_pattern = extra_params.get("model_pattern", "")
    temp_keys_file = extra_params.get("temp_keys_file", "")
    temp_links_file = extra_params.get("temp_links_file", "")

    if session:
        # Create process callback as closure function using partial
        callback = None
        if temp_keys_file or temp_links_file:
            # Process search page content and save extracted results
            def process(
                urls: list[str],
                text: str,
                pattern: str,
                address: str,
                endpoint: str,
                model: str,
                keys_file: str,
                links_file: str,
            ) -> list[Service]:
                results = list()

                try:
                    # Extract services from search page content
                    if text and pattern:
                        services = collect(
                            url="",
                            key_pattern=pattern,
                            address_pattern=address,
                            endpoint_pattern=endpoint,
                            model_pattern=model,
                            text=text,
                        )

                        if services:
                            results.extend(services)

                            # Save services to temporary file
                            if keys_file:
                                persist_keys(services=services, filepath=keys_file, overwrite=False)

                    # Save links to temporary file
                    if urls and links_file:
                        write_file(links_file, urls, overwrite=False)

                except Exception as e:
                    logging.error(f"[Search] process callback failed: {e}")

                return results

            # Use partial function to fix parameters
            callback = partial(
                process,
                pattern=regex,
                address=address_pattern,
                endpoint=endpoint_pattern,
                model=model_pattern,
                keys_file=temp_keys_file,
                links_file=temp_links_file,
            )

        # Search new links with query and session
        sources = batch_search_code(
            session=session,
            query=query,
            with_api=with_api,
            thread_num=thread_num,
            fast=fast,
            callback=callback,
        )
        if sources:
            links.update(sources)

    # Load temporary links from file if exists
    links.update(read_file(filepath=temp_links_file))

    # Load links from file if exists
    if not links:
        logging.warning(f"[Recall] cannot found any link with query: {query}")
        return []

    # save links to file
    if links_file and not write_file(directory=links_file, lines=list(links), overwrite=True):
        logging.warning(f"[Recall] failed to save links to file: {links_file}, links: {links}")

    logging.info(f"[Recall] start to extract candidates from {len(links)} links")

    tasks = [[x, regex, 3, address_pattern, endpoint_pattern, model_pattern] for x in links if x]
    result = multi_thread_run(func=collect, tasks=tasks, thread_num=thread_num)
    services = [] if not result else list(itertools.chain.from_iterable([x for x in result if x]))

    # Load temporary keys from file if exists
    if temp_keys_file and os.path.exists(temp_keys_file) and os.path.isfile(temp_keys_file):
        with open(temp_keys_file, "r", encoding="utf8") as f:
            for line in f.readlines():
                text = trim(line)
                if not text or text.startswith(";") or text.startswith("#"):
                    continue

                services.append(Service.deserialize(text=text))

    return services


def chat(
    url: str, headers: dict, model: str = "", params: dict = None, retries: int = 2, timeout: int = 10
) -> tuple[int, str]:
    def output(code: int, message: str, debug: bool = False) -> None:
        text = f"[Chat] failed to request url: {url}, headers: {headers}, status code: {code}, message: {message}"
        if debug:
            logging.debug(text)
        else:
            logging.error(text)

    url, model = trim(url), trim(model)
    if not url:
        logging.error(f"[Chat] url cannot be empty")
        return 400, None

    if not isinstance(headers, dict):
        logging.error(f"[Chat] headers must be a dict")
        return 400, None
    elif len(headers) == 0:
        headers["content-type"] = "application/json"

    if not params or not isinstance(params, dict):
        if not model:
            logging.error(f"[Chat] model cannot be empty")
            return 400, None

        params = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": DEFAULT_QUESTION}],
        }

    payload = json.dumps(params).encode("utf8")
    timeout = max(1, timeout)
    retries = max(1, retries)
    code, message, attempt = 400, None, 0

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    while attempt < retries:
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as response:
                code = 200
                message = response.read().decode("utf8")
                break
        except urllib.error.HTTPError as e:
            code = e.code
            if code != 401:
                try:
                    # read response body
                    message = e.read().decode("utf8")

                    # not a json string, use reason instead
                    if not message.startswith("{") or not message.endswith("}"):
                        message = e.reason
                except:
                    message = e.reason

                # print http status code and error message
                output(code=code, message=message, debug=False)

            if code in NO_RETRY_ERROR_CODES:
                break
        except Exception:
            output(code=code, message=traceback.format_exc(), debug=True)

        attempt += 1
        time.sleep(1)

    return code, message


def scan_anthropic_keys(
    session: str,
    with_api: bool = False,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
    model: str = "",
) -> None:
    regex = r"sk-ant-(?:sid01|api03)-[a-zA-Z0-9_\-]{93}AA"
    if with_api:
        conditions = [Condition(query="sk-ant-api03-", regex=regex), Condition(query="sk-ant-sid01-", regex=regex)]
    else:
        conditions = [Condition(query="", regex=regex)]

    default_model = trim(model) or "claude-3-5-sonnet-latest"
    provider = AnthropicProvider(conditions=conditions, default_model=default_model)

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
        workspace=workspace,
    )


def scan_azure_keys(
    session: str,
    with_api: bool = False,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
    model: str = "",
) -> None:
    # TODO: optimize query syntax for github api
    query = r"/https:\/\/[a-zA-Z0-9_\-\.]+.openai.azure.com\/openai\// AND /(?-i)[a-z0-9]{32}/"
    if with_api:
        query = "openai.azure.com/openai/deployments"

    regex = r"[a-z0-9]{32}"
    conditions = [Condition(query=query, regex=regex)]
    default_model = trim(model) or "gpt-4o"
    provider = AzureOpenAIProvider(conditions=conditions, default_model=default_model)

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
        workspace=workspace,
    )


def scan_gemini_keys(
    session: str,
    with_api: bool = False,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
    model: str = "",
) -> None:
    query = r'/AIzaSy[a-zA-Z0-9_\-]{33}/ AND content:"gemini"'
    if with_api:
        query = '"AIzaSy" AND "gemini"'

    regex = r"AIzaSy[a-zA-Z0-9_\-]{33}"
    conditions = [Condition(query=query, regex=regex)]
    default_model = trim(model) or "gemini-2.0-flash"
    provider = GeminiProvider(conditions=conditions, default_model=default_model)

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
        workspace=workspace,
    )


def scan_gooeyai_keys(
    session: str,
    with_api: bool = False,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
    model: str = "",
) -> None:
    query = '/sk-[a-zA-Z0-9]{48}/ AND "https://api.gooey.ai"'
    if with_api:
        query = '"https://api.gooey.ai" AND "sk-"'

    regex = r"sk-[a-zA-Z0-9]{48}"
    conditions = [Condition(query=query, regex=regex)]
    default_model = trim(model) or "gpt_4_o_mini"
    provider = GooeyAIProvider(conditions=conditions, default_model=default_model)

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
        workspace=workspace,
    )


def scan_openai_keys(
    session: str,
    with_api: bool = False,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
    model: str = "",
) -> None:
    # TODO: optimize query syntax for github api
    query = '"T3BlbkFJ"' if with_api else ""
    regex = r"sk(?:-proj)?-[a-zA-Z0-9]{20}T3BlbkFJ[a-zA-Z0-9]{20}|sk-proj-(?:[a-zA-Z0-9_\-]{91}|[a-zA-Z0-9_\-]{123}|[a-zA-Z0-9_\-]{155})A|sk-svcacct-[A-Za-z0-9_\-]+T3BlbkFJ[A-Za-z0-9_\-]+"

    conditions = [Condition(query=query, regex=regex)]
    provider = OpenAIProvider(conditions=conditions, default_model=(trim(model) or "gpt-4o-mini"))

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
        workspace=workspace,
    )


def scan_doubao_keys(
    session: str,
    with_api: bool = False,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
    model: str = "",
) -> None:
    # TODO: optimize query syntax for github api
    query = '"https://ark.cn-beijing.volces.com" AND /[a-z0-9]{8}(?:-[a-z0-9]{4}){3}-[a-z0-9]{12}/'
    if with_api:
        query = '"https://ark.cn-beijing.volces.com" AND "ep-"'

    regex = r"[a-z0-9]{8}(?:-[a-z0-9]{4}){3}-[a-z0-9]{12}"
    conditions = [Condition(query=query, regex=regex)]
    provider = DoubaoProvider(conditions=conditions, default_model=(trim(model) or "doubao-pro-32k"))

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
        workspace=workspace,
    )


def scan_qianfan_keys(
    session: str,
    with_api: bool = False,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
    model: str = "",
) -> None:
    query = '"bce-v3/ALTAK-"' if with_api else r"/bce-v3\/ALTAK-[a-zA-Z0-9]{21}\/[a-z0-9]{40}/"
    regex = r"bce-v3/ALTAK-[a-zA-Z0-9]{21}/[a-z0-9]{40}"
    conditions = [Condition(query=query, regex=regex)]
    provider = QianFanProvider(conditions=conditions, default_model=(trim(model) or "ernie-4.0-8k-latest"))

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
        workspace=workspace,
    )


def scan_stabilityai_keys(
    session: str,
    with_api: bool = False,
    fast: bool = False,
    skip: bool = False,
    thread_num: int = None,
    workspace: str = "",
    model: str = "",
) -> None:
    query = '"https://api.stability.ai" AND /sk-[a-zA-Z0-9]{48}/'
    if with_api:
        query = '"https://api.stability.ai" AND "sk-"'

    regex = r"sk-[a-zA-Z0-9]{48}"
    conditions = [Condition(query=query, regex=regex)]
    provider = StabilityAIProvider(conditions=conditions, default_model=(trim(model) or "core"))

    return scan(
        session=session,
        provider=provider,
        with_api=with_api,
        thread_num=thread_num,
        fast=fast,
        skip=skip,
        workspace=workspace,
    )


def scan_others(args: argparse.Namespace) -> None:
    if not args or not isinstance(args, argparse.Namespace):
        return

    default_model = trim(args.pm)
    if not default_model:
        logging.error(f"model name as default cannot be empty")
        return

    base_url = trim(args.pb)
    if not re.match(r"^https?://([\w\-_]+\.[\w\-_]+)+", base_url):
        logging.error(f"invalid base url: {base_url}")
        return

    pattern = trim(args.pp)
    if not pattern:
        logging.error(f"pattern for extracting keys cannot be empty")
        return

    queries = list()
    if isinstance(args.pq, str):
        keyword = trim(args.pq)
        if keyword:
            queries.append(keyword)
    elif isinstance(args.pq, list):
        for keyword in args.pq:
            query = trim(keyword)
            if query:
                queries.append(query)

    if not queries:
        if args.rest:
            logging.error(f"queries cannot be empty when using rest api")
            return
        else:
            queries = [""]

    conditions = [Condition(query=query, regex=pattern) for query in queries]

    name = trim(args.pn)
    if not name:
        start = base_url.find("//")
        if start == -1:
            start = -2

        end = base_url.find("/", start + 2)
        if end == -1:
            end = len(base_url)

        name = re.sub(r"[._:/\#]+", "-", trim(base_url[start + 2 : end]), flags=re.I).lower()
        logging.warning(f"provider name is not set, use {name} instead")

    provider = OpenAILikeProvider(
        name=name,
        base_url=base_url,
        default_model=default_model,
        conditions=conditions,
        completion_path=args.pc,
        model_path=args.pl,
        auth_key=args.pk,
    )

    return scan(
        session=args.session,
        provider=provider,
        with_api=args.rest,
        thread_num=args.thread,
        fast=args.fast,
        skip=args.elide,
        workspace=args.workspace,
    )


def trim(text: str) -> str:
    if not text or type(text) != str:
        return ""

    return text.strip()


def isblank(text: str) -> bool:
    return not text or type(text) != str or not text.strip()


def http_get(
    url: str,
    headers: dict = None,
    params: dict = None,
    retries: int = 3,
    interval: float = 0,
    timeout: float = 10,
) -> str:
    if isblank(text=url):
        logging.error(f"invalid url: {url}")
        return ""

    if retries <= 0:
        return ""

    headers = DEFAULT_HEADERS if not headers else headers

    interval = max(0, interval)
    timeout = max(1, timeout)
    try:
        url = encoding_url(url=url)
        if params and isinstance(params, dict):
            data = urllib.parse.urlencode(params)
            if "?" in url:
                url += f"&{data}"
            else:
                url += f"?{data}"

        request = urllib.request.Request(url=url, headers=headers)
        response = urllib.request.urlopen(request, timeout=timeout, context=CTX)
        content = response.read()
        status_code = response.getcode()
        try:
            content = str(content, encoding="utf8")
        except:
            content = gzip.decompress(content).decode("utf8")
        if status_code != 200:
            return ""

        return content
    except urllib.error.HTTPError as e:
        message = e.read()
        try:
            message = str(message, encoding="utf8")
        except:
            message = gzip.decompress(message).decode("utf8")

        if not (message.startswith("{") and message.endswith("}")):
            message = e.reason

        logging.error(f"failed to request url: {url}, status code: {e.code}, message: {message}")

        if e.code in NO_RETRY_ERROR_CODES:
            return ""
    except:
        logging.debug(f"failed to request url: {url}, message: {traceback.format_exc()}")

    time.sleep(interval)
    return http_get(
        url=url,
        headers=headers,
        params=params,
        retries=retries - 1,
        interval=interval,
        timeout=timeout,
    )


def multi_thread_run(func: Callable, tasks: list, thread_num: int = None) -> list:
    if not func or not tasks or not isinstance(tasks, list):
        return []

    if thread_num is None or thread_num <= 0:
        thread_num = min(len(tasks), (os.cpu_count() or 1) * 2)

    funcname = getattr(func, "__name__", repr(func))

    results, starttime = [None] * len(tasks), time.time()
    with futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
        if isinstance(tasks[0], (list, tuple)):
            collections = {executor.submit(func, *param): i for i, param in enumerate(tasks)}
        else:
            collections = {executor.submit(func, param): i for i, param in enumerate(tasks)}

        items = futures.as_completed(collections)
        for future in items:
            try:
                result = future.result()
                index = collections[future]
                results[index] = result
            except:
                logging.error(
                    f"function {funcname} execution generated an exception, message:\n{traceback.format_exc()}"
                )

    logging.info(
        f"[Concurrent] multi-threaded execute [{funcname}] finished, count: {len(tasks)}, cost: {time.time()-starttime:.2f}s"
    )

    return results


def encoding_url(url: str) -> str:
    if not url:
        return ""

    url = url.strip()
    cn_chars = re.findall("[\u4e00-\u9fa5]+", url)
    if not cn_chars:
        return url

    punycodes = list(map(lambda x: "xn--" + x.encode("punycode").decode("utf-8"), cn_chars))
    for c, pc in zip(cn_chars, punycodes):
        url = url[: url.find(c)] + pc + url[url.find(c) + len(c) :]

    return url


def read_file(filepath: str) -> list[str]:
    filepath = trim(filepath)
    if not filepath:
        logging.error(f"filepath cannot be empty")
        return []

    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        logging.error(f"file not found: {filepath}")
        return []

    lines = list()
    with open(filepath, "r", encoding="utf8") as f:
        for line in f.readlines():
            text = trim(line)
            if not text or text.startswith(";") or text.startswith("#"):
                continue

            lines.append(text)

    return lines


def write_file(directory: str, lines: str | list, overwrite: bool = True) -> bool:
    if not directory or not lines or not isinstance(lines, (str, list)):
        logging.error(f"filename or lines is invalid, filename: {directory}")
        return False

    try:
        if not isinstance(lines, str):
            lines = "\n".join(lines)

        filepath = os.path.abspath(os.path.dirname(directory))
        os.makedirs(filepath, exist_ok=True)

        mode = "w+" if overwrite else "a+"

        # waitting for lock
        FILE_LOCK.acquire(30)

        with open(directory, mode=mode, encoding="UTF8") as f:
            f.write(lines + "\n")
            f.flush()

        # release lock
        FILE_LOCK.release()

        return True
    except:
        return False


def main(args: argparse.Namespace) -> None:
    session = trim(args.session)

    if args.azure:
        scan_azure_keys(
            session=session,
            with_api=args.rest,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
            model=args.pm,
        )

    if args.doubao:
        scan_doubao_keys(
            session=session,
            with_api=args.rest,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
            model=args.pm,
        )

    if args.claude:
        scan_anthropic_keys(
            session=session,
            with_api=args.rest,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
            model=args.pm,
        )

    if args.gemini:
        scan_gemini_keys(
            session=session,
            with_api=args.rest,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
            model=args.pm,
        )

    if args.gooeyai:
        scan_gooeyai_keys(
            session=session,
            with_api=args.rest,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
            model=args.pm,
        )

    if args.openai:
        scan_openai_keys(
            session=session,
            with_api=args.rest,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
            model=args.pm,
        )

    if args.qianfan:
        scan_qianfan_keys(
            session=session,
            with_api=args.rest,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
            model=args.pm,
        )

    if args.stabilityai:
        scan_stabilityai_keys(
            session=session,
            with_api=args.rest,
            thread_num=args.thread,
            fast=args.fast,
            skip=args.elide,
            workspace=args.workspace,
            model=args.pm,
        )

    if args.variant:
        return scan_others(args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--azure",
        dest="azure",
        action="store_true",
        default=False,
        help="scan azure's openai api keys",
    )

    parser.add_argument(
        "-c",
        "--claude",
        dest="claude",
        action="store_true",
        default=False,
        help="scan claude api keys",
    )

    parser.add_argument(
        "-d",
        "--doubao",
        dest="doubao",
        action="store_true",
        default=False,
        help="scan doubao api keys",
    )

    parser.add_argument(
        "-e",
        "--elide",
        dest="elide",
        action="store_true",
        default=False,
        help="skip search new keys from github",
    )

    parser.add_argument(
        "-f",
        "--fast",
        dest="fast",
        action="store_true",
        default=False,
        help="concurrent request github rest api to search code for speed up but easy to fail",
    )

    parser.add_argument(
        "-g",
        "--gemini",
        dest="gemini",
        action="store_true",
        default=False,
        help="scan gemini api keys",
    )

    parser.add_argument(
        "-o",
        "--openai",
        dest="openai",
        action="store_true",
        default=False,
        help="scan openai api keys",
    )

    parser.add_argument(
        "-q",
        "--qianfan",
        dest="qianfan",
        action="store_true",
        default=False,
        help="scan qianfan api keys",
    )

    parser.add_argument(
        "-r",
        "--rest",
        dest="rest",
        action="store_true",
        default=False,
        help="search code through github rest api",
    )

    parser.add_argument(
        "-s",
        "--session",
        type=str,
        required=False,
        default="",
        help="github token if use rest api else user session key named 'user_session'",
    )

    parser.add_argument(
        "-t",
        "--thread",
        type=int,
        required=False,
        default=-1,
        help="concurrent thread number. default is -1, mean auto select",
    )

    parser.add_argument(
        "-v",
        "--variant",
        dest="variant",
        action="store_true",
        default=False,
        help="scan other api keys like openai",
    )

    parser.add_argument(
        "-w",
        "--workspace",
        type=str,
        default=PATH,
        required=False,
        help="workspace path",
    )

    parser.add_argument(
        "-x",
        "--stabilityai",
        dest="stabilityai",
        action="store_true",
        default=False,
        help="scan stability.ai api keys",
    )

    parser.add_argument(
        "-y",
        "--gooeyai",
        dest="gooeyai",
        action="store_true",
        default=False,
        help="scan gooeyai api keys",
    )

    parser.add_argument(
        "-pb",
        "--provider-base",
        dest="pb",
        type=str,
        default="",
        required=False,
        help="base url, must be a valid url start with 'http://' or 'https://'",
    )

    parser.add_argument(
        "-pc",
        "--provider-chat",
        dest="pc",
        type=str,
        default="",
        required=False,
        help="chat api path, default is '/v1/chat/completions'",
    )

    parser.add_argument(
        "-pk",
        "--provider-key",
        dest="pk",
        type=str,
        default="",
        required=False,
        help="The key name of the request header used for authentication, default is 'Authorization'",
    )

    parser.add_argument(
        "-pl",
        "--provider-list",
        dest="pl",
        type=str,
        default="",
        required=False,
        help="list models api path, default is '/v1/models'",
    )

    parser.add_argument(
        "-pm",
        "--provider-model",
        dest="pm",
        type=str,
        default="",
        required=False,
        help="default model name",
    )

    parser.add_argument(
        "-pn",
        "--provider-name",
        dest="pn",
        type=str,
        default="",
        required=False,
        help="provider name, contain only letters, numbers, '_' and '-'",
    )

    parser.add_argument(
        "-pq",
        "--provider-query",
        dest="pq",
        type=str,
        nargs="+",
        default="",
        required=False,
        help="query syntax for github search",
    )

    parser.add_argument(
        "-pp",
        "--provider-pattern",
        dest="pp",
        type=str,
        default="",
        required=False,
        help=r"pattern for extract keys from code, default is 'sk-[a-zA-Z0-9_\-]{48}'",
    )

    main(parser.parse_args())
