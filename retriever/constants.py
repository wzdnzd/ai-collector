#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constants used across the async pipeline system.
Only contains truly necessary constants that are used across multiple files.
"""

import ssl
from typing import Dict, List, Set

# Configuration files
DEFAULT_CONFIG_FILE: str = "config.yaml"
DEFAULT_WORKSPACE_DIR: str = "./data"

# Log levels
LOG_LEVEL_DEBUG: str = "DEBUG"
LOG_LEVEL_INFO: str = "INFO"
LOG_LEVEL_WARNING: str = "WARNING"
LOG_LEVEL_ERROR: str = "ERROR"

# Application banner
APPLICATION_BANNER: str = """
╔══════════════════════════════════════════════════════════════╗
║                    Async Pipeline System                     ║
║              Multi-Provider API Key Discovery                ║
╚══════════════════════════════════════════════════════════════╝
"""

# Default intervals and timeouts
DEFAULT_STATS_INTERVAL: int = 10
DEFAULT_SHUTDOWN_TIMEOUT: float = 30.0

# Provider types
PROVIDER_TYPE_OPENAI_LIKE: str = "openai_like"
PROVIDER_TYPE_ANTHROPIC: str = "anthropic"
PROVIDER_TYPE_BEDROCK: str = "bedrock"
PROVIDER_TYPE_GEMINI: str = "gemini"
PROVIDER_TYPE_GOOEY_AI: str = "gooey_ai"
PROVIDER_TYPE_STABILITY_AI: str = "stability_ai"
PROVIDER_TYPE_VERTEX: str = "vertex"
PROVIDER_TYPE_CUSTOM: str = "custom"

# Default configuration values
DEFAULT_BATCH_SIZE: int = 50
DEFAULT_SAVE_INTERVAL: int = 30
DEFAULT_QUEUE_INTERVAL: int = 60
DEFAULT_RETRIES: int = 3
DEFAULT_TIMEOUT: int = 30

# Pipeline stage names
STAGE_NAME_SEARCH: str = "search"
STAGE_NAME_COLLECT: str = "collect"
STAGE_NAME_CHECK: str = "check"
STAGE_NAME_MODELS: str = "models"

# Queue and thread defaults
DEFAULT_THREAD_COUNTS: Dict[str, int] = {
    STAGE_NAME_SEARCH: 1,
    STAGE_NAME_COLLECT: 8,
    STAGE_NAME_CHECK: 4,
    STAGE_NAME_MODELS: 2,
}

# Queue size for stage
DEFAULT_SEARCH_QUEUE_SIZE: int = 100000
DEFAULT_COLLECT_QUEUE_SIZE: int = 200000
DEFAULT_CHECK_QUEUE_SIZE: int = 500000
DEFAULT_MODELS_QUEUE_SIZE: int = 1000000

# Monitoring defaults
DEFAULT_MEMORY_THRESHOLD: int = 1024 * 1024 * 1024  # 1GB
DEFAULT_ERROR_RATE_THRESHOLD_APP: float = 0.15
DEFAULT_QUEUE_SIZE_THRESHOLD_APP: int = 1000

# Service type constants
SERVICE_TYPE_GITHUB_API: str = "github_api"
SERVICE_TYPE_GITHUB_WEB: str = "github_web"
PROVIDER_SERVICE_PREFIX: str = "provider"

# Queue state constants
QUEUE_STATE_PROVIDER_MULTI: str = "multi"
QUEUE_STATE_MAX_AGE_HOURS: int = 24

# Alert timing constants
ALERT_COOLDOWN_SECONDS: int = 300  # 5 minutes

# Load balancer constants
DEFAULT_MIN_WORKERS: int = 1
DEFAULT_MAX_WORKERS: int = 16
DEFAULT_TARGET_QUEUE_SIZE: int = 20
DEFAULT_ADJUSTMENT_INTERVAL: float = 30.0
DEFAULT_SCALE_UP_THRESHOLD: float = 0.8
DEFAULT_SCALE_DOWN_THRESHOLD: float = 0.3
LB_RECENT_HISTORY_SIZE: int = 5

# Default API paths
DEFAULT_COMPLETION_PATH: str = "/v1/chat/completions"
DEFAULT_MODEL_PATH: str = "/v1/models"
DEFAULT_AUTHORIZATION_HEADER: str = "Authorization"

# Progress display constants
PROGRESS_UPDATE_INTERVAL: int = 10

# Pattern keys
PATTERN_KEY: str = "key_pattern"
PATTERN_ADDRESS: str = "address_pattern"
PATTERN_ENDPOINT: str = "endpoint_pattern"
PATTERN_MODEL: str = "model_pattern"

# SSL Context
CTX: ssl.SSLContext = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

# HTTP Configuration
USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
}

# API Configuration
DEFAULT_QUESTION: str = "Hello"

# Error handling
NO_RETRY_ERROR_CODES: set = {400, 401, 402, 404, 422}

# Language popularity tiers for LLM/API key domain with time step multipliers
LANGUAGE_TIERS: Dict[str, Dict] = {
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
LANGUAGE_TO_TIER: Dict[str, str] = {}
for name, data in LANGUAGE_TIERS.items():
    for lang in data["languages"]:
        LANGUAGE_TO_TIER[lang] = name

# Flatten all languages for backward compatibility
POPULAR_LANGUAGES: List[str] = []
for data in LANGUAGE_TIERS.values():
    POPULAR_LANGUAGES.extend(data["languages"])

# Language to extension mapping to avoid invalid combinations
LANGUAGE_EXTENSIONS: Dict[str, List[str]] = {
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
SIZE_RANGES: List[str] = [
    "<1000",  # < 1KB
    "1000..5000",  # 1-5KB
    "5000..20000",  # 5-20KB
    "20000..100000",  # 20-100KB
    ">100000",  # > 100KB
]

# Search limits
API_MAX_PAGES: int = 10
WEB_MAX_PAGES: int = 5
API_RESULTS_PER_PAGE: int = 100
WEB_RESULTS_PER_PAGE: int = 20
API_LIMIT: int = API_MAX_PAGES * API_RESULTS_PER_PAGE
WEB_LIMIT: int = WEB_MAX_PAGES * WEB_RESULTS_PER_PAGE

# Result categories
RESULT_CATEGORY_CHECK_TASKS: str = "check_tasks"
RESULT_CATEGORY_COLLECT_TASKS: str = "collect_tasks"
RESULT_CATEGORY_MATERIAL_KEYS: str = "material_keys"
RESULT_CATEGORY_VALID_KEYS: str = "valid_keys"
RESULT_CATEGORY_INVALID_KEYS: str = "invalid_keys"
RESULT_CATEGORY_NO_QUOTA_KEYS: str = "no_quota_keys"
RESULT_CATEGORY_WAIT_CHECK_KEYS: str = "wait_check_keys"

# Github code search syntax
ALLOWED_OPERATORS: Set[str] = set(["AND", "OR", "NOT", "AND NOT"])

# Maximum number of times to re-queue when processing fails
DEFAULT_MAX_RETRIES_REQUEUED: int = 3
