#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration system for multi-provider API key scanner.
Supports YAML-based configuration with provider type enumeration and flexible conditions.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import constants
import models
import yaml

from logger import get_config_logger

# Get configuration module logger
logger = get_config_logger()


class ProviderType(Enum):
    """Provider interface type enumeration"""

    OPENAI_LIKE = constants.PROVIDER_TYPE_OPENAI_LIKE  # OpenAI compatible APIs (OpenAI, Azure, Doubao, QianFan)
    ANTHROPIC = constants.PROVIDER_TYPE_ANTHROPIC  # Anthropic Claude API
    GEMINI = constants.PROVIDER_TYPE_GEMINI  # Google Gemini API
    GOOEY_AI = constants.PROVIDER_TYPE_GOOEY_AI  # GooeyAI API
    STABILITY_AI = constants.PROVIDER_TYPE_STABILITY_AI  # StabilityAI API
    CUSTOM = constants.PROVIDER_TYPE_CUSTOM  # Custom API implementation


@dataclass
class ApiConfig:
    """API configuration for a provider"""

    base_url: str = ""
    completion_path: str = constants.DEFAULT_COMPLETION_PATH
    model_path: str = constants.DEFAULT_MODEL_PATH
    default_model: str = ""
    auth_key: str = constants.DEFAULT_AUTHORIZATION_HEADER
    extra_headers: Dict[str, str] = field(default_factory=dict)
    api_version: str = ""  # For Azure and other versioned APIs
    timeout: int = constants.DEFAULT_TIMEOUT
    retries: int = constants.DEFAULT_RETRIES


@dataclass
class Patterns:
    """Extraction patterns for keys and metadata"""

    key_pattern: str = ""
    address_pattern: str = ""
    endpoint_pattern: str = ""
    model_pattern: str = ""


@dataclass
class RateLimit:
    """Rate limiting configuration"""

    rate: float = 1.0  # requests per second
    burst: int = 5  # burst size
    adaptive: bool = True  # enable adaptive rate adjustment


@dataclass
class TaskConfig:
    """Configuration for a single provider task"""

    name: str = ""
    enabled: bool = True
    provider_type: str = ""
    use_api: bool = False
    skip_search: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)
    api: ApiConfig = field(default_factory=ApiConfig)
    patterns: Patterns = field(default_factory=Patterns)
    conditions: List[Dict[str, str]] = field(default_factory=list)
    rate_limit: RateLimit = field(default_factory=RateLimit)


@dataclass
class GlobalConfig:
    """Global application configuration"""

    workspace: str = constants.DEFAULT_WORKSPACE_DIR
    session: str = ""  # GitHub session for web search
    token: str = ""  # GitHub token for API search
    max_retry: int = 2  #  Maximum number of retries for failed requests


@dataclass
class PipelineConfig:
    """Pipeline stage configuration"""

    threads: Dict[str, int] = field(default_factory=lambda: constants.DEFAULT_THREAD_COUNTS.copy())
    queue_sizes: Dict[str, int] = field(
        default_factory=lambda: {
            constants.STAGE_NAME_SEARCH: constants.DEFAULT_SEARCH_QUEUE_SIZE,
            constants.STAGE_NAME_COLLECT: constants.DEFAULT_COLLECT_QUEUE_SIZE,
            constants.STAGE_NAME_CHECK: constants.DEFAULT_CHECK_QUEUE_SIZE,
            constants.STAGE_NAME_MODELS: constants.DEFAULT_MODELS_QUEUE_SIZE,
        }
    )


@dataclass
class PersistenceConfig:
    """Persistence and recovery configuration"""

    batch_size: int = constants.DEFAULT_BATCH_SIZE
    save_interval: int = constants.DEFAULT_SAVE_INTERVAL
    queue_interval: int = constants.DEFAULT_QUEUE_INTERVAL
    auto_restore: bool = True
    shutdown_timeout: int = constants.DEFAULT_SHUTDOWN_TIMEOUT


@dataclass
class MonitoringConfig:
    """Monitoring and statistics configuration"""

    stats_interval: int = constants.DEFAULT_STATS_INTERVAL
    show_stats: bool = True
    alert_error_rate: float = 0.1
    alert_queue_size: int = 1000


@dataclass
class Config:
    """Main configuration container"""

    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    rate_limits: Dict[str, RateLimit] = field(
        default_factory=lambda: {
            constants.SERVICE_TYPE_GITHUB_API: RateLimit(rate=0.15, burst=3, adaptive=True),  # ~9 RPM, conservative
            constants.SERVICE_TYPE_GITHUB_WEB: RateLimit(rate=0.5, burst=2, adaptive=True),
        }
    )
    tasks: List[TaskConfig] = field(default_factory=list)


class ConfigParser:
    """Configuration file parser and validator"""

    def __init__(self, config_file: str = constants.DEFAULT_CONFIG_FILE) -> None:
        self.config_file = config_file
        self.config: Optional[Config] = None

    def load(self) -> Config:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_file):
            self._create_default_config()

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            self.config = self._parse_config(data)
            self._validate_config()
            return self.config

        except Exception as e:
            logger.error(f"Failed to load config file {self.config_file}: {e}")
            raise

    def _parse_config(self, data: Dict[str, Any]) -> Config:
        """Parse YAML data into Config object"""
        config = Config()

        # Parse global configuration
        if "global" in data:
            global_data = data["global"]
            config.global_config = GlobalConfig(
                workspace=os.path.abspath(global_data.get("workspace", "./workspace")),
                session=global_data.get("session", ""),
                token=global_data.get("token", ""),
                max_retry=global_data.get("max_retry", constants.DEFAULT_MAX_RETRY),
            )

        # Parse pipeline configuration
        if "pipeline" in data:
            pipeline_data = data["pipeline"]
            config.pipeline = PipelineConfig(
                threads=pipeline_data.get("threads", config.pipeline.threads),
                queue_sizes=pipeline_data.get("queue_sizes", config.pipeline.queue_sizes),
            )

        # Parse persistence configuration
        if "persistence" in data:
            persist_data = data["persistence"]
            config.persistence = PersistenceConfig(
                batch_size=persist_data.get("batch_size", constants.DEFAULT_BATCH_SIZE),
                save_interval=persist_data.get("save_interval", constants.DEFAULT_SAVE_INTERVAL),
                queue_interval=persist_data.get("queue_interval", constants.DEFAULT_QUEUE_INTERVAL),
                auto_restore=persist_data.get("auto_restore", True),
                shutdown_timeout=persist_data.get("shutdown_timeout", constants.DEFAULT_SHUTDOWN_TIMEOUT),
            )

        # Parse monitoring configuration
        if "monitoring" in data:
            monitor_data = data["monitoring"]
            config.monitoring = MonitoringConfig(
                stats_interval=monitor_data.get("stats_interval", constants.DEFAULT_STATS_INTERVAL),
                show_stats=monitor_data.get("show_stats", True),
                alert_error_rate=monitor_data.get("alert_error_rate", 0.1),
                alert_queue_size=monitor_data.get("alert_queue_size", 1000),
            )

        # Parse rate limits
        if "rate_limits" in data:
            for name, limit_data in data["rate_limits"].items():
                config.rate_limits[name] = RateLimit(
                    rate=limit_data.get("rate", 1.0),
                    burst=limit_data.get("burst", 5),
                    adaptive=limit_data.get("adaptive", True),
                )

        # Parse tasks
        if "tasks" in data:
            for task_data in data["tasks"]:
                config.tasks.append(self._parse_task(task_data))

        return config

    def _parse_task(self, task_data: Dict[str, Any]) -> TaskConfig:
        """Parse a single task configuration"""
        task = TaskConfig()
        task.name = task_data.get("name", "")
        task.enabled = task_data.get("enabled", True)
        task.provider_type = task_data.get("provider_type", "")
        task.use_api = task_data.get("use_api", False)
        task.skip_search = task_data.get("skip_search", False)
        task.extras = task_data.get("extras", {})

        # Parse API configuration
        if "api" in task_data:
            api_data = task_data["api"]
            task.api = ApiConfig(
                base_url=api_data.get("base_url", ""),
                completion_path=api_data.get("completion_path", constants.DEFAULT_COMPLETION_PATH),
                model_path=api_data.get("model_path", constants.DEFAULT_MODEL_PATH),
                default_model=api_data.get("default_model", ""),
                auth_key=api_data.get("auth_key", constants.DEFAULT_AUTHORIZATION_HEADER),
                extra_headers=api_data.get("extra_headers", {}),
                api_version=api_data.get("api_version", ""),
                timeout=api_data.get("timeout", constants.DEFAULT_TIMEOUT),
                retries=api_data.get("retries", constants.DEFAULT_RETRIES),
            )

        # Parse extraction patterns
        if "patterns" in task_data:
            pattern_data = task_data["patterns"]
            task.patterns = Patterns(
                key_pattern=pattern_data.get(constants.PATTERN_KEY, ""),
                address_pattern=pattern_data.get(constants.PATTERN_ADDRESS, ""),
                endpoint_pattern=pattern_data.get(constants.PATTERN_ENDPOINT, ""),
                model_pattern=pattern_data.get(constants.PATTERN_MODEL, ""),
            )

        # Parse conditions with flexible regex support
        if "conditions" in task_data:
            task.conditions = task_data["conditions"]

        # Parse rate limit
        if "rate_limit" in task_data:
            limit_data = task_data["rate_limit"]
            task.rate_limit = RateLimit(
                rate=limit_data.get("rate", 1.0),
                burst=limit_data.get("burst", 5),
                adaptive=limit_data.get("adaptive", True),
            )

        return task

    def parse_conditions(self, task: TaskConfig) -> List[models.Condition]:
        """Parse flexible condition configuration"""
        conditions: List[models.Condition] = []
        global_pattern = task.patterns.key_pattern

        for condition_data in task.conditions:
            query = condition_data.get("query", "")
            # Use condition-specific pattern if provided, otherwise use global pattern
            regex = condition_data.get("key_pattern", global_pattern)

            if regex:  # regex is required, query is optional
                conditions.append(models.Condition(regex=regex, query=query))
            else:
                logger.warning(f"Invalid condition (missing regex pattern): {condition_data}")

        return conditions

    def _validate_config(self) -> None:
        """Validate configuration completeness and correctness"""
        if not self.config:
            raise ValueError("Configuration is None")

        if not self.config.tasks:
            raise ValueError("At least one task must be configured")

        enabled_tasks = [t for t in self.config.tasks if t.enabled]
        if not enabled_tasks:
            raise ValueError("At least one task must be enabled")

        # Validate provider types
        valid_types = [t.value for t in ProviderType]
        for task in self.config.tasks:
            if task.provider_type not in valid_types:
                raise ValueError(
                    f"Invalid provider_type '{task.provider_type}' in task '{task.name}'. "
                    f"Valid types: {valid_types}"
                )

            if not task.skip_search and not self.config.global_config.session and not self.config.global_config.token:
                raise ValueError("Either GitHub session or token is required in global config")

        # Validate task names are unique
        names = [t.name for t in self.config.tasks if t.enabled]
        if len(names) != len(set(names)):
            raise ValueError("Task names must be unique")

    def _create_default_config(self) -> None:
        """Create a default configuration file"""
        default_config = {
            "global": {
                "workspace": constants.DEFAULT_WORKSPACE_DIR,
                "session": "your_github_session_here",
                "token": "your_github_token_here",
                "max_retry": constants.DEFAULT_MAX_RETRY,
            },
            "pipeline": {"threads": constants.DEFAULT_THREAD_COUNTS.copy()},
            "persistence": {
                "batch_size": constants.DEFAULT_BATCH_SIZE,
                "save_interval": constants.DEFAULT_SAVE_INTERVAL,
                "queue_interval": constants.DEFAULT_QUEUE_INTERVAL,
                "auto_restore": True,
                "shutdown_timeout": constants.DEFAULT_SHUTDOWN_TIMEOUT,
            },
            "rate_limits": {
                constants.SERVICE_TYPE_GITHUB_API: {"rate": 0.15, "burst": 3, "adaptive": True},  # ~9 RPM, conservative
                constants.SERVICE_TYPE_GITHUB_WEB: {"rate": 0.5, "burst": 2, "adaptive": True},
            },
            "tasks": [
                {
                    "name": "openai",
                    "enabled": True,
                    "provider_type": "openai_like",
                    "use_api": False,
                    "skip_search": False,
                    "api": {
                        "base_url": "https://api.openai.com",
                        "completion_path": "/v1/chat/completions",
                        "model_path": "/v1/models",
                        "default_model": "gpt-4o-mini",
                        "auth_key": "Authorization",
                    },
                    "patterns": {"key_pattern": "sk(?:-proj)?-[a-zA-Z0-9]{20}T3BlbkFJ[a-zA-Z0-9]{20}"},
                    "conditions": [{"query": '"T3BlbkFJ"'}],
                    "rate_limit": {"rate": 2.0, "burst": 10},
                }
            ],
            "monitoring": {"stats_interval": 5, "show_stats": True},
        }

        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True, indent=2)

        logger.info(f"Created default configuration file: {self.config_file}")
        logger.info(
            "Please edit the configuration file and set your GitHub session (for web search) and/or token (for API search)"
        )


def load_config(config_file: str = constants.DEFAULT_CONFIG_FILE) -> Config:
    """Convenience function to load configuration"""
    parser = ConfigParser(config_file)
    return parser.load()


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config()
        logger.info(f"Loaded configuration with {len(config.tasks)} tasks")
        for task in config.tasks:
            if task.enabled:
                logger.info(f"  - {task.name} ({task.provider_type})")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
