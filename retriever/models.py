#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data models and enums for the async pipeline system.
Contains data classes for structured data transfer between components.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Optional


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
class KeyDetail(object):
    # token
    key: str

    # available
    available: bool = False

    # models that the key can access
    models: List[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
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

    def __hash__(self) -> int:
        return hash((self.address, self.endpoint, self.key, self.model))

    def __eq__(self, other: object) -> bool:
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
    def deserialize(cls, text: str) -> Optional["Service"]:
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


@dataclass
class CheckResult(object):
    # whether the key can be used now
    available: bool = False

    # error message if the key cannot be used
    reason: ErrorReason = ErrorReason.UNKNOWN

    @staticmethod
    def ok() -> "CheckResult":
        return CheckResult(available=True, reason=ErrorReason.NONE)

    @staticmethod
    def fail(reason: ErrorReason) -> "CheckResult":
        return CheckResult(available=False, reason=reason)


@dataclass
class Condition(object):
    # pattern for extract key from code
    regex: str

    # search keyword or pattern
    query: str = ""

    def __hash__(self) -> int:
        return hash((self.query, self.regex))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Condition):
            return False

        return self.query == other.query and self.regex == other.regex


@dataclass
class ProviderPatterns:
    """Patterns for provider data extraction"""

    key_pattern: str = ""
    address_pattern: str = ""
    endpoint_pattern: str = ""
    model_pattern: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert patterns to dictionary format"""
        return {
            "key_pattern": self.key_pattern,
            "address_pattern": self.address_pattern,
            "endpoint_pattern": self.endpoint_pattern,
            "model_pattern": self.model_pattern,
        }


@dataclass
class TaskRecoveryInfo:
    """Information for task recovery"""

    queue_tasks: Dict[str, List[Any]] = field(default_factory=dict)
    result_tasks: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)
    total_queue_tasks: int = 0
    total_result_tasks: int = 0


@dataclass
class TaskManagerStats:
    """Task manager statistics"""

    providers: List[str] = field(default_factory=list)
    running: bool = False
    pipeline_stats: Optional["PipelineAllStats"] = None
    result_stats: Optional[Dict[str, "ProviderResultStats"]] = None


@dataclass
class LogFileInfo:
    """Information about a log file"""

    filename: str
    size: str
    modified: str
    path: str
    error: Optional[str] = None


@dataclass
class LoggingStats:
    """Logging system statistics"""

    active_loggers: int
    log_files: Dict[str, LogFileInfo] = field(default_factory=dict)
    logs_directory: Optional[str] = None


@dataclass
class PipelineStageStats:
    """Statistics for a single pipeline stage"""

    name: str
    queue_size: int = 0
    total_processed: int = 0
    total_errors: int = 0
    error_rate: float = 0.0
    runtime: float = 0.0
    processing_rate: float = 0.0
    last_activity: float = 0.0
    workers: int = 0
    running: bool = False


@dataclass
class PipelineAllStats:
    """Statistics for all pipeline stages"""

    search: PipelineStageStats
    collect: PipelineStageStats
    check: PipelineStageStats
    models: PipelineStageStats
    runtime: float = 0.0
    rate_limiter_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ProviderResultStats:
    """Statistics for provider results"""

    valid_keys: int = 0
    invalid_keys: int = 0
    no_quota_keys: int = 0
    wait_check_keys: int = 0
    total_links: int = 0
    total_models: int = 0


@dataclass
class ApplicationStatus:
    """Application status information"""

    timestamp: float
    running: bool
    runtime: float
    shutdown_requested: bool
    task_manager_status: Optional[Any] = None
    monitoring_status: Optional[Any] = None
    load_balancer_status: Optional[Any] = None


@dataclass
class LoadBalancerConfig:
    """Load balancer configuration"""

    min_workers: int
    max_workers: int
    target_queue_size: int
    adjustment_interval: float
    scale_up_threshold: float
    scale_down_threshold: float


@dataclass
class LoadBalancerMetricsUpdate:
    """Load balancer metrics update"""

    queue_size: int
    active_workers: int
    processing_rate: float
    avg_processing_time: float


@dataclass
class MonitoringConfig:
    """Monitoring system configuration"""

    update_interval: float
    error_rate_threshold: float
    queue_size_threshold: int
    memory_threshold: int


@dataclass
class PipelineStatsUpdate:
    """Pipeline statistics update"""

    search_queue: int
    collect_queue: int
    check_queue: int
    models_queue: int
    active_workers: int
    total_workers: int
    is_finished: bool


@dataclass
class ProviderStatsUpdate:
    """Provider statistics update"""

    valid_keys: int
    invalid_keys: int
    no_quota_keys: int
    wait_check_keys: int
    total_links: int
    total_models: int


@dataclass
class ProviderStats:
    """Statistics for a single provider"""

    name: str = ""

    # Task counts
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

    # Result counts
    valid_keys: int = 0
    invalid_keys: int = 0
    no_quota_keys: int = 0
    wait_check_keys: int = 0
    total_links: int = 0
    total_models: int = 0

    # Performance metrics
    avg_processing_time: float = 0.0
    success_rate: float = 0.0
    last_activity: float = 0.0

    # Rate limiting
    rate_limited_count: int = 0
    current_rate: float = 0.0


@dataclass
class PipelineStats:
    """Overall pipeline statistics"""

    start_time: float = field(default_factory=time.time)

    # Stage statistics
    search_queue: int = 0
    collect_queue: int = 0
    check_queue: int = 0
    models_queue: int = 0

    # Worker statistics
    active_workers: int = 0
    total_workers: int = 0

    # Performance
    total_throughput: float = 0.0
    memory_usage: float = 0.0

    # Status
    is_running: bool = False
    is_finished: bool = False


@dataclass
class WorkerMetrics:
    """Metrics for worker performance analysis"""

    stage_name: str = ""
    current_workers: int = 0
    target_workers: int = 0
    queue_size: int = 0
    processing_rate: float = 0.0
    avg_processing_time: float = 0.0
    utilization: float = 0.0
    last_adjustment: float = 0.0


@dataclass
class ResultStats:
    """Statistics for result processing"""

    valid_keys: int = 0
    no_quota_keys: int = 0
    wait_check_keys: int = 0
    invalid_keys: int = 0
    material_keys: int = 0
    total_links: int = 0
    models_count: int = 0
    start_time: float = field(default_factory=time.time)
    last_save: float = field(default_factory=time.time)


@dataclass
class SearchTaskResult:
    """Result from search task execution"""

    links: List[str] = field(default_factory=list)
    total: Optional[int] = None


@dataclass
class CollectTaskResult:
    """Result from collect task execution"""

    services: List[Service] = field(default_factory=list)


@dataclass
class CheckTaskResult:
    """Result from check task execution"""

    valid_keys: List[Service] = field(default_factory=list)
    invalid_keys: List[Service] = field(default_factory=list)
    no_quota_keys: List[Service] = field(default_factory=list)
    wait_check_keys: List[Service] = field(default_factory=list)


@dataclass
class ModelsTaskResult:
    """Result from models task execution"""

    models: List[str] = field(default_factory=list)
