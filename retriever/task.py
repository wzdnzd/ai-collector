#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task classes for the asynchronous pipeline system.
Each task carries provider identification for proper routing and result isolation.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

from constants import PATTERN_ADDRESS, PATTERN_ENDPOINT, PATTERN_KEY, PATTERN_MODEL
from models import Service

from logger import get_task_manager_logger

# Get task manager logger
logger = get_task_manager_logger()


@dataclass
class ProviderTask(ABC):
    """Base class for all provider-specific tasks"""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    provider: str = ""  # Provider name for routing and isolation
    created_at: float = field(default_factory=time.time)
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary for persistence"""
        return {
            "type": self.__class__.__name__,
            "task_id": self.task_id,
            "provider": self.provider,
            "created_at": self.created_at,
            "attempts": self.attempts,
            "data": self._serialize_data(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderTask":
        """Deserialize task from dictionary"""
        instance = cls.__new__(cls)
        instance.task_id = data["task_id"]
        instance.provider = data["provider"]
        instance.created_at = data["created_at"]
        instance.attempts = data["attempts"]
        instance._deserialize_data(data["data"])
        return instance

    @abstractmethod
    def _serialize_data(self) -> Dict[str, Any]:
        """Serialize task-specific data"""
        pass

    @abstractmethod
    def _deserialize_data(self, data: Dict[str, Any]) -> None:
        """Deserialize task-specific data"""
        pass


@dataclass
class SearchTask(ProviderTask):
    """Task for searching GitHub for potential API keys"""

    query: str = ""
    regex: str = ""
    page: int = 1
    use_api: bool = False
    address_pattern: str = ""
    endpoint_pattern: str = ""
    model_pattern: str = ""

    def _serialize_data(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "regex": self.regex,
            "page": self.page,
            "use_api": self.use_api,
            "address_pattern": self.address_pattern,
            "endpoint_pattern": self.endpoint_pattern,
            "model_pattern": self.model_pattern,
        }

    def _deserialize_data(self, data: Dict[str, Any]) -> None:
        self.query = data["query"]
        self.regex = data.get("regex", "")
        self.page = data["page"]
        self.use_api = data.get("use_api", False)
        self.address_pattern = data.get("address_pattern", "")
        self.endpoint_pattern = data.get("endpoint_pattern", "")
        self.model_pattern = data.get("model_pattern", "")


@dataclass
class CollectTask(ProviderTask):
    """Task for collecting API keys from discovered URLs"""

    url: str = ""
    key_pattern: str = ""
    address_pattern: str = ""
    endpoint_pattern: str = ""
    model_pattern: str = ""
    retries: int = 3

    def _serialize_data(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            PATTERN_KEY: self.key_pattern,
            PATTERN_ADDRESS: self.address_pattern,
            PATTERN_ENDPOINT: self.endpoint_pattern,
            PATTERN_MODEL: self.model_pattern,
            "retries": self.retries,
        }

    def _deserialize_data(self, data: Dict[str, Any]) -> None:
        self.url = data["url"]
        self.key_pattern = data[PATTERN_KEY]
        self.address_pattern = data.get(PATTERN_ADDRESS, "")
        self.endpoint_pattern = data.get(PATTERN_ENDPOINT, "")
        self.model_pattern = data.get(PATTERN_MODEL, "")
        self.retries = data.get("retries", 3)


@dataclass
class CheckTask(ProviderTask):
    """Task for validating API keys"""

    service: Optional[Service] = None
    custom_url: str = ""
    custom_headers: Dict[str, str] = field(default_factory=dict)

    def _serialize_data(self) -> Dict[str, Any]:
        return {
            "service": self.service.serialize() if self.service else "",
            "custom_url": self.custom_url,
            "custom_headers": self.custom_headers,
        }

    def _deserialize_data(self, data: Dict[str, Any]) -> None:
        service_data = data.get("service", "")
        self.service = Service.deserialize(service_data) if service_data else None
        self.custom_url = data.get("custom_url", "")
        self.custom_headers = data.get("custom_headers", {})


@dataclass
class ModelsTask(ProviderTask):
    """Task for retrieving available models for a validated API key"""

    service: Optional[Service] = None

    def _serialize_data(self) -> Dict[str, Any]:
        return {"service": self.service.serialize() if self.service else ""}

    def _deserialize_data(self, data: Dict[str, Any]) -> None:
        service_data = data.get("service", "")
        self.service = Service.deserialize(service_data) if service_data else None


class TaskFactory:
    """Factory for creating tasks from configuration and serialized data"""

    @staticmethod
    def create_search_task(
        provider: str,
        query: str,
        regex: str = "",
        page: int = 1,
        use_api: bool = False,
        address_pattern: str = "",
        endpoint_pattern: str = "",
        model_pattern: str = "",
    ) -> SearchTask:
        """Create a search task"""
        return SearchTask(
            provider=provider,
            query=query,
            regex=regex,
            page=page,
            use_api=use_api,
            address_pattern=address_pattern,
            endpoint_pattern=endpoint_pattern,
            model_pattern=model_pattern,
        )

    @staticmethod
    def create_collect_task(provider: str, url: str, patterns: Dict[str, str]) -> CollectTask:
        """Create a collect task with extraction patterns"""
        return CollectTask(
            provider=provider,
            url=url,
            key_pattern=patterns.get(PATTERN_KEY, ""),
            address_pattern=patterns.get(PATTERN_ADDRESS, ""),
            endpoint_pattern=patterns.get(PATTERN_ENDPOINT, ""),
            model_pattern=patterns.get(PATTERN_MODEL, ""),
        )

    @staticmethod
    def create_check_task(provider: str, service: Service) -> CheckTask:
        """Create a check task for API key validation"""
        return CheckTask(provider=provider, service=service)

    @staticmethod
    def create_models_task(provider: str, service: Service) -> ModelsTask:
        """Create a models task for retrieving available models"""
        return ModelsTask(provider=provider, service=service)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ProviderTask:
        """Create task from serialized dictionary"""
        task_type = data.get("type", "")

        if task_type == "SearchTask":
            return SearchTask.from_dict(data)
        elif task_type == "CollectTask":
            return CollectTask.from_dict(data)
        elif task_type == "CheckTask":
            return CheckTask.from_dict(data)
        elif task_type == "ModelsTask":
            return ModelsTask.from_dict(data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")


# Task type registry for serialization
TASK_TYPES: Dict[str, Type[ProviderTask]] = {
    "SearchTask": SearchTask,
    "CollectTask": CollectTask,
    "CheckTask": CheckTask,
    "ModelsTask": ModelsTask,
}


def get_task_class(task_type: str) -> Optional[Type[ProviderTask]]:
    """Get task class by type name"""
    return TASK_TYPES.get(task_type)


if __name__ == "__main__":
    # Test task serialization
    task = TaskFactory.create_search_task("openai", '"T3BlbkFJ"', 1)

    # Serialize
    data = task.to_dict()
    logger.info(f"Serialized: {data}")

    # Deserialize
    restored = TaskFactory.from_dict(data)
    logger.info(f"Restored: {restored}")

    assert task.task_id == restored.task_id
    assert task.provider == restored.provider
    assert task.query == restored.query
    logger.info("Task serialization test passed!")
