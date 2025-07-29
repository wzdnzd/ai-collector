#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration settings for regex engine components.
"""

import os
from dataclasses import dataclass


@dataclass
class RefineEngineConfig:
    """Configuration for regex engine components."""

    # RefineEngine settings
    max_queries: int = int(os.getenv("REGEX_MAX_QUERIES", "100000000"))
    max_depth: int = int(os.getenv("REGEX_MAX_DEPTH", "3"))
    max_quantifier_length: int = int(os.getenv("REGEX_MAX_QUANTIFIER_LENGTH", "150"))

    # SplittabilityAnalyzer settings - use pure mathematical approach by default
    enable_recursion_limit: bool = os.getenv("REGEX_ENABLE_RECURSION_LIMIT", "false").lower() == "true"
    enable_value_threshold: bool = os.getenv("REGEX_ENABLE_VALUE_THRESHOLD", "false").lower() == "true"
    enable_resource_limit: bool = os.getenv("REGEX_ENABLE_RESOURCE_LIMIT", "false").lower() == "true"
    max_recursion_depth: int = int(os.getenv("REGEX_MAX_RECURSION_DEPTH", "10"))
    min_enumeration_value: float = float(os.getenv("REGEX_MIN_ENUMERATION_VALUE", "0.01"))
    max_resource_cost: float = float(os.getenv("REGEX_MAX_RESOURCE_COST", "50.0"))

    def __post_init__(self):
        """Validate configuration values after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration values."""
        # Validate positive integers
        for field_name in ["max_queries", "max_depth", "max_quantifier_length", "max_recursion_depth"]:
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Configuration '{field_name}' must be a positive integer, got: {value}")

        # Validate positive floats
        for field_name in ["min_enumeration_value", "max_resource_cost"]:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"Configuration '{field_name}' must be a positive number, got: {value}")

        # Validate booleans
        for field_name in ["enable_recursion_limit", "enable_value_threshold", "enable_resource_limit"]:
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise ValueError(f"Configuration '{field_name}' must be a boolean, got: {value}")

    @classmethod
    def create_with_overrides(cls, **overrides) -> "RefineEngineConfig":
        """
        Create configuration instance with optional overrides.

        Args:
            **overrides: Configuration field overrides

        Returns:
            RegexEngineConfig: Configuration instance with overrides applied
        """
        # Get default values
        config = cls()

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown configuration field: {key}")

        # Re-validate after overrides
        config._validate()

        return config
