#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Provider implementations for search engine.
"""

from .anthropic import AnthropicProvider
from .azure import AzureOpenAIProvider
from .base import Provider
from .doubao import DoubaoProvider
from .gemini import GeminiProvider
from .gooeyai import GooeyAIProvider
from .openai import OpenAIProvider
from .openai_like import OpenAILikeProvider
from .qianfan import QianFanProvider
from .stabilityai import StabilityAIProvider

__all__ = [
    "Provider",
    "OpenAILikeProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "DoubaoProvider",
    "GeminiProvider",
    "GooeyAIProvider",
    "QianFanProvider",
    "StabilityAIProvider",
]
