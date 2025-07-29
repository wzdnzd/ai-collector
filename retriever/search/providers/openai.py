#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenAI provider implementation.
"""

from typing import List

from models import Condition
from utils import trim

from .openai_like import OpenAILikeProvider


class OpenAIProvider(OpenAILikeProvider):
    """OpenAI provider implementation."""

    def __init__(self, conditions: List[Condition], default_model: str = "", skip_search: bool = False):
        default_model = trim(default_model) or "gpt-4o-mini"
        base_url = "https://api.openai.com"

        super().__init__("openai", base_url, default_model, conditions, skip_search=skip_search)
