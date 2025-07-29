#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GooeyAI provider implementation.
"""

import urllib.parse
from typing import Dict, List, Optional

from constants import USER_AGENT
from models import CheckResult, Condition, ErrorReason

from utils import trim

from ..client import chat
from .base import Provider


class GooeyAIProvider(Provider):
    """GooeyAI provider implementation."""

    def __init__(self, conditions: List[Condition], default_model: str = "", skip_search: bool = False):
        default_model = trim(default_model) or "gpt_4_o_mini"
        base_url = "https://api.gooey.ai"
        sub_path = "/v2/google-gpt"

        super().__init__("gooeyai", base_url, sub_path, "", default_model, conditions, skip_search=skip_search)

    def _get_headers(self, token: str, additional: Optional[Dict] = None) -> Optional[Dict]:
        """Get headers for GooeyAI API requests."""
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
        """Check GooeyAI token validity."""
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

    def list_models(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available GooeyAI models."""
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
