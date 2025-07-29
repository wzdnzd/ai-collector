#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the search engine.
"""

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import get_utils_logger

# Get utils logger
logger = get_utils_logger()


def trim(text: str) -> str:
    """Trim whitespace from text, return empty string if invalid."""
    if not text or type(text) != str:
        return ""
    return text.strip()


def isblank(text: str) -> bool:
    """Check if text is blank or invalid."""
    return not text or type(text) != str or not text.strip()


def encoding_url(url: str) -> str:
    """Encode Chinese characters in URL to punycode."""
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
