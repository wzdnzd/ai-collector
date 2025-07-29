"""
Advanced regex pattern parsing and enumeration system for search optimization.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced regex pattern parsing and enumeration system for search optimization.
"""

from .engine import RefineEngine
from .generator import QueryGenerator
from .optimizer import EnumerationOptimizer
from .parser import RegexParser
from .segments import (
    CharClassSegment,
    EnumerationStrategy,
    FixedSegment,
    GroupSegment,
    OptionalSegment,
    Segment,
)
from .splittability import SplittabilityAnalyzer

__all__ = [
    "RegexParser",
    "EnumerationOptimizer",
    "QueryGenerator",
    "RefineEngine",
    "SplittabilityAnalyzer",
    "Segment",
    "FixedSegment",
    "CharClassSegment",
    "OptionalSegment",
    "GroupSegment",
    "EnumerationStrategy",
]
