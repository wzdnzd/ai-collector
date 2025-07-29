#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main regex engine interface for pattern analysis and query generation.
"""

import re
import threading
from typing import List, Optional

from constants import ALLOWED_OPERATORS, POPULAR_LANGUAGES, SIZE_RANGES
from logger import get_engine_logger

from .config import RefineEngineConfig
from .generator import QueryGenerator
from .optimizer import EnumerationOptimizer
from .parser import RegexParser
from .segments import CharClassSegment, FixedSegment, GroupSegment, OptionalSegment
from .splittability import SplittabilityAnalyzer

logger = get_engine_logger()


class RefineEngine:
    """Main interface for regex pattern processing - Singleton pattern."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, config: Optional[RefineEngineConfig] = None):
        if config is None:
            config = RefineEngineConfig()

        self.config = config
        self.parser = RegexParser.get_instance(config.max_quantifier_length)
        self.optimizer = EnumerationOptimizer.get_instance(config.max_queries)
        self.generator = QueryGenerator.get_instance(config.max_depth)
        self.splittability = SplittabilityAnalyzer.get_instance(
            enable_recursion_limit=config.enable_recursion_limit,
            enable_value_threshold=config.enable_value_threshold,
            enable_resource_limit=config.enable_resource_limit,
            max_recursion_depth=config.max_recursion_depth,
            min_enumeration_value=config.min_enumeration_value,
            max_resource_cost=config.max_resource_cost,
        )

    @classmethod
    def get_instance(cls, config: Optional[RefineEngineConfig] = None) -> "RefineEngine":
        """Get singleton instance with optional configuration."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance.__init__(config)
                    logger.debug("RefineEngine singleton instance initialized")
        return cls._instance

    def _extract_regex_pattern(self, query: str) -> Optional[str]:
        """
        Extract regex pattern from GitHub search format, handling escaped slashes.

        Args:
            query: Query string in format "/pattern/" or "/pattern/ other terms" or "terms AND /pattern/"

        Returns:
            str: Extracted pattern or None if not found
        """
        if not query:
            return None

        # For simple /pattern/ format, use direct extraction
        if query.startswith("/") and query.count("/") >= 2:
            # Find the closing slash, accounting for escaped slashes
            i = 1  # Start after the opening slash
            while i < len(query):
                if query[i] == "/":
                    # Check if this slash is escaped
                    escape_count = 0
                    j = i - 1
                    while j >= 0 and query[j] == "\\":
                        escape_count += 1
                        j -= 1

                    # If even number of backslashes (including 0), slash is not escaped
                    if escape_count % 2 == 0:
                        # Found the closing slash
                        return query[1:i]
                i += 1

        # Look for patterns that are likely regex (contain character classes, quantifiers, etc.)
        patterns = list(re.finditer(r"/([^/]*[\[\]{}+*?\\][^/]*)/", query))
        if patterns:
            # Return the longest match (most likely to be the complete regex pattern)
            longest = max(patterns, key=lambda m: len(m.group(1)))
            return longest.group(1)

        # Fallback: find any /pattern/ occurrence
        patterns = list(re.finditer(r"/([^/]+)/", query))
        if patterns:
            # Return the longest match
            longest = max(patterns, key=lambda m: len(m.group(1)))
            return longest.group(1)

        return None

    def has_pattern(self, query: str) -> bool:
        """Check if query contains enumerable regex patterns."""
        if not query:
            return False

        # Extract regex pattern from GitHub search format
        pattern = self._extract_regex_pattern(query)
        if not pattern:
            return False

        try:
            segments = self.parser.parse(pattern)

            # Check for variable segments (including inside groups)
            def find_variable_segments(segs):
                variables = []
                for seg in segs:
                    if isinstance(seg, CharClassSegment):
                        variables.append(seg)
                    elif isinstance(seg, (GroupSegment, OptionalSegment)):
                        variables.extend(find_variable_segments(seg.content))
                return variables

            variable_segments = find_variable_segments(segments)

            result = len(variable_segments) > 0
            logger.debug(f"Pattern check: '{pattern}' -> {result} ({len(variable_segments)} variables)")
            return result

        except Exception as e:
            logger.warning(f"Pattern check failed for '{pattern}': {e}")
            return False

    def generate_queries(self, query: str, partitions: int) -> List[str]:
        """Generate refined queries from a query string."""
        if partitions <= 0 or not query:
            logger.debug(f"Invalid partitions={partitions} or query: {query}")
            return []

        # Try to divide the original query into multiple smaller intervals
        queries = self._divide(query, partitions)
        if not queries:
            logger.debug("No queries generated from divide with regex")
            return []

        if len(queries) >= partitions:
            logger.debug(f"Already have enough queries: {len(queries)} >= {partitions}")
            return queries

        candidates = set()

        # Try to divide with language
        for item in queries:
            results = self._divide_with_language(item)
            if not results:
                logger.debug("No queries generated from divide with language")
                continue

            for result in results:
                if result:
                    candidates.add(result)

        conditions = list(candidates)
        logger.debug(f"Generated {len(conditions)} queries from query: {query}, partitions: {partitions}")

        return conditions

    def _divide(self, query: str, partitions: int) -> List[str]:
        """
        Split a broad regular expression query into multiple regular expressions that match smaller ranges

        Args:
            query: Query string in format "/pattern/" or "/pattern/ other terms" or "terms AND /pattern/"
            partitions: Number of partitions to divide the query into

        Returns:
            List[str]: List of refined queries
        """

        if partitions <= 0 or not query:
            return []

        # Extract pattern and base query
        pattern = self._extract_regex_pattern(query)
        if not pattern:
            return [query]

        try:
            # Parse pattern
            segments = self.parser.parse(pattern)

            # Check if pattern can be split safely
            enabled, reason = self.splittability.can_split_further(pattern, segments)
            if not enabled:
                logger.info(f"Pattern cannot be split further: {reason}")

                # Return original query
                return [query]

            # Evaluate strategies based on partitions requirement
            strategy, found = self.optimizer.evaluate_strategies_for_partitions(segments, partitions)

            if found:
                # Found strategy that can generate >= partitions queries
                # Use minimum enumeration depth approach
                logger.info(f"Found suitable strategy for {partitions} partitions, " f"using minimum enumeration depth")
                refined_patterns = self.generator.generate(segments, strategy, partitions)
            else:
                # No suitable strategy found, use the one that generates most queries
                logger.info(
                    f"No strategy found for {partitions} partitions, "
                    f"using strategy with maximum queries ({strategy.queries})"
                )
                refined_patterns = self.generator.generate(segments, strategy)

            # Reconstruct queries preserving original structure
            queries = []
            for item in refined_patterns:
                # Replace the original pattern with the refined pattern in the original query
                text = query.replace(f"/{pattern}/", f"/{item}/")
                queries.append(text)

            logger.info(
                f"Generated {len(queries)} queries from pattern: '{pattern}', requested partitions: {partitions}"
            )
            return queries

        except Exception as e:
            logger.warning(f"Query generation failed for '{pattern}': {e}")
            return [query]

    def _divide_with_language(self, query: str) -> List[str]:
        """Generate refined queries with adaptive refinement language level."""
        base = query.strip() if query else ""
        if not base:
            logger.debug("No query provided for language refinement")
            return []

        queries = set()
        if not re.match(r" language:[a-zA-Z0-9#]+ ", base, flags=re.I):
            # Language-based refinement
            for lang in POPULAR_LANGUAGES:
                queries.add(f"{base} language:{lang}")
        elif not re.match(r" size:[a-zA-Z0-9#=<>.]+ ", base, flags=re.I):
            # Sise-based refinement
            for size in SIZE_RANGES:
                queries.add(f"{base} size:{size}")
        else:
            logger.debug("Cannot refine with language or sie refinement due to existing refinement criteria")
            queries.add(base)

        return list(queries)

    def analyze_pattern(self, pattern: str) -> dict:
        """Analyze pattern and return detailed information."""
        try:
            segments = self.parser.parse(pattern)
            strategy = self.optimizer.optimize(segments)

            # Count segments recursively
            def count_segments_recursive(segs, seg_type):
                count = 0
                for seg in segs:
                    if isinstance(seg, seg_type):
                        count += 1
                    elif isinstance(seg, (GroupSegment, OptionalSegment)):
                        count += count_segments_recursive(seg.content, seg_type)
                return count

            analysis = {
                "segments": len(segments),
                "fixed_segments": count_segments_recursive(segments, FixedSegment),
                "variable_segments": count_segments_recursive(segments, CharClassSegment),
                "optional_segments": count_segments_recursive(segments, OptionalSegment),
                "group_segments": count_segments_recursive(segments, GroupSegment),
                "enumeration_segments": len(strategy.segments),
                "enumeration_value": strategy.value,
                "estimated_queries": strategy.queries,
                "parseable": True,
            }

            return analysis

        except Exception as e:
            logger.warning(f"Pattern analysis failed for '{pattern}': {e}")
            return {"parseable": False, "error": str(e)}

    def can_split_safely(
        self, query: str, recursion_depth: int = 0, parent_pattern: Optional[str] = None
    ) -> tuple[bool, str]:
        """Check if a query can be split safely without infinite loops."""
        # Extract pattern from GitHub search format
        match = re.search(r"/([^/]+)/", query)
        if not match:
            return False, "No regex pattern found in query"

        pattern = match.group(1)

        try:
            segments = self.parser.parse(pattern)
            return self.splittability.can_split_further(pattern, segments, recursion_depth, parent_pattern)
        except Exception as e:
            logger.warning(f"Splittability check failed for '{pattern}': {e}")
            return False, f"Analysis failed: {str(e)}"

    def reset_split_history(self):
        """Reset the split history for a new analysis session."""
        self.splittability.reset_history()

    def get_split_analysis_summary(self) -> dict:
        """Get a summary of the current split analysis state."""
        return self.splittability.get_analysis_summary()

    def clean_regex(self, query: str, separator: str = "AND") -> str:
        """
        Clean regex query by extracting fixed strings from regex patterns.

        Args:
            query: Input query string containing regex patterns
            separator: Separator to use ("AND", "OR", "NOT", "AND NOT"), defaults to "AND"

        Returns:
            str: Cleaned query with fixed strings extracted from regex patterns
        """
        if not query:
            return ""

        # Validate separator
        if separator not in ALLOWED_OPERATORS:
            separator = "AND"

        # Create the actual separator with spaces
        delimiter = f" {separator} "

        # Split by the actual separator
        parts = query.split(delimiter)

        results = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Skip if the part is exactly the separator
            if part == separator:
                continue

            # Check if part is already quoted
            if part.startswith('"') and part.endswith('"'):
                results.append(part)
                continue

            # Check if part matches pattern [a-zA-Z]+:.*\S.*
            if re.match(r"^[a-zA-Z]+:.*\S.*$", part):
                results.append(part)
                continue

            # Check if part is a regex pattern (starts and ends with /)
            is_regex = part.startswith("/") and part.endswith("/")

            if is_regex:
                # Extract pattern without slashes
                pattern = part[1:-1]

                try:
                    # Parse the pattern
                    segments = self.parser.parse(pattern)

                    # Extract fixed strings from FixedSegment
                    fixed = []
                    self._extract_fixed_strings(segments, fixed)

                    # Process fixed strings
                    processed = []
                    for text in fixed:
                        # Remove escape characters if original part was wrapped in slashes
                        if is_regex:
                            # Remove backslash escapes
                            text = text.replace("\\/", "/").replace("\\\\", "\\")

                        # Skip if same as separator or length < 3
                        if text == separator or len(text) < 3:
                            continue

                        # Add quotes if not already quoted and doesn't match search syntax pattern
                        # Pattern should match things like content:"value" but not URLs like https://
                        if not (text.startswith('"') and text.endswith('"')) and not re.match(
                            r'^[a-zA-Z]+:[\"\'"].*[\"\'"]$', text
                        ):
                            text = f'"{text}"'

                        processed.append(text)

                    # Add processed strings to cleaned parts
                    results.extend(processed)

                except Exception as e:
                    logger.warning(f"Failed to parse regex pattern '{pattern}': {e}")
                    # If parsing fails, skip this part
                    continue
            else:
                # For non-regex parts, try to parse anyway to extract fixed strings
                try:
                    segments = self.parser.parse(part)
                    fixed = []
                    self._extract_fixed_strings(segments, fixed)

                    processed = []
                    for text in fixed:
                        # Skip if same as separator or length < 3
                        if text == separator or len(text) < 3:
                            continue

                        # Add quotes if not already quoted and doesn't match search syntax pattern
                        # Pattern should match things like content:"value" but not URLs like https://
                        if not (text.startswith('"') and text.endswith('"')) and not re.match(
                            r'^[a-zA-Z]+:[\"\'"].*[\"\'"]$', text
                        ):
                            text = f'"{text}"'

                        processed.append(text)

                    if processed:
                        results.extend(processed)
                    else:
                        # If no fixed strings found, treat as regular string
                        if not re.match(r"^[a-zA-Z]+:.*\S.*$", part):
                            part = f'"{part}"'
                        results.append(part)

                except Exception:
                    # If parsing fails, treat as regular string
                    if not re.match(r"^[a-zA-Z]+:.*\S.*$", part):
                        part = f'"{part}"'
                    results.append(part)

        # Return result
        if not results:
            return ""
        elif len(results) == 1:
            return results[0]
        else:
            return delimiter.join(results)

    def _extract_fixed_strings(self, segments: List, fixed: List[str]) -> None:
        """
        Recursively extract fixed strings from segments.

        Args:
            segments: List of segments to process
            fixed: List to append found fixed strings to
        """
        # First, collect all fixed segments and merge consecutive ones
        merged = []
        current = ""

        for seg in segments:
            if isinstance(seg, FixedSegment):
                if seg.content:
                    # Handle escape sequences
                    content = seg.content
                    if content == "\\/":
                        content = "/"
                    elif content == "\\\\":
                        content = "\\"

                    # Skip pure special characters
                    if content in [".", "*", "+", "?", "{", "}", "^", "$", "|", "(", ")", "[", "]"]:
                        # If we have accumulated string, save it
                        if current:
                            merged.append(current)
                            current = ""
                    else:
                        # Include "." as part of the string since it's often part of domain names
                        current += content
            elif isinstance(seg, GroupSegment):
                # Save current accumulated string before processing group
                if current:
                    merged.append(current)
                    current = ""

                # For groups, check if it's a simple choice pattern
                if len(seg.content) == 1 and isinstance(seg.content[0], FixedSegment):
                    content = seg.content[0].content
                    # Check if it's a simple choice pattern like "sid01|api03"
                    if "|" in content and not any(char in content for char in "[]{}*+?()\\"):
                        # Skip choice patterns as they're not fixed strings
                        continue
                    else:
                        # It's a regular fixed string in a group
                        merged.append(content)
                else:
                    # Recursively process group content for other cases
                    self._extract_fixed_strings(seg.content, merged)
            elif isinstance(seg, OptionalSegment):
                # Save current accumulated string before skipping optional
                if current:
                    merged.append(current)
                    current = ""
                # Skip optional segments as they're not guaranteed to be present
                continue
            else:
                # For other segment types (like CharClassSegment), break the current string
                if current:
                    merged.append(current)
                    current = ""

        # Don't forget the last accumulated string
        if current:
            merged.append(current)

        # Add all merged strings to the result
        fixed.extend(merged)
        # Don't forget the last accumulated string
        if current:
            merged.append(current)
