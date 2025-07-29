#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Query generator for enumerated regex patterns.
"""

import itertools
import math
import threading
from typing import List, Set

from logger import get_engine_logger

from .segments import (
    CharClassSegment,
    EnumerationStrategy,
    FixedSegment,
    GroupSegment,
    OptionalSegment,
    Segment,
)

logger = get_engine_logger()


class QueryGenerator:
    """Generate queries from enumeration strategy - Singleton pattern."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    @classmethod
    def get_instance(cls, max_depth: int = 3) -> "QueryGenerator":
        """Get singleton instance with optional configuration."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance.__init__(max_depth)
                    logger.debug("QueryGenerator singleton instance initialized")
        return cls._instance

    def generate(self, segments: List[Segment], strategy: EnumerationStrategy, partitions: int = -1) -> List[str]:
        """Generate queries from enumeration strategy."""
        if not strategy.segments:
            # No enumeration needed
            query = self._reconstruct_pattern(segments)
            return [query] if query else []

        try:
            if partitions > 0:
                # Use depth-based generation logic
                # Calculate minimum depth needed for target queries
                min_depth = self._calculate_min_depth_for_target(strategy.segments, partitions)

                queries = set()

                # Generate queries for each enumerable part with calculated depth
                for target_segment in strategy.segments:
                    part_queries = self._generate_queries_for_single_part(segments, target_segment, min_depth)
                    queries.update(part_queries)

                result = list(queries)
                logger.info(
                    f"Generated {len(result)} unique queries with depth {min_depth} "
                    f"from {len(strategy.segments)} parts"
                )
                return result
            else:
                # Use original generation logic
                # If strategy contains multiple segments, we should enumerate them together (Cartesian product)
                # If strategy contains only one segment, enumerate that segment only
                if len(strategy.segments) == 1:
                    # Single segment enumeration - this is the most common case
                    queries = self._generate_queries_for_single_part(segments, strategy.segments[0])
                    logger.info(f"Generated {len(queries)} queries for single segment enumeration")
                    return queries
                else:
                    # Multiple segment enumeration - enumerate each segment SEPARATELY (not together)
                    # This ensures the union of results represents the original regex space
                    queries = self._generate_queries_for_multiple_parts(segments, strategy.segments, separate=True)
                    logger.info(
                        f"Generated {len(queries)} queries for {len(strategy.segments)} segments enumeration (separately)"
                    )
                    return queries

        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
            query = self._reconstruct_pattern(segments)
            return [] if not query else [query]

    def _calculate_min_depth_for_target(self, segments: List[CharClassSegment], target_queries: int) -> int:
        """Calculate minimum enumeration depth needed to reach target query count."""
        if not segments or target_queries <= 1:
            return 1

        # For multiple segments, we need to calculate the depth such that
        # the total combinations from all segments >= target_queries
        # Each segment contributes charset_size^depth combinations
        # Total combinations = sum(charset_size^depth for each segment)

        # Find the minimum charset size to use as base
        min_charset_size = min(len(segment.charset) for segment in segments if len(segment.charset) > 0)

        if min_charset_size <= 1:
            return 1

        # Calculate depth needed: min_charset_size^depth * num_segments >= target_queries
        # Solve for depth: depth >= log(target_queries / num_segments) / log(min_charset_size)
        required_per_segment = target_queries / len(segments)

        if required_per_segment <= 1:
            return 1

        min_depth = math.ceil(math.log(required_per_segment) / math.log(min_charset_size))

        # Ensure depth is reasonable (between 1 and max_depth)
        min_depth = max(1, min(min_depth, self.max_depth))

        logger.debug(
            f"Calculated min depth {min_depth} for target {target_queries} queries "
            f"with {len(segments)} segments, min_charset_size={min_charset_size}"
        )

        return min_depth

    def _generate_queries_for_single_part(
        self, segments: List[Segment], target: CharClassSegment, depth: int = -1
    ) -> List[str]:
        """Generate queries by enumerating target segment with optional specific depth."""
        if depth > 0:
            # Use specific depth logic
            combinations = self._generate_segment_combinations(target, depth)
        else:
            # Use original logic
            combinations = self._generate_segment_combinations(target)

        queries = []
        for combo in combinations:
            # Apply enumeration only to target segment
            new_segments = self._apply_single_enumeration(segments, target, combo)
            pattern = self._reconstruct_pattern(new_segments)
            if pattern and pattern != self._reconstruct_pattern(segments):
                # Only add if pattern actually changed (was enumerated)
                queries.append(pattern)

        # If no enumerated queries generated, force enumeration
        if not queries and combinations:
            # Force enumeration for all combinations (up to reasonable limit)
            for combo in combinations:
                new_segments = self._apply_single_enumeration(segments, target, combo)
                pattern = self._reconstruct_pattern(new_segments)
                if pattern:
                    queries.append(pattern)

        return queries

    def _generate_queries_for_multiple_parts(
        self, segments: List[Segment], targets: List[CharClassSegment], separate: bool = False
    ) -> List[str]:
        """Generate queries by enumerating multiple segments.

        Args:
            segments: List of all segments
            targets: List of target segments to enumerate
            separate: If True, enumerate each segment separately (union).
                     If False, enumerate all segments together (Cartesian product).
        """
        if separate:
            # Enumerate each target segment separately to ensure union represents original regex space
            all_queries = []

            # Enumerate each target segment separately
            for i, target in enumerate(targets):
                logger.debug(
                    f"Enumerating target segment {i}: position={target.position}, value={getattr(target, 'value', 'N/A')}"
                )
                target_queries = self._generate_queries_for_single_part(segments, target)
                logger.debug(f"Generated {len(target_queries)} queries for segment {i}")
                if target_queries:
                    logger.debug(f"Sample query for segment {i}: {target_queries[0]}")
                all_queries.extend(target_queries)

            logger.debug(f"Total queries before deduplication: {len(all_queries)}")

            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for query in all_queries:
                if query not in seen:
                    seen.add(query)
                    unique_queries.append(query)

            logger.debug(f"Unique queries after deduplication: {len(unique_queries)}")
            return unique_queries
        else:
            # Original logic: enumerate all segments together (Cartesian product)
            # Generate combinations for each target segment
            all_combinations = []
            for target in targets:
                combinations = self._generate_segment_combinations(target)
                all_combinations.append(combinations)

            # Generate Cartesian product of all combinations
            queries = []
            for combo_tuple in itertools.product(*all_combinations):
                # Apply enumeration to all target segments simultaneously
                new_segments = segments.copy()
                for i, target in enumerate(targets):
                    enum_value = combo_tuple[i]
                    new_segments = self._apply_single_enumeration(new_segments, target, enum_value)

                pattern = self._reconstruct_pattern(new_segments)
                if pattern and pattern != self._reconstruct_pattern(segments):
                    queries.append(pattern)

            return queries

    def _generate_segment_combinations(self, segment: CharClassSegment, depth: int = -1) -> List[str]:
        """Generate combinations for single segment with optional specific depth."""
        # Expand shortcuts like \w and \d
        expanded_charset = self._expand_charset_shortcuts(segment.original_charset_str)

        # Parse charset considering case sensitivity
        charset = self._parse_charset_to_set(expanded_charset, segment.case_sensitive)
        charset = sorted(list(charset))

        if depth > 0:
            # Use specific depth logic
            if not charset:
                return [""]

            # Generate combinations with specified depth
            combinations = []
            for combo in itertools.product(charset, repeat=depth):
                combinations.append("".join(combo))

            logger.info(f"Generated {len(combinations)} combinations for depth {depth}")
            return combinations
        else:
            # Calculate optimal enumeration depth (1-3 chars as requested)
            optimal_depth = min(3, max(1, self._calculate_optimal_depth(segment)))

            if optimal_depth == 0:
                return [""]

            # Generate combinations
            combinations = []
            for combo in itertools.product(charset, repeat=optimal_depth):
                combinations.append("".join(combo))

            logger.info(f"Generated {len(combinations)} combinations for depth {optimal_depth}")

            return combinations

    def _calculate_optimal_depth(self, segment: CharClassSegment) -> int:
        """Calculate optimal enumeration depth based on mathematical analysis."""
        charset_size = len(segment.charset)

        if charset_size == 0:
            return 0

        # Calculate depth based on enumeration value and efficiency
        # Higher enumeration value segments deserve deeper enumeration
        if hasattr(segment, "value") and segment.value > 0:
            # Use enumeration value to determine depth
            if segment.value > 20:  # Very high value
                target_depth = min(4, self.max_depth)
            elif segment.value > 10:  # High value
                target_depth = min(3, self.max_depth)
            elif segment.value > 5:  # Medium value
                target_depth = min(2, self.max_depth)
            else:  # Low value
                target_depth = min(1, self.max_depth)
        else:
            # Fallback to conservative depth
            target_depth = min(2, self.max_depth)

        # Ensure the depth makes mathematical sense
        # Don't enumerate more characters than the minimum length
        if hasattr(segment, "min_length"):
            target_depth = min(target_depth, segment.min_length)

        # For very small charsets, we can afford deeper enumeration
        if charset_size <= 10:
            target_depth = min(target_depth + 1, self.max_depth)
        elif charset_size >= 50:
            target_depth = max(1, target_depth - 1)

        logger.debug(
            f"Calculated optimal depth {target_depth} for charset_size={charset_size}, "
            f"value={getattr(segment, 'value', 0):.3f}"
        )

        return target_depth

    def _apply_single_enumeration(
        self, segments: List[Segment], target: CharClassSegment, enum_value: str
    ) -> List[Segment]:
        """Apply enumeration to only the target segment, including nested segments."""
        new_segments = []

        for segment in segments:
            if self._is_target_segment(segment, target):
                # This is the target segment to enumerate
                if enum_value:
                    # Create fixed prefix segment
                    fixed_seg = FixedSegment()
                    fixed_seg.position = segment.position
                    fixed_seg.content = enum_value
                    new_segments.append(fixed_seg)

                    # Create remaining character class segment
                    remaining_min = max(0, segment.min_length - len(enum_value))
                    remaining_max = max(0, segment.max_length - len(enum_value))

                    # Always create remaining segment for + and *, even if min=0
                    should_create_remaining = segment.original_quantifier in ["+", "*"] or remaining_max > 0

                    if should_create_remaining:
                        remaining_seg = CharClassSegment()
                        remaining_seg.position = segment.position
                        remaining_seg.charset = segment.charset.copy()
                        remaining_seg.original_charset_str = segment.original_charset_str
                        remaining_seg.case_sensitive = segment.case_sensitive
                        remaining_seg.min_length = remaining_min
                        remaining_seg.max_length = remaining_max
                        remaining_seg.original_quantifier = self._adjust_quantifier(
                            segment.original_quantifier, len(enum_value), segment
                        )
                        new_segments.append(remaining_seg)
                else:
                    # Keep original if no enumeration value
                    new_segments.append(segment)
            elif isinstance(segment, GroupSegment):
                # Recursively process group content
                new_group = GroupSegment()
                new_group.position = segment.position
                new_group.capturing = segment.capturing
                if hasattr(segment, "original_prefix"):
                    new_group.original_prefix = segment.original_prefix
                if hasattr(segment, "quantifier"):
                    new_group.quantifier = segment.quantifier
                new_group.content = self._apply_single_enumeration(segment.content, target, enum_value)
                new_segments.append(new_group)
            elif isinstance(segment, OptionalSegment):
                # Recursively process optional content
                new_optional = OptionalSegment()
                new_optional.position = segment.position
                new_optional.content = self._apply_single_enumeration(segment.content, target, enum_value)
                new_segments.append(new_optional)
            else:
                # Keep original segment
                new_segments.append(segment)

        return new_segments

    def _is_target_segment(self, segment: Segment, target: CharClassSegment) -> bool:
        """Check if segment is the target segment to enumerate."""
        if not isinstance(segment, CharClassSegment):
            return False

        # Compare by position and content characteristics
        return (
            segment.position == target.position
            and segment.charset == target.charset
            and segment.min_length == target.min_length
            and segment.max_length == target.max_length
        )

    def _calculate_remaining_length(self, segment: CharClassSegment, enum_length: int) -> int:
        """Calculate remaining minimum length after enumeration."""
        if segment.original_quantifier == "+":
            # For +: {1,max} - enum_length = {max(0, 1-enum), max-enum}
            remaining_min = max(0, segment.min_length - enum_length)
            return remaining_min
        elif segment.original_quantifier == "*":
            # For *: {0,max} - enum_length = {max(0, 0-enum), max-enum}
            remaining_min = max(0, segment.min_length - enum_length)
            return remaining_min
        elif segment.original_quantifier == "?":
            # For ?: {0,1} - enum_length
            return max(0, 1 - enum_length)
        elif segment.original_quantifier.startswith("{"):
            # For {n,m}: calculate remaining based on actual range
            remaining_min = max(0, segment.min_length - enum_length)
            return remaining_min
        else:
            # For no quantifier (single char)
            return max(0, 1 - enum_length)

    def _adjust_quantifier(self, original_quantifier: str, enum_length: int, segment: CharClassSegment) -> str:
        """Adjust quantifier after enumeration."""
        if original_quantifier == "+":
            # For +: {1,max} - enum_length = {max(0, 1-enum), max-enum}
            remaining_min = max(0, segment.min_length - enum_length)
            remaining_max = max(0, segment.max_length - enum_length)

            if remaining_min == 0 and remaining_max > 0:
                return "*"  # Zero or more
            elif remaining_min == 1 and remaining_max > 1:
                return "+"  # One or more
            elif remaining_min == remaining_max and remaining_min > 0:
                return f"{{{remaining_min}}}"  # Exact count
            elif remaining_min != remaining_max and remaining_max > 0:
                return f"{{{remaining_min},{remaining_max}}}"  # Range
            else:
                return ""  # No remaining chars needed

        elif original_quantifier == "*":
            # For *: {0,max} - enum_length = {max(0, 0-enum), max-enum}
            remaining_min = max(0, segment.min_length - enum_length)
            remaining_max = max(0, segment.max_length - enum_length)

            if remaining_min == 0 and remaining_max > 0:
                return "*"  # Keep *
            elif remaining_min == remaining_max and remaining_min > 0:
                return f"{{{remaining_min}}}"  # Exact count
            elif remaining_min != remaining_max and remaining_max > 0:
                return f"{{{remaining_min},{remaining_max}}}"  # Range
            else:
                return ""  # No remaining chars needed

        elif original_quantifier == "?":
            # For ?: {0,1} - enum_length
            if enum_length >= 1:
                return ""  # No more chars needed
            else:
                return "?"  # Keep ?

        elif original_quantifier.startswith("{"):
            # For {n,m}: calculate remaining range
            remaining_min = max(0, segment.min_length - enum_length)
            remaining_max = max(0, segment.max_length - enum_length)

            if remaining_min == remaining_max and remaining_min > 0:
                return f"{{{remaining_min}}}"
            elif remaining_min != remaining_max and remaining_max > 0:
                return f"{{{remaining_min},{remaining_max}}}"
            else:
                return ""
        else:
            return original_quantifier

    def _reconstruct_pattern(self, segments: List[Segment]) -> str:
        """Reconstruct regex pattern from segments preserving original structure."""
        try:
            result = ""

            for segment in segments:
                if isinstance(segment, FixedSegment):
                    result += self._preserve_escapes(segment.content)
                elif isinstance(segment, CharClassSegment):
                    result += self._reconstruct_charclass(segment)
                elif isinstance(segment, GroupSegment):
                    group_content = self._reconstruct_pattern(segment.content)
                    group_str = ""
                    if segment.capturing:
                        group_str = f"({group_content})"
                    else:
                        # Preserve original group structure including special flags
                        if hasattr(segment, "original_prefix") and segment.original_prefix:
                            group_str = f"({segment.original_prefix}{group_content})"
                        else:
                            group_str = f"(?:{group_content})"

                    # Add quantifier if present
                    if hasattr(segment, "quantifier") and segment.quantifier:
                        group_str += segment.quantifier

                    result += group_str
                elif isinstance(segment, OptionalSegment):
                    optional_content = self._reconstruct_pattern(segment.content)
                    result += f"(?:{optional_content})?"

            return result

        except Exception as e:
            logger.warning(f"Pattern reconstruction failed: {e}")
            return ""

    def _preserve_escapes(self, content: str) -> str:
        """Preserve original escape sequences in fixed content."""
        # Don't re-escape already escaped characters
        # This preserves original escapes like \/, \-, \.
        return content

    def _expand_charset_shortcuts(self, charset_str: str) -> str:
        """Expand \\w and \\d shortcuts in charset string."""
        if not charset_str:
            return charset_str

        # Expand \w to [a-zA-Z0-9_]
        expanded = charset_str.replace("\\w", "a-zA-Z0-9_")
        # Expand \d to [0-9]
        expanded = expanded.replace("\\d", "0-9")

        return expanded

    def _detect_case_sensitivity(self, pattern: str) -> bool:
        """Detect if pattern has (?-i) case sensitive flag."""
        return "(?-i)" in pattern

    def _parse_charset_to_set(self, charset_str: str, case_sensitive: bool) -> Set[str]:
        """Parse charset string to character set, handling case sensitivity."""
        chars = set()

        # Remove brackets if present
        if charset_str.startswith("[") and charset_str.endswith("]"):
            charset_str = charset_str[1:-1]

        i = 0
        while i < len(charset_str):
            if charset_str[i] == "\\" and i + 1 < len(charset_str):
                # Handle escape sequences
                next_char = charset_str[i + 1]
                if next_char == "-":
                    chars.add("-")  # Escaped dash
                elif next_char == "\\":
                    chars.add("\\")  # Escaped backslash
                elif next_char == "]":
                    chars.add("]")  # Escaped bracket
                else:
                    chars.add(next_char)  # Other escaped chars
                i += 2
            elif i + 2 < len(charset_str) and charset_str[i + 1] == "-" and charset_str[i + 2] != "]":
                # Handle range like a-z, A-Z, 0-9 (but not at end)
                start_char = charset_str[i]
                end_char = charset_str[i + 2]
                for code in range(ord(start_char), ord(end_char) + 1):
                    chars.add(chr(code))
                i += 3
            else:
                # Single character
                chars.add(charset_str[i])
                i += 1

        # Handle GitHub case insensitivity (default behavior)
        if not case_sensitive:
            # GitHub treats [a-zA-Z0-9] as [a-z0-9] (36 chars total)
            normalized_chars = set()
            for char in chars:
                if char.isalpha():
                    normalized_chars.add(char.lower())
                else:
                    normalized_chars.add(char)
            chars = normalized_chars

        return chars

    def _reconstruct_charclass(self, segment: CharClassSegment) -> str:
        """Reconstruct character class pattern preserving original escapes."""
        # Always use original charset string to preserve escapes like \-
        if hasattr(segment, "original_charset_str") and segment.original_charset_str:
            charset_part = segment.original_charset_str
        else:
            # Fallback to reconstructing from charset with proper escaping
            charset = sorted(list(segment.charset))
            class_content = ""
            i = 0
            while i < len(charset):
                char = charset[i]

                # Check for consecutive characters to create ranges
                if i + 2 < len(charset):
                    if ord(charset[i + 1]) == ord(char) + 1 and ord(charset[i + 2]) == ord(char) + 2:
                        # Find end of range
                        end_i = i + 2
                        while end_i + 1 < len(charset) and ord(charset[end_i + 1]) == ord(charset[end_i]) + 1:
                            end_i += 1

                        class_content += f"{char}-{charset[end_i]}"
                        i = end_i + 1
                        continue

                # Preserve escape sequences for special characters
                if char in r"\-]^":
                    class_content += f"\\{char}"
                else:
                    class_content += char
                i += 1

            charset_part = f"[{class_content}]"

        # Always use original quantifier to preserve exact format
        if hasattr(segment, "original_quantifier") and segment.original_quantifier:
            quantifier = segment.original_quantifier
        else:
            # Fallback quantifier reconstruction
            if segment.min_length == segment.max_length:
                if segment.min_length == 1:
                    quantifier = ""
                else:
                    quantifier = f"{{{segment.min_length}}}"
            elif segment.max_length == float("inf") or segment.max_length >= 150:
                if segment.min_length == 0:
                    quantifier = "*"
                elif segment.min_length == 1:
                    quantifier = "+"
                else:
                    quantifier = f"{{{segment.min_length},}}"
            else:
                quantifier = f"{{{segment.min_length},{int(segment.max_length)}}}"

        return f"{charset_part}{quantifier}"
