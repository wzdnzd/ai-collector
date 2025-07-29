#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regex pattern parser with support for complex patterns.
"""

import threading
from typing import List, Optional, Set, Tuple, Union

from logger import get_engine_logger

from .segments import (
    CharClassSegment,
    FixedSegment,
    GroupSegment,
    OptionalSegment,
    Segment,
)

logger = get_engine_logger()


class RegexParser:
    """Parse regex patterns into segment sequences - Singleton pattern."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, max_quantifier_length: int = 150):
        self.max_quantifier_length = max_quantifier_length
        self.pattern = ""
        self.pos = 0
        self.length = 0

    @classmethod
    def get_instance(cls, max_quantifier_length: int = 150) -> "RegexParser":
        """Get singleton instance with optional configuration."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance.__init__(max_quantifier_length)
                    logger.debug("RegexParser singleton instance initialized")
        return cls._instance

    def parse(self, pattern: str) -> List[Segment]:
        """Parse regex pattern into segments."""
        if not pattern:
            return []

        self.pattern = pattern
        self.pos = 0
        self.length = len(pattern)
        segments = []

        try:
            while self.pos < self.length:
                segment = self._parse_next()
                if segment:
                    segment.position = len(segments)
                    segments.append(segment)
                else:
                    break

            self._calculate_prefix_lengths(segments)
            return segments

        except Exception as e:
            logger.warning(f"Failed to parse pattern '{pattern}': {e}")
            return []

    def _parse_next(self) -> Optional[Segment]:
        """Parse next segment from current position."""
        if self.pos >= self.length:
            return None

        char = self.pattern[self.pos]

        if char == "(":
            return self._parse_group()
        elif char == "[":
            return self._parse_charclass()
        elif char in r".*+?{}^$|\\":
            return self._parse_special()
        else:
            return self._parse_fixed()

    def _parse_group(self) -> Optional[Segment]:
        """Parse group patterns (...) or (?:...) or (?-i) etc."""
        start_pos = self.pos
        self.pos += 1  # Skip '('

        if self.pos >= self.length:
            return None

        # Check for special group prefixes like ?:, ?-i, etc.
        non_capturing = False
        original_prefix = ""

        if self.pos < self.length and self.pattern[self.pos] == "?":
            # Parse special group syntax
            prefix_start = self.pos
            self.pos += 1  # Skip '?'

            if self.pos < self.length:
                if self.pattern[self.pos] == ":":
                    # Non-capturing group (?:...)
                    non_capturing = True
                    original_prefix = "?:"
                    self.pos += 1
                elif self.pattern[self.pos] == "-" and self.pos + 1 < self.length and self.pattern[self.pos + 1] == "i":
                    # Case sensitive flag (?-i)
                    original_prefix = "?-i"
                    self.pos += 2
                    non_capturing = True
                else:
                    # Other special syntax - preserve as-is
                    while self.pos < self.length and self.pattern[self.pos] not in "):":
                        self.pos += 1
                    original_prefix = self.pattern[prefix_start : self.pos]
                    if self.pos < self.length and self.pattern[self.pos] == ":":
                        self.pos += 1
                        non_capturing = True

        # Find matching closing parenthesis
        paren_count = 1
        group_start = self.pos

        while self.pos < self.length and paren_count > 0:
            if self.pattern[self.pos] == "(":
                paren_count += 1
            elif self.pattern[self.pos] == ")":
                paren_count -= 1
            self.pos += 1

        if paren_count > 0:
            logger.warning("Unmatched parentheses in pattern")
            return None

        # Parse group content - preserve original for complex structures
        group_pattern = self.pattern[group_start : self.pos - 1]

        # For choice patterns like (sid01|api03), treat as single fixed content
        if "|" in group_pattern and not any(char in group_pattern for char in "[]{}*+?()"):
            # This is a simple choice pattern, create a single fixed segment
            choice_segment = FixedSegment()
            choice_segment.position = 0
            choice_segment.content = group_pattern
            group_content = [choice_segment]
        else:
            # Parse normally for other patterns
            sub_parser = RegexParser()
            group_content = sub_parser.parse(group_pattern)

        # Check for quantifier
        quantifier = self._parse_quantifier()

        if quantifier == "?":
            segment = OptionalSegment()
            segment.position = start_pos
            segment.content = group_content
            return segment
        else:
            segment = GroupSegment()
            segment.position = start_pos
            segment.content = group_content
            segment.capturing = not non_capturing
            # Store original prefix to preserve special flags
            if original_prefix:
                segment.original_prefix = original_prefix
            # Store quantifier to preserve group repetition like {3}
            if quantifier:
                segment.quantifier = quantifier
            return segment

    def _parse_charclass(self) -> Optional[CharClassSegment]:
        """Parse character class [...]"""
        start_pos = self.pos
        self.pos += 1  # Skip '['

        if self.pos >= self.length:
            return None

        # Find closing bracket
        class_content = ""
        while self.pos < self.length and self.pattern[self.pos] != "]":
            class_content += self.pattern[self.pos]
            self.pos += 1

        if self.pos >= self.length:
            logger.warning("Unclosed character class")
            return None

        self.pos += 1  # Skip ']'

        # Parse character set
        charset = self._parse_charset(class_content)
        if not charset:
            return None

        # Parse quantifier
        quantifier = self._parse_quantifier()
        min_len, max_len = self._quantifier_to_range(quantifier)

        # Detect case sensitivity from pattern
        case_sensitive = self._detect_case_sensitivity()

        segment = CharClassSegment()
        segment.position = start_pos
        segment.charset = charset
        segment.min_length = min_len
        segment.max_length = max_len
        segment.original_quantifier = quantifier
        segment.original_charset_str = f"[{class_content}]"  # Store original with escapes
        segment.case_sensitive = case_sensitive
        return segment

    def _parse_charset(self, content: str) -> Set[str]:
        """Parse character class content into character set."""
        chars = set()
        i = 0
        length = len(content)

        while i < length:
            if i + 2 < length and content[i + 1] == "-":
                # Handle ranges like a-z, A-Z, 0-9
                start = content[i]
                end = content[i + 2]

                # Handle escaped characters
                if start == "\\" and i + 1 < length:
                    start = self._unescape_char(content[i + 1])
                    i += 1
                if end == "\\" and i + 3 < length:
                    end = self._unescape_char(content[i + 3])
                    i += 1

                # Add range characters
                try:
                    for c in range(ord(start), ord(end) + 1):
                        chars.add(chr(c))
                except ValueError:
                    logger.warning(f"Invalid character range: {start}-{end}")

                i += 3
            elif content[i] == "\\" and i + 1 < length:
                # Handle escaped characters
                escaped = self._unescape_char(content[i + 1])
                if escaped == "d":
                    chars.update("0123456789")
                elif escaped == "w":
                    chars.update("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
                elif escaped == "s":
                    chars.update(" \t\n\r\f\v")
                else:
                    chars.add(escaped)
                i += 2
            else:
                # Regular character
                chars.add(content[i])
                i += 1

        return chars

    def _unescape_char(self, char: str) -> str:
        """Unescape special characters."""
        escape_map = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "\\": "\\", "-": "-", "]": "]", "[": "["}
        return escape_map.get(char, char)

    def _parse_quantifier(self) -> str:
        """Parse quantifier {n,m}, +, *, ?"""
        if self.pos >= self.length:
            return ""

        char = self.pattern[self.pos]

        if char == "{":
            start = self.pos
            while self.pos < self.length and self.pattern[self.pos] != "}":
                self.pos += 1
            if self.pos < self.length:
                self.pos += 1  # Skip '}'
                return self.pattern[start : self.pos]
        elif char in "+*?":
            self.pos += 1
            return char

        return ""

    def _quantifier_to_range(self, quantifier: str) -> Tuple[int, Union[int, float]]:
        """Convert quantifier to length range."""
        if not quantifier:
            return (1, 1)
        elif quantifier == "+":
            return (1, self.max_quantifier_length)
        elif quantifier == "*":
            return (0, self.max_quantifier_length)
        elif quantifier == "?":
            return (0, 1)
        elif quantifier.startswith("{") and quantifier.endswith("}"):
            content = quantifier[1:-1]
            try:
                if "," in content:
                    parts = content.split(",")
                    min_val = int(parts[0]) if parts[0] else 0
                    max_val = int(parts[1]) if parts[1] else float("inf")
                    return (min_val, max_val)
                else:
                    val = int(content)
                    return (val, val)
            except ValueError:
                logger.warning(f"Invalid quantifier: {quantifier}")
                return (1, 1)
        else:
            return (1, 1)

    def _parse_special(self) -> Optional[Segment]:
        """Parse special characters and escape sequences preserving original format."""
        char = self.pattern[self.pos]

        if char == "\\" and self.pos + 1 < self.length:
            next_char = self.pattern[self.pos + 1]

            # Handle \w and \d as character classes
            if next_char == "w":
                self.pos += 2
                # Parse quantifier
                quantifier = self._parse_quantifier()
                min_len, max_len = self._quantifier_to_range(quantifier)

                # Create character class for \w
                segment = CharClassSegment()
                segment.position = self.pos - 2
                segment.charset = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
                segment.min_length = min_len
                segment.max_length = max_len
                segment.original_quantifier = quantifier
                segment.original_charset_str = "[a-zA-Z0-9_]"
                segment.case_sensitive = self._detect_case_sensitivity()
                return segment

            elif next_char == "d":
                self.pos += 2
                # Parse quantifier
                quantifier = self._parse_quantifier()
                min_len, max_len = self._quantifier_to_range(quantifier)

                # Create character class for \d
                segment = CharClassSegment()
                segment.position = self.pos - 2
                segment.charset = set("0123456789")
                segment.min_length = min_len
                segment.max_length = max_len
                segment.original_quantifier = quantifier
                segment.original_charset_str = "[0-9]"
                segment.case_sensitive = self._detect_case_sensitivity()
                return segment
            else:
                # Handle other escape sequences - preserve original escape
                original_escape = self.pattern[self.pos : self.pos + 2]
                self.pos += 2
                segment = FixedSegment()
                segment.position = self.pos - 2
                segment.content = original_escape  # Preserve original escape like \/
                return segment
        else:
            # Handle other special characters as fixed content
            self.pos += 1
            segment = FixedSegment()
            segment.position = self.pos - 1
            segment.content = char
            return segment

    def _parse_fixed(self) -> FixedSegment:
        """Parse fixed string segment preserving escape sequences."""
        start_pos = self.pos
        content = ""

        while self.pos < self.length and self.pattern[self.pos] not in r"()[].*+?{}^$|\\":
            content += self.pattern[self.pos]
            self.pos += 1

        segment = FixedSegment()
        segment.position = start_pos
        segment.content = content
        return segment

    def _detect_case_sensitivity(self) -> bool:
        """Detect if pattern has (?-i) case sensitive flag."""
        return "(?-i)" in self.pattern

    def _calculate_prefix_lengths(self, segments: List[Segment]) -> None:
        """Calculate fixed prefix length for each segment."""
        prefix_length = 0

        for segment in segments:
            segment.prefix_length = prefix_length

            if isinstance(segment, FixedSegment):
                prefix_length += len(segment.content)
            elif isinstance(segment, GroupSegment):
                # For groups, calculate internal prefix lengths
                self._calculate_prefix_lengths(segment.content)
            elif isinstance(segment, OptionalSegment):
                # For optional segments, prefix length doesn't increase
                self._calculate_prefix_lengths(segment.content)
                # Don't add to prefix_length since it's optional
