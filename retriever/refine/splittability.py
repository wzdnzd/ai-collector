"""
Splittability analyzer to determine if a regex pattern can be further split safely.
This module prevents infinite loops and ensures convergence in regex enumeration.
"""

import math
import re
import threading
from typing import Dict, List, Optional, Set, Tuple

from logger import get_engine_logger

from .segments import (
    CharClassSegment,
    FixedSegment,
    GroupSegment,
    OptionalSegment,
    Segment,
)

logger = get_engine_logger()


class SplittabilityAnalyzer:
    """Analyze whether a regex pattern can be safely split further based on mathematical principles - Singleton pattern."""

    _instance = None
    _lock = threading.Lock()

    def __init__(
        self,
        enable_recursion_limit: bool = False,
        enable_value_threshold: bool = False,
        enable_resource_limit: bool = False,
        max_recursion_depth: int = 10,
        min_enumeration_value: float = 0.01,
        max_resource_cost: float = 50.0,
    ):
        # Core mathematical conditions (always enabled)
        self.split_history: List[str] = []  # Track splitting history for convergence analysis
        self.seen_patterns: Set[str] = set()  # Fast cycle detection

        # Optional safety conditions (configurable)
        self.enable_recursion_limit = enable_recursion_limit
        self.enable_value_threshold = enable_value_threshold
        self.enable_resource_limit = enable_resource_limit

        # Configuration parameters
        self.max_recursion_depth = max_recursion_depth
        self.min_enumeration_value = min_enumeration_value
        self.max_resource_cost = max_resource_cost

    @classmethod
    def get_instance(
        cls,
        enable_recursion_limit: bool = False,
        enable_value_threshold: bool = False,
        enable_resource_limit: bool = False,
        max_recursion_depth: int = 10,
        min_enumeration_value: float = 0.01,
        max_resource_cost: float = 50.0,
    ) -> "SplittabilityAnalyzer":
        """Get singleton instance with optional configuration."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance.__init__(
                        enable_recursion_limit=enable_recursion_limit,
                        enable_value_threshold=enable_value_threshold,
                        enable_resource_limit=enable_resource_limit,
                        max_recursion_depth=max_recursion_depth,
                        min_enumeration_value=min_enumeration_value,
                        max_resource_cost=max_resource_cost,
                    )
                    logger.debug("SplittabilityAnalyzer singleton instance initialized")
        return cls._instance

    def can_split_further(
        self, pattern: str, segments: List[Segment], current_depth: int = 0, parent_pattern: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Determine if a regex pattern can be split further based on mathematical principles.
        Time complexity: O(n) where n is the number of segments.

        Core conditions (always checked):
        1. Unsupported features check
        2. Cycle detection
        3. Enumerability check
        4. Convergence analysis

        Returns:
            Tuple[bool, str]: (can_split, reason)
        """
        logger.debug(f"Analyzing pattern: {pattern[:50]}{'...' if len(pattern) > 50 else ''}")

        # FAST CHECK 1: Unsupported features (O(1) amortized)
        if self._has_unsupported_features(pattern):
            return False, "Pattern contains unsupported features"

        # FAST CHECK 2: Cycle detection (O(1) average)
        if self._detect_cycle(pattern):
            return False, "Cycle detected in pattern analysis"

        # FAST CHECK 3: Find enumerable segments (O(n))
        enumerable_segments = self._find_enumerable_segments(segments)
        if not enumerable_segments:
            return False, "No enumerable segments found"

        # FAST CHECK 4: Convergence analysis (O(n))
        if not self._will_converge(pattern, segments, enumerable_segments):
            return False, "Enumeration will not converge"

        # Optional checks (only if enabled)
        if self.enable_recursion_limit and current_depth >= self.max_recursion_depth:
            return False, f"Recursion limit reached: {current_depth}"

        if self.enable_value_threshold:
            total_value = self._calc_total_value(enumerable_segments, segments)
            if total_value < self.min_enumeration_value:
                return False, f"Value threshold not met: {total_value:.3f}"

        if self.enable_resource_limit:
            cost = self._estimate_cost(enumerable_segments)
            if cost > self.max_resource_cost:
                return False, f"Resource limit exceeded: {cost:.2f}"

        # Record successful analysis
        self.split_history.append(pattern)
        self.seen_patterns.add(pattern)

        reason = f"Splittable: {len(enumerable_segments)} enumerable segments"
        logger.debug(f"Pattern analysis passed: {reason}")
        return True, reason

    def _has_unsupported_features(self, pattern: str) -> bool:
        """
        Check for truly unsupported regex features that prevent enumeration entirely.
        Time complexity: O(1) amortized.

        Note: Many advanced features like lookahead, lookbehind, word boundaries, and
        possessive quantifiers are actually supported if there are enumerable segments.
        """
        if "\\" in pattern:
            # Backreferences create dependencies that prevent safe splitting
            if re.search(r"\\[1-9]", pattern):
                logger.debug("Backreference detected - prevents enumeration")
                return True

            # Unicode properties are not supported (ASCII only)
            if "\\p{" in pattern or "\\P{" in pattern:
                logger.debug("Unicode property detected - not supported")
                return True

        # Conditional expressions create complex control flow
        if "(?(" in pattern:
            logger.debug("Conditional expression detected - not supported")
            return True

        # Recursive patterns are not supported
        if "(?R)" in pattern or "(?0)" in pattern:
            logger.debug("Recursive pattern detected - not supported")
            return True

        return False

    def _detect_cycle(self, pattern: str) -> bool:
        """Fast cycle detection using set lookup. Time complexity: O(1) average."""
        # Only detect true cycles - when same pattern appears multiple times
        if len(self.split_history) >= 5:
            # Check if pattern appears multiple times in recent history
            recent_patterns = self.split_history[-5:]
            if recent_patterns.count(pattern) >= 3:
                logger.debug("True cycle detected: pattern appears 3+ times")
                return True

        return False

    def _find_enumerable_segments(self, segments: List[Segment]) -> List[CharClassSegment]:
        """Find segments that can be enumerated. Time complexity: O(n)."""
        enumerable = []

        # Iterative traversal to avoid recursion overhead
        stack = list(segments)

        while stack:
            segment = stack.pop()

            if isinstance(segment, CharClassSegment):
                if self._is_enumerable(segment):
                    enumerable.append(segment)
            elif isinstance(segment, (GroupSegment, OptionalSegment)):
                # Add nested segments to stack
                if hasattr(segment, "content") and segment.content:
                    stack.extend(segment.content)

        return enumerable

    def _is_enumerable(self, segment: CharClassSegment) -> bool:
        """
        Check if a segment can be enumerated to reduce search space.
        Focus on whether enumerating 2-3 characters would be beneficial.
        Time complexity: O(1).
        """
        # Must have charset
        if not hasattr(segment, "charset") or not segment.charset:
            return False

        charset_size = len(segment.charset)

        # Must have multiple characters to benefit from enumeration
        if charset_size <= 1:
            return False

        # For enumeration to be worthwhile, we need reasonable charset size
        # Even larger charsets can be enumerable if we only enumerate a few positions
        if charset_size > 128:  # ASCII limit
            return False

        # Check length constraints if available
        if hasattr(segment, "min_length") and hasattr(segment, "max_length"):
            min_len = getattr(segment, "min_length", 1)
            max_len = getattr(segment, "max_length", float("inf"))

            # Zero-length segments don't contribute to enumeration
            if max_len == 0:
                return False

            # Only check if the segment can contribute at least one character for enumeration
            # Even if min_len is very large, we can still enumerate the first few positions
            # The remaining length will be reduced accordingly (e.g., min_len=100 -> min_len=98 after enumerating 2 chars)

        return True

    def _will_converge(
        self, pattern: str, segments: List[Segment], enumerable_segments: List[CharClassSegment]
    ) -> bool:
        """
        Check if enumeration will converge based on mathematical principles.
        Core conditions: 1) Enumerable charset exists 2) Search space reduces after enumeration
        Time complexity: O(1) - simplified for mathematical convergence only.
        """
        # Must have enumerable segments to proceed
        if not enumerable_segments:
            return False

        # Mathematical convergence: if we can enumerate characters from a charset > 1,
        # the search space will always reduce when we fix 2-3 positions
        for segment in enumerable_segments:
            charset_size = len(segment.charset)
            if charset_size > 1:
                # Enumeration always converges when charset > 1
                # because fixing positions reduces total combinations
                logger.debug(f"Convergence guaranteed: charset_size={charset_size}")
                return True

        return False

    def _calc_log_search_space(self, segments: List[Segment]) -> float:
        """Calculate search space in logarithmic space to avoid overflow. Time complexity: O(n)."""
        log_total = 0.0

        for segment in segments:
            if isinstance(segment, CharClassSegment):
                charset_size = len(getattr(segment, "charset", set()))
                if charset_size <= 1:
                    continue  # No contribution to search space

                # Estimate average length for quantified segments
                avg_length = 1.0
                if hasattr(segment, "min_length") and hasattr(segment, "max_length"):
                    min_len = getattr(segment, "min_length", 1)
                    max_len = getattr(segment, "max_length", float("inf"))

                    if max_len == float("inf"):
                        # Conservative estimate for unbounded quantifiers
                        avg_length = max(min_len, 3)  # Reasonable default
                    else:
                        avg_length = (min_len + max_len) / 2

                # Add log(charset_size^avg_length) = avg_length * log(charset_size)
                if charset_size > 1:  # Ensure we have valid charset
                    log_total += avg_length * math.log(charset_size)

            elif isinstance(segment, OptionalSegment):
                # Optional: log(1 + inner_space) ≈ log(2) for simplicity
                log_total += math.log(2)
                # Note: We skip recursive calculation for efficiency

            elif isinstance(segment, GroupSegment):
                # Groups: process inner content (simplified)
                if hasattr(segment, "content") and segment.content:
                    log_total += self._calc_log_search_space(segment.content)

        return log_total

    def _calc_enumeration_benefit(self, segment: CharClassSegment, all_segments: List[Segment]) -> float:
        """
        Calculate the benefit of enumerating this segment.
        Returns the expected reduction in search space complexity.
        Time complexity: O(1).
        """
        charset_size = len(segment.charset)

        if charset_size <= 1:
            return 0.0  # No benefit from single character

        # Calculate the context value - how much fixed content surrounds this segment
        context_value = self._calc_context_value(segment, all_segments)

        # Enumeration benefit is higher when:
        # 1. There's good context (fixed content around the segment)
        # 2. The charset is not too large
        # 3. We're only enumerating a few positions

        # Base benefit from charset reduction
        # We typically enumerate 2-3 positions, so benefit = charset_size^2 or charset_size^3
        enum_positions = min(3, getattr(segment, "min_length", 1))
        enum_combinations = charset_size**enum_positions

        # Benefit calculation favors smaller charsets for enumeration
        # Larger charsets are harder to enumerate effectively
        if charset_size <= 10:
            base_benefit = 1.0  # Small charset, high benefit
        elif charset_size <= 26:
            base_benefit = 0.8  # Alphabet size, good benefit
        elif charset_size <= 62:
            base_benefit = 0.6  # Alphanumeric, moderate benefit
        else:
            base_benefit = 0.3  # Large charset, low benefit

        # Scale by enumeration positions (more positions = more benefit)
        base_benefit *= enum_positions * 0.2

        # Scale by context - more context means more benefit
        context_multiplier = 1.0 + (context_value * 0.2)

        total_benefit = base_benefit * context_multiplier

        logger.debug(f"Enumeration benefit for charset size {charset_size}: {total_benefit:.3f}")
        return total_benefit

    def _calc_context_value(self, segment: CharClassSegment, all_segments: List[Segment]) -> float:
        """Calculate how much fixed context surrounds this segment."""
        if not hasattr(segment, "position"):
            return 0.0

        position = segment.position
        context_value = 0.0

        # Count fixed content before and after this segment
        for other_segment in all_segments:
            if isinstance(other_segment, FixedSegment):
                other_pos = getattr(other_segment, "position", 0)
                content_len = len(getattr(other_segment, "content", ""))

                # Weight nearby fixed content more heavily
                distance = abs(other_pos - position)
                weight = 1.0 / (1.0 + distance)
                context_value += content_len * weight

        return min(context_value, 5.0)  # Cap the context value

    def _making_progress(self, current_pattern: str) -> bool:
        """Fast progress check using simple heuristics. Time complexity: O(1)."""
        if not self.split_history:
            return True  # First pattern, assume progress

        prev_pattern = self.split_history[-1]

        # Quick length-based check
        curr_len = len(current_pattern)
        prev_len = len(prev_pattern)

        # Progress if pattern is getting more specific (longer) but not too long
        if curr_len > prev_len and curr_len < prev_len * 2:
            return True

        # Progress if pattern complexity is changing appropriately
        curr_brackets = current_pattern.count("[")
        prev_brackets = prev_pattern.count("[")

        # Good if we're reducing variable parts
        if curr_brackets < prev_brackets:
            return True

        # More lenient similarity check - only reject if patterns are nearly identical
        if abs(curr_len - prev_len) <= 1 and curr_brackets == prev_brackets:
            # Check character-level similarity for very similar lengths
            common = sum(1 for a, b in zip(current_pattern, prev_pattern) if a == b)
            similarity = common / max(curr_len, prev_len, 1)
            if similarity > 0.95:  # Much more lenient threshold
                return False

        return True  # Default to allowing progress

    def _calc_total_value(self, enumerable_segments: List[CharClassSegment], all_segments: List[Segment]) -> float:
        """Fast calculation of total enumeration value. Time complexity: O(n)."""
        total_value = 0.0

        for seg in enumerable_segments:
            # Use cached value if available
            if hasattr(seg, "value") and seg.value > 0:
                total_value += seg.value
            else:
                # Fast estimation based on charset size and position
                charset_size = len(seg.charset)
                if charset_size > 1:
                    # Simple value estimation: log-based to avoid large numbers
                    base_value = math.log(charset_size)

                    # Boost for segments with good context (position-based heuristic)
                    if hasattr(seg, "position"):
                        position_boost = min(seg.position * 0.1, 1.0)
                        base_value *= 1 + position_boost

                    total_value += base_value

        return total_value

    def _estimate_cost(self, enumerable_segments: List[CharClassSegment]) -> float:
        """Fast cost estimation. Time complexity: O(n)."""
        if not enumerable_segments:
            return float("inf")

        total_cost = 0.0

        for seg in enumerable_segments:
            charset_size = len(seg.charset)

            # Base cost: logarithmic in charset size
            base_cost = math.log(charset_size + 1)

            # Length complexity cost
            if hasattr(seg, "max_length"):
                max_len = getattr(seg, "max_length", 1)
                if max_len == float("inf"):
                    base_cost += 3.0  # High cost for unbounded
                else:
                    base_cost += math.log(max_len + 1) * 0.5

            total_cost += base_cost

        return total_cost

    def reset_history(self):
        """Reset analysis state for new session."""
        self.split_history.clear()
        self.seen_patterns.clear()
        logger.debug("Analysis state reset")

    def get_stats(self) -> Dict:
        """Get analysis statistics."""
        return {
            "patterns_analyzed": len(self.split_history),
            "unique_patterns": len(self.seen_patterns),
            "max_recursion_depth": self.max_recursion_depth,
            "min_enumeration_value": self.min_enumeration_value,
            "max_resource_cost": self.max_resource_cost,
        }
