#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enumeration strategy optimizer for regex patterns.
"""

import itertools
import math
import threading
from typing import List, Tuple

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


class EnumerationOptimizer:
    """Optimize enumeration strategy for regex patterns - Singleton pattern."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, max_queries: int = 100000000):
        self.max_queries = max_queries

    @classmethod
    def get_instance(cls, max_queries: int = 100000000) -> "EnumerationOptimizer":
        """Get singleton instance with optional configuration."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    cls._instance.__init__(max_queries)
                    logger.debug("EnumerationOptimizer singleton instance initialized")
        return cls._instance

    def optimize(self, segments: List[Segment]) -> EnumerationStrategy:
        """Find optimal enumeration strategy."""
        if not segments:
            return EnumerationStrategy([], segments, 0.0, 1)

        try:
            # Expand all variants from optional segments
            variants = self._expand_variants(segments)

            best_strategy = None
            best_value = 0.0

            # Find best strategy across all variants
            for variant in variants:
                strategy = self._optimize_variant(variant)
                if strategy.value > best_value:
                    best_strategy = strategy
                    best_value = strategy.value

            # If no strategy found, ensure we at least try to enumerate available segments
            if best_strategy is None or best_strategy.value == 0.0:
                # Find any enumerable segments and create a basic strategy
                def find_vars(segs):
                    vars = []
                    for seg in segs:
                        if isinstance(seg, CharClassSegment):
                            vars.append(seg)
                        elif isinstance(seg, (GroupSegment, OptionalSegment)):
                            vars.extend(find_vars(seg.content))
                    return vars

                vars = find_vars(segments)
                if vars:
                    # Calculate values for all segments
                    for segment in vars:
                        segment.value = self._calculate_value(segment, segments)

                    # Sort by value and select the best one
                    sorted_segs = sorted(vars, key=lambda s: s.value, reverse=True)
                    best_segment = sorted_segs[0]

                    # Create a strategy with the best segment
                    optimal_depth = self._calculate_segment_optimal_depth(best_segment)
                    queries = len(best_segment.charset) ** optimal_depth if optimal_depth > 0 else 1

                    best_strategy = EnumerationStrategy([best_segment], segments, best_segment.value, queries)
                    logger.info(
                        f"Fallback strategy selected: 1 segment, value={best_strategy.value:.3f}, queries={best_strategy.queries}"
                    )

            return best_strategy or EnumerationStrategy([], segments, 0.0, 1)

        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            return EnumerationStrategy([], segments, 0.0, 1)

    def evaluate_strategies_for_partitions(
        self, segments: List[Segment], partitions: int
    ) -> tuple[EnumerationStrategy, bool]:
        """
        Evaluate strategies to find one that generates >= partitions queries.

        Returns:
            tuple: (best_strategy, found_suitable_strategy)
        """
        if not segments or partitions <= 0:
            return EnumerationStrategy([], segments, 0.0, 1), False

        try:
            # Expand all variants from optional segments
            variants = self._expand_variants(segments)

            suitable = []
            all_strats = []

            # Evaluate all possible strategies
            for variant in variants:
                strategies = self._generate_all_strategies(variant)
                all_strats.extend(strategies)

                # Find strategies that meet partition requirement
                for strategy in strategies:
                    if strategy.queries >= partitions:
                        suitable.append(strategy)

            # If we found suitable strategies, return the one with minimum enumeration depth
            if suitable:
                # Prioritize single-segment strategies over multi-segment ones
                single = [s for s in suitable if len(s.segments) == 1]
                if single:
                    best = self._select_strategy_with_min_depth(single, partitions)
                else:
                    best = self._select_strategy_with_min_depth(suitable, partitions)
                return best, True

            # If no suitable strategy found, return the one that generates most queries
            if all_strats:
                # Prioritize single-segment strategies
                single = [s for s in all_strats if len(s.segments) == 1]
                if single:
                    best = max(single, key=lambda s: s.queries)
                else:
                    best = max(all_strats, key=lambda s: s.queries)
                return best, False

            return EnumerationStrategy([], segments, 0.0, 1), False

        except Exception as e:
            logger.warning(f"Strategy evaluation failed: {e}")
            return EnumerationStrategy([], segments, 0.0, 1), False

    def _generate_all_strategies(self, segments: List[Segment]) -> List[EnumerationStrategy]:
        """Generate all possible enumeration strategies for given segments."""

        # Find all variable segments
        def find_vars(segs):
            vars = []
            for seg in segs:
                if isinstance(seg, CharClassSegment):
                    vars.append(seg)
                elif isinstance(seg, (GroupSegment, OptionalSegment)):
                    vars.extend(find_vars(seg.content))
            return vars

        vars = find_vars(segments)

        if not vars:
            return [EnumerationStrategy([], segments, 0.0, 1)]

        # Calculate enumeration values
        for segment in vars:
            segment.value = self._calculate_value(segment, segments)

        strategies = []

        # Generate strategies for different combinations and depths
        # Prioritize single-segment strategies first, then multi-segment
        for combo_size in range(1, min(4, len(vars) + 1)):
            for combo in itertools.combinations(vars, combo_size):
                # Try different enumeration depths
                for depth in range(1, 5):  # depths 1-4
                    strategy = self._create_strategy_with_depth(list(combo), segments, depth)
                    if strategy.queries > 0:
                        strategies.append(strategy)

        return strategies

    def _create_strategy_with_depth(
        self, segments_to_enum: List[CharClassSegment], all_segments: List[Segment], depth: int
    ) -> EnumerationStrategy:
        """Create enumeration strategy with specific depth."""
        total_queries = 1
        total_value = 0.0

        for segment in segments_to_enum:
            charset_size = len(segment.charset)
            if charset_size > 0:
                segment_queries = charset_size**depth
                total_queries *= segment_queries
                total_value += segment.value

            # Early termination if queries become too large
            if total_queries > self.max_queries * 100:
                break

        return EnumerationStrategy(segments_to_enum, all_segments, total_value, total_queries)

    def _select_strategy_with_min_depth(
        self, strategies: List[EnumerationStrategy], partitions: int
    ) -> EnumerationStrategy:
        """Select strategy with minimum enumeration depth that meets partition requirement."""

        # Calculate effective depth for each strategy
        def calculate_effective_depth(strategy: EnumerationStrategy) -> float:
            if not strategy.segments:
                return 0.0

            total_depth = 0.0
            for segment in strategy.segments:
                charset_size = len(segment.charset)
                if charset_size > 0 and strategy.queries > 0:
                    # Calculate depth from queries = charset_size^depth
                    segment_contribution = strategy.queries ** (1.0 / len(strategy.segments))
                    if segment_contribution > 0:
                        depth = math.log(segment_contribution) / math.log(charset_size)
                        total_depth += depth

            return total_depth / len(strategy.segments) if strategy.segments else 0.0

        # Calculate fixed context length for enumeration segment
        def calc_context_length(strategy: EnumerationStrategy, depth: int) -> int:
            """Calculate the length of fixed context that can be formed with enumeration segment."""
            if not strategy.segments or len(strategy.segments) != 1:
                return 0

            segment = strategy.segments[0]
            segments = strategy.original
            pos = segment.position

            # Calculate preceding fixed content length
            before = 0
            for i in range(pos - 1, -1, -1):
                if isinstance(segments[i], FixedSegment):
                    before += len(segments[i].content)
                elif isinstance(segments[i], OptionalSegment):
                    continue  # Skip optional segments
                else:
                    break  # Stop at variable/group segments

            # Calculate following fixed content length
            # Only if enumeration fully covers current segment
            after = 0
            remaining = segment.min_length - depth
            if remaining <= 0:
                for i in range(pos + 1, len(segments)):
                    if isinstance(segments[i], FixedSegment):
                        after += len(segments[i].content)
                    else:
                        break

            return before + depth + after

        # Calculate total length of segments being enumerated
        def calc_segment_length(strategy: EnumerationStrategy) -> int:
            if not strategy.segments:
                return 0
            return sum(seg.min_length for seg in strategy.segments)

        # Calculate strategy score for selection
        def calc_score(strategy: EnumerationStrategy) -> tuple:
            depth = calculate_effective_depth(strategy)
            length = calc_segment_length(strategy)
            excess = strategy.queries - partitions
            context = calc_context_length(strategy, max(1, int(math.ceil(depth))))

            # Find immediate preceding fixed segment length for tie-breaking
            before = 0
            if strategy.segments:
                pos = strategy.segments[0].position
                for i in range(pos - 1, -1, -1):
                    if isinstance(strategy.original[i], FixedSegment):
                        before = len(strategy.original[i].content)
                        break
                    elif isinstance(strategy.original[i], OptionalSegment):
                        continue
                    else:
                        break

            # Prefer shorter segments, longer context, longer immediate before
            return (length, -context, -before, depth, excess)

        suitable = [s for s in strategies if s.queries >= partitions]
        if suitable:
            return min(suitable, key=calc_score)

        return strategies[0] if strategies else EnumerationStrategy([], [], 0.0, 1)

    def _expand_variants(self, segments: List[Segment]) -> List[List[Segment]]:
        """Expand all possible variants from optional segments."""
        variants = [[]]

        for segment in segments:
            new_variants = []

            for variant in variants:
                if isinstance(segment, OptionalSegment):
                    # Add both empty and content variants
                    new_variants.append(variant.copy())  # Empty
                    new_variants.append(variant + segment.content)  # Content
                elif isinstance(segment, GroupSegment):
                    # Flatten group content for processing
                    flattened = segment.flatten()
                    new_variants.append(variant + flattened)
                else:
                    # Regular segment
                    new_variants.append(variant + [segment])

            variants = new_variants

            # Check for exponential growth and optimize if needed
            if len(variants) > 1000:
                # Instead of limiting, optimize by selecting most valuable variants
                variants = self._optimize_variants(variants)
                logger.info(f"Optimized {len(new_variants)} variants to {len(variants)} most valuable ones")

        return variants

    def _optimize_variants(self, variants: List[List[Segment]]) -> List[List[Segment]]:
        """Optimize variants by selecting the most valuable ones."""
        if len(variants) <= 1000:
            return variants

        # Score each variant based on fixed prefix length and complexity
        scored_variants = []
        for variant in variants:
            score = self._calculate_variant_score(variant)
            scored_variants.append((score, variant))

        # Sort by score (descending) and take top variants
        scored_variants.sort(key=lambda x: x[0], reverse=True)

        # Take top 80% of reasonable size, but ensure we don't lose valuable variants
        target_size = min(1000, max(100, len(variants) // 2))
        optimized = [variant for score, variant in scored_variants[:target_size]]

        logger.info(f"Selected {len(optimized)} highest-value variants from {len(variants)}")
        return optimized

    def _calculate_variant_score(self, variant: List[Segment]) -> float:
        """Calculate score for a variant based on its enumeration potential."""
        score = 0.0

        # Add points for fixed prefix length
        fixed_length = 0
        for segment in variant:
            if isinstance(segment, FixedSegment):
                fixed_length += len(segment.content)
            else:
                break  # Stop at first non-fixed segment

        score += fixed_length * 2  # Fixed prefix is valuable

        # Add points for variable segments with good enumeration potential
        for segment in variant:
            if isinstance(segment, CharClassSegment):
                if hasattr(segment, "value"):
                    score += segment.value
                else:
                    # Estimate value based on charset size and length
                    charset_size = len(segment.charset)
                    if charset_size > 0 and charset_size <= 100:
                        score += 1.0 / charset_size  # Smaller charset = higher value

        return score

    def _optimize_variant(self, segments: List[Segment]) -> EnumerationStrategy:
        """Optimize enumeration for single variant."""

        # Find all variable segments (including inside groups)
        def find_vars(segs):
            vars = []
            for seg in segs:
                if isinstance(seg, CharClassSegment):
                    vars.append(seg)
                elif isinstance(seg, (GroupSegment, OptionalSegment)):
                    vars.extend(find_vars(seg.content))
            return vars

        vars = find_vars(segments)

        if not vars:
            return EnumerationStrategy([], segments, 0.0, 1)

        # Calculate enumeration values
        for segment in vars:
            segment.value = self._calculate_value(segment, segments)

        # Select best enumeration combination
        return self._select_combination(vars, segments)

    def _calculate_value(self, segment: CharClassSegment, all_segments: List[Segment]) -> float:
        """Calculate enumeration value for segment."""
        try:
            # Fixed prefix length
            prefix_length = segment.prefix_length

            # Fixed suffix length
            suffix_length = 0
            for i in range(segment.position + 1, len(all_segments)):
                if isinstance(all_segments[i], FixedSegment):
                    suffix_length += len(all_segments[i])

            # Use actual combination count without artificial limits
            combinations = segment.combinations

            # Value calculation with proper mathematical scaling
            prefix_weight = math.log(max(1, prefix_length + 1))
            suffix_weight = math.log(max(1, suffix_length + 1)) * 0.3

            # Use log scaling to handle large combination counts gracefully
            if combinations > 0:
                cost_weight = math.log(combinations)
            else:
                cost_weight = 1.0

            value = (prefix_weight + suffix_weight) / max(0.1, cost_weight)

            logger.debug(
                f"Segment value: prefix={prefix_length}, suffix={suffix_length}, "
                f"combinations={combinations}, value={value:.3f}"
            )

            return value

        except Exception as e:
            logger.warning(f"Value calculation failed: {e}")
            return 0.0

    def _select_combination(self, vars: List[CharClassSegment], all_segments: List[Segment]) -> EnumerationStrategy:
        """Select best enumeration combination."""
        if not vars:
            return EnumerationStrategy([], all_segments, 0.0, 1)

        # Sort by value descending
        sorted_segs = sorted(vars, key=lambda s: s.value, reverse=True)

        best_strategy = EnumerationStrategy([], all_segments, 0.0, 1)

        # Try single segment enumeration
        for segment in sorted_segs:
            # Calculate optimal enumeration depth based on segment characteristics
            optimal_depth = self._calculate_segment_optimal_depth(segment)
            queries = len(segment.charset) ** optimal_depth if optimal_depth > 0 else 1

            # Consider this strategy if it's valuable and feasible
            if self._is_strategy_feasible(queries) and segment.value > best_strategy.value:
                best_strategy = EnumerationStrategy([segment], all_segments, segment.value, queries)

        # Try multi-segment combinations with intelligent depth selection
        max_combo_size = min(3, len(sorted_segs))  # Reasonable limit for combinations
        for combo_size in range(2, max_combo_size + 1):
            for combo in itertools.combinations(sorted_segs, combo_size):
                total_queries, total_value = self._evaluate_combination(combo)

                if self._is_strategy_feasible(total_queries) and total_value > best_strategy.value:
                    best_strategy = EnumerationStrategy(list(combo), all_segments, total_value, total_queries)

        logger.info(
            f"Selected strategy: {len(best_strategy.segments)} segments, "
            f"value={best_strategy.value:.3f}, queries={best_strategy.queries}"
        )

        return best_strategy

    def _calculate_segment_optimal_depth(self, segment: CharClassSegment) -> int:
        """Calculate optimal enumeration depth for a single segment."""
        charset_size = len(segment.charset)

        if charset_size == 0:
            return 0

        # Base depth on enumeration value and charset characteristics
        if segment.value > 15:  # Very high value
            base_depth = 3
        elif segment.value > 8:  # High value
            base_depth = 2
        else:  # Medium/low value
            base_depth = 1

        # Adjust based on charset size
        if charset_size <= 10:  # Small charset, can afford deeper enumeration
            depth = min(base_depth + 1, 4)
        elif charset_size <= 30:  # Medium charset
            depth = base_depth
        else:  # Large charset, use shallower enumeration
            depth = max(1, base_depth - 1)

        # Don't enumerate more than minimum required length
        if hasattr(segment, "min_length") and segment.min_length > 0:
            depth = min(depth, segment.min_length)

        return depth

    def _is_strategy_feasible(self, query_count: int) -> bool:
        """Check if a strategy with given query count is feasible."""
        # Instead of hard limits, use exponential cost function
        if query_count <= 0:
            return False

        # Allow larger query counts but with exponentially decreasing preference
        # This ensures we don't artificially limit but prefer smaller counts
        return query_count <= self.max_queries or (
            query_count <= self.max_queries * 10 and query_count <= 50000  # Reasonable upper bound for practical use
        )

    def _evaluate_combination(self, combo: tuple) -> Tuple[int, float]:
        """Evaluate a combination of segments for total queries and value."""
        total_queries = 1
        total_value = 0.0

        for segment in combo:
            # Use intelligent depth calculation
            depth = max(1, self._calculate_segment_optimal_depth(segment) - 1)  # Reduce for combinations
            segment_queries = len(segment.charset) ** depth

            total_queries *= segment_queries
            total_value += segment.value

            # Early termination if queries become too large
            if total_queries > self.max_queries * 100:  # Much more generous limit
                break

        return total_queries, total_value
