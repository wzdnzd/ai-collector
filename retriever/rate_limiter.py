#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rate limiting system using Token Bucket algorithm.
Supports per-service rate limits with adaptive adjustment and burst handling.
"""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import constants
from config import RateLimit

from logger import get_rate_limiter_logger

# Get rate limiter logger
logger = get_rate_limiter_logger()


@dataclass
class TokenBucket:
    """Token bucket for rate limiting with burst support"""

    def __init__(self, rate: float, burst: int, adaptive: bool = True):
        self.rate = rate  # tokens per second
        self.burst = burst  # maximum tokens
        self.adaptive = adaptive  # enable adaptive rate adjustment
        self.tokens = float(burst)  # current tokens
        self.last_update = time.time()
        self.lock = threading.Lock()

        # Adaptive rate adjustment
        self.original_rate = rate
        self.consecutive_success = 0
        self.consecutive_failures = 0
        self.last_adjustment = time.time()

    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from bucket, return True if successful"""
        with self.lock:
            now = time.time()

            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            # Check if enough tokens available
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time needed to acquire tokens"""
        with self.lock:
            if self.tokens >= tokens:
                return 0.0

            needed = tokens - self.tokens
            return needed / self.rate

    def adjust_rate(self, success: bool):
        """Adjust rate based on success/failure feedback"""
        if not self.adaptive:
            return

        with self.lock:
            now = time.time()

            if success:
                self.consecutive_success += 1
                self.consecutive_failures = 0

                # Gradually increase rate after sustained success
                if (
                    self.consecutive_success >= 10
                    and now - self.last_adjustment > 60
                    and self.rate < self.original_rate
                ):

                    old_rate = self.rate
                    self.rate = min(self.original_rate, self.rate * 1.1)
                    self.last_adjustment = now

                    if old_rate != self.rate:
                        logger.info(f"Rate increased from {old_rate:.2f} to {self.rate:.2f}")

            else:
                self.consecutive_failures += 1
                self.consecutive_success = 0

                # Decrease rate on failure
                if self.consecutive_failures >= 2:
                    old_rate = self.rate
                    self.rate = max(0.1, self.rate * 0.5)  # Reduce by 50%, minimum 0.1
                    self.last_adjustment = now
                    self.consecutive_failures = 0  # Reset to avoid further reduction

                    if old_rate != self.rate:
                        logger.warning(f"Rate decreased from {old_rate:.2f} to {self.rate:.2f}")

    def get_stats(self) -> Dict[str, float]:
        """Get current bucket statistics"""
        with self.lock:
            return {
                "rate": self.rate,
                "burst": self.burst,
                "tokens": self.tokens,
                "utilization": 1.0 - (self.tokens / self.burst),
            }


class RateLimiter:
    """Multi-service rate limiter with Token Bucket algorithm"""

    def __init__(self, rate_limits: Dict[str, RateLimit]):
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

        # Initialize buckets for each service
        for service, limit in rate_limits.items():
            self.buckets[service] = TokenBucket(rate=limit.rate, burst=limit.burst, adaptive=limit.adaptive)

        logger.info(f"Initialized rate limiter with {len(self.buckets)} services")

    def acquire(self, service: str, tokens: int = 1) -> bool:
        """Acquire tokens for a service"""
        bucket = self._get_bucket(service)
        if not bucket:
            return True  # No limit configured, allow request

        return bucket.acquire(tokens)

    def wait_time(self, service: str, tokens: int = 1) -> float:
        """Get wait time needed for tokens"""
        bucket = self._get_bucket(service)
        if not bucket:
            return 0.0

        return bucket.wait_time(tokens)

    def report_result(self, service: str, success: bool):
        """Report request result for adaptive rate adjustment"""
        bucket = self._get_bucket(service)
        if bucket:
            bucket.adjust_rate(success)

    def add_service(self, service: str, rate_limit: RateLimit):
        """Add a new service rate limit"""
        with self.lock:
            self.buckets[service] = TokenBucket(
                rate=rate_limit.rate, burst=rate_limit.burst, adaptive=rate_limit.adaptive
            )
            logger.info(f"Added rate limit for service: {service}")

    def update_service(self, service: str, rate_limit: RateLimit):
        """Update existing service rate limit"""
        with self.lock:
            if service in self.buckets:
                bucket = self.buckets[service]
                bucket.rate = rate_limit.rate
                bucket.burst = rate_limit.burst
                bucket.adaptive = rate_limit.adaptive
                bucket.original_rate = rate_limit.rate
                logger.info(f"Updated rate limit for service: {service}")

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all services"""
        stats = {}
        with self.lock:
            for service, bucket in self.buckets.items():
                stats[service] = bucket.get_stats()
        return stats

    def _get_bucket(self, service: str) -> Optional[TokenBucket]:
        """Get bucket for service, thread-safe"""
        with self.lock:
            return self.buckets.get(service)


class AsyncRateLimiter:
    """Async wrapper for rate limiter with automatic waiting"""

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter

    async def acquire(self, service: str, tokens: int = 1) -> bool:
        """Acquire tokens, waiting if necessary"""

        # Try immediate acquisition
        if self.rate_limiter.acquire(service, tokens):
            return True

        # Wait for tokens to become available
        wait_time = self.rate_limiter.wait_time(service, tokens)
        if wait_time > 0:
            logger.debug(f"Rate limit hit for {service}, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            return self.rate_limiter.acquire(service, tokens)

        return False

    def report_result(self, service: str, success: bool):
        """Report request result"""
        self.rate_limiter.report_result(service, success)


def create_rate_limiter(rate_limits: Dict[str, RateLimit]) -> RateLimiter:
    """Factory function to create rate limiter"""
    return RateLimiter(rate_limits)


if __name__ == "__main__":
    # Test rate limiter

    limits = {
        constants.SERVICE_TYPE_GITHUB_API: RateLimit(rate=2.0, burst=5, adaptive=True),
        "openai": RateLimit(rate=1.0, burst=3, adaptive=True),
    }

    limiter = create_rate_limiter(limits)

    # Test acquisition
    logger.info("Testing rate limiter...")

    # Burst test
    for i in range(7):
        success = limiter.acquire(constants.SERVICE_TYPE_GITHUB_API)
        logger.info(f"Request {i+1}: {'✓' if success else '✗'}")

    # Wait and try again
    time.sleep(1)
    success = limiter.acquire(constants.SERVICE_TYPE_GITHUB_API)
    logger.info(f"After 1s wait: {'✓' if success else '✗'}")

    # Test adaptive adjustment
    logger.info("Testing adaptive adjustment...")
    for i in range(5):
        limiter.report_result(constants.SERVICE_TYPE_GITHUB_API, False)  # Report failures

    stats = limiter.get_stats()
    logger.info(f"Stats after failures: {stats}")

    logger.info("Rate limiter test completed!")
