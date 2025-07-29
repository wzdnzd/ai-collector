#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load balancing system for dynamic worker thread management.
Automatically adjusts worker counts based on queue sizes and processing speeds.
"""

import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

from constants import (
    DEFAULT_ADJUSTMENT_INTERVAL,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MIN_WORKERS,
    DEFAULT_SCALE_DOWN_THRESHOLD,
    DEFAULT_SCALE_UP_THRESHOLD,
    DEFAULT_TARGET_QUEUE_SIZE,
    LB_RECENT_HISTORY_SIZE,
    STAGE_NAME_CHECK,
    STAGE_NAME_COLLECT,
    STAGE_NAME_MODELS,
    STAGE_NAME_SEARCH,
)
from models import LoadBalancerConfig

from logger import get_load_balancer_logger

# Get load balancer logger
logger = get_load_balancer_logger()


@dataclass
class WorkerMetrics:
    """Metrics for worker performance analysis"""

    stage_name: str = ""
    current_workers: int = 0
    target_workers: int = 0
    queue_size: int = 0
    processing_rate: float = 0.0
    avg_processing_time: float = 0.0
    utilization: float = 0.0
    last_adjustment: float = 0.0


class LoadBalancer:
    """Dynamic load balancer for pipeline worker threads"""

    def __init__(self, config: LoadBalancerConfig):
        self.worker_metrics: Dict[str, WorkerMetrics] = {}
        self.stages: Dict[str, Any] = {}

        # Load balancing parameters
        self.min_workers = config.min_workers
        self.max_workers = config.max_workers
        self.target_queue_size = config.target_queue_size
        self.scale_up_threshold = config.scale_up_threshold
        self.scale_down_threshold = config.scale_down_threshold
        self.adjustment_interval = config.adjustment_interval

        # Monitoring
        self.running = False
        self.balancer_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # History for trend analysis
        self.metrics_history: Dict[str, deque] = {}

        logger.info("Initialized load balancer")

    def register_stage(self, stage_name: str, stage_instance: Any):
        """Register a pipeline stage for load balancing"""
        with self.lock:
            self.stages[stage_name] = stage_instance
            self.worker_metrics[stage_name] = WorkerMetrics(stage_name=stage_name)
            self.metrics_history[stage_name] = deque(maxlen=50)

        logger.info(f"Registered stage for load balancing: {stage_name}")

    def start(self):
        """Start load balancing thread"""
        if self.running:
            return

        self.running = True
        self.balancer_thread = threading.Thread(target=self._balancing_loop, name="load-balancer", daemon=True)
        self.balancer_thread.start()

        logger.info("Started load balancer")

    def stop(self):
        """Stop load balancing thread"""
        self.running = False

        if self.balancer_thread and self.balancer_thread.is_alive():
            self.balancer_thread.join(timeout=2.0)
            if self.balancer_thread.is_alive():
                logger.warning("Load balancer thread did not stop gracefully")

        logger.info("Stopped load balancer")

    def update_metrics(self, stage_name: str, metrics_data):
        """Update metrics for a specific stage"""
        with self.lock:
            if stage_name not in self.worker_metrics:
                return

            metrics = self.worker_metrics[stage_name]

            # Update metrics from data class
            metrics.queue_size = metrics_data.queue_size
            metrics.current_workers = metrics_data.active_workers
            metrics.processing_rate = metrics_data.processing_rate
            metrics.avg_processing_time = metrics_data.avg_processing_time

            # Calculate utilization
            if metrics.current_workers > 0:
                metrics.utilization = min(1.0, metrics.queue_size / (metrics.current_workers * self.target_queue_size))

            # Store in history
            self.metrics_history[stage_name].append(
                {
                    "timestamp": time.time(),
                    "queue_size": metrics.queue_size,
                    "workers": metrics.current_workers,
                    "utilization": metrics.utilization,
                    "processing_rate": metrics.processing_rate,
                }
            )

    def get_recommended_workers(self, stage_name: str) -> int:
        """Get recommended worker count for a stage"""
        with self.lock:
            if stage_name not in self.worker_metrics:
                return self.min_workers

            metrics = self.worker_metrics[stage_name]

            # Base calculation on queue size and processing rate
            if metrics.processing_rate > 0:
                # Calculate workers needed to maintain target queue size
                target_workers = max(1, int(metrics.queue_size / self.target_queue_size))
            else:
                # Fallback to utilization-based calculation
                if metrics.utilization > self.scale_up_threshold:
                    target_workers = min(self.max_workers, metrics.current_workers + 1)
                elif metrics.utilization < self.scale_down_threshold:
                    target_workers = max(self.min_workers, metrics.current_workers - 1)
                else:
                    target_workers = metrics.current_workers

            # Apply constraints
            target_workers = max(self.min_workers, min(self.max_workers, target_workers))

            # Consider trend analysis
            target_workers = self._apply_trend_analysis(stage_name, target_workers)

            metrics.target_workers = target_workers
            return target_workers

    def should_adjust_workers(self, stage_name: str) -> bool:
        """Check if worker count should be adjusted"""
        with self.lock:
            if stage_name not in self.worker_metrics:
                return False

            metrics = self.worker_metrics[stage_name]
            current_time = time.time()

            # Check if enough time has passed since last adjustment
            if current_time - metrics.last_adjustment < self.adjustment_interval:
                return False

            # Check if adjustment is needed
            recommended = self.get_recommended_workers(stage_name)
            return recommended != metrics.current_workers

    def adjust_workers(self, stage_name: str) -> bool:
        """Adjust worker count for a stage"""
        with self.lock:
            if stage_name not in self.stages:
                return False

            stage = self.stages[stage_name]
            metrics = self.worker_metrics[stage_name]
            target_workers = self.get_recommended_workers(stage_name)

            if target_workers == metrics.current_workers:
                return False

            try:
                # Attempt to adjust workers (this depends on stage implementation)
                if hasattr(stage, "adjust_workers"):
                    success = stage.adjust_workers(target_workers)
                elif hasattr(stage, "set_worker_count"):
                    success = stage.set_worker_count(target_workers)
                else:
                    # Fallback: log recommendation
                    logger.info(
                        f"Recommend adjusting {stage_name} workers: {metrics.current_workers} -> {target_workers}"
                    )
                    success = False

                if success:
                    metrics.last_adjustment = time.time()
                    logger.info(f"Adjusted {stage_name} workers: {metrics.current_workers} -> {target_workers}")
                    return True

            except Exception as e:
                logger.error(f"Failed to adjust workers for {stage_name}: {e}")

            return False

    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get current load balancing statistics"""
        with self.lock:
            stats = {
                "timestamp": time.time(),
                "stages": {},
                "total_workers": 0,
                "total_target_workers": 0,
                "total_queue_size": 0,
            }

            for stage_name, metrics in self.worker_metrics.items():
                stats["stages"][stage_name] = {
                    "current_workers": metrics.current_workers,
                    "target_workers": metrics.target_workers,
                    "queue_size": metrics.queue_size,
                    "utilization": metrics.utilization,
                    "processing_rate": metrics.processing_rate,
                    "last_adjustment": metrics.last_adjustment,
                }

                stats["total_workers"] += metrics.current_workers
                stats["total_target_workers"] += metrics.target_workers
                stats["total_queue_size"] += metrics.queue_size

            return stats

    def _balancing_loop(self):
        """Main load balancing loop"""
        while self.running:
            try:
                # Check each stage for adjustment needs
                for stage_name in list(self.worker_metrics.keys()):
                    if self.should_adjust_workers(stage_name):
                        self.adjust_workers(stage_name)

                # Sleep until next check
                time.sleep(self.adjustment_interval / 2)

            except Exception as e:
                logger.error(f"Load balancing loop error: {e}")
                time.sleep(5.0)

    def _apply_trend_analysis(self, stage_name: str, target_workers: int) -> int:
        """Apply trend analysis to worker count recommendation"""
        if stage_name not in self.metrics_history:
            return target_workers

        history = self.metrics_history[stage_name]
        if len(history) < 5:
            return target_workers

        # Analyze recent trend in queue size
        recent_queue_sizes = [entry["queue_size"] for entry in list(history)[-LB_RECENT_HISTORY_SIZE:]]

        # If queue is consistently growing, be more aggressive in scaling up
        if len(recent_queue_sizes) >= 3:
            if all(recent_queue_sizes[i] <= recent_queue_sizes[i + 1] for i in range(len(recent_queue_sizes) - 1)):
                # Queue is growing - scale up more aggressively
                target_workers = min(self.max_workers, target_workers + 1)
            elif all(recent_queue_sizes[i] >= recent_queue_sizes[i + 1] for i in range(len(recent_queue_sizes) - 1)):
                # Queue is shrinking - be more conservative about scaling down
                if target_workers < self.worker_metrics[stage_name].current_workers:
                    target_workers = self.worker_metrics[stage_name].current_workers

        return target_workers


def create_load_balancer(config: LoadBalancerConfig) -> LoadBalancer:
    """Factory function to create load balancer"""
    return LoadBalancer(config)


if __name__ == "__main__":
    # Test load balancer

    config = LoadBalancerConfig(
        min_workers=DEFAULT_MIN_WORKERS,
        max_workers=DEFAULT_MAX_WORKERS,
        target_queue_size=DEFAULT_TARGET_QUEUE_SIZE,
        adjustment_interval=DEFAULT_ADJUSTMENT_INTERVAL,
        scale_up_threshold=DEFAULT_SCALE_UP_THRESHOLD,
        scale_down_threshold=DEFAULT_SCALE_DOWN_THRESHOLD,
    )

    balancer = create_load_balancer(config)

    # Mock stage class
    class MockStage:
        def __init__(self, name):
            self.name = name
            self.worker_count = 2

        def adjust_workers(self, count):
            logger.info(f"Adjusting {self.name} workers: {self.worker_count} -> {count}")
            self.worker_count = count
            return True

    # Register mock stages
    for stage_name in [STAGE_NAME_SEARCH, STAGE_NAME_COLLECT, STAGE_NAME_CHECK, STAGE_NAME_MODELS]:
        balancer.register_stage(stage_name, MockStage(stage_name))

    balancer.start()

    try:
        # Simulate varying load
        for i in range(20):
            for stage_name in [STAGE_NAME_SEARCH, STAGE_NAME_COLLECT, STAGE_NAME_CHECK, STAGE_NAME_MODELS]:
                # Simulate different load patterns
                if stage_name == STAGE_NAME_SEARCH:
                    queue_size = random.randint(0, 50)
                elif stage_name == STAGE_NAME_COLLECT:
                    queue_size = random.randint(10, 100)
                elif stage_name == STAGE_NAME_CHECK:
                    queue_size = random.randint(50, 200)
                else:  # models
                    queue_size = random.randint(0, 20)

                from models import LoadBalancerMetricsUpdate

                balancer.update_metrics(
                    stage_name,
                    LoadBalancerMetricsUpdate(
                        queue_size=queue_size,
                        active_workers=balancer.worker_metrics[stage_name].current_workers,
                        processing_rate=random.uniform(0.5, 5.0),
                        avg_processing_time=random.uniform(0.1, 2.0),
                    ),
                )

            # Log stats
            stats = balancer.get_load_balancing_stats()
            logger.info(f"Iteration {i+1}:")
            for stage_name, stage_stats in stats["stages"].items():
                logger.info(
                    f"  {stage_name}: workers={stage_stats['current_workers']}->{stage_stats['target_workers']}, "
                    f"queue={stage_stats['queue_size']}, util={stage_stats['utilization']:.2f}"
                )

            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("Stopping load balancer test...")

    finally:
        balancer.stop()
        logger.info("Load balancer test completed!")
