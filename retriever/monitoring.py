#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monitoring system for multi-provider pipeline processing.
Provides real-time statistics, alerts, and performance monitoring.
"""

import random
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from constants import ALERT_COOLDOWN_SECONDS
from models import MonitoringConfig, PipelineStats, ProviderStats

from logger import get_monitoring_logger

# Get monitoring logger
logger = get_monitoring_logger()


class AlertManager:
    """Manages alerts and notifications for pipeline issues"""

    def __init__(self, config: MonitoringConfig):
        self.alert_handlers: List[Callable] = []
        self.alert_history: deque = deque(maxlen=100)
        self.lock = threading.Lock()

        # Alert thresholds
        self.error_rate_threshold = config.error_rate_threshold
        self.queue_size_threshold = config.queue_size_threshold
        self.memory_threshold = config.memory_threshold

        logger.info("Initialized alert manager")

    def add_handler(self, handler: Callable[[str, str, Dict[str, Any]], None]) -> None:
        """Add alert handler function"""
        self.alert_handlers.append(handler)

    def check_alerts(self, provider_stats: Dict[str, ProviderStats], pipeline_stats: PipelineStats) -> None:
        """Check for alert conditions and trigger notifications"""

        # Check provider error rates
        for provider_name, stats in provider_stats.items():
            if stats.total_tasks > 10:  # Only check if we have enough data
                error_rate = stats.failed_tasks / stats.total_tasks
                if error_rate > self.error_rate_threshold:
                    self._trigger_alert(
                        "HIGH_ERROR_RATE",
                        f"Provider {provider_name} has high error rate: {error_rate:.2%}",
                        {"provider": provider_name, "error_rate": error_rate},
                    )

        # Check queue sizes
        total_queue_size = (
            pipeline_stats.search_queue
            + pipeline_stats.collect_queue
            + pipeline_stats.check_queue
            + pipeline_stats.models_queue
        )

        if total_queue_size > self.queue_size_threshold:
            self._trigger_alert(
                "HIGH_QUEUE_SIZE",
                f"Total queue size is high: {total_queue_size}",
                {"queue_size": total_queue_size},
            )

        # Check memory usage
        if pipeline_stats.memory_usage > self.memory_threshold:
            self._trigger_alert(
                "HIGH_MEMORY_USAGE",
                f"Memory usage is high: {pipeline_stats.memory_usage / (1024*1024):.1f} MB",
                {"memory_usage": pipeline_stats.memory_usage},
            )

    def _trigger_alert(self, alert_type: str, message: str, data: Dict[str, Any]) -> None:
        """Trigger an alert"""
        with self.lock:
            alert = {"type": alert_type, "message": message, "data": data, "timestamp": time.time()}

            # Check if we've already sent this alert recently (within 5 minutes)
            recent_alerts = [
                a
                for a in self.alert_history
                if a["type"] == alert_type and time.time() - a["timestamp"] < ALERT_COOLDOWN_SECONDS
            ]

            if not recent_alerts:
                self.alert_history.append(alert)

                # Send to all handlers
                for handler in self.alert_handlers:
                    try:
                        handler(alert_type, message, data)
                    except Exception as e:
                        logger.error(f"Alert handler error: {e}")


class MultiProviderMonitoring:
    """Main monitoring system for multi-provider pipeline"""

    def __init__(self, config: MonitoringConfig):
        self.provider_stats: Dict[str, ProviderStats] = {}
        self.pipeline_stats = PipelineStats()
        self.alert_manager = AlertManager(config)

        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        self.update_interval = config.update_interval

        # Statistics history for trend analysis
        self.stats_history: deque = deque(maxlen=100)
        self.lock = threading.Lock()

        # Add default console alert handler
        self.alert_manager.add_handler(self._console_alert_handler)

        logger.info("Initialized multi-provider monitoring")

    def start(self) -> None:
        """Start monitoring thread"""
        if self.running:
            return

        self.running = True
        self.pipeline_stats.is_running = True
        self.pipeline_stats.start_time = time.time()

        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, name="monitoring-thread", daemon=True)
        self.monitoring_thread.start()

        logger.info("Started monitoring system")

    def stop(self) -> None:
        """Stop monitoring thread"""
        self.running = False
        self.pipeline_stats.is_running = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
            if self.monitoring_thread.is_alive():
                logger.warning("Monitoring thread did not stop gracefully")

        logger.info("Stopped monitoring system")

    def update_provider_stats(self, provider_name: str, stats_data: Any) -> None:
        """Update statistics for a specific provider"""
        with self.lock:
            if provider_name not in self.provider_stats:
                self.provider_stats[provider_name] = ProviderStats(name=provider_name)

            stats = self.provider_stats[provider_name]

            # Update from stats_data (data class or dict)
            if hasattr(stats_data, "valid_keys"):
                stats.valid_keys = stats_data.valid_keys
            if hasattr(stats_data, "invalid_keys"):
                stats.invalid_keys = stats_data.invalid_keys
            if hasattr(stats_data, "no_quota_keys"):
                stats.no_quota_keys = stats_data.no_quota_keys
            if hasattr(stats_data, "wait_check_keys"):
                stats.wait_check_keys = stats_data.wait_check_keys
            if hasattr(stats_data, "total_links"):
                stats.total_links = stats_data.total_links
            if hasattr(stats_data, "total_models"):
                stats.total_models = stats_data.total_models

            stats.last_activity = time.time()

            # Calculate derived metrics
            if stats.total_tasks > 0:
                stats.success_rate = (stats.total_tasks - stats.failed_tasks) / stats.total_tasks

    def update_pipeline_stats(self, stats_data: Any) -> None:
        """Update overall pipeline statistics"""
        with self.lock:
            # Update from stats_data (data class)
            if hasattr(stats_data, "search_queue"):
                self.pipeline_stats.search_queue = stats_data.search_queue
            if hasattr(stats_data, "collect_queue"):
                self.pipeline_stats.collect_queue = stats_data.collect_queue
            if hasattr(stats_data, "check_queue"):
                self.pipeline_stats.check_queue = stats_data.check_queue
            if hasattr(stats_data, "models_queue"):
                self.pipeline_stats.models_queue = stats_data.models_queue
            if hasattr(stats_data, "active_workers"):
                self.pipeline_stats.active_workers = stats_data.active_workers
            if hasattr(stats_data, "total_workers"):
                self.pipeline_stats.total_workers = stats_data.total_workers
            if hasattr(stats_data, "is_finished"):
                self.pipeline_stats.is_finished = stats_data.is_finished

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics snapshot"""
        with self.lock:
            return {
                "timestamp": time.time(),
                "pipeline": self.pipeline_stats,
                "providers": dict(self.provider_stats),
                "runtime": time.time() - self.pipeline_stats.start_time,
            }

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summarized statistics"""
        with self.lock:
            total_tasks = sum(stats.total_tasks for stats in self.provider_stats.values())
            total_completed = sum(stats.completed_tasks for stats in self.provider_stats.values())
            total_failed = sum(stats.failed_tasks for stats in self.provider_stats.values())
            total_valid_keys = sum(stats.valid_keys for stats in self.provider_stats.values())
            total_links = sum(stats.total_links for stats in self.provider_stats.values())

            runtime = time.time() - self.pipeline_stats.start_time

            return {
                "total_tasks": total_tasks,
                "completed_tasks": total_completed,
                "failed_tasks": total_failed,
                "success_rate": (total_completed / max(total_tasks, 1)),
                "total_valid_keys": total_valid_keys,
                "total_links": total_links,
                "runtime": runtime,
                "throughput": total_completed / max(runtime, 1),
                "active_providers": len(self.provider_stats),
                "is_running": self.pipeline_stats.is_running,
                "is_finished": self.pipeline_stats.is_finished,
            }

    def print_status(self, detailed: bool = False) -> None:
        """Print current status to console"""
        stats = self.get_current_stats()
        summary = self.get_stats_summary()

        logger.info("=" * 80)
        logger.info(f"{'Pipeline Status':^80}")
        logger.info("=" * 80)

        # Overall summary
        logger.info(
            f"Runtime: {summary['runtime']:.1f}s | "
            f"Tasks: {summary['completed_tasks']}/{summary['total_tasks']} | "
            f"Success: {summary['success_rate']:.1%} | "
            f"Keys: {summary['total_valid_keys']} | "
            f"Links: {summary['total_links']}"
        )

        logger.info(
            f"Throughput: {summary['throughput']:.1f} tasks/sec | "
            f"Active Workers: {stats['pipeline'].active_workers}/{stats['pipeline'].total_workers}"
        )

        if detailed:
            logger.info("-" * 80)
            logger.info("Provider Details:")

            for provider_name, provider_stats in stats["providers"].items():
                logger.info(
                    f"  {provider_name:>12}: "
                    f"tasks={provider_stats.completed_tasks}/{provider_stats.total_tasks} | "
                    f"valid={provider_stats.valid_keys} | "
                    f"links={provider_stats.total_links} | "
                    f"rate={provider_stats.success_rate:.1%}"
                )

            logger.info("-" * 80)
            logger.info("Queue Status:")
            pipeline = stats["pipeline"]
            logger.info(
                f"  Search: {pipeline.search_queue:>4} | "
                f"Collect: {pipeline.collect_queue:>4} | "
                f"Check: {pipeline.check_queue:>4} | "
                f"Models: {pipeline.models_queue:>4}"
            )

        logger.info("=" * 80)

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect current stats
                current_stats = self.get_current_stats()

                # Store in history
                with self.lock:
                    self.stats_history.append(current_stats)

                # Check for alerts
                self.alert_manager.check_alerts(self.provider_stats, self.pipeline_stats)

                # Sleep until next update
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1.0)

    def _console_alert_handler(self, alert_type: str, message: str, data: Dict[str, Any]) -> None:
        """Default console alert handler"""
        timestamp = time.strftime("%H:%M:%S")
        logger.warning(f"[{timestamp}] ALERT ({alert_type}): {message}")


def create_monitoring_system(config: MonitoringConfig) -> MultiProviderMonitoring:
    """Factory function to create monitoring system"""
    return MultiProviderMonitoring(config)


if __name__ == "__main__":
    # Test monitoring system

    config = MonitoringConfig(
        update_interval=2.0,
        error_rate_threshold=0.1,
        queue_size_threshold=100,
        memory_threshold=1024 * 1024 * 1024,  # 1GB
    )

    monitoring = create_monitoring_system(config)
    monitoring.start()

    try:
        # Simulate some statistics updates
        for i in range(10):
            # Update provider stats
            monitoring.update_provider_stats(
                "test_provider",
                {
                    "total_tasks": i * 10,
                    "completed_tasks": i * 9,
                    "failed_tasks": i,
                    "valid_keys": i * 2,
                    "total_links": i * 5,
                },
            )

            # Update pipeline stats
            monitoring.update_pipeline_stats(
                {
                    "search_queue": random.randint(0, 50),
                    "collect_queue": random.randint(0, 100),
                    "check_queue": random.randint(0, 200),
                    "models_queue": random.randint(0, 20),
                    "active_workers": random.randint(5, 15),
                    "total_workers": 16,
                }
            )

            # Print status
            monitoring.print_status(detailed=True)

            time.sleep(3)

    except KeyboardInterrupt:
        logger.info("Stopping monitoring test...")

    finally:
        monitoring.stop()
        logger.info("Monitoring test completed!")
