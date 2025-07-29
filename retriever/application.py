#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main application class for the async pipeline system.
Integrates all components: task management, monitoring, load balancing, and graceful shutdown.
"""

import argparse
import os
import signal
import sys
import threading
import time
import traceback
from typing import Any, Optional

from config import Config, load_config
from constants import (
    DEFAULT_ADJUSTMENT_INTERVAL,
    DEFAULT_CONFIG_FILE,
    DEFAULT_ERROR_RATE_THRESHOLD_APP,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MEMORY_THRESHOLD,
    DEFAULT_MIN_WORKERS,
    DEFAULT_QUEUE_SIZE_THRESHOLD_APP,
    DEFAULT_SCALE_DOWN_THRESHOLD,
    DEFAULT_SCALE_UP_THRESHOLD,
    DEFAULT_SHUTDOWN_TIMEOUT,
    DEFAULT_STATS_INTERVAL,
    DEFAULT_TARGET_QUEUE_SIZE,
    STAGE_NAME_CHECK,
    STAGE_NAME_COLLECT,
    STAGE_NAME_MODELS,
    STAGE_NAME_SEARCH,
)
from load_balancer import LoadBalancer, create_load_balancer
from logger import get_application_logger, init_logging
from models import (
    ApplicationStatus,
    LoadBalancerConfig,
    LoadBalancerMetricsUpdate,
    MonitoringConfig,
    PipelineStatsUpdate,
    ProviderStatsUpdate,
)
from monitoring import MonitoringSystem, create_monitoring_system
from task_manager import TaskManager, create_task_manager

# Get application logger
logger = get_application_logger()


class AsyncPipelineApplication:
    """Main application class for the async pipeline system"""

    def __init__(self, config_path: str = DEFAULT_CONFIG_FILE) -> None:
        self.config_path = config_path
        self.config: Optional[Config] = None
        self.task_manager: Optional[TaskManager] = None
        self.monitoring: Optional[MonitoringSystem] = None
        self.load_balancer: Optional[LoadBalancer] = None

        # Application state
        self.running = False
        self.shutdown_requested = False
        self.start_time = 0.0

        # Thread management
        self.main_thread: Optional[threading.Thread] = None
        self.shutdown_lock = threading.Lock()

        # Statistics
        self.stats_display_interval = float(DEFAULT_STATS_INTERVAL)
        self.last_stats_display = 0.0

        logger.info(f"Initialized application with config: {config_path}")

    def initialize(self) -> bool:
        """Initialize all application components"""
        try:
            # Load configuration
            self.config = load_config(self.config_path)
            logger.info("Configuration loaded successfully")

            # Create task manager
            self.task_manager = create_task_manager(self.config_path)
            logger.info(f"Task manager created with {len(self.task_manager.providers)} providers")

            # Create monitoring system
            monitoring_config = MonitoringConfig(
                update_interval=self.config.monitoring.stats_interval,
                error_rate_threshold=DEFAULT_ERROR_RATE_THRESHOLD_APP,
                queue_size_threshold=DEFAULT_QUEUE_SIZE_THRESHOLD_APP,
                memory_threshold=DEFAULT_MEMORY_THRESHOLD,
            )
            self.monitoring = create_monitoring_system(monitoring_config)
            logger.info("Monitoring system created")

            # Create load balancer
            load_balancer_config = LoadBalancerConfig(
                min_workers=DEFAULT_MIN_WORKERS,
                max_workers=DEFAULT_MAX_WORKERS,
                target_queue_size=DEFAULT_TARGET_QUEUE_SIZE,
                adjustment_interval=DEFAULT_ADJUSTMENT_INTERVAL,
                scale_up_threshold=DEFAULT_SCALE_UP_THRESHOLD,
                scale_down_threshold=DEFAULT_SCALE_DOWN_THRESHOLD,
            )
            self.load_balancer = create_load_balancer(load_balancer_config)
            logger.info("Load balancer created")

            # Register pipeline stages with load balancer
            if self.task_manager.pipeline:
                pipeline = self.task_manager.pipeline
                if hasattr(pipeline, f"{STAGE_NAME_SEARCH}_stage"):
                    self.load_balancer.register_stage(STAGE_NAME_SEARCH, pipeline.search_stage)
                if hasattr(pipeline, f"{STAGE_NAME_COLLECT}_stage"):
                    self.load_balancer.register_stage(STAGE_NAME_COLLECT, pipeline.collect_stage)
                if hasattr(pipeline, f"{STAGE_NAME_CHECK}_stage"):
                    self.load_balancer.register_stage(STAGE_NAME_CHECK, pipeline.check_stage)
                if hasattr(pipeline, f"{STAGE_NAME_MODELS}_stage"):
                    self.load_balancer.register_stage(STAGE_NAME_MODELS, pipeline.models_stage)

            # Set up signal handlers
            self._setup_signal_handlers()

            logger.info("Application initialization completed")
            return True

        except Exception as e:
            logger.error(f"Application initialization failed: {e}")
            return False

    def start(self) -> bool:
        """Start the application"""
        if self.running:
            logger.warning("Application is already running")
            return False

        if not self.initialize():
            return False

        self.running = True
        self.start_time = time.time()

        try:
            # Start monitoring
            self.monitoring.start()

            # Start load balancer
            self.load_balancer.start()

            # Start task manager
            self.task_manager.start()

            # Start main application loop
            self.main_thread = threading.Thread(target=self._main_loop, name="application-main", daemon=True)
            self.main_thread.start()

            logger.info("Application started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            self.stop()
            return False

    def stop(self, timeout: float = DEFAULT_SHUTDOWN_TIMEOUT) -> None:
        """Stop the application gracefully"""
        with self.shutdown_lock:
            if not self.running:
                return

            logger.info("Stopping application...")
            self.shutdown_requested = True
            self.running = False

            try:
                # Stop task manager first (most important)
                if self.task_manager:
                    logger.info("Stopping task manager...")
                    self.task_manager.stop(timeout * 0.6)

                # Stop load balancer
                if self.load_balancer:
                    logger.info("Stopping load balancer...")
                    self.load_balancer.stop()

                # Stop monitoring
                if self.monitoring:
                    logger.info("Stopping monitoring...")
                    self.monitoring.stop()

                # Wait for main thread
                if self.main_thread and self.main_thread.is_alive():
                    self.main_thread.join(timeout=2.0)
                    if self.main_thread.is_alive():
                        logger.warning("Main thread did not stop gracefully")

                logger.info("Application stopped successfully")

            except Exception as e:
                logger.error(f"Error during application shutdown: {e}")

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for application to complete processing"""
        if not self.running:
            return False

        start_time = time.time()

        while self.running and not self.shutdown_requested:
            # Check if processing is complete
            if self.task_manager and self.task_manager.pipeline:
                if self.task_manager.pipeline.is_finished():
                    logger.info("Processing completed")
                    return True

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Wait for completion timed out after {timeout} seconds")
                return False

            time.sleep(1.0)  # Main loop sleep interval

        return False

    def get_status(self) -> ApplicationStatus:
        """Get comprehensive application status"""
        status = ApplicationStatus(
            timestamp=time.time(),
            running=self.running,
            runtime=time.time() - self.start_time if self.start_time > 0 else 0,
            shutdown_requested=self.shutdown_requested,
        )

        # Get task manager status
        if self.task_manager:
            try:
                status.task_manager_status = self.task_manager.get_stats()
            except Exception as e:
                status.task_manager_status = {"error": str(e)}

        # Get monitoring status
        if self.monitoring:
            try:
                status.monitoring_status = self.monitoring.get_stats_summary()
            except Exception as e:
                status.monitoring_status = {"error": str(e)}

        # Get load balancer status
        if self.load_balancer:
            try:
                status.load_balancer_status = self.load_balancer.get_load_balancing_stats()
            except Exception as e:
                status.load_balancer_status = {"error": str(e)}

        return status

    def print_status(self, detailed: bool = False) -> None:
        """Print current application status"""
        if self.monitoring:
            self.monitoring.print_status(detailed)
        else:
            status = self.get_status()
            logger.info(f"Application Status: {'Running' if status.running else 'Stopped'}")
            logger.info(f"Runtime: {status.runtime:.1f} seconds")

            if status.task_manager_status and hasattr(status.task_manager_status, "providers"):
                tm_stats = status.task_manager_status
                logger.info(f"Providers: {len(tm_stats.providers)}")
                if tm_stats.pipeline_stats:
                    logger.info("Pipeline stages running")

    def _main_loop(self) -> None:
        """Main application loop"""
        logger.info("Starting main application loop")

        while self.running and not self.shutdown_requested:
            try:
                current_time = time.time()

                # Update monitoring with current statistics
                self._update_monitoring_stats()

                # Update load balancer metrics
                self._update_load_balancer_metrics()

                # Display status periodically
                if current_time - self.last_stats_display >= self.stats_display_interval:
                    if self.config.monitoring.show_stats:
                        self.print_status(detailed=True)
                    self.last_stats_display = current_time

                # Check if processing is complete
                if self.task_manager and self.task_manager.pipeline:
                    if self.task_manager.pipeline.is_finished():
                        logger.info("All processing completed, shutting down")
                        break

                time.sleep(1.0)  # Main loop sleep interval

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5.0)  # Error sleep interval

        logger.info("Main application loop ended")

    def _update_monitoring_stats(self) -> None:
        """Update monitoring system with current statistics"""
        if not self.monitoring or not self.task_manager:
            return

        try:
            # Get task manager stats
            tm_stats = self.task_manager.get_stats()

            # Update provider statistics
            if tm_stats.result_stats:
                for provider_name, result_stats in tm_stats.result_stats.items():
                    provider_update = ProviderStatsUpdate(
                        valid_keys=result_stats.valid_keys,
                        invalid_keys=result_stats.invalid_keys,
                        no_quota_keys=result_stats.no_quota_keys,
                        wait_check_keys=result_stats.wait_check_keys,
                        total_links=result_stats.total_links,
                        total_models=result_stats.total_models,
                    )
                    self.monitoring.update_provider_stats(provider_name, provider_update)

            # Update pipeline statistics
            if tm_stats.pipeline_stats:
                pipeline_stats = tm_stats.pipeline_stats

                total_workers = (
                    pipeline_stats.search.workers
                    + pipeline_stats.collect.workers
                    + pipeline_stats.check.workers
                    + pipeline_stats.models.workers
                )

                pipeline_update = PipelineStatsUpdate(
                    search_queue=pipeline_stats.search.queue_size,
                    collect_queue=pipeline_stats.collect.queue_size,
                    check_queue=pipeline_stats.check.queue_size,
                    models_queue=pipeline_stats.models.queue_size,
                    active_workers=total_workers,
                    total_workers=total_workers,
                    is_finished=(self.task_manager.pipeline.is_finished() if self.task_manager.pipeline else False),
                )
                self.monitoring.update_pipeline_stats(pipeline_update)

        except Exception as e:
            logger.debug(f"Error updating monitoring stats: {e}")

    def _update_load_balancer_metrics(self) -> None:
        """Update load balancer with current metrics"""
        if not self.load_balancer or not self.task_manager:
            return

        try:
            tm_stats = self.task_manager.get_stats()

            if tm_stats.pipeline_stats:
                pipeline_stats = tm_stats.pipeline_stats
                for stage_name, stage_stats in [
                    ("search", pipeline_stats.search),
                    ("collect", pipeline_stats.collect),
                    ("check", pipeline_stats.check),
                    ("models", pipeline_stats.models),
                ]:
                    metrics_update = LoadBalancerMetricsUpdate(
                        queue_size=stage_stats.queue_size,
                        active_workers=stage_stats.workers,
                        processing_rate=stage_stats.processing_rate,
                        avg_processing_time=0.0,  # Not available in current stats
                    )
                    self.load_balancer.update_metrics(stage_name, metrics_update)

        except Exception as e:
            logger.debug(f"Error updating load balancer metrics: {e}")

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown"""

        def signal_handler(signum: int, frame: Optional[Any]) -> None:
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


def create_application(config_path: str = DEFAULT_CONFIG_FILE) -> AsyncPipelineApplication:
    """Factory function to create application instance"""
    return AsyncPipelineApplication(config_path)


if __name__ == "__main__":
    init_logging("INFO")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Async Pipeline Application")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG_FILE, help="Configuration file path")
    parser.add_argument("--timeout", type=float, help="Maximum runtime in seconds")
    parser.add_argument(
        "--stats-interval", type=float, default=float(DEFAULT_STATS_INTERVAL), help="Statistics display interval"
    )

    args = parser.parse_args()

    # Create and run application
    app = create_application(args.config)
    app.stats_display_interval = args.stats_interval

    try:
        logger.info("Starting Async Pipeline Application")
        logger.info("=" * 60)

        if not app.start():
            logger.error("Failed to start application")
            sys.exit(1)

        logger.info("Application started successfully")
        logger.info("Press Ctrl+C to stop gracefully")

        # Wait for completion or timeout
        completed = app.wait_for_completion(args.timeout)

        if completed:
            logger.info("Processing completed successfully!")
        elif args.timeout:
            logger.info(f"Processing timed out after {args.timeout} seconds")
        else:
            logger.info("🛑 Processing stopped by user")

        # Print final status
        logger.info("Final Status:")
        app.print_status(detailed=True)

    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")

    except Exception as e:
        logger.error(f"Application error: {e}")
        traceback.print_exc()

    finally:
        logger.info("Shutting down...")
        try:
            app.stop()
            logger.info("Application stopped")

            # Try to get summary with timeout
            try:
                # Use a simple timeout mechanism for status retrieval
                status_result = [None]
                status_error = [None]

                def get_status_with_timeout():
                    try:
                        status_result[0] = app.get_status()
                    except Exception as e:
                        status_error[0] = e

                status_thread = threading.Thread(target=get_status_with_timeout, daemon=True)
                status_thread.start()
                status_thread.join(timeout=2.0)  # Status thread timeout

                if status_result[0] is not None:
                    status = status_result[0]
                    logger.info(f"Summary: Runtime {status.runtime:.1f}s")

                    if status.monitoring_status:
                        monitoring_stats = status.monitoring_status
                        if isinstance(monitoring_stats, dict):
                            logger.info(
                                f"Results: {monitoring_stats.get('total_valid_keys', 0)} valid keys, "
                                f"{monitoring_stats.get('total_links', 0)} links processed"
                            )
                elif status_error[0] is not None:
                    logger.warning(f"Could not retrieve final status: {status_error[0]}")
                else:
                    logger.warning("Status retrieval timed out")

            except Exception as e:
                logger.warning(f"Could not retrieve final status: {e}")

        except Exception as e:
            logger.error(f"Error during final cleanup: {e}")

        # Force exit after a short delay to allow log flushing
        def force_exit():
            time.sleep(0.1)  # Force exit delay
            os._exit(0)  # Force exit without cleanup

        # Start force exit thread
        exit_thread = threading.Thread(target=force_exit, daemon=True)
        exit_thread.start()

        # Try normal exit first
        sys.exit(0)
