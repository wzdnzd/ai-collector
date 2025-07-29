#!/usr/bin/env python3
"""
Main entry point for the async pipeline system
"""

import argparse
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from config import load_config
from constants import (
    APPLICATION_BANNER,
    DEFAULT_CONFIG_FILE,
    DEFAULT_STATS_INTERVAL,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
)
from logger import flush_logs, get_logging_stats, get_main_logger, init_logging
from task_manager import TaskManager, create_task_manager

# Get main program logger
logger = get_main_logger()


def setup_logging(level: str = LOG_LEVEL_INFO) -> None:
    """Setup logging configuration"""
    init_logging(level)

    # Show log file information
    stats = get_logging_stats()

    if stats.log_files:
        logger.info("Log files initialized:")
        for filename, info in stats.log_files.items():
            if not info.error:
                logger.info(f"  - {filename} ({info.size})")

    if stats.logs_directory:
        logger.info(f"Logs directory: {stats.logs_directory}")


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")

    # Flush logs before exit
    flush_logs()

    sys.exit(0)


def print_stats(manager: TaskManager, interval: int = 10) -> None:
    """Print periodic statistics"""
    try:
        stats = manager.get_stats()

        separator_long = "=" * 80
        separator_short = "-" * 80

        logger.info("\n" + separator_long)
        logger.info(f"{'Pipeline Status':^80}")
        logger.info(separator_long)

        if stats.pipeline_stats:
            # Pipeline stage stats
            pipeline_stats = stats.pipeline_stats
            for stage_name, stage_stats in [
                ("search", pipeline_stats.search),
                ("collect", pipeline_stats.collect),
                ("check", pipeline_stats.check),
                ("models", pipeline_stats.models),
            ]:
                logger.info(
                    f"{stage_name:<10} | Queue: {stage_stats.queue_size:<6} | "
                    f"Processed: {stage_stats.total_processed:<8} | "
                    f"Errors: {stage_stats.total_errors:<4} | "
                    f"Workers: {stage_stats.workers}"
                )

        logger.info(separator_short)

        if stats.result_stats:
            # Provider result stats
            for provider_name, result_stats in stats.result_stats.items():
                logger.info(
                    f"{provider_name:<10} | Valid: {result_stats.valid_keys:<6} | "
                    f"No Quota: {result_stats.no_quota_keys:<4} | "
                    f"Wait: {result_stats.wait_check_keys:<4} | "
                    f"Invalid: {result_stats.invalid_keys:<6} | "
                    f"Links: {result_stats.total_links}"
                )

        logger.info(separator_long)

    except Exception as e:
        logger.debug(f"Error printing stats: {e}")


def run_pipeline(config_file: str, log_level: str, stats_interval: int) -> None:
    """Run the main pipeline"""
    setup_logging(log_level)
    logger.info(APPLICATION_BANNER)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    manager = None

    try:
        # Create and start task manager
        logger.info(f"Loading configuration from: {config_file}")
        manager = create_task_manager(config_file)

        logger.info(f"Initialized with {len(manager.providers)} providers:")
        for provider_name in manager.providers.keys():
            logger.info(f"  - {provider_name}")

        logger.info("Starting pipeline...")
        manager.start()

        logger.info("Pipeline started successfully!")
        logger.info(f"Statistics will be displayed every {stats_interval} seconds")
        logger.info("Press Ctrl+C to stop gracefully")

        # Main monitoring loop
        last_stats_time = 0

        while True:
            current_time = time.time()

            # Print stats periodically
            if current_time - last_stats_time >= stats_interval:
                print_stats(manager, stats_interval)
                last_stats_time = current_time

            # Check if processing is complete
            if manager.pipeline and manager.pipeline.is_finished():
                logger.info("All processing completed!")
                break

            time.sleep(1.0)  # Main loop sleep interval

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        traceback.print_exc()

    finally:
        if manager:
            logger.info("Stopping pipeline...")
            manager.stop()
            logger.info("Pipeline stopped")

        # Flush all logs before exit
        flush_logs()
        logger.info("Logs flushed to disk")

        # Ensure program exits
        sys.exit(0)


def validate_config(config_file: str) -> bool:
    """Validate configuration file"""
    logger.info(f"Validating configuration: {config_file}")

    try:
        manager = create_task_manager(config_file)

        logger.info("Configuration is valid")
        logger.info(f"Found {len(manager.providers)} providers:")

        for provider_name, provider in manager.providers.items():
            conditions_count = len(provider.conditions) if provider.conditions else 0
            logger.info(f"   - {provider_name}: {conditions_count} search conditions")

        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def create_sample_config() -> None:
    """Create a sample configuration file"""
    config_path = Path(DEFAULT_CONFIG_FILE)

    if config_path.exists():
        logger.info(f"Configuration file already exists: {config_path}")
        return

    try:
        # This will create a default config
        load_config(DEFAULT_CONFIG_FILE)

        logger.info(f"Created sample configuration: {config_path}")
        logger.info("Please edit the configuration file to:")
        logger.info("  1. Set your GitHub session token or API key")
        logger.info("  2. Configure provider search patterns")
        logger.info("  3. Adjust rate limits and thread counts")

    except Exception as e:
        logger.error(f"Failed to create sample config: {e}")


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Async Pipeline System for Multi-Provider API Key Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run with default config.yaml
  %(prog)s -c custom.yaml           # Run with custom config
  %(prog)s --validate               # Validate configuration
  %(prog)s --create-config          # Create sample configuration
  %(prog)s --log-level DEBUG        # Enable debug logging
        """,
    )

    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_FILE, help=f"Configuration file path (default: {DEFAULT_CONFIG_FILE})"
    )

    parser.add_argument(
        "--log-level",
        choices=[LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_WARNING, LOG_LEVEL_ERROR],
        default=LOG_LEVEL_INFO,
        help=f"Logging level (default: {LOG_LEVEL_INFO})",
    )

    parser.add_argument(
        "--stats-interval",
        type=int,
        default=DEFAULT_STATS_INTERVAL,
        help=f"Statistics display interval in seconds (default: {DEFAULT_STATS_INTERVAL})",
    )

    parser.add_argument("--validate", action="store_true", help="Validate configuration and exit")

    parser.add_argument("--create-config", action="store_true", help="Create sample configuration file and exit")

    args = parser.parse_args()

    # Handle special commands
    if args.create_config:
        create_sample_config()
        return

    if args.validate:
        success = validate_config(args.config)
        sys.exit(0 if success else 1)

    # Check if config file exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.error("Use --create-config to create a sample configuration")
        sys.exit(1)

    # Run the pipeline
    run_pipeline(args.config, args.log_level, args.stats_interval)


if __name__ == "__main__":
    main()
