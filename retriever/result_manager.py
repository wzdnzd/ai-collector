#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Result management system with real-time persistence.
Supports batch saving for keys, links, and other results with atomic file operations.
"""

import datetime
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import models
from models import ResultStats

from logger import get_result_manager_logger

# Get result manager logger
logger = get_result_manager_logger()

from search import Provider


@dataclass
class ResultStats:
    """Statistics for result processing"""

    valid_keys: int = 0
    no_quota_keys: int = 0
    wait_check_keys: int = 0
    invalid_keys: int = 0
    material_keys: int = 0
    total_links: int = 0
    models_count: int = 0
    start_time: float = field(default_factory=time.time)
    last_save: float = field(default_factory=time.time)


class AtomicFileWriter:
    """Atomic file operations to prevent corruption during writes"""

    @staticmethod
    def write_atomic(filepath: str, content: str):
        """Write file atomically using temporary file and rename"""
        temp_path = filepath + ".tmp"
        backup_path = filepath + ".bak"

        try:
            # Write to temporary file
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # Backup existing file if it exists
            if os.path.exists(filepath):
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(filepath, backup_path)

            # Atomic rename
            os.rename(temp_path, filepath)

            # Remove backup on success
            if os.path.exists(backup_path):
                os.remove(backup_path)

        except Exception as e:
            # Cleanup on failure
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Restore backup if needed
            if os.path.exists(backup_path) and not os.path.exists(filepath):
                os.rename(backup_path, filepath)
            raise e

    @staticmethod
    def append_atomic(filepath: str, lines: List[str]):
        """Append lines to file atomically"""
        if not lines:
            return

        # Read existing content
        existing = ""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                existing = f.read()

        # Prepare new content
        new_content = existing + "\n".join(lines) + "\n"

        # Write atomically
        AtomicFileWriter.write_atomic(filepath, new_content)


class ResultBuffer:
    """Buffer for batching results before writing to files"""

    def __init__(self, result_type: str, batch_size: int = 50, flush_interval: float = 30.0):
        self.result_type = result_type
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffer: List[Any] = []
        self.last_flush = time.time()
        self.lock = threading.Lock()

    def add(self, item: Any) -> bool:
        """Add item to buffer, return True if flush is needed"""
        with self.lock:
            self.buffer.append(item)

            # Check if flush is needed
            return len(self.buffer) >= self.batch_size or time.time() - self.last_flush >= self.flush_interval

    def flush(self) -> List[Any]:
        """Flush buffer and return items"""
        with self.lock:
            if not self.buffer:
                return []

            items = self.buffer.copy()
            self.buffer.clear()
            self.last_flush = time.time()
            return items

    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)


class ResultManager:
    """Manages results for a single provider with real-time persistence"""

    def __init__(self, provider: Provider, workspace: str, batch_size: int = 50, save_interval: float = 30.0):
        self.name = provider.name
        self.provider = provider
        self.workspace = workspace
        self.batch_size = batch_size
        self.save_interval = save_interval

        # Create provider directory
        self.directory = os.path.join(workspace, "providers", self.provider.directory)
        os.makedirs(self.directory, exist_ok=True)

        # Get file paths from provider instance
        self.files = {
            "valid_keys": os.path.join(self.directory, provider.keys_filename),
            "no_quota_keys": os.path.join(self.directory, provider.no_quota_filename),
            "wait_check_keys": os.path.join(self.directory, provider.wait_check_filename),
            "invalid_keys": os.path.join(self.directory, provider.invalid_keys_filename),
            "material_keys": os.path.join(self.directory, provider.material_filename),
            "links": os.path.join(self.directory, provider.links_filename),
            "summary": os.path.join(self.directory, provider.summary_filename),
        }

        # Result buffers
        self.buffers = {
            result_type: ResultBuffer(result_type, batch_size, save_interval)
            for result_type in self.files.keys()
            if result_type != "summary"
        }

        # Models data (not buffered, updated directly)
        self.models_data: Dict[str, Any] = {}

        # Statistics
        self.stats = ResultStats()

        # Thread safety
        self.lock = threading.Lock()

        # Start periodic flush thread
        self.running = True
        self.flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self.flush_thread.start()

        logger.info(f"Initialized result manager for provider: {self.name}")

    def add_result(self, result_type: str, data: Any):
        """Add result to appropriate buffer"""
        if result_type not in self.buffers:
            logger.warning(f"Unknown result type: {result_type}")
            return

        # Handle different data types
        items = []
        if isinstance(data, list):
            items = data
        else:
            items = [data]

        # Add to buffer and check if flush is needed
        buffer = self.buffers[result_type]
        needs_flush = False

        for item in items:
            if buffer.add(item):
                needs_flush = True

        # Update statistics
        with self.lock:
            if result_type == "valid_keys":
                self.stats.valid_keys += len(items)
            elif result_type == "no_quota_keys":
                self.stats.no_quota_keys += len(items)
            elif result_type == "wait_check_keys":
                self.stats.wait_check_keys += len(items)
            elif result_type == "invalid_keys":
                self.stats.invalid_keys += len(items)
            elif result_type == "material_keys":
                self.stats.material_keys += len(items)
            elif result_type == "links":
                self.stats.total_links += len(items)

        # Immediate flush if needed
        if needs_flush:
            self._flush_buffer(result_type)

        logger.debug(f"Added {len(items)} {result_type} for {self.name}")

    def add_links(self, links: List[str]):
        """Convenience method for adding links with validation"""
        if not links:
            return

        # Filter valid links
        valid_links = [link for link in links if link and isinstance(link, str) and link.startswith("http")]

        if valid_links:
            self.add_result("links", valid_links)
            logger.debug(f"Added {len(valid_links)} links for {self.name}")

    def add_models(self, key: str, models: List[str]):
        """Add model list for a key (not buffered, saved immediately)"""
        with self.lock:
            self.models_data[key] = {"models": models, "timestamp": time.time()}
            self.stats.models_count += 1

        # Save models data immediately
        self._save_models()
        logger.debug(f"Added {len(models)} models for key in {self.name}")

    def flush_all(self):
        """Flush all buffers immediately"""
        for result_type in self.buffers.keys():
            self._flush_buffer(result_type)

        # Save models data
        self._save_models()

        logger.info(f"Flushed all buffers for {self.name}")

    def get_stats(self) -> ResultStats:
        """Get current statistics"""
        with self.lock:
            return self.stats

    def backup_existing_files(self) -> None:
        """Backup existing result files to timestamped folder"""

        # Check if any files exist
        existing_files = []
        for file_type, filepath in self.files.items():
            if os.path.exists(filepath):
                existing_files.append((file_type, filepath))

        if not existing_files:
            logger.debug(f"No existing files to backup for {self.name}")
            return

        # Create backup folder with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_dir = os.path.join(self.directory, f"backup-{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)

        # Move existing files to backup folder
        for file_type, filepath in existing_files:
            try:
                backup_path = os.path.join(backup_dir, os.path.basename(filepath))
                os.rename(filepath, backup_path)
                logger.debug(f"Backed up {file_type} file for {self.name}")
            except Exception as e:
                logger.error(f"Failed to backup {file_type} for {self.name}: {e}")

        logger.info(f"Backed up {len(existing_files)} files for {self.name} to {backup_dir}")

    def recover_tasks(self) -> Dict[str, List]:
        """Recover tasks from existing result files"""
        recovered = {"check_tasks": [], "collect_tasks": []}

        # Recover check tasks from material file
        material_path = self.files["material_keys"]
        if os.path.exists(material_path):
            try:
                with open(material_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                # Deserialize service object
                                service = self._deserialize_service(line)
                                if service:
                                    recovered["check_tasks"].append(service)
                            except Exception as e:
                                logger.warning(f"Failed to deserialize service from {self.name}: {e}")

                logger.info(f"Recovered {len(recovered['check_tasks'])} check tasks from {self.name}")
            except Exception as e:
                logger.error(f"Failed to read material file for {self.name}: {e}")

        # Recover collect tasks from links file (skip if provider has skip_search enabled)
        if not self.provider.skip_search:
            links_path = self.files["links"]
            if os.path.exists(links_path):
                try:
                    with open(links_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and line.startswith("http"):
                                recovered["collect_tasks"].append(line)

                    logger.info(f"Recovered {len(recovered['collect_tasks'])} collect tasks from {self.name}")
                except Exception as e:
                    logger.error(f"Failed to read links file for {self.name}: {e}")
        else:
            logger.info(f"Skipping collect task recovery for {self.name} (skip_search enabled)")

        return recovered

    def _deserialize_service(self, line: str) -> Optional[Any]:
        """Deserialize service object from string"""
        try:
            return models.Service.deserialize(line)
        except Exception as e:
            logger.warning(f"Failed to deserialize service: {e}")
            return None

    def stop(self):
        """Stop the result manager and flush all data"""
        self.running = False
        if self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5)

        self.flush_all()
        logger.info(f"Stopped result manager for {self.name}")

    def _periodic_flush(self):
        """Periodic flush thread"""
        while self.running:
            try:
                time.sleep(self.save_interval)

                # Check each buffer for time-based flush
                for result_type, buffer in self.buffers.items():
                    if buffer.size() > 0 and time.time() - buffer.last_flush >= self.save_interval:
                        self._flush_buffer(result_type)

            except Exception as e:
                logger.error(f"Error in periodic flush for {self.name}: {e}")

    def _flush_buffer(self, result_type: str):
        """Flush a specific buffer to file"""
        buffer = self.buffers.get(result_type)
        if not buffer:
            return

        items = buffer.flush()
        if not items:
            return

        try:
            filepath = self.files[result_type]

            # Convert items to strings
            lines = []
            for item in items:
                if hasattr(item, "serialize"):
                    lines.append(item.serialize())
                else:
                    lines.append(str(item))

            # Atomic append to file
            AtomicFileWriter.append_atomic(filepath, lines)

            with self.lock:
                self.stats.last_save = time.time()

            logger.info(f"Saved {len(lines)} {result_type} for {self.name}")

        except Exception as e:
            logger.error(f"Failed to save {result_type} for {self.name}: {e}")

    def _save_models(self):
        """Save models data to JSON file"""
        if not self.models_data:
            return

        try:
            filepath = self.files["summary"]

            # Prepare summary data
            summary = {
                "provider": self.name,
                "updated_at": time.time(),
                "models": self.models_data,
                "stats": {
                    "total_keys": len(self.models_data),
                    "total_models": sum(len(data["models"]) for data in self.models_data.values()),
                },
            }

            # Write atomically
            content = json.dumps(summary, indent=2, ensure_ascii=False)
            AtomicFileWriter.write_atomic(filepath, content)

            logger.debug(f"Saved models summary for {self.name}")

        except Exception as e:
            logger.error(f"Failed to save models for {self.name}: {e}")


class MultiResultManager:
    """Manages results for multiple providers"""

    def __init__(
        self, workspace: str, providers: Dict[str, Any] = None, batch_size: int = 50, save_interval: float = 30.0
    ):
        self.workspace = workspace
        self.providers = providers or {}
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.managers: Dict[str, ResultManager] = {}
        self.lock = threading.Lock()

        # Create workspace directory
        os.makedirs(workspace, exist_ok=True)
        os.makedirs(os.path.join(workspace, "providers"), exist_ok=True)

    def get_manager(self, name: str) -> ResultManager:
        """Get or create result manager for provider"""
        with self.lock:
            if name not in self.managers:
                provider = self.providers.get(name)
                if not provider:
                    raise ValueError(f"Provider instance not found: {name}")
                self.managers[name] = ResultManager(provider, self.workspace, self.batch_size, self.save_interval)
            return self.managers[name]

    def add_result(self, provider: str, result_type: str, data: Any):
        """Add result for a specific provider"""
        manager = self.get_manager(provider)
        manager.add_result(result_type, data)

    def add_links(self, provider: str, links: List[str]):
        """Add links for a specific provider"""
        manager = self.get_manager(provider)
        manager.add_links(links)

    def add_models(self, provider: str, key: str, models: List[str]):
        """Add models for a specific provider"""
        manager = self.get_manager(provider)
        manager.add_models(key, models)

    def flush_all(self):
        """Flush all providers"""
        with self.lock:
            for manager in self.managers.values():
                manager.flush_all()

    def get_all_stats(self) -> Dict[str, ResultStats]:
        """Get statistics for all providers"""
        stats = {}
        with self.lock:
            for provider, manager in self.managers.items():
                stats[provider] = manager.get_stats()
        return stats

    def recover_all_tasks(self) -> Dict[str, Dict[str, List]]:
        """Recover tasks from all providers' result files"""
        all_recovered = {}

        for name in self.providers.keys():
            try:
                manager = self.get_manager(name)
                recovered = manager.recover_tasks()
                if recovered["check_tasks"] or recovered["collect_tasks"]:
                    all_recovered[name] = recovered
            except Exception as e:
                logger.error(f"Failed to recover tasks for {name}: {e}")

        total_check = sum(len(tasks["check_tasks"]) for tasks in all_recovered.values())
        total_collect = sum(len(tasks["collect_tasks"]) for tasks in all_recovered.values())

        if total_check > 0 or total_collect > 0:
            logger.info(f"Recovered {total_check} check tasks and {total_collect} collect tasks from all providers")

        return all_recovered

    def backup_all_existing_files(self) -> None:
        """Backup existing files for all providers"""
        for name in self.providers.keys():
            try:
                manager = self.get_manager(name)
                manager.backup_existing_files()
            except Exception as e:
                logger.error(f"Failed to backup files for {name}: {e}")

    def stop_all(self):
        """Stop all result managers"""
        with self.lock:
            for manager in self.managers.values():
                manager.stop()

        logger.info("Stopped all result managers")


if __name__ == "__main__":
    # Test result manager
    # Create temporary workspace
    workspace = tempfile.mkdtemp()
    logger.info(f"Testing in workspace: {workspace}")

    try:
        # Test single provider
        manager = ResultManager("test_provider", workspace, batch_size=3, save_interval=1)

        # Add some results
        manager.add_result("valid_keys", ["key1", "key2"])
        manager.add_links(["http://example.com/1", "http://example.com/2"])
        manager.add_result("valid_keys", "key3")  # Should trigger flush

        # Wait for periodic flush
        time.sleep(2)

        # Check files
        links_file = os.path.join(workspace, "providers", "test_provider", "links.txt")
        if os.path.exists(links_file):
            with open(links_file, "r") as f:
                content = f.read()
                logger.info(f"Links file content:\n{content}")

        # Get stats
        stats = manager.get_stats()
        logger.info(f"Stats: valid_keys={stats.valid_keys}, links={stats.total_links}")

        # Stop manager
        manager.stop()

        logger.info("Result manager test completed!")

    finally:
        # Cleanup
        shutil.rmtree(workspace)
