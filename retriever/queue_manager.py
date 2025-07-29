#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Queue persistence system for task recovery and state management.

Handles serialization/deserialization of task queues with atomic operations.
"""


import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import constants
from result_manager import AtomicFileWriter
from task import ProviderTask, TaskFactory

from logger import get_queue_manager_logger

# Get queue manager logger
logger = get_queue_manager_logger()


@dataclass
class QueueState:
    """State information for a task queue"""

    stage: str = ""
    provider: str = ""
    task_count: int = 0
    saved_at: float = 0.0

    tasks: List[Dict[str, Any]] = None

    def __post_init__(self):

        if self.tasks is None:

            self.tasks = []


class QueueManager:
    """Manages persistence and recovery of task queues"""

    def __init__(self, workspace: str, save_interval: float = 60.0):
        self.workspace = workspace
        self.save_interval = save_interval

        # Create persistence directory
        self.persistence_dir = os.path.join(workspace, "queue_state")
        os.makedirs(self.persistence_dir, exist_ok=True)

        # Queue file paths
        self.queue_files = {
            "search": os.path.join(self.persistence_dir, "search_queue.json"),
            "collect": os.path.join(self.persistence_dir, "collect_queue.json"),
            "check": os.path.join(self.persistence_dir, "check_queue.json"),
            "models": os.path.join(self.persistence_dir, "models_queue.json"),
        }

        # Thread safety
        self.lock = threading.Lock()

        # Periodic save
        self.running = True
        self.save_thread = None

        logger.info(f"Initialized queue manager with persistence at: {self.persistence_dir}")

    def _get_queue_filepath(self, stage_name: str) -> str:
        """Get filepath for a stage, supporting dynamic stage names"""
        # Check predefined stages first
        if stage_name in self.queue_files:
            return self.queue_files[stage_name]

        # Create filepath for dynamic stage names
        return os.path.join(self.persistence_dir, f"{stage_name}_queue.json")

    def start_periodic_save(self, stages: Dict[str, Any]):
        """Start periodic queue state saving"""
        if self.save_thread and self.save_thread.is_alive():
            return

        self.stages = stages
        self.save_thread = threading.Thread(target=self._periodic_save_loop, daemon=True)
        self.save_thread.start()

        logger.info(f"Started periodic queue saving (interval: {self.save_interval}s)")

    def save_queue_state(self, stage_name: str, task_list: List[ProviderTask]) -> None:
        """Save queue state for a specific stage"""
        if not task_list:
            # Save empty state to indicate stage is clean
            self._save_empty_state(stage_name)
            return

        try:
            # Group tasks by provider
            provider_tasks = {}
            for task in task_list:
                provider = task.provider
                if provider not in provider_tasks:
                    provider_tasks[provider] = []

                provider_tasks[provider].append(task.to_dict())

            # Create queue state
            state = QueueState(
                stage=stage_name,
                provider=constants.QUEUE_STATE_PROVIDER_MULTI,  # Multiple providers
                task_count=len(task_list),
                saved_at=time.time(),
                tasks=[],
            )

            # Add all tasks
            for provider, provider_task_list in provider_tasks.items():
                state.tasks.extend(provider_task_list)

            # Save to file - support dynamic stage names
            filepath = self._get_queue_filepath(stage_name)
            if filepath:
                content = json.dumps(
                    {
                        "stage": state.stage,
                        "provider": state.provider,
                        "task_count": state.task_count,
                        "saved_at": state.saved_at,
                        "tasks": state.tasks,
                    },
                    indent=2,
                    ensure_ascii=False,
                )

                AtomicFileWriter.write_atomic(filepath, content)
                logger.info(f"Saved {len(task_list)} tasks for {stage_name} stage")

        except Exception as e:
            logger.error(f"Failed to save queue state for {stage_name}: {e}")

    def load_queue_state(self, stage_name: str) -> List[ProviderTask]:
        """Load queue state for a specific stage"""
        filepath = self._get_queue_filepath(stage_name)
        if not filepath or not os.path.exists(filepath):
            return []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            task_list = []
            for task_data in data.get("tasks", []):
                try:
                    task = TaskFactory.from_dict(task_data)
                    task_list.append(task)
                except Exception as e:
                    logger.warning(f"Failed to deserialize task: {e}")
                    continue

            # Check if state is recent (within 24 hours)
            saved_at = data.get("saved_at", 0)
            age_hours = (time.time() - saved_at) / 3600

            if age_hours > constants.QUEUE_STATE_MAX_AGE_HOURS:
                logger.warning(f"Queue state for {stage_name} is {age_hours:.1f} hours old, skipping recovery")
                return []

            if task_list:
                logger.info(f"Loaded {len(task_list)} tasks for {stage_name} stage (age: {age_hours:.1f}h)")

            return task_list
        except Exception as e:
            logger.error(f"Failed to load queue state for {stage_name}: {e}")
            return []

    def save_all_queues(self, stages: Dict[str, Any]):
        """Save state for all queues"""
        for stage_name, stage in stages.items():
            if hasattr(stage, "get_pending_tasks"):
                tasks = stage.get_pending_tasks()
                self.save_queue_state(stage_name, tasks)
            else:
                # Fallback: try to extract tasks from queue
                tasks = self._extract_tasks_from_queue(stage)
                self.save_queue_state(stage_name, tasks)

    def load_all_queues(self) -> Dict[str, List[ProviderTask]]:
        """Load state for all queues"""
        all_tasks = {}

        for stage_name in self.queue_files.keys():
            task_list = self.load_queue_state(stage_name)
            if task_list:
                all_tasks[stage_name] = task_list

        total_tasks = sum(len(task_list) for task_list in all_tasks.values())
        if total_tasks > 0:
            logger.info(f"Loaded {total_tasks} total tasks from previous session")

        return all_tasks

    def clear_queue_state(self, stage_name: str):
        """Clear saved state for a stage"""
        filepath = self.queue_files.get(stage_name)
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleared queue state for {stage_name}")
            except Exception as e:
                logger.error(f"Failed to clear queue state for {stage_name}: {e}")

    def clear_all_states(self):
        """Clear all saved queue states"""
        for stage_name in self.queue_files.keys():
            self.clear_queue_state(stage_name)

    def get_state_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about saved queue states"""
        info = {}

        for stage_name, filepath in self.queue_files.items():
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    info[stage_name] = {
                        "task_count": data.get("task_count", 0),
                        "saved_at": data.get("saved_at", 0),
                        "age_hours": (time.time() - data.get("saved_at", 0)) / 3600,
                        "file_size": os.path.getsize(filepath),
                    }
                except Exception as e:
                    info[stage_name] = {"error": str(e)}
            else:
                info[stage_name] = {"task_count": 0, "saved_at": 0}

        return info

    def stop(self):
        """Stop the queue manager"""
        self.running = False
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join(timeout=5)

        logger.info("Stopped queue manager")

    def _periodic_save_loop(self):
        """Periodic save loop"""
        while self.running:
            try:
                time.sleep(self.save_interval)

                if hasattr(self, "stages") and self.stages:
                    self.save_all_queues(self.stages)

            except Exception as e:
                logger.error(f"Error in periodic queue save: {e}")

    def _save_empty_state(self, stage_name: str):
        """Save empty state to indicate clean stage"""
        filepath = self._get_queue_filepath(stage_name)
        if filepath:
            try:
                content = json.dumps(
                    {
                        "stage": stage_name,
                        "provider": constants.QUEUE_STATE_PROVIDER_MULTI,
                        "task_count": 0,
                        "saved_at": time.time(),
                        "tasks": [],
                    },
                    indent=2,
                )

                AtomicFileWriter.write_atomic(filepath, content)

            except Exception as e:
                logger.error(f"Failed to save empty state for {stage_name}: {e}")

    def _extract_tasks_from_queue(self, stage) -> List[ProviderTask]:
        """Extract tasks from a queue object (fallback method)"""
        task_list = []
        if hasattr(stage, "queue"):
            # Try to get tasks without removing them
            temp_tasks = []

            # Extract all tasks
            while not stage.queue.empty():
                try:
                    task = stage.queue.get_nowait()
                    if isinstance(task, ProviderTask):
                        task_list.append(task)
                        temp_tasks.append(task)
                except:
                    break

            # Put tasks back
            for task in temp_tasks:
                try:
                    stage.queue.put_nowait(task)
                except:
                    pass

        return task_list


class GracefulShutdown:
    """Handles graceful shutdown with queue state preservation"""

    def __init__(self, queue_manager: QueueManager, result_manager: Any, stages: Dict[str, Any]):
        self.queue_manager = queue_manager
        self.result_manager = result_manager
        self.stages = stages
        self.shutdown_timeout = 30

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Registered graceful shutdown handlers")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()

    def shutdown(self):
        """Perform graceful shutdown"""
        start_time = time.time()

        try:
            # 1. Stop accepting new tasks
            logger.info("Stopping task acceptance...")
            for stage in self.stages.values():
                if hasattr(stage, "stop_accepting"):
                    stage.stop_accepting()

            # 2. Flush all result buffers
            logger.info("Flushing result buffers...")
            if hasattr(self.result_manager, "flush_all"):
                self.result_manager.flush_all()

            # 3. Save all queue states
            logger.info("Saving queue states...")
            self.queue_manager.save_all_queues(self.stages)

            # 4. Wait for current tasks to complete (with timeout)
            remaining_time = self.shutdown_timeout - (time.time() - start_time)
            if remaining_time > 0:
                logger.info(f"Waiting up to {remaining_time:.1f}s for tasks to complete...")
                self._wait_for_completion(remaining_time)

            # 5. Force save final state
            logger.info("Final state save...")
            self.queue_manager.save_all_queues(self.stages)

            # 6. Stop managers
            self.queue_manager.stop()
            if hasattr(self.result_manager, "stop_all"):
                self.result_manager.stop_all()

            logger.info("Graceful shutdown completed")

        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")

        finally:
            sys.exit(0)

    def _wait_for_completion(self, timeout: float):
        """Wait for tasks to complete"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_idle = True

            for stage in self.stages.values():
                if hasattr(stage, "is_busy") and stage.is_busy():
                    all_idle = False
                    break
                elif hasattr(stage, "queue") and not stage.queue.empty():
                    all_idle = False
                    break

            if all_idle:
                logger.info("All stages idle, shutdown can proceed")
                break

            time.sleep(1)


if __name__ == "__main__":
    # Create temporary workspace
    workspace = tempfile.mkdtemp()
    logger.info(f"Testing in workspace: {workspace}")

    try:
        # Create queue manager
        qm = QueueManager(workspace, save_interval=2)

        # Create some test tasks
        test_tasks = [
            TaskFactory.create_search_task("openai", '"test"', 1),
            TaskFactory.create_search_task("gemini", '"test"', 2),
            TaskFactory.create_collect_task("openai", "http://example.com", {"key_pattern": "sk-.*"}),
        ]

        # Save queue state
        qm.save_queue_state("search", test_tasks[:2])
        qm.save_queue_state("collect", test_tasks[2:])

        # Load queue state
        loaded_search = qm.load_queue_state("search")
        loaded_collect = qm.load_queue_state("collect")

        logger.info(f"Saved {len(test_tasks[:2])} search tasks, loaded {len(loaded_search)}")
        logger.info(f"Saved {len(test_tasks[2:])} collect tasks, loaded {len(loaded_collect)}")

        # Check state info
        info = qm.get_state_info()
        logger.info(f"State info: {info}")

        # Stop manager
        qm.stop()

        logger.info("Queue manager test completed!")
    finally:
        # Cleanup
        shutil.rmtree(workspace)
