#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline system for asynchronous multi-provider task processing.
Implements producer-consumer pattern with configurable worker threads.
"""

import math
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import constants
import models
from config import Config
from queue_manager import QueueManager
from rate_limiter import RateLimiter
from refine import RefineEngine
from result_manager import MultiResultManager
from search import client
from task import (
    CheckTask,
    CollectTask,
    ModelsTask,
    ProviderTask,
    SearchTask,
    TaskFactory,
)

import logger
import utils
from logger import (
    get_check_logger,
    get_collect_logger,
    get_models_logger,
    get_pipeline_logger,
    get_search_logger,
)

# Get pipeline logger
pipeline_logger = get_pipeline_logger()


class PipelineStage(ABC):
    """Base class for pipeline stages with worker thread management"""

    def __init__(
        self,
        name: str,
        worker_func: Callable[..., Any],
        upstream: Optional["PipelineStage"] = None,
        thread_count: int = 1,
        queue_size: int = 1000,
        max_retries_requeued: int = 0,
    ) -> None:
        self.name = name
        self.worker_func = worker_func
        self.upstream = upstream
        self.thread_count = thread_count

        # Task queue
        self.task_queue = queue.Queue(maxsize=queue_size)

        # Task deduplication
        self.processed_tasks: set = set()
        self.dedup_lock = threading.Lock()

        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False
        self.accepting_tasks = True

        # Maximum number of requeues
        self.max_retries_requeued = max(max_retries_requeued, 0)

        # Statistics
        self.total_processed = 0
        self.total_errors = 0
        self.last_activity = time.time()
        self.start_time = time.time()

        # Work state tracking
        self.active_workers = 0
        self.work_lock = threading.Lock()

        # Thread safety
        self.stats_lock = threading.Lock()

        pipeline_logger.info(f"Created pipeline stage: {name}, threads: {thread_count}, queue: {queue_size}")

    def start(self) -> None:
        """Start worker threads"""
        if self.running:
            return

        self.running = True
        self.accepting_tasks = True

        for i in range(self.thread_count):
            worker = threading.Thread(target=self._worker_loop, name=f"{self.name}-worker-{i+1}", daemon=True)
            worker.start()
            self.workers.append(worker)

        pipeline_logger.info(f"[{self.name}] started {len(self.workers)} workers")

    def stop(self, timeout: float = constants.DEFAULT_SHUTDOWN_TIMEOUT) -> None:
        """Stop worker threads gracefully"""
        if not self.running:
            return

        # Stop accepting new tasks
        self.accepting_tasks = False

        # Wait for current tasks to complete
        start_time = time.time()
        while not self.task_queue.empty() and time.time() - start_time < timeout:
            time.sleep(0.1)

        # Stop workers
        self.running = False

        # Wait for workers to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5.0)

        pipeline_logger.info(f"[{self.name}] all workers is stoped")

    def put_task(self, task: ProviderTask) -> bool:
        """Add task to queue with deduplication check"""
        if not self.accepting_tasks:
            pipeline_logger.warning(f"[{self.name}] queue is not accepting new tasks, discard: {task}")
            return False

        # Generate task ID for deduplication
        task_id = self._generate_task_id(task)

        # Check if task already processed
        with self.dedup_lock:
            # Logic: attempts == 0 means it is a new task, but the same task has already been queued.
            if task_id in self.processed_tasks and (task.attempts == 0 or task.attempts > self.max_retries_requeued):
                if task.attempts > self.max_retries_requeued:
                    pipeline_logger.warning(
                        f"[{self.name}] task=[{task_id}] will be discarded because the maximum number of retries=[{self.max_retries_requeued}] has been reached"
                    )
                return False

        # Try to add to queue
        try:
            self.task_queue.put(task, timeout=1.0)
            with self.dedup_lock:
                self.processed_tasks.add(task_id)

            return True
        except queue.Full:
            pipeline_logger.warning(f"[{self.name}] queue is full")
            return False

    def is_finished(self) -> bool:
        """Check if stage is finished processing"""
        # Stage is finished if:
        # 1. Not accepting new tasks AND queue is empty
        # 2. OR upstream is finished AND queue is empty
        queue_empty = self.task_queue.empty()

        if not self.accepting_tasks and queue_empty:
            return True

        if self.upstream and self.upstream.is_finished() and queue_empty:
            return True

        return False

    def get_stats(self) -> models.PipelineStageStats:
        """Get stage statistics"""
        with self.stats_lock:
            runtime = time.time() - self.start_time
            return models.PipelineStageStats(
                name=self.name,
                queue_size=self.task_queue.qsize(),
                total_processed=self.total_processed,
                total_errors=self.total_errors,
                error_rate=self.total_errors / max(self.total_processed, 1),
                runtime=runtime,
                processing_rate=self.total_processed / max(runtime, 1),
                last_activity=self.last_activity,
                workers=len(self.workers),
                running=self.running,
            )

    def get_pending_tasks(self) -> List[ProviderTask]:
        """Get all pending tasks (for persistence)"""
        tasks = []
        temp_tasks = []

        # Extract all tasks without blocking
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
                tasks.append(task)
                temp_tasks.append(task)
            except queue.Empty:
                break

        # Put tasks back
        for task in temp_tasks:
            try:
                self.task_queue.put_nowait(task)
            except queue.Full:
                pipeline_logger.warning(f"[{self.name}] lost task during persistence: {task.task_id}")

        return tasks

    def is_busy(self) -> bool:
        """Check if stage is currently processing tasks"""
        return not self.task_queue.empty() or self._has_active_workers()

    def stop_accepting(self) -> None:
        """Stop accepting new tasks"""
        self.accepting_tasks = False

    @abstractmethod
    def process_task(self, task: ProviderTask) -> Tuple[bool, Any]:
        """Process a single task (implemented by subclasses)"""
        pass

    @abstractmethod
    def _generate_task_id(self, task: ProviderTask) -> str:
        """Generate unique task identifier for deduplication"""
        pass

    def _worker_loop(self) -> None:
        """Main worker thread loop"""
        while self.running:
            try:
                # Get task with timeout
                task = self.task_queue.get(timeout=1.0)

                # Mark worker as active
                with self.work_lock:
                    self.active_workers += 1

                # Update activity time
                with self.stats_lock:
                    self.last_activity = time.time()

                # Process task
                try:
                    requeue, result = self.process_task(task)
                    if requeue and isinstance(result, ProviderTask):
                        result.attempts += 1
                        success = self.put_task(result)

                        status = "successfully" if success else "failed"
                        pipeline_logger.warning(f"[{self.name}] Requeued queue {status}, task: {result}")

                    # Update success statistics
                    with self.stats_lock:
                        self.total_processed += 1

                    # Handle result if returned
                    if not requeue and result:
                        self._handle_result(task, result)

                except Exception as e:
                    pipeline_logger.error(f"[{self.name}] error processing task, message: {e}")

                    # Update error statistics
                    with self.stats_lock:
                        self.total_errors += 1
                        self.total_processed += 1

                finally:
                    # Mark worker as inactive
                    with self.work_lock:
                        self.active_workers -= 1

                    # Mark task as done
                    self.task_queue.task_done()

            except queue.Empty:
                # Timeout waiting for task, continue loop
                continue
            except Exception as e:
                pipeline_logger.error(f"[{self.name}] worker occur error, message: {e}")

    def _handle_result(self, task: ProviderTask, result: Any) -> None:
        """Handle task result (can be overridden by subclasses)"""
        pass

    def _has_active_workers(self) -> bool:
        """Check if any workers are currently active"""
        return any(worker.is_alive() for worker in self.workers)


class SearchStage(PipelineStage):
    """Pipeline stage for searching GitHub"""

    def __init__(
        self,
        collect_stage: "CollectStage",
        check_stage: "CheckStage",
        rate_limiter: RateLimiter,
        result_manager: MultiResultManager,
        config: Config,
        **kwargs: Any,
    ) -> None:
        super().__init__("search", self._search_worker, **kwargs)
        self.collect_stage = collect_stage
        self.check_stage = check_stage
        self.rate_limiter = rate_limiter
        self.result_manager = result_manager
        self.config = config
        self.logger = get_search_logger()

        # Whether any worker is generating new tasks
        self.generating_tasks = False

    def is_likely_finished(self) -> bool:
        """Fast lockless check for completion (may have slight inaccuracy but safe for pre-filtering)"""
        return self.task_queue.empty() and self.active_workers == 0 and not self.generating_tasks

    def is_truly_finished(self) -> bool:
        """Precise locked check for completion, only called when pre-check passes"""
        if not self.is_likely_finished():
            return False

        # Only acquire lock when likely finished for precise confirmation
        with self.work_lock:
            return self.task_queue.empty() and self.active_workers == 0 and not self.generating_tasks

    def _generate_task_id(self, task: ProviderTask) -> str:
        """Generate unique task identifier for deduplication"""
        return f"search:{task.provider}:{getattr(task, 'query', '')}:{getattr(task, 'page', 1)}:{getattr(task, 'regex', '')}"

    def process_task(self, task: ProviderTask) -> Tuple[bool, Optional[models.SearchTaskResult]]:
        """Process search task"""
        if not isinstance(task, SearchTask):
            self.logger.error(f"[{self.name}] invalid task type: {type(task)}")
            return False, None

        return self._search_worker(task)

    def _search_worker(self, task: SearchTask) -> Tuple[bool, models.SearchTaskResult]:
        """Optimized search worker using single network request per search"""
        try:
            # Execute search based on page number
            if task.page == 1:
                results, content, total = self._execute_first_page_search(task)
                # Store total for later task generation
                data = models.SearchTaskResult(links=results or [], total=total)
            else:
                results, content = self._execute_page_search(task)
                data = models.SearchTaskResult(links=results or [])

            # Extract keys directly from search content
            keys: List[models.Service] = []
            if content and task.regex:
                keys = self._extract_keys_from_content(content, task)
                for key_service in keys:
                    check_task = TaskFactory.create_check_task(task.provider, key_service)
                    self.check_stage.put_task(check_task)

                if keys:
                    self.logger.info(
                        f"[{self.name}] extracted {len(keys)} keys directly from search content, provider: {task.provider}"
                    )
            elif not content:
                return True, task

            # Create collect tasks for links
            if results:
                for link in results:
                    patterns = {
                        constants.PATTERN_KEY: task.regex,
                        constants.PATTERN_ADDRESS: task.address_pattern,
                        constants.PATTERN_ENDPOINT: task.endpoint_pattern,
                        constants.PATTERN_MODEL: task.model_pattern,
                    }
                    collect_task = TaskFactory.create_collect_task(task.provider, link, patterns)
                    self.collect_stage.put_task(collect_task)

                # Save links to results
                self.result_manager.add_links(task.provider, results)

            data.keys_extracted = len(keys)

            self.logger.info(
                f"[{self.name}] search completed for {task.provider}: {len(results) if results else 0} links, {len(keys)} keys"
            )

            return False, data

        except Exception as e:
            self.logger.error(f"[{self.name}] occur error, provider: {task.provider}, task: {task}, message: {e}")
            return True, task

    def _execute_first_page_search(self, task: SearchTask) -> Tuple[List[str], str, int]:
        """Execute first page search and get total count in single request"""
        # Apply rate limiting
        self._apply_rate_limit(task.use_api)

        # Get config and select auth method based on use_api
        auth_token = self.config.global_config.token if task.use_api else self.config.global_config.session

        # Execute search with count - now returns content as well
        results, total, content = client.search_with_count(
            query=self._preprocess_query(task.query, task.use_api),
            session=auth_token,
            page=task.page,
            with_api=task.use_api,
            peer_page=constants.API_RESULTS_PER_PAGE if task.use_api else constants.WEB_RESULTS_PER_PAGE,
        )

        return results, content, total

    def _preprocess_query(self, query: str, use_api: bool) -> str:
        """Github Rest API search syntax don't support regex, so we need remove it if exists"""
        if use_api:
            keyword = RefineEngine.get_instance().clean_regex(query=query)
            if keyword:
                query = keyword

        return query

    def _execute_page_search(self, task: SearchTask) -> Tuple[List[str], str]:
        """Execute subsequent page search in single request"""
        # Apply rate limiting
        self._apply_rate_limit(task.use_api)

        # Get config and select auth method based on use_api
        auth_token = self.config.global_config.token if task.use_api else self.config.global_config.session

        # Execute search - now returns content as well
        results, content = client.search_code(
            query=self._preprocess_query(task.query, task.use_api),
            session=auth_token,
            page=task.page,
            with_api=task.use_api,
            peer_page=constants.API_RESULTS_PER_PAGE if task.use_api else constants.WEB_RESULTS_PER_PAGE,
        )

        return results, content

    def _apply_rate_limit(self, use_api: bool) -> bool:
        """Apply rate limiting for GitHub requests"""
        service_type = constants.SERVICE_TYPE_GITHUB_API if use_api else constants.SERVICE_TYPE_GITHUB_WEB
        if not self.rate_limiter.acquire(service_type):
            wait_time = self.rate_limiter.wait_time(service_type)
            if wait_time > 0:
                time.sleep(wait_time)
                if not self.rate_limiter.acquire(service_type):
                    self.logger.warning(
                        f'[{self.name}] rate limit exceeded for Github {"Rest API" if use_api else "Web"}'
                    )
                    return False
        return True

    def _handle_first_page_results(self, task: SearchTask, total: int) -> None:
        """Handle first page results - decide pagination or refinement"""
        limit = constants.API_LIMIT if task.use_api else constants.WEB_LIMIT
        per_page = constants.API_RESULTS_PER_PAGE if task.use_api else constants.WEB_RESULTS_PER_PAGE

        # If needs refine query
        if total > limit:
            # Regenerate the query with less data
            partitions = int(math.ceil(total / limit))
            queries = RefineEngine.get_instance().generate_queries(query=task.query, partitions=partitions)

            # Add new query to task queue
            for query in queries:
                if not query:
                    self.logger.warning(
                        f"[{self.name}] skip to add to task queue due to refined query is empty for query: {task.query}, provider: {task.provider}"
                    )
                    continue
                elif query == task.query:
                    self.logger.warning(
                        f"[{self.name}] discard due to refined query is the same as original query: {query}, provider: {task.provider}"
                    )
                    continue

                item = SearchTask(
                    provider=task.provider,
                    query=query,
                    regex=task.regex,
                    page=1,
                    use_api=task.use_api,
                    address_pattern=task.address_pattern,
                    endpoint_pattern=task.endpoint_pattern,
                    model_pattern=task.model_pattern,
                )

                self.put_task(item)

            self.logger.info(
                f"[{self.name}] generated {len(queries)} refined tasks for provider: {task.provider}, query: {task.query}"
            )

        # If needs pagination and not refining
        elif total > per_page:
            page_tasks = self._generate_page_tasks(task, total, per_page)
            for page_task in page_tasks:
                self.put_task(page_task)
            self.logger.info(
                f"[{self.name}] generated {len(page_tasks)} page tasks for provider: {task.provider}, query: {task.query}"
            )

    def _generate_page_tasks(self, task: SearchTask, total: int, per_page: int) -> List[SearchTask]:
        """Generate pagination tasks"""
        # Limit max pages
        max_pages = min(
            math.ceil(total / per_page),
            constants.API_MAX_PAGES if task.use_api else constants.WEB_MAX_PAGES,
        )

        page_tasks: List[SearchTask] = []
        for page in range(2, max_pages + 1):  # Start from page 2
            page_task = SearchTask(
                provider=task.provider,
                query=task.query,
                regex=task.regex,
                page=page,
                use_api=task.use_api,
                address_pattern=task.address_pattern,
                endpoint_pattern=task.endpoint_pattern,
                model_pattern=task.model_pattern,
            )
            page_tasks.append(page_task)

        return page_tasks

    def _handle_result(self, task: SearchTask, result: models.SearchTaskResult) -> None:
        """Handle task result and generate new tasks with proper state tracking"""
        # Check if this is a first page task that might generate new tasks
        if task.page == 1 and result.total > 0:
            # Mark as generating tasks
            with self.work_lock:
                self.generating_tasks = True

            try:
                # Generate refined or pagination tasks
                self._handle_first_page_results(task, result.total)
            finally:
                # Mark generation complete
                with self.work_lock:
                    self.generating_tasks = False

    def _extract_keys_from_content(self, content: str, task: SearchTask) -> List[models.Service]:
        """Extract keys directly from search content"""
        services = client.collect(
            key_pattern=task.regex,
            address_pattern=task.address_pattern,
            endpoint_pattern=task.endpoint_pattern,
            model_pattern=task.model_pattern,
            text=content,
        )

        return services


class CollectStage(PipelineStage):
    """Pipeline stage for collecting keys from URLs"""

    def __init__(self, check_stage: "CheckStage", result_manager: MultiResultManager, **kwargs: Any) -> None:
        super().__init__("collect", self._collect_worker, **kwargs)
        self.check_stage = check_stage
        self.result_manager = result_manager
        self.logger = get_collect_logger()

    def _generate_task_id(self, task: ProviderTask) -> str:
        """Generate unique task identifier for deduplication"""
        return f"collect:{task.provider}:{getattr(task, 'url', '')}"

    def process_task(self, task: ProviderTask) -> Tuple[bool, Optional[models.CollectTaskResult]]:
        """Process collect task"""
        if not isinstance(task, CollectTask):
            self.logger.error(f"[{self.name}] invalid task type: {type(task)}")
            return False, None

        return self._collect_worker(task)

    def _collect_worker(self, task: CollectTask) -> Tuple[bool, Optional[models.CollectTaskResult]]:
        """Collect worker implementation with advanced collection"""
        try:
            # Execute collection using global collect function
            services = client.collect(
                key_pattern=task.key_pattern,
                url=task.url,
                retries=task.retries,
                address_pattern=task.address_pattern,
                endpoint_pattern=task.endpoint_pattern,
                model_pattern=task.model_pattern,
            )

            # Create check tasks for found services
            if services:
                for service in services:
                    check_task = TaskFactory.create_check_task(task.provider, service)
                    self.check_stage.put_task(check_task)

                # Save material keys
                self.result_manager.add_result(task.provider, constants.RESULT_CATEGORY_MATERIAL_KEYS, services)

            # Save the processed link
            self.result_manager.add_links(task.provider, [task.url])

            return False, models.CollectTaskResult(services=services)

        except Exception as e:
            self.logger.error(f"[{self.name}] occur error for provider: {task.provider}, task: {task}, message: {e}")
            return True, task


class CheckStage(PipelineStage):
    """Pipeline stage for validating API keys"""

    def __init__(
        self,
        models_stage: "ModelsStage",
        result_manager: MultiResultManager,
        rate_limiter: RateLimiter,
        providers: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__("check", self._check_worker, **kwargs)
        self.models_stage = models_stage
        self.result_manager = result_manager
        self.rate_limiter = rate_limiter
        self.providers = providers
        self.logger = get_check_logger()

    def _generate_task_id(self, task: ProviderTask) -> str:
        """Generate unique task identifier for deduplication"""
        service = getattr(task, "service", None)
        if service:
            return f"check:{task.provider}:{service.key}:{service.address}:{service.endpoint}"

        return f"check:{task.provider}:unknown"

    def process_task(self, task: ProviderTask) -> Tuple[bool, Optional[models.CheckTaskResult]]:
        """Process check task"""
        if not isinstance(task, CheckTask):
            self.logger.error(f"[{self.name}] invalid task type: {type(task)}")
            return False, None

        return self._check_worker(task)

    def _check_worker(self, task: CheckTask) -> Tuple[bool, Optional[models.CheckTaskResult]]:
        """Check worker implementation"""
        try:
            # Get provider instance
            provider = self.providers.get(task.provider)
            if not provider:
                self.logger.error(f"[{self.name}] unknown provider: {task.provider}")
                return False, None

            # Apply rate limiting
            service_type = Pipeline.get_service_name(task.provider)
            if not self.rate_limiter.acquire(service_type):
                wait_time = self.rate_limiter.wait_time(service_type)
                if wait_time > 0:
                    time.sleep(wait_time)
                    if not self.rate_limiter.acquire(service_type):
                        self.logger.warning(f"[{self.name}] rate limit exceeded for provider: {task.provider}")
                        return True, task

            # Execute check
            result = provider.check(
                token=task.service.key,
                address=task.custom_url or task.service.address,
                endpoint=task.service.endpoint,
                model=task.service.model,
            )

            # Report rate limit success
            self.rate_limiter.report_result(service_type, True)

            # Handle result based on availability
            if result.available:
                # Create models task
                models_task = TaskFactory.create_models_task(task.provider, task.service)
                self.models_stage.put_task(models_task)

                # Save as valid key
                self.result_manager.add_result(task.provider, constants.RESULT_CATEGORY_VALID_KEYS, [task.service])
                return False, models.CheckTaskResult(valid_keys=[task.service])

            else:
                # Categorize based on error reason
                if result.reason == models.ErrorReason.NO_QUOTA:
                    self.result_manager.add_result(
                        task.provider, constants.RESULT_CATEGORY_NO_QUOTA_KEYS, [task.service]
                    )
                    return False, models.CheckTaskResult(no_quota_keys=[task.service])

                elif result.reason in [
                    models.ErrorReason.RATE_LIMITED,
                    models.ErrorReason.NO_MODEL,
                    models.ErrorReason.NO_ACCESS,
                ]:
                    self.result_manager.add_result(
                        task.provider, constants.RESULT_CATEGORY_WAIT_CHECK_KEYS, [task.service]
                    )
                    return False, models.CheckTaskResult(wait_check_keys=[task.service])

                else:
                    self.result_manager.add_result(
                        task.provider, constants.RESULT_CATEGORY_INVALID_KEYS, [task.service]
                    )
                    return False, models.CheckTaskResult(invalid_keys=[task.service])

        except Exception as e:
            # Report rate limit failure
            self.rate_limiter.report_result(Pipeline.get_service_name(task.provider), False)
            self.logger.error(f"[{self.name}] occur error for provider: {task.provider}, task: {task}, message: {e}")

            return True, task


class ModelsStage(PipelineStage):
    """Pipeline stage for retrieving model lists"""

    def __init__(self, result_manager: MultiResultManager, providers: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__("models", self._models_worker, **kwargs)
        self.result_manager = result_manager
        self.providers = providers
        self.logger = get_models_logger()

    def _generate_task_id(self, task: ProviderTask) -> str:
        """Generate unique task identifier for deduplication"""
        service = getattr(task, "service", None)
        if service:
            return f"models:{task.provider}:{service.key}:{service.address}"

        return f"models:{task.provider}:unknown"

    def process_task(self, task: ProviderTask) -> Tuple[bool, Optional[models.ModelsTaskResult]]:
        """Process models task"""
        if not isinstance(task, ModelsTask):
            self.logger.error(f"[{self.name}] invalid task type: {type(task)}")
            return False, None

        return self._models_worker(task)

    def _models_worker(self, task: ModelsTask) -> Tuple[bool, Optional[models.ModelsTaskResult]]:
        """Models worker implementation"""
        try:
            # Get provider instance
            provider = self.providers.get(task.provider)
            if not provider:
                self.logger.error(f"[{self.name}] unknown provider: {task.provider}")
                return False, None

            # Get model list
            model_list = provider.list_models(
                token=task.service.key, address=task.service.address, endpoint=task.service.endpoint
            )

            # Save models to result manager
            if model_list:
                self.result_manager.add_models(task.provider, task.service.key, model_list)

            return False, models.ModelsTaskResult(models=model_list)

        except Exception as e:
            self.logger.error(f"[{self.name}] list models error, provider: {task.provider}, task: {task}, message: {e}")
            return True, task


class Pipeline:
    """Main pipeline coordinator"""

    def __init__(self, config: Config, providers: Dict[str, Any]):
        self.config = config
        self.providers = providers

        # Create shared components
        self.result_manager = MultiResultManager(
            workspace=config.global_config.workspace,
            providers=providers,
            batch_size=config.persistence.batch_size,
            save_interval=config.persistence.save_interval,
        )

        self.rate_limiter = RateLimiter(config.rate_limits)

        # Initialize GitHub client rate limiter
        client.init_github_client(config.rate_limits)

        self.queue_manager = QueueManager(
            workspace=config.global_config.workspace, save_interval=config.persistence.queue_interval
        )

        # Create pipeline stages
        self._create_stages()

        # Statistics and completion tracking
        self.start_time = time.time()
        self.initial_tasks_count = 0
        self.initial_tasks_completed = 0
        self.completion_check_thread = None
        self.completion_check_running = False

        pipeline_logger.info("Initialized pipeline with 4 stages")

    def _create_stages(self) -> None:
        """Create pipeline stages in dependency order"""
        thread_config = self.config.pipeline.threads
        queue_config = self.config.pipeline.queue_sizes

        # Create stages (reverse dependency order)
        self.models_stage = ModelsStage(
            result_manager=self.result_manager,
            providers=self.providers,
            thread_count=thread_config.get("models", 2),
            queue_size=queue_config.get("models", constants.DEFAULT_MODELS_QUEUE_SIZE),
            max_retries_requeued=self.config.global_config.max_retries_requeued,
        )

        self.check_stage = CheckStage(
            models_stage=self.models_stage,
            result_manager=self.result_manager,
            rate_limiter=self.rate_limiter,
            providers=self.providers,
            thread_count=thread_config.get("check", 8),
            queue_size=queue_config.get("check", constants.DEFAULT_CHECK_QUEUE_SIZE),
            max_retries_requeued=self.config.global_config.max_retries_requeued,
        )

        self.collect_stage = CollectStage(
            check_stage=self.check_stage,
            result_manager=self.result_manager,
            thread_count=thread_config.get("collect", 4),
            queue_size=queue_config.get("collect", constants.DEFAULT_COLLECT_QUEUE_SIZE),
            max_retries_requeued=self.config.global_config.max_retries_requeued,
        )

        self.search_stage = SearchStage(
            collect_stage=self.collect_stage,
            check_stage=self.check_stage,
            rate_limiter=self.rate_limiter,
            result_manager=self.result_manager,
            config=self.config,
            thread_count=thread_config.get("search", 2),
            queue_size=queue_config.get("search", constants.DEFAULT_SEARCH_QUEUE_SIZE),
            max_retries_requeued=self.config.global_config.max_retries_requeued,
        )

        # Set upstream relationships
        self.collect_stage.upstream = self.search_stage
        self.check_stage.upstream = self.collect_stage
        self.models_stage.upstream = self.check_stage

    def start(self) -> None:
        """Start all pipeline stages"""
        stages = [self.search_stage, self.collect_stage, self.check_stage, self.models_stage]

        for stage in stages:
            stage.start()

        # Start queue manager periodic save
        stage_dict = {
            "search": self.search_stage,
            "collect": self.collect_stage,
            "check": self.check_stage,
            "models": self.models_stage,
        }
        self.queue_manager.start_periodic_save(stage_dict)

        pipeline_logger.info("Started all pipeline stages")

    def stop(self, timeout: float = 30.0) -> None:
        """Stop all pipeline stages"""
        # Stop completion monitoring
        self.completion_check_running = False
        if self.completion_check_thread and self.completion_check_thread.is_alive():
            self.completion_check_thread.join(timeout=5.0)

        stages = [self.search_stage, self.collect_stage, self.check_stage, self.models_stage]

        for stage in stages:
            stage.stop(timeout / len(stages))

        # Stop managers
        self.queue_manager.stop()
        self.result_manager.stop_all()

        pipeline_logger.info("Stopped all pipeline stages")

    def is_finished(self) -> bool:
        """Check if entire pipeline is finished"""
        # If no initial tasks were added (e.g., all providers have skip_search=True),
        # check if all queues are empty and no work is being done
        if not hasattr(self, "initial_tasks_count") or self.initial_tasks_count == 0:
            all_queues_empty = (
                self.search_stage.task_queue.empty()
                and self.collect_stage.task_queue.empty()
                and self.check_stage.task_queue.empty()
                and self.models_stage.task_queue.empty()
            )

            # If all queues are empty, consider pipeline finished
            if all_queues_empty:
                return True

        # Normal completion check for cases with initial tasks
        return (
            self.search_stage.is_finished()
            and self.collect_stage.is_finished()
            and self.check_stage.is_finished()
            and self.models_stage.is_finished()
        )

    def get_all_stats(self) -> models.PipelineAllStats:
        """Get statistics for all stages"""
        return models.PipelineAllStats(
            search=self.search_stage.get_stats(),
            collect=self.collect_stage.get_stats(),
            check=self.check_stage.get_stats(),
            models=self.models_stage.get_stats(),
            runtime=time.time() - self.start_time,
            rate_limiter_stats=client.get_github_stats(),
        )

    def add_initial_tasks(self, initial_tasks: List[ProviderTask]) -> None:
        """Add initial search tasks to pipeline"""
        self.initial_tasks_count = len(initial_tasks)
        self.initial_tasks_completed = 0

        for task in initial_tasks:
            self.search_stage.put_task(task)

        # Start completion monitoring thread
        self._start_completion_monitoring()

        pipeline_logger.info(f"Added {len(initial_tasks)} initial tasks to pipeline")

    def _start_completion_monitoring(self) -> None:
        """Start thread to monitor completion of initial tasks"""
        if self.completion_check_running:
            return

        self.completion_check_running = True
        self.completion_check_thread = threading.Thread(
            target=self._completion_monitor_loop, name="completion-monitor", daemon=True
        )
        self.completion_check_thread.start()

    def _completion_monitor_loop(self) -> None:
        """Monitor completion using precise work state detection"""
        while self.completion_check_running:
            try:
                # Use optimized completion detection with lockless pre-check
                if self.search_stage.is_likely_finished():
                    # Only perform precise check when likely finished
                    if self.search_stage.is_truly_finished():
                        self._wait_for_downstream_completion()
                        break

                time.sleep(5)  # Check every 5 seconds (reduced frequency)

            except Exception as e:
                pipeline_logger.error(f"Completion monitor error: {e}")
                time.sleep(5)

    def _wait_for_downstream_completion(self) -> None:
        """Wait for downstream stages to complete and then stop accepting tasks"""
        pipeline_logger.info("SearchStage truly finished, waiting for downstream stages...")

        # Now safe to stop search stage from accepting new tasks
        self.search_stage.stop_accepting()

        # Wait for downstream stages to finish processing
        max_wait_time = 300  # 5 minutes max wait
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            # Check if all stages are idle (no pending tasks)
            if (
                self.search_stage.task_queue.empty()
                and self.collect_stage.task_queue.empty()
                and self.check_stage.task_queue.empty()
                and self.models_stage.task_queue.empty()
            ):

                # Wait a bit more to ensure no new tasks are being generated
                time.sleep(5)

                # Double check
                if (
                    self.search_stage.task_queue.empty()
                    and self.collect_stage.task_queue.empty()
                    and self.check_stage.task_queue.empty()
                    and self.models_stage.task_queue.empty()
                ):

                    pipeline_logger.info("All stages completed, stopping task acceptance...")

                    # Stop all stages from accepting new tasks
                    self.collect_stage.stop_accepting()
                    self.check_stage.stop_accepting()
                    self.models_stage.stop_accepting()

                    break

            time.sleep(2)

        self.completion_check_running = False
        pipeline_logger.info("Pipeline completion monitoring finished")

    @staticmethod
    def get_service_name(provider: str) -> str:
        name = utils.trim(provider)
        if not name:
            return ""

        return f"{constants.PROVIDER_SERVICE_PREFIX}:{name}"
