#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task manager for coordinating multi-provider pipeline processing.
Creates provider instances from configuration and manages task distribution.
"""

import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional

import constants
import search
from config import Config, ConfigParser, TaskConfig, load_config
from models import Condition, ProviderPatterns, TaskManagerStats, TaskRecoveryInfo
from task import ProviderTask, SearchTask, TaskFactory

from logger import get_task_manager_logger
from pipeline import Pipeline

# Get task manager logger
logger = get_task_manager_logger()


class ProviderFactory:
    """Factory for creating provider instances from configuration"""

    @staticmethod
    def create_provider(task_config: TaskConfig, conditions: List[Condition]) -> search.Provider:
        """Create provider instance based on configuration"""
        provider_type = task_config.provider_type
        name = task_config.name
        api_config = task_config.api
        skip_search = task_config.skip_search
        extras = task_config.extras

        if provider_type == constants.PROVIDER_TYPE_OPENAI_LIKE:
            return ProviderFactory._create_openai_like(name, api_config, conditions, skip_search, **(extras or {}))

        elif provider_type == constants.PROVIDER_TYPE_ANTHROPIC:
            return ProviderFactory._create_anthropic(api_config, conditions, skip_search)

        elif provider_type == constants.PROVIDER_TYPE_GEMINI:
            return ProviderFactory._create_gemini(api_config, conditions, skip_search)

        elif provider_type == constants.PROVIDER_TYPE_GOOEY_AI:
            return ProviderFactory._create_gooey_ai(api_config, conditions, skip_search)

        elif provider_type == constants.PROVIDER_TYPE_STABILITY_AI:
            return ProviderFactory._create_stability_ai(api_config, conditions, skip_search)

        elif provider_type == constants.PROVIDER_TYPE_CUSTOM:
            return ProviderFactory._create_custom(name, api_config, conditions, skip_search)

        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    @staticmethod
    def _create_openai_like(
        name: str,
        api_config: Any,
        conditions: List[Condition],
        skip_search: bool,
        **kwargs,
    ) -> search.OpenAILikeProvider:
        """Create OpenAI-compatible provider"""
        # Handle special cases
        name_lower = name.lower()
        if name_lower == "azure":
            return search.AzureOpenAIProvider(
                conditions=conditions, default_model=api_config.default_model, skip_search=skip_search
            )
        elif name_lower == "doubao":
            return search.DoubaoProvider(
                conditions=conditions, default_model=api_config.default_model, skip_search=skip_search
            )
        elif name_lower == "qianfan":
            return search.QianFanProvider(
                conditions=conditions, default_model=api_config.default_model, skip_search=skip_search
            )
        elif name_lower == "openai":
            return search.OpenAIProvider(
                conditions=conditions, default_model=api_config.default_model, skip_search=skip_search
            )
        else:
            # Generic OpenAI-like provider
            if not kwargs:
                kwargs = {}
            if api_config.auth_key != "Authorization":
                kwargs["auth_key"] = api_config.auth_key

            return search.OpenAILikeProvider(
                name=name,
                base_url=api_config.base_url,
                default_model=api_config.default_model,
                conditions=conditions,
                completion_path=api_config.completion_path,
                model_path=api_config.model_path,
                skip_search=skip_search,
                **kwargs,
            )

    @staticmethod
    def _create_anthropic(api_config: Any, conditions: List[Condition], skip_search: bool) -> search.AnthropicProvider:
        """Create Anthropic provider"""
        return search.AnthropicProvider(
            conditions=conditions, default_model=api_config.default_model, skip_search=skip_search
        )

    @staticmethod
    def _create_gemini(api_config: Any, conditions: List[Condition], skip_search: bool) -> search.GeminiProvider:
        """Create Gemini provider"""
        return search.GeminiProvider(
            conditions=conditions, default_model=api_config.default_model, skip_search=skip_search
        )

    @staticmethod
    def _create_gooey_ai(api_config: Any, conditions: List[Condition], skip_search: bool) -> search.GooeyAIProvider:
        """Create GooeyAI provider"""
        return search.GooeyAIProvider(
            conditions=conditions, default_model=api_config.default_model, skip_search=skip_search
        )

    @staticmethod
    def _create_stability_ai(
        api_config: Any, conditions: List[Condition], skip_search: bool
    ) -> search.StabilityAIProvider:
        """Create StabilityAI provider"""
        return search.StabilityAIProvider(
            conditions=conditions, default_model=api_config.default_model, skip_search=skip_search
        )

    @staticmethod
    def _create_custom(
        name: str, api_config: Any, conditions: List[Condition], skip_search: bool
    ) -> search.OpenAILikeProvider:
        """Create custom provider (based on OpenAI-like)"""
        kwargs = {}

        # Add custom configuration
        if api_config.auth_key != "Authorization":
            kwargs["auth_key"] = api_config.auth_key

        if api_config.extra_headers:
            kwargs["extra_headers"] = api_config.extra_headers

        return search.OpenAILikeProvider(
            name=name,
            base_url=api_config.base_url,
            default_model=api_config.default_model,
            conditions=conditions,
            completion_path=api_config.completion_path,
            model_path=api_config.model_path,
            skip_search=skip_search,
            **kwargs,
        )


class TaskManager:
    """Main task manager for multi-provider coordination"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.providers: Dict[str, search.Provider] = {}
        self.pipeline: Optional[Pipeline] = None
        self.running = False

        # Initialize providers
        self._initialize_providers()

        # Create pipeline
        self._create_pipeline()

        logger.info(f"Initialized task manager with {len(self.providers)} providers")

    def _initialize_providers(self) -> None:
        """Initialize all enabled providers from configuration"""
        parser = ConfigParser()

        for task_config in self.config.tasks:
            if not task_config.enabled:
                logger.info(f"Skipping disabled provider: {task_config.name}")
                continue

            try:
                # Parse conditions with flexible regex support
                conditions = parser.parse_conditions(task_config)

                if not conditions:
                    logger.warning(f"No valid conditions for provider {task_config.name}, skipping")
                    continue

                # Create provider instance
                provider = ProviderFactory.create_provider(task_config, conditions)
                self.providers[task_config.name] = provider

                logger.info(
                    f"Created provider: {task_config.name} ({task_config.provider_type}) "
                    f"with {len(conditions)} conditions"
                )

            except Exception as e:
                logger.error(f"Failed to create provider {task_config.name}: {e}")
                continue

        if not self.providers:
            raise ValueError("No valid providers configured")

    def _create_pipeline(self) -> None:
        """Create pipeline with all components"""
        # Add provider-specific rate limits
        rate_limits = self.config.rate_limits.copy()

        for task_config in self.config.tasks:
            if task_config.enabled:
                service_name = f"{constants.PROVIDER_SERVICE_PREFIX}{task_config.name}"
                rate_limits[service_name] = task_config.rate_limit

        # Update config with provider rate limits
        self.config.rate_limits = rate_limits

        # Create pipeline
        self.pipeline = Pipeline(self.config, self.providers)

        logger.info("Created pipeline with all providers")

    def start(self) -> None:
        """Start the task manager and pipeline"""
        if self.running:
            return

        # 1. Start pipeline (creates ResultManager without backup)
        self.pipeline.start()

        # 2. Recover queue tasks
        recoverd_tasks = self.pipeline.queue_manager.load_all_queues()

        # 3. Filetr out undo tasks with providers
        undo_tasks: Dict[str, List[ProviderTask]] = defaultdict(list)
        for stage, tasks in recoverd_tasks.items():
            for task in tasks:
                if not task or task.provider not in self.providers:
                    continue

                provider = self.providers.get(task.provider)
                if (
                    stage == constants.STAGE_NAME_SEARCH or stage == constants.STAGE_NAME_COLLECT
                ) and provider.skip_search:
                    logger.debug(f"Skipping recover {stage} task for provider {task.provider} due to skip_search flag")
                    continue

                undo_tasks[stage].append(task)

        # 4. Recover result file tasks (material.txt, links.txt)
        old_tasks = self.pipeline.result_manager.recover_all_tasks()

        # 5. Add recovered tasks to appropriate queues
        recovery_info = TaskRecoveryInfo(
            queue_tasks=undo_tasks,
            result_tasks=old_tasks,
            total_queue_tasks=sum(len(tasks) for tasks in undo_tasks.values()),
            total_result_tasks=sum(
                len(provider_tasks[constants.RESULT_CATEGORY_CHECK_TASKS])
                + len(provider_tasks[constants.RESULT_CATEGORY_COLLECT_TASKS])
                for provider_tasks in old_tasks.values()
            ),
        )
        self._add_recovered_tasks(recovery_info)

        # 6. Backup existing files (after recovery is complete)
        self.pipeline.result_manager.backup_all_existing_files()

        # 7. Add initial search tasks
        initial_tasks = self._create_initial_tasks()
        if initial_tasks:
            self.pipeline.add_initial_tasks(initial_tasks)

        self.running = True

        # Log recovery and startup info
        logger.info(
            f"Started task manager: {recovery_info.total_queue_tasks} queue tasks, {recovery_info.total_result_tasks} result tasks, {len(initial_tasks)} initial tasks"
        )

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the task manager gracefully"""
        if not self.running:
            return

        self.running = False

        if self.pipeline:
            self.pipeline.stop(timeout)

        logger.info("Stopped task manager")

    def wait_completion(self) -> None:
        """Wait for all tasks to complete"""
        if not self.pipeline:
            return

        logger.info("Waiting for pipeline completion...")

        while not self.pipeline.is_finished():
            time.sleep(1)

            # Print progress periodically
            if hasattr(self, "_last_progress_time"):
                if time.time() - self._last_progress_time > constants.PROGRESS_UPDATE_INTERVAL:
                    self._print_progress()
                    self._last_progress_time = time.time()
            else:
                self._last_progress_time = time.time()

        logger.info("Pipeline completed")

    def get_stats(self) -> TaskManagerStats:
        """Get comprehensive statistics"""
        stats = TaskManagerStats(providers=list(self.providers.keys()), running=self.running)

        if self.pipeline:
            stats.pipeline_stats = self.pipeline.get_all_stats()
            stats.result_stats = self.pipeline.result_manager.get_all_stats()

        return stats

    def _create_initial_tasks(self) -> List[SearchTask]:
        """Create initial search tasks for all providers"""
        tasks = []

        for task_config in self.config.tasks:
            if not task_config.enabled:
                continue

            if task_config.skip_search:
                logger.warning(f"Skipping search for provider {task_config.name} due to skip_search is True")
                continue

            if not self.config.global_config.token and not self.config.global_config.session:
                logger.warning(
                    f"Skipping search for provider {task_config.name} as no github token or session is provided"
                )
                continue

            provider = self.providers.get(task_config.name)
            if not provider:
                continue

            for condition in provider.conditions:
                # Create search task for each condition
                task = TaskFactory.create_search_task(
                    provider=task_config.name,
                    query=condition.query or condition.regex,
                    regex=condition.regex,
                    page=1,
                    use_api=task_config.use_api,
                    address_pattern=task_config.patterns.address_pattern
                    or provider.extras.get(constants.PATTERN_ADDRESS, ""),
                    endpoint_pattern=task_config.patterns.endpoint_pattern
                    or provider.extras.get(constants.PATTERN_ENDPOINT, ""),
                    model_pattern=task_config.patterns.model_pattern
                    or provider.extras.get(constants.PATTERN_MODEL, ""),
                )
                tasks.append(task)

        return tasks

    def _add_recovered_tasks(self, recovery_info: TaskRecoveryInfo) -> None:
        """Add recovered tasks to appropriate pipeline stages"""

        # Add recovered queue tasks
        for stage_name, tasks in recovery_info.queue_tasks.items():
            if not tasks:
                continue

            stage = getattr(self.pipeline, f"{stage_name}_stage", None)
            if stage:
                for task in tasks:
                    stage.put_task(task)
                logger.info(f"Added {len(tasks)} recovered {stage_name} tasks")

        # Add recovered result tasks
        for provider_name, provider_tasks in recovery_info.result_tasks.items():
            # Add check tasks from material.txt
            for service in provider_tasks.get(constants.RESULT_CATEGORY_CHECK_TASKS, []):
                try:
                    check_task = TaskFactory.create_check_task(provider_name, service)
                    self.pipeline.check_stage.put_task(check_task)
                except Exception as e:
                    logger.warning(f"Failed to create check task for {provider_name}: {e}")

            # Add collect tasks from links.txt
            provider = self.providers.get(provider_name)
            if provider:
                for url in provider_tasks.get(constants.RESULT_CATEGORY_COLLECT_TASKS, []):
                    try:
                        # Get patterns from provider conditions
                        patterns = self._get_provider_patterns(provider)
                        collect_task = TaskFactory.create_collect_task(provider_name, url, patterns.to_dict())
                        self.pipeline.collect_stage.put_task(collect_task)
                    except Exception as e:
                        logger.warning(f"Failed to create collect task for {provider_name}: {e}")

    def _get_provider_patterns(self, provider: search.Provider) -> ProviderPatterns:
        """Extract patterns from provider conditions"""
        patterns = ProviderPatterns()

        # Use first condition's regex as key pattern
        if provider.conditions:
            patterns.key_pattern = provider.conditions[0].regex

        return patterns

    def _print_progress(self) -> None:
        """Print current progress"""
        if not self.pipeline:
            return

        try:
            pipeline_stats = self.pipeline.get_all_stats()
            result_stats = self.pipeline.result_manager.get_all_stats()

            separator_long = "=" * 80
            logger.info(separator_long)
            logger.info(f"{'Pipeline Progress':^80}")
            logger.info(separator_long)

            # Pipeline stage stats
            for stage_name, stats in pipeline_stats.items():
                if isinstance(stats, dict) and "queue_size" in stats:
                    logger.info(
                        f"{stage_name:>10}: queue={stats['queue_size']:>4}, "
                        f"processed={stats['total_processed']:>6}, "
                        f"errors={stats['total_errors']:>3}"
                    )

            logger.info("-" * 80)

            # Provider result stats
            for provider_name, stats in result_stats.items():
                logger.info(
                    f"{provider_name:>10}: valid={stats.valid_keys:>4}, "
                    f"no_quota={stats.no_quota_keys:>3}, "
                    f"wait={stats.wait_check_keys:>3}, "
                    f"invalid={stats.invalid_keys:>4}, "
                    f"links={stats.total_links:>5}"
                )

            logger.info(separator_long)

        except Exception as e:
            logger.debug(f"Error printing progress: {e}")


def create_task_manager(config_file: str = constants.DEFAULT_CONFIG_FILE) -> TaskManager:
    """Factory function to create task manager from configuration"""
    config = load_config(config_file)
    return TaskManager(config)


if __name__ == "__main__":
    # Test task manager creation
    try:
        # Create task manager
        manager = create_task_manager()

        logger.info(f"Created task manager with providers: {list(manager.providers.keys())}")

        # Test provider creation
        for name, provider in manager.providers.items():
            logger.info(f"  {name}: {provider.__class__.__name__} with {len(provider.conditions)} conditions")

        # Test stats
        stats = manager.get_stats()
        logger.info(f"Manager stats: {stats.providers}")

        logger.info("Task manager test completed!")

    except Exception as e:
        logger.error(f"Task manager test failed: {e}")
        traceback.print_exc()
