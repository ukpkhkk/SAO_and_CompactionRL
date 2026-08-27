import asyncio
import atexit
import queue
import threading
import time
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any
from slime.utils.data import Dataset
import copy
import inspect
import logging
import uuid

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Import core functions from sglang_rollout directly to avoid code duplication
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group, generate
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.async_utils import run
from slime.utils.types import Sample
from slime.rollout.base_types import RolloutFnEvalOutput
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from slime.rollout.rm_hub import async_rm, batched_async_rm

# Global worker manager
_global_worker = None
_worker_lock = threading.Lock()

# Global WindowedFIFOBuffer instance (shared across calls within the same process)
_global_fifo_buffer = None


def get_global_worker(args, data_buffer, current_rollout_id: int | None = None):
    """Get or create global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=args.sglang_server_concurrency)
            if current_rollout_id is not None:
                _global_worker.current_rollout_id = current_rollout_id
            _global_worker.start()
        else:
            _global_worker.data_buffer = data_buffer
            if current_rollout_id is not None:
                _global_worker.current_rollout_id = current_rollout_id
        return _global_worker


def stop_global_worker():
    """Stop global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


async def generate_and_rm(
        args: Namespace,
        sample: Sample | list[Sample],
        sampling_params: dict[str, Any],
        evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.get_semaphore():
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            else:
                sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

    return sample


class WindowedFIFOBuffer:
    """
    FIFO buffer with windowed delay control for fully-async rollout.

    ``entry_step`` is a **persistent property** of each sample, set once
    when the sample first enters the buffer and never changed afterward —
    similar to ``sample.completed_step``.

    It is determined by the sample's ``group_index`` and
    ``rollout_batch_size``:  ``entry_step = group_index // rollout_batch_size``.

    For example, if ``rollout_batch_size=32``:
    - group_index 0–31  → entry_step = 0
    - group_index 32–63 → entry_step = 1
    - group_index 64–95 → entry_step = 2

    This is **deterministic**: it depends only on the sample's position
    in the initial dataset (group_index), NOT on when the sample is
    fetched from the buffer or how many times it has been re-queued.

    If a sample is partially completed, aborted, and put back into the
    buffer, its ``entry_step`` is **preserved** — it is never
    recalculated on re-entry.

    **FIFO ordering guarantee**: The buffer is a strict FIFO queue.
    Normal data is appended to the tail (tail-insert), so later-arriving
    data enters inference later.  Aborted data is inserted at the head
    (head-insert) because its entry_step is earlier, ensuring it gets
    re-dispatched to inference before newer data.

    There is **no entry_step-based filtering** when pulling data from the
    buffer — all data is dispatched to inference in FIFO order regardless
    of its entry_step relative to the current training step.  The
    windowed FIFO constraint (max_delay_step) is enforced at the
    *consumption* side in ``generate_rollout_async``, not at the
    *dispatch* side.

    When ``current_step - entry_step >= max_delay_step``, the group is
    considered "expired" and must be included in the current training batch.

    Expired-priority strategy (two cases based on expired count m vs batch size n):

    - Case 1 (m <= n): Wait for ALL m expired groups to complete, include them
      in training, then fill remaining n-m slots with non-expired groups.
      If total available exceeds n during the wait, extras are left for the
      next training step.
    - Case 2 (m > n): Stop launching new inference tasks, wait for n expired
      groups to complete, then start training. Remaining m-n expired groups
      are left for the next round.
    """

    def __init__(self, data_buffer, max_delay_step: int, rollout_batch_size: int):
        self._data_buffer = data_buffer
        self.max_delay_step = max_delay_step
        self.rollout_batch_size = rollout_batch_size
        self.current_step = 0
        self._pending: list[tuple[list[Sample], int]] = []

    def _compute_entry_step_for_group(self, group: list[Sample]) -> int:
        """Compute entry_step based on group_index and rollout_batch_size.

        entry_step = group_index // rollout_batch_size

        This is deterministic: it depends only on the sample's position in
        the initial dataset, NOT on when the sample is fetched from the
        buffer or how many times it has been re-queued after abort.

        If any sample in the group already has an entry_step (re-entered
        after abort), that value is used for the whole group.

        If group_index is None (shouldn't happen in normal flow), falls
        back to 0.
        """
        existing = next((s.entry_step for s in group if s.entry_step is not None), None)
        if existing is not None:
            return existing
        group_index = next((s.group_index for s in group if s.group_index is not None), None)
        if group_index is not None:
            es = group_index // self.rollout_batch_size
        else:
            es = 0
        for s in group:
            s.entry_step = es

        return es

    def add_samples(self, groups: list[list[Sample]], entry_step: int | None = None):
        """Add sample groups to the tail of the FIFO buffer (tail-insert).

        Normal data is always appended to the tail, maintaining FIFO order.
        If ``entry_step`` is provided (e.g. putting back a non-aborted group),
        the group retains its original entry_step.  Otherwise, the entry_step
        is read from the samples themselves (if already set) or computed from
        group_index and rollout_batch_size.
        """
        for group in groups:
            if entry_step is not None:
                for s in group:
                    s.entry_step = entry_step
                self._pending.append((group, entry_step))
            else:
                es = self._compute_entry_step_for_group(group)
                self._pending.append((group, es))

    def get_samples(self, num_samples: int) -> list[tuple[list[Sample], int]]:
        """Pop up to num_samples groups from the head of the FIFO queue.

        No entry_step-based filtering is applied — data is dispatched to
        inference in strict FIFO order.  Falls through to the underlying
        data_buffer when the FIFO pending list is empty.

        entry_step is computed from group_index and rollout_batch_size.
        If the sample already has an entry_step (re-entered after abort),
        it is preserved and not recalculated.
        """
        result = []
        while len(result) < num_samples:
            if self._pending:
                group, es = self._pending.pop(0)
                result.append((group, es))
            else:
                raw_groups = self._data_buffer.get_samples(1)
                if not raw_groups:
                    break
                for group in raw_groups:
                    es = self._compute_entry_step_for_group(group)
                    result.append((group, es))
                if not result:
                    break
        return result

    def put_back(self, group: list[Sample], entry_step: int):
        """Put a group back to the head of the FIFO buffer (head-insert).

        Used for aborted groups — their entry_step is earlier, so they
        should be re-dispatched to inference before newer data.  This
        ensures the inference engine consumes data in a reasonable order.

        The entry_step is preserved on the sample objects so it will never
        be recalculated on re-entry.
        """
        for s in group:
            s.entry_step = entry_step
        self._pending.insert(0, (group, entry_step))

    def get_expired_groups(self) -> list[tuple[list[Sample], int]]:
        """Return all groups whose staleness (current_step - entry_step) >= max_delay_step."""
        expired = []
        remaining = []
        for group, entry_step in self._pending:
            if self.current_step - entry_step >= self.max_delay_step:
                expired.append((group, entry_step))
            else:
                remaining.append((group, entry_step))
        self._pending = remaining
        return expired

    def advance_step(self):
        """Increment the training step counter. Called after each rollout round completes."""
        self.current_step += 1

    def __len__(self):
        return len(self._pending)

    @property
    def data_buffer(self):
        return self._data_buffer


class AsyncRolloutWorker:
    """
    Asynchronous rollout worker with windowed FIFO support.

    Architecture overview:
    - Runs in a dedicated daemon thread with its own asyncio event loop.
    - Continuously pulls sample groups from the data source (DataBuffer or
      WindowedFIFOBuffer), launches inference tasks via SGLang, and pushes
      completed results to ``output_queue``.
    - Each completed result is tagged with its ``entry_step`` so that the
      rollout function can enforce the maximum delay window.

    Key data structures:
    - ``output_queue``: Thread-safe bounded queue (maxsize=1000) where
      completed inference results are placed by the worker loop. Items are
      ``(entry_step, group)`` tuples.
    - ``_pending_completed``: Persistent dict that maps ``entry_step`` →
      list of ``(entry_step, group)`` tuples. Populated by
      ``drain_output_queue()`` and consumed by ``generate_rollout_async()``.
      Unlike a local variable, this dict survives across training steps,
      ensuring that unconsumed data is not lost between steps.
    - ``_inflight_registry``: Thread-safe dict mapping ``task_local_id`` →
      ``entry_step`` for all currently in-flight inference tasks. Used by
      ``generate_rollout_async()`` to determine which entry_steps are
      still being processed and whether they are expired.
    - ``_max_entry_step``: Monotonically increasing counter tracking the
      highest ``entry_step`` of any group pulled from the buffer. Used in
      conjunction with ``windowed_fifo_max_prefetch_steps`` to limit how
      far ahead the worker prefetches data relative to the current
      training step.

    Prefetch throttling:
    When using WindowedFIFOBuffer, the worker limits prefetching via
    ``--windowed-fifo-max-prefetch-steps`` (default 8). If
    ``_max_entry_step - current_rollout_id > max_prefetch_steps``, the
    worker stops pulling new data from the buffer. This prevents the
    worker from pulling too far ahead of training, which would cause
    excessive staleness (large negative ``entry_delay_step``).
    """

    def __init__(self, args, data_buffer, concurrency=10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        # Bounded output queue for completed inference results.
        # Items are (entry_step, list[Sample]) tuples.
        self.output_queue = queue.Queue(maxsize=1000)
        self.worker_thread = None
        self.state = None
        self._loop_ready = threading.Event()
        # Tracks the current training step (rollout_id), set by
        # generate_rollout_fully_async before each rollout round.
        self.current_rollout_id = None
        self.pause_new_tasks = False
        # Thread-safe registry: {task_local_id: entry_step} for in-flight tasks.
        self._inflight_registry: dict[int, int] = {}
        self._inflight_lock = threading.Lock()
        self._task_counter = 0
        # Highest entry_step pulled from buffer so far (monotonically increasing).
        # Used for prefetch throttling: stops pulling when
        # _max_entry_step - current_rollout_id > max_prefetch_steps.
        self._max_entry_step = 0
        # Persistent cache for completed inference results, keyed by entry_step.
        # Populated by drain_output_queue(), consumed by generate_rollout_async().
        # Survives across training steps — unconsumed data is NOT discarded.
        self._pending_completed: dict[int, list[tuple[int, list[Sample]]]] = {}

    def _register_inflight(self, entry_step: int) -> int:
        """Register an in-flight task, return its local task id."""
        with self._inflight_lock:
            tid = self._task_counter
            self._task_counter += 1
            self._inflight_registry[tid] = entry_step
            return tid

    def _unregister_inflight(self, tid: int):
        """Unregister an in-flight task when it completes."""
        with self._inflight_lock:
            self._inflight_registry.pop(tid, None)

    def get_inflight_entry_steps(self) -> dict[int, int]:
        """Return a snapshot of {task_local_id: entry_step} for all in-flight tasks."""
        with self._inflight_lock:
            return dict(self._inflight_registry)

    async def continuous_worker_loop(self):
        """Continuous work loop — pull data from buffer and launch inference tasks.

        This loop runs indefinitely in the worker's daemon thread. Each iteration:
        1. Cleans up completed tasks (unregisters from inflight registry,
           assigns completed_step, pushes results to output_queue).
        2. Launches new inference tasks if under the concurrency limit
           (rollout_batch_size). In WindowedFIFOBuffer mode, prefetch
           throttling limits how far ahead data is pulled.
        3. Sleeps briefly (1s) before the next iteration.

        Prefetch throttling (WindowedFIFOBuffer only):
        When _max_entry_step - current_rollout_id > max_prefetch_steps,
        the worker stops pulling new data. This prevents excessive
        staleness by ensuring the worker does not prefetch data that is
        too many steps ahead of the current training progress.
        """
        self.state = GenerateState(self.args)
        self.loop = asyncio.get_running_loop()
        self._loop_ready.set()

        use_windowed_fifo = isinstance(self.data_buffer, WindowedFIFOBuffer)
        print(f"Continuous async rollout worker started (windowed_fifo={'on' if use_windowed_fifo else 'off'})")

        # Tracks currently running asyncio inference tasks.
        # Key: asyncio.Task, Value: (local_tid, entry_step)
        active_tasks: dict[asyncio.Task, tuple[int, int]] = {}

        while self.running:
            try:
                # Maximum number of concurrent inference tasks.
                # In Case 2 (m > n expired groups), this is temporarily set to 0
                # to pause new inference launches.
                max_concurrent_tasks = self.args.rollout_batch_size

                # --- Phase 1: Clean up completed tasks ---
                # For each done task: unregister from inflight registry,
                # assign completed_step using current_rollout_id at completion
                # time, and push the result group to output_queue.
                done_tasks = {task for task in active_tasks if task.done()}
                for task in done_tasks:
                    local_tid, entry_step = active_tasks.pop(task)
                    self._unregister_inflight(local_tid)
                    try:
                        result = task.result()
                    except Exception as e:
                        print(f"Task failed with exception: {e}")
                        continue
                    # completed_step records the training step at which the
                    # inference task completed. This is used to compute
                    # off_step = used_step - completed_step (how many steps
                    # the data was "off-policy").
                    if self.current_rollout_id is not None:
                        for sample in result:
                            if sample.status != Sample.Status.ABORTED and sample.completed_step is None:
                                sample.completed_step = self.current_rollout_id
                    self.output_queue.put((entry_step, result))

                # --- Phase 2: Launch new inference tasks ---
                # Pull data from the buffer and create asyncio tasks for
                # inference. In WindowedFIFOBuffer mode, prefetch throttling
                # limits how far ahead we pull data.
                while len(active_tasks) < max_concurrent_tasks and self.running and not self.pause_new_tasks:
                    if isinstance(self.data_buffer, WindowedFIFOBuffer):
                        # Prefetch throttling: stop pulling if the worker has
                        # already pulled data that is more than max_prefetch_steps
                        # ahead of the current training step. This prevents
                        # excessive staleness in the training data.
                        max_prefetch_steps = getattr(self.args, "windowed_fifo_max_prefetch_steps", 8)
                        current_rollout_id = self.current_rollout_id if self.current_rollout_id is not None else 0
                        if self._max_entry_step - current_rollout_id >= max_prefetch_steps:
                            print(f"=======>max_entry_step({self._max_entry_step}) - current_rollout_id({self.current_rollout_id}) > max_prefetch_steps({max_prefetch_steps}), do not add new data!!! ")
                            break
                        entries = self.data_buffer.get_samples(1)
                        if not entries:
                            break
                        group, entry_step = entries[0]
                        # Update _max_entry_step to track the highest entry_step
                        # pulled so far. Since FIFO ordering guarantees
                        # monotonically increasing entry_step, this value only
                        # increases.
                        self._max_entry_step = entry_step
                    else:
                        # Non-windowed-FIFO mode: pull directly from DataBuffer.
                        # entry_step is always 0 (no staleness tracking).
                        samples = self.data_buffer.get_samples(1)
                        if not samples:
                            break
                        group = samples[0]
                        entry_step = 0

                    local_tid = self._register_inflight(entry_step)

                    task = asyncio.create_task(
                        generate_and_rm_group(
                            self.args,
                            group,
                            sampling_params=self.state.sampling_params.copy(),
                            evaluation=False,
                        )
                    )
                    active_tasks[task] = (local_tid, entry_step)

                await asyncio.sleep(1)

            except Exception as e:
                print(f"Error in continuous worker loop: {e}")
                await asyncio.sleep(1)

        if active_tasks:
            print(f"Waiting for {len(active_tasks)} continuous tasks to complete...")
            await asyncio.wait(active_tasks.keys())

        print("Continuous async rollout worker stopped")

    def worker_thread_func(self):
        """Worker function running in independent thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.continuous_worker_loop())
        finally:
            loop.close()

    def start(self):
        """Start continuous work mode"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            self._loop_ready.wait()
            print("Started continuous async worker thread")

    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            self.worker_thread.join(timeout=5)
        print("Stopped async worker thread")

    def run_coro(self, coro):
        """
        Run a coroutine in the worker's event loop and wait for result.
        This allows evaluation and training to share the same loop context.
        """
        if not self.loop or not self.loop.is_running():
            raise RuntimeError("Worker loop is not running")

        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result()
        except Exception as e:
            raise e

    def get_completed_groups(self) -> list[tuple]:
        """Get completed sample groups from the output queue.
        DEPRECATED for windowed FIFO mode — use drain_output_queue() instead.
        """
        completed = []
        while True:
            try:
                result = self.output_queue.get_nowait()
                completed.append(result)
            except queue.Empty:
                break
        return completed

    def drain_output_queue(self) -> bool:
        """Drain output_queue into the persistent _pending_completed cache.

        ABORTED groups are put back to the data_buffer via head-insert
        (put_back) for regeneration instead of entering the training pipeline.
        Only fully-completed groups (no ABORTED samples) are added to
        _pending_completed.

        Returns True if any new data was drained (including ABORTED groups
        that were redirected to data_buffer).
        """
        drained = False
        while True:
            try:
                entry_step, group = self.output_queue.get_nowait()
                try:
                    is_aborted = any(s.status == Sample.Status.ABORTED for s in group)
                except Exception:
                    is_aborted = False

                if is_aborted:
                    if isinstance(self.data_buffer, WindowedFIFOBuffer):
                        self.data_buffer.put_back(group, entry_step)
                        print(
                            f"[drain_output_queue] ABORTED group (entry_step={entry_step}) "
                            f"put back to FIFO buffer for regeneration",
                            flush=True,
                        )
                    else:
                        try:
                            self.data_buffer.add_samples([group])
                            print(
                                f"[drain_output_queue] ABORTED group (entry_step={entry_step}) "
                                f"returned to data buffer",
                                flush=True,
                            )
                        except Exception as e:
                            print(f"Failed to return aborted group to buffer: {e}", flush=True)
                else:
                    if entry_step not in self._pending_completed:
                        self._pending_completed[entry_step] = []
                    self._pending_completed[entry_step].append((entry_step, group))
                drained = True
            except queue.Empty:
                break
        return drained

    def get_queue_size(self) -> int:
        """Get current output queue size"""
        return self.output_queue.qsize()


def _ensure_fifo_buffer(args, data_buffer):
    """
    Create or return the global WindowedFIFOBuffer wrapping data_buffer.

    If --windowed-fifo-max-delay-step is set (>=1), wraps the original
    data_buffer in a WindowedFIFOBuffer. The wrapper is cached globally so
    that it persists across rollout calls.

    entry_step is a persistent property on each Sample object, set once
    when the sample first enters the buffer and never changed afterward.
    It is computed from group_index and rollout_batch_size:
    entry_step = group_index // rollout_batch_size.
    For example, with rollout_batch_size=32,
    group_index 0-31 get entry_step=0, group_index 32-63 get entry_step=1, etc.
    If a sample is aborted and put back, its entry_step is preserved.
    """
    global _global_fifo_buffer
    use_windowed_fifo = (
            getattr(args, "windowed_fifo_max_delay_step", None) is not None
            and args.windowed_fifo_max_delay_step >= 1
    )
    if not use_windowed_fifo:
        return data_buffer, False

    max_delay_step = args.windowed_fifo_max_delay_step
    rollout_batch_size = args.rollout_batch_size

    if _global_fifo_buffer is None:
        fifo_buffer = WindowedFIFOBuffer(
            data_buffer, max_delay_step=max_delay_step,
            rollout_batch_size=rollout_batch_size,
        )
        _global_fifo_buffer = fifo_buffer
        print(
            f"WindowedFIFOBuffer created with max_delay_step={max_delay_step}, "
            f"rollout_batch_size={rollout_batch_size}"
        )
    else:
        _global_fifo_buffer._data_buffer = data_buffer

    return _global_fifo_buffer, True


async def generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    """
    Asynchronous rollout generation with windowed FIFO support.

    When ``--windowed-fifo-max-delay-step`` is set (>= 1), samples that have
    been in the buffer for more than that many training steps are considered
    "expired" and must be included in the current training batch.

    **FIFO ordering**: Data is dispatched to inference in FIFO order by the
    buffer (tail-insert for normal data, head-insert for aborted data).
    Completed groups are consumed in ascending entry_step order, ensuring
    earlier data is trained first.  There is **no entry_step-based filtering**
    — all completed groups are eligible for consumption regardless of their
    entry_step relative to the current training step.

    Expired-priority strategy (two cases, m = expired count, n = batch size):

    Case 1 (m <= n):
      - Wait for ALL m expired groups (completed + in-flight) to finish.
      - Must include all m expired groups in the current training step.
      - Fill remaining n-m slots with non-expired completed groups (in
        ascending entry_step order).
      - If more than n groups become available during the wait, extras are
        left in completed_groups for the next training step.

    Case 2 (m > n):
      - Stop launching new inference (pause the worker from adding new tasks).
      - Wait for n expired groups to complete, then start training.
      - Remaining m-n expired groups are left for the next round.
      - Next round re-evaluates case 1 or case 2 with the remaining expired count.
    """
    assert args.rollout_global_dataset

    use_windowed_fifo = isinstance(data_buffer, WindowedFIFOBuffer)
    max_delay_step = getattr(args, "windowed_fifo_max_delay_step", 0) if use_windowed_fifo else 0

    worker = get_global_worker(args, data_buffer)

    target_data_size = args.rollout_batch_size

    data: list[list[Sample]] = []
    data_entry_steps: list[int] = []
    if use_windowed_fifo:
        # In windowed FIFO mode, completed_groups is the worker's persistent
        # cache (_pending_completed), which survives across training steps.
        # Unconsumed data from previous steps is retained and available here.
        completed_groups = worker._pending_completed
    else:
        # In non-windowed mode, completed_groups is a local dict that is
        # re-initialized each call. Unconsumed data is lost between steps.
        completed_groups: dict[int, list[tuple[int, list[Sample]]]] = {}
    do_print = True

    print(
        f"Starting async rollout generation for {target_data_size} groups "
        f"(windowed_fifo={'on, max_delay=' + str(max_delay_step) if use_windowed_fifo else 'off'})"
    )
    print(f"Global worker queue size: {worker.get_queue_size()}")

    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0

    def _accept_group(group, entry_step):
        """Add a completed group to the training data, handling aborted samples.

        Aborted groups are put back to the FIFO buffer via head-insert
        (put_back) so they are re-dispatched to inference before newer data.
        """
        nonlocal do_print
        try:
            any_aborted = any(s.status == Sample.Status.ABORTED for s in group)
        except Exception:
            any_aborted = False

        if any_aborted:
            try:
                data_buffer.put_back(group, entry_step)
                print(f"Head-inserted aborted group (entry_step={entry_step}) to FIFO buffer", flush=True)
            except Exception as e:
                print(f"Failed to return aborted group to buffer: {e}", flush=True)
            return False

        if do_print:
            print(
                f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                f"label: {group[0].label}, reward: {group[0].reward}",
                flush=True,
            )
            do_print = False

        for sample in group:
            sample.used_step = rollout_id
            assert sample.completed_step is not None, (
                "[Warning] In fully async mode, sample.completed_step should not be None!!!"
            )
            sample.off_step = sample.used_step - sample.completed_step
            if sample.entry_step is not None:
                sample.entry_delay_step = sample.used_step - sample.entry_step
            print(
                f"====> calc off steps, sample.completed_step:{sample.completed_step}, "
                f"sample.used_step:{sample.used_step}, sample.off_step:{sample.off_step}, "
                f"sample.entry_step:{sample.entry_step}, sample.entry_delay_step:{sample.entry_delay_step}"
            )

        data.append(group)
        data_entry_steps.append(entry_step)
        return True

    while len(data) < target_data_size:
        # --- Drain phase: Move completed inference results from output_queue
        # to completed_groups (or _pending_completed in windowed FIFO mode).
        # In windowed FIFO mode, drain_output_queue() also handles ABORTED
        # groups by putting them back to the FIFO buffer for regeneration.
        if use_windowed_fifo:
            made_progress = worker.drain_output_queue()
        else:
            # Non-windowed mode: manually transfer results from output_queue
            # to the local completed_groups dict.
            completed = worker.get_completed_groups()
            made_progress = False
            for entry_step, group in completed:
                if entry_step not in completed_groups:
                    completed_groups[entry_step] = []
                completed_groups[entry_step].append((entry_step, group))
                made_progress = True

        if made_progress:
            last_progress_time = time.time()

        processed_any = False

        if use_windowed_fifo:
            # --- Windowed FIFO expired-priority strategy ---
            # Classify completed and in-flight groups as "expired" (staleness
            # >= max_delay_step) or "fresh". Expired groups must be
            # force-included in the current batch to prevent starvation.
            current_step = data_buffer.current_step

            all_entry_steps = sorted(completed_groups.keys())

            # Identify completed groups whose entry_step is expired
            # (current_step - entry_step >= max_delay_step).
            expired_completed_entry_steps = sorted(
                [es for es in all_entry_steps if current_step - es >= max_delay_step]
            )
            expired_completed_count = sum(
                len(completed_groups[es]) for es in expired_completed_entry_steps
            )

            inflight_snapshot = worker.get_inflight_entry_steps()
            expired_in_flight_tasks = [
                tid for tid, es in inflight_snapshot.items()
                if current_step - es >= max_delay_step
            ]
            expired_in_flight_count = len(expired_in_flight_tasks)
            expired_in_flight_entry_steps = set(
                es for es in inflight_snapshot.values()
                if current_step - es >= max_delay_step
            )

            total_expired = expired_completed_count + expired_in_flight_count

            if total_expired == 0:
                # No expired groups — fill with completed groups in FIFO order
                # (ascending entry_step, no entry_step-based filtering)
                for es in all_entry_steps:
                    if len(data) >= target_data_size:
                        break
                    groups_at_step = completed_groups.pop(es)
                    remaining = []
                    for _, group in groups_at_step:
                        if len(data) >= target_data_size:
                            remaining.append((es, group))
                            continue
                        if _accept_group(group, es):
                            processed_any = True
                    if remaining:
                        completed_groups[es] = remaining

            elif total_expired <= target_data_size:
                # Case 1: m <= n
                # Must wait for ALL m expired groups to complete, then include them.
                # Fill remaining slots with non-expired in FIFO order. Extras left for next round.

                # First, consume all expired completed groups
                for es in expired_completed_entry_steps:
                    if len(data) >= target_data_size:
                        break
                    groups_at_step = completed_groups.pop(es)
                    remaining = []
                    for _, group in groups_at_step:
                        if len(data) >= target_data_size:
                            remaining.append((es, group))
                            continue
                        if _accept_group(group, es):
                            processed_any = True
                    if remaining:
                        completed_groups[es] = remaining

                # If there are expired in-flight groups, wait for them
                if expired_in_flight_count > 0 and len(data) < target_data_size:
                    print(
                        f"[Case 1: m={total_expired} <= n={target_data_size}] "
                        f"Waiting for {expired_in_flight_count} expired in-flight tasks "
                        f"to complete (entry_steps: {sorted(expired_in_flight_entry_steps)})...",
                        flush=True,
                    )
                    await asyncio.sleep(0.5)
                    continue

                # All expired groups are now consumed. Fill remaining with non-expired in FIFO order.
                non_expired_entry_steps = [
                    es for es in all_entry_steps
                    if es in completed_groups and current_step - es < max_delay_step
                ]
                for es in non_expired_entry_steps:
                    if len(data) >= target_data_size:
                        break
                    groups_at_step = completed_groups.pop(es)
                    remaining = []
                    for _, group in groups_at_step:
                        if len(data) >= target_data_size:
                            remaining.append((es, group))
                            continue
                        if _accept_group(group, es):
                            processed_any = True
                    if remaining:
                        completed_groups[es] = remaining

            else:
                # Case 2: m > n
                # Stop launching new inference. Wait for n expired groups to complete.
                # Remaining m-n expired groups left for next round.

                worker.pause_new_tasks = True

                print(
                    f"[Case 2: m={total_expired} > n={target_data_size}] "
                    f"Pausing new inference. Waiting for {target_data_size} expired groups "
                    f"to complete (expired_in_flight={expired_in_flight_count}, "
                    f"expired_completed={expired_completed_count})...",
                    flush=True,
                )

                # Consume expired completed groups first
                for es in expired_completed_entry_steps:
                    if len(data) >= target_data_size:
                        break
                    groups_at_step = completed_groups.pop(es)
                    remaining = []
                    for _, group in groups_at_step:
                        if len(data) >= target_data_size:
                            remaining.append((es, group))
                            continue
                        if _accept_group(group, es):
                            processed_any = True
                    if remaining:
                        completed_groups[es] = remaining

                # If we still need more and there are in-flight expired, wait
                if len(data) < target_data_size and expired_in_flight_entry_steps:
                    await asyncio.sleep(0.5)
                    continue

                # We have n groups — restore inference for next round
                if worker.pause_new_tasks:
                    worker.pause_new_tasks = False
                    print(
                        f"[Case 2] Collected {len(data)} groups, restoring inference "
                        f"(rollout_batch_size={worker.args.rollout_batch_size}).",
                        flush=True,
                    )

        else:
            # Original non-windowed-FIFO logic
            available_ids = list(completed_groups.keys())
            for group_id in available_ids:
                if len(data) >= target_data_size:
                    break

                groups_at_id = completed_groups.pop(group_id)
                for _, group in groups_at_id:
                    if len(data) >= target_data_size:
                        if group_id not in completed_groups:
                            completed_groups[group_id] = []
                        completed_groups[group_id].append((group_id, group))
                        break

                    try:
                        any_aborted = any(s.status == Sample.Status.ABORTED for s in group)
                    except Exception:
                        any_aborted = False

                    if any_aborted:
                        try:
                            data_buffer.add_samples([group])
                            print(f"Returned aborted group {group_id} to data buffer", flush=True)
                        except Exception as e:
                            print(f"Failed to return aborted group {group_id} to buffer: {e}", flush=True)
                        continue

                    if do_print:
                        print(
                            f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                            f"label: {group[0].label}, reward: {group[0].reward}",
                            flush=True,
                        )
                        do_print = False

                    for sample in group:
                        sample.used_step = rollout_id
                        assert sample.completed_step is not None, (
                            "[Warning] In fully async mode, sample.completed_step should not be None!!!"
                        )
                        sample.off_step = sample.used_step - sample.completed_step
                        if sample.entry_step is not None:
                            sample.entry_delay_step = sample.used_step - sample.entry_step
                        print(
                            f"====> calc off steps, sample.completed_step:{sample.completed_step}, "
                            f"sample.used_step:{sample.used_step}, sample.off_step:{sample.off_step}, "
                            f"sample.entry_step:{sample.entry_step}, sample.entry_delay_step:{sample.entry_delay_step}"
                        )

                    data.append(group)
                    processed_any = True

        # Check progress
        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            print(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Collected: {len(data)}/{target_data_size}"
            )
            last_progress_time = current_time

        if not processed_any:
            await asyncio.sleep(0.01)

    # Advance FIFO step counter and re-queue expired pending groups via
    # head-insert. Expired pending groups are those in the FIFO buffer
    # whose entry_step is too old (staleness >= max_delay_step). They
    # are put back at the head so they get re-dispatched to inference
    # before newer data.
    if use_windowed_fifo:
        data_buffer.advance_step()
        expired_pending = data_buffer.get_expired_groups()
        for group, entry_step in expired_pending:
            data_buffer.put_back(group, entry_step)

    duration = time.time() - start_time
    print(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")

    if data:
        print(
            f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
            flush=True,
        )

    data = sorted(data, key=lambda group: group[0].index)
    return data


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


EVAL_PROMPT_DATASET = {}


async def eval_rollout_single_dataset(
        args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(sample.prompt) + sample.response]} "
                f"reward={sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    """
    Entry point for fully-async rollout generation.

    This function is called once per training step by the training loop.
    It sets up the WindowedFIFOBuffer (if configured), obtains the global
    AsyncRolloutWorker, and runs the async rollout collection coroutine
    in the worker's event loop.

    The worker runs continuously in a background daemon thread, pulling
    data from the buffer and launching inference tasks independently.
    Prefetch throttling (--windowed-fifo-max-prefetch-steps, default 8)
    limits how far ahead the worker pulls data relative to the current
    training step, preventing excessive staleness.

    When --windowed-fifo-max-delay-step is set, wraps the data_buffer in
    a WindowedFIFOBuffer that enforces the maximum delay window.
    """
    data_buffer, use_windowed_fifo = _ensure_fifo_buffer(args, data_buffer)
    if use_windowed_fifo:
        data_buffer.current_step = rollout_id

    worker = get_global_worker(args, data_buffer, current_rollout_id=rollout_id)

    if evaluation:
        print(f'worker:{worker}, rollout_id:{rollout_id}, data_buffer:{data_buffer}')
        output, _ = worker.run_coro(eval_rollout(args, rollout_id))
        return output
    else:
        completed_samples = worker.run_coro(generate_rollout_async(args, rollout_id, data_buffer))
        return completed_samples


# Register exit cleanup function

atexit.register(stop_global_worker)
