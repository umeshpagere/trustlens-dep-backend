"""
TrustLens Job Registry

Tracks in-flight asyncio Tasks by a caller-supplied job ID so that the
/api/cancel endpoint can cancel a running analysis on demand.

Thread-safety note:
  All operations access a plain dict. Flask/Hypercorn's asyncio event loop
  runs on a single thread, so no locking is needed.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

# job_id (str) -> asyncio.Task
_jobs: dict[str, asyncio.Task] = {}


def register(job_id: str, task: asyncio.Task) -> None:
    """Register an asyncio Task under *job_id*."""
    _jobs[job_id] = task
    logger.debug("Job registered: %s", job_id)


def cancel(job_id: str) -> bool:
    """
    Cancel the task registered under *job_id*.

    Returns True if a running task was found and cancelled,
    False if no such job exists or the task already finished.
    """
    task = _jobs.get(job_id)
    if task and not task.done():
        task.cancel()
        logger.info("Job cancelled: %s", job_id)
        return True
    logger.debug("Cancel request for unknown/finished job: %s", job_id)
    return False


def unregister(job_id: str) -> None:
    """Remove the job from the registry (called in the route's finally block)."""
    _jobs.pop(job_id, None)
    logger.debug("Job unregistered: %s", job_id)
