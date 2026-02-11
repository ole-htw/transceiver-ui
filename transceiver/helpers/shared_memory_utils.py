"""Shared-memory helper utilities used across UI and worker modules."""

from __future__ import annotations

import contextlib
from multiprocessing import shared_memory


SHM_REGISTRY: set[str] = set()


def create_shared_memory(size: int, *, registry: set[str] | None = None) -> shared_memory.SharedMemory:
    """Create shared memory and register it for later cleanup."""
    target_registry = SHM_REGISTRY if registry is None else registry
    try:
        shm = shared_memory.SharedMemory(create=True, size=size, track=False)  # type: ignore[call-arg]
    except TypeError:
        shm = shared_memory.SharedMemory(create=True, size=size)
    target_registry.add(shm.name)
    return shm


def cleanup_shared_memory(*, registry: set[str] | None = None) -> None:
    """Unlink all registered shared-memory segments."""
    target_registry = SHM_REGISTRY if registry is None else registry
    if not target_registry:
        return
    for shm_name in sorted(target_registry):
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
        except (FileNotFoundError, OSError, ValueError):
            continue
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        finally:
            with contextlib.suppress(Exception):
                shm.close()
    target_registry.clear()
