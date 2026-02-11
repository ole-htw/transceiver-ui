"""Shared-memory helper utilities with explicit ownership tracking."""

from __future__ import annotations

import contextlib
from multiprocessing import shared_memory


class SharedMemoryRegistry:
    """Track owned shared-memory segments and unlink them on cleanup."""

    def __init__(self) -> None:
        self._registry: set[str] = set()

    def create(self, size: int) -> shared_memory.SharedMemory:
        """Create shared memory with tracking disabled when supported."""
        try:
            shm = shared_memory.SharedMemory(create=True, size=size, track=False)  # type: ignore[call-arg]
        except TypeError:
            shm = shared_memory.SharedMemory(create=True, size=size)
        self._registry.add(shm.name)
        return shm

    def cleanup(self) -> None:
        """Unlink any shared-memory segments created by this registry."""
        if not self._registry:
            return
        for shm_name in sorted(self._registry):
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
        self._registry.clear()


def open_shared_array(*, shm_name: str, shape: tuple[int, ...], dtype: str):
    """Open a shared memory segment and present it as a NumPy array view."""
    import numpy as np

    shm = shared_memory.SharedMemory(name=shm_name)
    array = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
    return shm, array
