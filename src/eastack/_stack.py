"""High-level stacking API."""
import numpy as np

from . import _bindings as _b


def stack(frames):
    """Accumulate frames using batched SIMD kernels.

    Args:
        frames: list of numpy float32 arrays (same shape), or 3D array (N, H, W).

    Returns:
        Accumulated sum as flat float32 array. Divide by len(frames) for mean,
        or use stack_mean() instead.
    """
    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        frames = [frames[i] for i in range(frames.shape[0])]

    n = len(frames)
    if n == 0:
        raise ValueError("no frames to stack")

    flat = [np.ascontiguousarray(f, dtype=np.float32).ravel() for f in frames]
    size = flat[0].size
    acc = np.zeros(size, dtype=np.float32)

    i = 0
    while i + 8 <= n:
        _b.accumulate_batch8_f32x8(
            acc, flat[i], flat[i+1], flat[i+2], flat[i+3],
            flat[i+4], flat[i+5], flat[i+6], flat[i+7],
        )
        i += 8

    while i + 4 <= n:
        _b.accumulate_batch4_f32x8(acc, flat[i], flat[i+1], flat[i+2], flat[i+3])
        i += 4

    while i < n:
        _b.accumulate_f32x8(acc, flat[i])
        i += 1

    return acc.reshape(frames[0].shape)


def stack_mean(frames):
    """Stack frames and compute mean.

    Args:
        frames: list of numpy float32 arrays (same shape), or 3D array (N, H, W).

    Returns:
        Mean of stacked frames as float32 array (same shape as input frames).
    """
    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        n = frames.shape[0]
    else:
        n = len(frames)

    acc = stack(frames)
    out = np.empty_like(acc)
    _b.scale_f32x8(acc.ravel(), out.ravel(), 1.0 / n)
    return out.reshape(acc.shape)
