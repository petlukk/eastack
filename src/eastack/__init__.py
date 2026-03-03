"""eastack — SIMD-accelerated frame stacking powered by Ea kernels."""

__version__ = "0.1.0"

from eastack._stack import stack, stack_mean
from eastack._bindings import (
    accumulate_f32x8,
    accumulate_batch4_f32x8,
    accumulate_batch8_f32x8,
    scale_f32x8,
    frame_stats,
)

__all__ = [
    "stack",
    "stack_mean",
    "accumulate_f32x8",
    "accumulate_batch4_f32x8",
    "accumulate_batch8_f32x8",
    "scale_f32x8",
    "frame_stats",
]
