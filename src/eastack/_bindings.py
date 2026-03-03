"""Low-level ctypes bindings for stack kernels."""
import ctypes as _ct

import numpy as np

from ._loader import load_library

_lib = load_library("stack")

# accumulate_f32x8(acc, frame, len)
_lib.accumulate_f32x8.argtypes = [
    _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float), _ct.c_int32,
]
_lib.accumulate_f32x8.restype = None

# accumulate_batch4_f32x8(acc, f0, f1, f2, f3, len)
_lib.accumulate_batch4_f32x8.argtypes = [
    _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float),
    _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float),
    _ct.POINTER(_ct.c_float), _ct.c_int32,
]
_lib.accumulate_batch4_f32x8.restype = None

# accumulate_batch8_f32x8(acc, f0..f7, len)
_lib.accumulate_batch8_f32x8.argtypes = [
    _ct.POINTER(_ct.c_float),
    _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float),
    _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float),
    _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float),
    _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float),
    _ct.c_int32,
]
_lib.accumulate_batch8_f32x8.restype = None

# scale_f32x8(data, out, len, factor)
_lib.scale_f32x8.argtypes = [
    _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float),
    _ct.c_int32, _ct.c_float,
]
_lib.scale_f32x8.restype = None

# frame_stats(data, len, out_min, out_max, out_sum)
_lib.frame_stats.argtypes = [
    _ct.POINTER(_ct.c_float), _ct.c_int32,
    _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float), _ct.POINTER(_ct.c_float),
]
_lib.frame_stats.restype = None


def _fptr(arr):
    return arr.ctypes.data_as(_ct.POINTER(_ct.c_float))


def _check_f32(arr, name):
    if arr.dtype != np.float32:
        raise TypeError(f"{name}: expected float32, got {arr.dtype}")


def accumulate_f32x8(acc, frame):
    """acc[i] += frame[i] using SIMD."""
    _check_f32(acc, "acc")
    _check_f32(frame, "frame")
    _lib.accumulate_f32x8(_fptr(acc), _fptr(frame), _ct.c_int32(frame.size))


def accumulate_batch4_f32x8(acc, f0, f1, f2, f3):
    """acc[i] += f0[i] + f1[i] + f2[i] + f3[i] in one pass."""
    for name, a in [("acc", acc), ("f0", f0), ("f1", f1), ("f2", f2), ("f3", f3)]:
        _check_f32(a, name)
    _lib.accumulate_batch4_f32x8(
        _fptr(acc), _fptr(f0), _fptr(f1), _fptr(f2), _fptr(f3),
        _ct.c_int32(f3.size),
    )


def accumulate_batch8_f32x8(acc, f0, f1, f2, f3, f4, f5, f6, f7):
    """acc[i] += f0[i] + ... + f7[i] in one pass."""
    for name, a in [("acc", acc), ("f0", f0), ("f1", f1), ("f2", f2), ("f3", f3),
                     ("f4", f4), ("f5", f5), ("f6", f6), ("f7", f7)]:
        _check_f32(a, name)
    _lib.accumulate_batch8_f32x8(
        _fptr(acc), _fptr(f0), _fptr(f1), _fptr(f2), _fptr(f3),
        _fptr(f4), _fptr(f5), _fptr(f6), _fptr(f7),
        _ct.c_int32(f7.size),
    )


def scale_f32x8(data, out, factor):
    """out[i] = data[i] * factor using SIMD."""
    _check_f32(data, "data")
    _check_f32(out, "out")
    _lib.scale_f32x8(_fptr(data), _fptr(out), _ct.c_int32(out.size),
                      _ct.c_float(float(factor)))


def frame_stats(data):
    """Single-pass min/max/sum. Returns (min, max, sum)."""
    _check_f32(data, "data")
    out_min = np.zeros(1, dtype=np.float32)
    out_max = np.zeros(1, dtype=np.float32)
    out_sum = np.zeros(1, dtype=np.float32)
    _lib.frame_stats(_fptr(data), _ct.c_int32(data.size),
                      _fptr(out_min), _fptr(out_max), _fptr(out_sum))
    return float(out_min[0]), float(out_max[0]), float(out_sum[0])
