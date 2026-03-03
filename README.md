# eastack

SIMD-accelerated frame stacking powered by [Ea](https://github.com/petlukk/eacompute) kernels.

Stack N noisy exposures into a clean result. Signal reinforces, noise cancels by sqrt(N). Useful for astronomy, microscopy, video denoising, or any workflow that averages multiple frames.

## Install

```bash
pip install eastack
```

Pre-built wheels include compiled SIMD kernels for Linux x86_64, Linux aarch64, and Windows x86_64.

## Usage

```python
import numpy as np
from eastack import stack_mean

# Stack 16 noisy frames into a clean mean
frames = [np.random.rand(1024, 1024).astype(np.float32) for _ in range(16)]
result = stack_mean(frames)

# Also accepts 3D arrays (N, H, W)
data = np.random.rand(16, 1024, 1024).astype(np.float32)
result = stack_mean(data)
```

## API

| Function | Description |
|----------|-------------|
| `stack_mean(frames)` | Stack and compute mean. Returns f32 array. |
| `stack(frames)` | Accumulate without dividing. Returns sum. |
| `frame_stats(data)` | Single-pass `(min, max, sum)` of a float32 array. |

Low-level kernel access:

| Function | Description |
|----------|-------------|
| `accumulate_f32x8(acc, frame)` | `acc += frame` using SIMD |
| `accumulate_batch4_f32x8(acc, f0, f1, f2, f3)` | 4 frames in one pass |
| `accumulate_batch8_f32x8(acc, f0..f7)` | 8 frames in one pass |
| `scale_f32x8(data, out, factor)` | `out = data * factor` using SIMD |

## How it works

Batched accumulation reduces memory traffic by processing multiple frames per pass over the accumulator:

- **Single-frame**: `acc += frame` — 3N memory transactions per element (read acc, read frame, write acc) repeated N times
- **Batched (K=8)**: `acc += f0 + f1 + ... + f7` — one acc read/write per 8 frames

The `stack()` function automatically dispatches: batch8 first, then batch4 for remainder, then singles.

## Performance

On 4096x4096 frames (64 MB accumulator, single-threaded):

```
NumPy streaming (np.add loop) : 119 ms
Ea single-frame (same loop)   : 110 ms
Ea batched (8 frames/pass)    :  68 ms  — 1.76x faster
```

The speedup grows with frame size because the accumulator exceeds L3 cache — reducing acc traffic has maximum impact in DRAM.

## Building from source

```bash
EA_BIN=./ea ./build_kernels.sh
pip install -e .
```

## Requirements

- Python 3.9+
- NumPy
