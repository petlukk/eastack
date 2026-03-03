"""End-to-end tests for eastack."""
import numpy as np
import pytest

from eastack import stack, stack_mean, frame_stats


def test_stack_basic():
    """Stack 4 identical frames, result should equal frame * 4."""
    frame = np.ones((64, 64), dtype=np.float32) * 10.0
    frames = [frame] * 4
    result = stack(frames)
    assert result.shape == (64, 64)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 40.0, rtol=1e-6)


def test_stack_mean_basic():
    """Mean of identical frames should equal original frame."""
    frame = np.full((128, 128), 7.5, dtype=np.float32)
    result = stack_mean([frame] * 8)
    np.testing.assert_allclose(result, 7.5, rtol=1e-6)


def test_stack_3d_input():
    """Accept 3D array (N, H, W) directly."""
    data = np.random.rand(16, 32, 32).astype(np.float32)
    result = stack_mean(data)
    expected = np.mean(data, axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_stack_batching():
    """Exercise all batch paths: batch8, batch4, single."""
    # 13 frames = 1x batch8 + 1x batch4 + 1x single
    n_frames = 13
    frame = np.ones(256, dtype=np.float32) * 3.0
    frames = [frame] * n_frames
    result = stack(frames)
    np.testing.assert_allclose(result, 3.0 * n_frames, rtol=1e-5)


def test_stack_mean_matches_numpy():
    """Stack mean should match np.mean within f32 precision."""
    rng = np.random.default_rng(42)
    frames = [rng.standard_normal(1024).astype(np.float32) for _ in range(20)]
    result = stack_mean(frames)
    expected = np.mean(np.array(frames), axis=0)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_frame_stats():
    """Single-pass min/max/sum."""
    data = np.array([1.0, -3.0, 5.0, 2.0, 0.5], dtype=np.float32)
    mn, mx, s = frame_stats(data)
    assert mn == pytest.approx(-3.0)
    assert mx == pytest.approx(5.0)
    assert s == pytest.approx(5.5)


def test_stack_preserves_shape():
    """Output shape matches input frame shape."""
    frames = [np.zeros((100, 200), dtype=np.float32) for _ in range(5)]
    result = stack(frames)
    assert result.shape == (100, 200)


def test_stack_empty_raises():
    """Stacking zero frames should raise."""
    with pytest.raises(ValueError):
        stack([])
