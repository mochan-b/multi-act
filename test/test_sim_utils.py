import pytest
import numpy as np
import sys
import os

from sim_utils import NumpyRingBuffer

def test_empty_buffer_get_raises():
    """
    Test that getting from an empty buffer raises ValueError.
    """
    buf = NumpyRingBuffer(capacity=3, shape=(14,))
    with pytest.raises(ValueError):
        buf.get([0])


def test_negative_index_raises():
    """
    Test that negative indices raise IndexError.
    """
    buf = NumpyRingBuffer(capacity=3, shape=(14,))
    buf.append(np.zeros(14))
    with pytest.raises(IndexError):
        buf.get([-1])


def test_index_out_of_capacity_raises():
    """
    Test that indices out of capacity raise IndexError.
    """
    buf = NumpyRingBuffer(capacity=3, shape=(14,))
    buf.append(np.zeros(14))
    with pytest.raises(IndexError):
        buf.get([3])


def test_clamping_behavior():
    """
    Test that indices beyond current size clamp to oldest element.
    """
    buf = NumpyRingBuffer(capacity=5, shape=(2,))
    arr1 = np.array([1, 1])
    arr2 = np.array([2, 2])
    buf.append(arr1)
    buf.append(arr2)
    # size=2, so indices >=2 clamp to oldest (arr1)
    result = buf.get([0, 1, 2, 4])  # request beyond size
    expected = [arr2, arr1, arr1, arr1]
    for res, exp in zip(result, expected):
        assert np.array_equal(res, exp)


def test_overflow_and_ordering():
    """
    Test that buffer correctly overwrites oldest data and maintains order.
    """
    cap = 3
    buf = NumpyRingBuffer(capacity=cap, shape=(3,))
    data = [np.array([i, i, i]) for i in range(5)]
    for x in data:
        buf.append(x)
    # buffer should now contain last 3 entries: 2,3,4
    result = buf.get([0, 1, 2])
    assert np.array_equal(result[0], data[-1])  # last appended
    assert np.array_equal(result[1], data[-2])
    assert np.array_equal(result[2], data[-3])


def test_vector_buffer_shape_and_dtype():
    """
    Test that the buffer correctly stores and retrieves vector data.
    """
    buf = NumpyRingBuffer(capacity=2, shape=(14,))
    arr = np.arange(14, dtype=np.float32)
    buf.append(arr)
    out = buf.get([0])[0]
    assert out.dtype == np.float32
    assert out.shape == (14,)
    assert np.array_equal(out, arr)


def test_image_buffer_shape_and_dtype():
    """
    Test that the buffer correctly stores and retrieves image data.
    """
    shape = (3, 3, 640, 480)
    buf = NumpyRingBuffer(capacity=2, shape=shape, dtype=np.uint8)
    img = np.full(shape, 7, dtype=np.uint8)
    buf.append(img)
    out = buf.get([0])[0]
    assert out.dtype == np.uint8
    assert out.shape == shape
    assert np.all(out == 7)


def test_image_history_eval_workflow():
    """
    Test image history buffer workflow matching eval_bc function.
    Tests the exact format used in eval: (num_cameras, C, H, W) with float32.
    """
    # Simulate the eval_bc workflow
    num_cameras = 3
    C, H, W = 3, 224, 224  # RGB images
    image_history_list = [0, 2]  # 1 step back and 3 steps back
    
    # Create buffer with same capacity logic as eval_bc
    buffer_capacity = max(image_history_list) + 2
    buf = NumpyRingBuffer(buffer_capacity, shape=(num_cameras, C, H, W), dtype=np.float32)
    
    # Simulate adding images over time (like in eval loop)
    test_images = []
    for t in range(5):
        # Create unique test image for each timestep
        img = np.full((num_cameras, C, H, W), t * 10, dtype=np.float32)
        test_images.append(img)
        buf.append(img)
        
        # Test retrieval after buffer has some data
        if t >= 2:  # Enough data for history
            # Get current + history: [0] for current, [1,2,3...] for history (shift indices by 1)
            adjusted_indices = [0] + [i + 1 for i in image_history_list]
            result = buf.get(adjusted_indices)
            
            # Check shape: (num_frames, num_cameras, C, H, W)
            expected_num_frames = len(image_history_list) + 1
            assert result.shape == (expected_num_frames, num_cameras, C, H, W)
            
            # Check that current frame is at index 0
            assert np.array_equal(result[0], test_images[t]), f"Current frame mismatch at t={t}"
            
            # Check that history frames are correctly retrieved
            for i, hist_idx in enumerate(image_history_list):
                expected_t = max(0, t - (hist_idx + 1))  # Clamp to first frame if needed
                assert np.array_equal(result[i + 1], test_images[expected_t]), \
                    f"History frame {i} mismatch at t={t}, hist_idx={hist_idx}"


def test_image_history_with_insufficient_data():
    """
    Test image history behavior when buffer doesn't have enough history data.
    This tests the clamping behavior that ensures we always get valid data.
    """
    num_cameras = 2
    C, H, W = 3, 64, 64
    image_history_list = [1, 3]  # 2 steps back and 4 steps back
    
    buffer_capacity = max(image_history_list) + 2
    buf = NumpyRingBuffer(buffer_capacity, shape=(num_cameras, C, H, W), dtype=np.float32)
    
    # Add only 2 images (less than max history index + 1)
    img1 = np.full((num_cameras, C, H, W), 100, dtype=np.float32)
    img2 = np.full((num_cameras, C, H, W), 200, dtype=np.float32)
    
    buf.append(img1)
    buf.append(img2)
    
    # Request current + history even though we don't have enough history
    adjusted_indices = [0] + [i + 1 for i in image_history_list]
    result = buf.get(adjusted_indices)
    
    # Should still return the right shape
    expected_num_frames = len(image_history_list) + 1
    assert result.shape == (expected_num_frames, num_cameras, C, H, W)
    
    # Current frame should be img2 (latest)
    assert np.array_equal(result[0], img2)
    
    # History frames should clamp to oldest available (img1)
    # Because we don't have enough history, it clamps to the oldest element
    assert np.array_equal(result[1], img1)  # hist_idx=1 -> clamps to oldest
    assert np.array_equal(result[2], img1)  # hist_idx=3 -> clamps to oldest
