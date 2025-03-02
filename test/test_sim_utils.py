import pytest
import numpy as np
import sys
import os

from sim_utils import QposHistoryManager


def test_initialization_default():
    """Test initialization with default values."""
    qpos_mgr = QposHistoryManager({})
    assert qpos_mgr.qpos_history_min == 1
    assert len(qpos_mgr.qpos_history) == 0


def test_initialization_custom():
    """Test initialization with custom values."""
    qpos_mgr = QposHistoryManager({'qpos_history_min': 5})
    assert qpos_mgr.qpos_history_min == 5
    assert len(qpos_mgr.qpos_history) == 0


def test_update():
    """Test updating with new qpos values."""
    qpos_mgr = QposHistoryManager({})
    qpos1 = np.array([1.0, 2.0, 3.0])
    qpos2 = np.array([4.0, 5.0, 6.0])
    
    qpos_mgr.update(qpos1)
    assert len(qpos_mgr.qpos_history) == 1
    np.testing.assert_array_equal(qpos_mgr.qpos_history[0], qpos1)
    
    qpos_mgr.update(qpos2)
    assert len(qpos_mgr.qpos_history) == 2
    np.testing.assert_array_equal(qpos_mgr.qpos_history[0], qpos1)
    np.testing.assert_array_equal(qpos_mgr.qpos_history[1], qpos2)


def test_get_padded_history_empty():
    """Test getting padded history when history is empty."""
    qpos_mgr = QposHistoryManager({})
    padded_history = qpos_mgr.get_padded_history()
    assert len(padded_history) == 0


def test_get_padded_history_shorter_than_min():
    """Test getting padded history when history is shorter than minimum required."""
    qpos_mgr = QposHistoryManager({'qpos_history_min': 3})
    qpos = np.array([1.0, 2.0, 3.0])
    
    qpos_mgr.update(qpos)
    padded_history = qpos_mgr.get_padded_history()
    
    assert len(padded_history) == 3
    for i in range(3):
        np.testing.assert_array_equal(padded_history[i], qpos)


def test_get_padded_history_equal_to_min():
    """Test getting padded history when history length equals minimum required."""
    qpos_mgr = QposHistoryManager({'qpos_history_min': 2})
    qpos1 = np.array([1.0, 2.0, 3.0])
    qpos2 = np.array([4.0, 5.0, 6.0])
    
    qpos_mgr.update(qpos1)
    qpos_mgr.update(qpos2)
    padded_history = qpos_mgr.get_padded_history()
    
    assert len(padded_history) == 2
    np.testing.assert_array_equal(padded_history[0], qpos1)
    np.testing.assert_array_equal(padded_history[1], qpos2)


def test_get_padded_history_longer_than_min():
    """Test getting padded history when history is longer than minimum required."""
    qpos_mgr = QposHistoryManager({'qpos_history_min': 1})
    qpos1 = np.array([1.0, 2.0, 3.0])
    qpos2 = np.array([4.0, 5.0, 6.0])
    qpos3 = np.array([7.0, 8.0, 9.0])
    
    qpos_mgr.update(qpos1)
    qpos_mgr.update(qpos2)
    qpos_mgr.update(qpos3)
    padded_history = qpos_mgr.get_padded_history()
    
    assert len(padded_history) == 2
    np.testing.assert_array_equal(padded_history[0], qpos2)
    np.testing.assert_array_equal(padded_history[1], qpos3)

def test_get_history_with_larget_inputs():
    """
    Test when we insert 5 qpos values.
    """
    qpos_mgr = QposHistoryManager({'qpos_history_min': 1})
    qpos1 = np.array([1.0, 2.0, 3.0])
    qpos2 = np.array([4.0, 5.0, 6.0])
    qpos3 = np.array([7.0, 8.0, 9.0])
    qpos4 = np.array([10.0, 11.0, 12.0])
    qpos5 = np.array([13.0, 14.0, 15.0])

    qpos_mgr.update(qpos1)
    qpos_mgr.update(qpos2)
    qpos_mgr.update(qpos3)
    qpos_mgr.update(qpos4)
    qpos_mgr.update(qpos5)

    history = qpos_mgr.get_padded_history()
    assert len(history) == 2

    np.testing.assert_array_equal(history[0], qpos4)
    np.testing.assert_array_equal(history[1], qpos5)

def test_history_size_limit():
    """Test that history size is limited to prevent unbounded growth."""
    qpos_mgr = QposHistoryManager({'qpos_history_min': 2})
    max_history = qpos_mgr.qpos_history_min + 1  # Updated to match implementation
    
    # Add max_history + 10 items
    for i in range(max_history + 10):
        qpos_mgr.update(np.array([float(i), float(i), float(i)]))
    
    # Check that only the last max_history items are kept
    assert len(qpos_mgr.qpos_history) == max_history
    
    # Check that the oldest items are removed
    first_qpos_index = (max_history + 10) - max_history
    np.testing.assert_array_equal(
        qpos_mgr.qpos_history[0], 
        np.array([float(first_qpos_index), float(first_qpos_index), float(first_qpos_index)])
    )
