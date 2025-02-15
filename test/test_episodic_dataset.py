import numpy as np
import torch
import h5py
import cv2

from utils import EpisodicDataset, get_norm_stats


def get_dataset_data(fix_start_ts=None, query_history=[], action_history=[], image_history=[]):
    """
    Retrieves dataset data (sim_insertion_scripted) for a specified number of episodes (10).

    Args:
        query_history (list, optional): List of history indices for qpos. Defaults to [].
        action_history (list, optional): List of history indices for actions. Defaults to [].
        image_history (list, optional): List of history indices for images. Defaults to [].

    Returns:
        tuple: A tuple containing:
            - image_data (ndarray): The image data from the dataset.
            - qpos_data (ndarray): The position data from the dataset.
            - action_data (ndarray): The action data from the dataset.
        """
    num_episodes = 50
    episode_ids = list(zip([0] * num_episodes, np.arange(0, num_episodes)))
    dataset_dirs = 'data/sim_insertion_scripted/'
    camera_names = ['top', 'angle', 'vis']
    norm_stats = get_norm_stats([dataset_dirs], [num_episodes])

    dataset = EpisodicDataset(episode_ids, [dataset_dirs], camera_names, norm_stats, fix_start_ts=fix_start_ts,
                               qpos_history=query_history, action_history=action_history, image_history=image_history)
    data = dataset[0]

    image_data = data['image_data']
    qpos_data = data['qpos_data']
    action_data = data['action_data']
    action_history_data = data['action_history_data']
    return image_data, qpos_data, action_data, action_history_data


def test_episodic_dataset():
    """
    Test the Episodic dataset and that it can retrieve the data from where we want.
    """
    image_data, qpos_data, action_data, action_history_data = get_dataset_data()

    assert image_data.shape == (1, 3, 3, 480, 640)
    assert qpos_data.shape == (1, 14)
    assert action_data.shape[1] == 14

    # Check that action_history_data is empty
    assert action_history_data.shape == torch.Size([0, 14])


def test_episodic_dataset_with_history():
    """
    Test the Episodic dataset and that it can retrieve the data from where we want.
    """
    fix_start_ts = 20
    qpos_history = [0, 1, 2]  # 1, 2, 3 steps back
    action_history = [0, 1, 2, 3]  # 1, 2, 3, 4 steps back
    image_history = [0, 1, 2, 3, 4]  # 1, 2, 3, 4, 5 steps back

    image_data, qpos_data, action_data, action_history_data = get_dataset_data(
        fix_start_ts=fix_start_ts,
        query_history=qpos_history,
        action_history=action_history,
        image_history=image_history
    )

    assert image_data.shape == (len(image_history) + 1, 3, 3, 480, 640)
    assert qpos_data.shape == (len(qpos_history) + 1, 14)
    assert action_history_data.shape == (len(action_history), 14)

    # Get the norm stats since we need that to normalize the data
    num_episodes = 50
    dataset_dirs = 'data/sim_insertion_scripted/'
    norm_stats = get_norm_stats([dataset_dirs], [num_episodes])
    camera_names = ['top', 'angle', 'vis']

    # Read data from HDF5 file and compare it to the data we have
    with h5py.File('data/sim_insertion_scripted/episode_0.hdf5', 'r') as f:
        # Read the value at fix_start_ts from the hdf5 file and compare it with the data from the dataloader
        start_ts_qpos = f['observations/qpos'][fix_start_ts]

        # Normalize the data
        start_ts_qpos = (start_ts_qpos - norm_stats['qpos_mean']) / norm_stats['qpos_std']

        # Check that the current data matches (current frame is at index 0)
        assert np.allclose(start_ts_qpos, qpos_data[0])

        # Check history data - qpos_data[i+1] should match history at query_history[i] steps back
        for i, hist_idx in enumerate(qpos_history):
            hist_ts = fix_start_ts - (hist_idx + 1)  # hist_idx 0 means 1 step back
            start_i_qpos = f['observations/qpos'][hist_ts]
            start_i_qpos = (start_i_qpos - norm_stats['qpos_mean']) / norm_stats['qpos_std']
            assert np.allclose(start_i_qpos, qpos_data[i + 1])

        # Read the action history data from the HDF5 file and compare it with the data from the dataloader
        for i, hist_idx in enumerate(action_history):
            hist_ts = fix_start_ts - (hist_idx + 1)  # hist_idx 0 means 1 step back
            start_i_action = f['action'][hist_ts]
            start_i_action = (start_i_action - norm_stats['action_mean']) / norm_stats['action_std']
            assert np.allclose(start_i_action, action_history_data[i])

        hdf5_image_data = np.stack([
            f[f'observations/images/{camera_name}'][fix_start_ts] for camera_name in camera_names
        ], axis=0)
        # Convert to torch
        hdf5_image_data = torch.tensor(hdf5_image_data)

        hdf5_image_data = torch.einsum('k h w c -> k c h w', hdf5_image_data)
        hdf5_image_data = hdf5_image_data / 255.0

        # Compare current frame data from dataloader and HDF5 file (current frame is at index 0)
        assert np.allclose(image_data[0], hdf5_image_data)

        # Read the image history data from the HDF5 file and compare it with the data from the dataloader
        for i, hist_idx in enumerate(image_history):
            hist_ts = fix_start_ts - (hist_idx + 1)  # hist_idx 0 means 1 step back
            hdf5_image_data = np.stack([
            f[f'observations/images/{camera_name}'][hist_ts] for camera_name in camera_names
            ], axis=0)
            # Convert to torch
            hdf5_image_data = torch.tensor(hdf5_image_data)

            hdf5_image_data = torch.einsum('k h w c -> k c h w', hdf5_image_data)
            hdf5_image_data = hdf5_image_data / 255.0

            # Compare data from dataloader and HDF5 file
            assert np.allclose(image_data[i + 1], hdf5_image_data)


def test_episodic_dataset_with_history_repeat():
    """
    Test the Episodic dataset and that it can retrieve the data from where we want with a repeat of the first data.
    """
    fix_start_ts = 1
    qpos_history = [0, 1]  # 1, 2 steps back
    action_history = [0, 1]  # 1, 2 steps back
    image_history = [0, 1]  # 1, 2 steps back

    image_data, qpos_data, action_data, action_history_data = get_dataset_data(
        fix_start_ts=fix_start_ts,
        query_history=qpos_history,
        action_history=action_history,
        image_history=image_history
    )

    assert image_data.shape == (3, 3, 3, 480, 640)
    assert qpos_data.shape == (3, 14)
    assert action_history_data.shape == (2, 14)

    # Check that qpos_data - index 1 and 2 should be same (both come from ts=0 due to clamping)
    # but index 0 should be different (current frame at ts=1)
    assert np.all(qpos_data[1].numpy() == qpos_data[2].numpy())  # Both history frames from ts=0
    assert np.any(qpos_data[0].numpy() != qpos_data[1].numpy())  # Current vs history should differ

    # Check that action_history_data both frames should be same (both come from ts=0 due to clamping)
    assert np.all(action_history_data[0].numpy() == action_history_data[1].numpy())

    # Check that image_data history frames should be same (both come from ts=0 due to clamping)
    # but current frame should be different
    assert np.all(image_data[1].numpy() == image_data[2].numpy())  # Both history frames from ts=0
    assert np.any(image_data[0].numpy() != image_data[1].numpy())  # Current vs history should differ

    # Get the norm stats since we need that to normalize the data
    num_episodes = 50
    dataset_dirs = 'data/sim_insertion_scripted/'
    norm_stats = get_norm_stats([dataset_dirs], [num_episodes])
    camera_names = ['top', 'angle', 'vis']

    # Read data from HDF5 file and compare it to the data we have
    with h5py.File('data/sim_insertion_scripted/episode_0.hdf5', 'r') as f:
        # Read the value at fix_start_ts from the hdf5 file and compare it with the data from the dataloader
        start_ts_qpos = f['observations/qpos'][fix_start_ts]

        # Normalize the data
        start_ts_qpos = (start_ts_qpos - norm_stats['qpos_mean']) / norm_stats['qpos_std']

        # Check that the current frame data matches (current frame is at index 0)
        assert np.allclose(start_ts_qpos, qpos_data[0])

        # Check that the history data is from ts=0 (due to clamping)
        start_0_qpos = f['observations/qpos'][0]
        start_0_qpos = (start_0_qpos - norm_stats['qpos_mean']) / norm_stats['qpos_std']
        assert np.allclose(start_0_qpos, qpos_data[1])
        assert np.allclose(start_0_qpos, qpos_data[2])

        hdf5_image_data = np.stack([
            f[f'observations/images/{camera_name}'][fix_start_ts] for camera_name in camera_names
        ], axis=0)
        # Convert to torch
        hdf5_image_data = torch.tensor(hdf5_image_data)

        hdf5_image_data = torch.einsum('k h w c -> k c h w', hdf5_image_data)
        hdf5_image_data = hdf5_image_data / 255.0

        # Compare current frame data from dataloader and HDF5 file (current frame is at index 0)
        assert np.allclose(image_data[0], hdf5_image_data)


def test_episodic_dataset_action_history_at_ts0():
    """
    Test that when the timestamp is 0, the action history is padded with the first qpos.
    """
    fix_start_ts = 0
    action_history = [0, 1, 2]  # 1, 2, 3 steps back

    # We need qpos_data to compare, so we request at least one qpos frame.
    # We expect qpos_data to have shape (1, 14) since no history is requested.
    _, qpos_data, _, action_history_data = get_dataset_data(
        fix_start_ts=fix_start_ts,
        action_history=action_history,
    )

    # Check that action_history_data has the correct shape
    assert action_history_data.shape == (len(action_history), 14)

    # Check that all items in action_history_data are the same
    for i in range(len(action_history) - 1):
        assert np.allclose(action_history_data[i], action_history_data[i+1])

    # Get the norm stats to normalize the data
    num_episodes = 50
    dataset_dirs = 'data/sim_insertion_scripted/'
    norm_stats = get_norm_stats([dataset_dirs], [num_episodes])

    # Read the qpos at ts=0 from the HDF5 file
    with h5py.File('data/sim_insertion_scripted/episode_0.hdf5', 'r') as f:
        qpos_ts0 = f['observations/qpos'][0]

    # Normalize the qpos data with action stats, because that's what the dataset does
    qpos_ts0_normalized_with_action_stats = (qpos_ts0 - norm_stats['action_mean']) / norm_stats['action_std']

    # Check that the action history is padded with the normalized qpos at ts=0
    assert np.allclose(action_history_data[0], qpos_ts0_normalized_with_action_stats)

    # Also check against the returned qpos_data (which should be the qpos at ts=0, but normalized with qpos stats)
    qpos_ts0_normalized_with_qpos_stats = (qpos_ts0 - norm_stats['qpos_mean']) / norm_stats['qpos_std']
    assert np.allclose(qpos_data[0], qpos_ts0_normalized_with_qpos_stats)


def test_episodic_dataset_with_non_successive_history():
    """
    Test the Episodic dataset with non-successive history indices.
    """
    fix_start_ts = 20
    qpos_history = [0, 2, 5]  # 1, 3, 6 steps back
    action_history = [1, 3, 5, 7]  # 2, 4 steps back
    image_history = [0, 4]  # 1, 5 steps back

    image_data, qpos_data, action_data, action_history_data = get_dataset_data(
        fix_start_ts=fix_start_ts,
        query_history=qpos_history,
        action_history=action_history,
        image_history=image_history
    )

    assert image_data.shape == (len(image_history) + 1, 3, 3, 480, 640)
    assert qpos_data.shape == (len(qpos_history) + 1, 14)
    assert action_history_data.shape == (len(action_history), 14)

    # Get the norm stats since we need that to normalize the data
    num_episodes = 50
    dataset_dirs = 'data/sim_insertion_scripted/'
    norm_stats = get_norm_stats([dataset_dirs], [num_episodes])
    camera_names = ['top', 'angle', 'vis']

    # Read data from HDF5 file and compare it to the data we have
    with h5py.File('data/sim_insertion_scripted/episode_0.hdf5', 'r') as f:
        # Read the value at fix_start_ts from the hdf5 file and compare it with the data from the dataloader
        start_ts_qpos = f['observations/qpos'][fix_start_ts]

        # Normalize the data
        start_ts_qpos = (start_ts_qpos - norm_stats['qpos_mean']) / norm_stats['qpos_std']

        # Check that the current data matches (current frame is at index 0)
        assert np.allclose(start_ts_qpos, qpos_data[0])

        # Check history data - qpos_data[i+1] should match history at query_history[i] steps back
        for i, hist_idx in enumerate(qpos_history):
            hist_ts = fix_start_ts - (hist_idx + 1)  # hist_idx 0 means 1 step back
            start_i_qpos = f['observations/qpos'][hist_ts]
            start_i_qpos = (start_i_qpos - norm_stats['qpos_mean']) / norm_stats['qpos_std']
            assert np.allclose(start_i_qpos, qpos_data[i + 1])

        # Read the action history data from the HDF5 file and compare it with the data from the dataloader
        for i, hist_idx in enumerate(action_history):
            hist_ts = fix_start_ts - (hist_idx + 1)  # hist_idx 0 means 1 step back
            start_i_action = f['action'][hist_ts]
            start_i_action = (start_i_action - norm_stats['action_mean']) / norm_stats['action_std']
            assert np.allclose(start_i_action, action_history_data[i])

        hdf5_image_data = np.stack([
            f[f'observations/images/{camera_name}'][fix_start_ts] for camera_name in camera_names
        ], axis=0)
        # Convert to torch
        hdf5_image_data = torch.tensor(hdf5_image_data)

        hdf5_image_data = torch.einsum('k h w c -> k c h w', hdf5_image_data)
        hdf5_image_data = hdf5_image_data / 255.0

        # Compare current frame data from dataloader and HDF5 file (current frame is at index 0)
        assert np.allclose(image_data[0], hdf5_image_data)

        # Read the image history data from the HDF5 file and compare it with the data from the dataloader
        for i, hist_idx in enumerate(image_history):
            hist_ts = fix_start_ts - (hist_idx + 1)  # hist_idx 0 means 1 step back
            hdf5_image_data = np.stack([
            f[f'observations/images/{camera_name}'][hist_ts] for camera_name in camera_names
            ], axis=0)
            # Convert to torch
            hdf5_image_data = torch.tensor(hdf5_image_data)

            hdf5_image_data = torch.einsum('k h w c -> k c h w', hdf5_image_data)
            hdf5_image_data = hdf5_image_data / 255.0

            # Compare data from dataloader and HDF5 file
            assert np.allclose(image_data[i + 1], hdf5_image_data)
