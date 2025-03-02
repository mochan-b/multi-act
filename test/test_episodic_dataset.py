import numpy as np
import torch
import h5py
import cv2

from utils import EpisodicDataset, get_norm_stats


def get_dataset_data(fix_start_ts=None, query_history=0, action_history=0, image_history=0):
    """
    Retrieves dataset data (sim_insertion_scripted) for a specified number of episodes (10).

    Args:
        query_history (int, optional): The history length for querying the dataset. Defaults to 0.
        action_history (int, optional): The history length for the actions. Defaults to 0.
        image_history (int, optional): The history length for the images. Defaults to 0.

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

    image_data, qpos_data, action_data, task_name, is_pad, action_history_data = data
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
    assert action_history_data.shape == torch.Size([0])


def test_episodic_dataset_with_history():
    """
    Test the Episodic dataset and that it can retrieve the data from where we want.
    """
    fix_start_ts = 20
    query_history = 3
    action_history = 4
    image_history = 5

    image_data, qpos_data, action_data, action_history_data = get_dataset_data(
        fix_start_ts=fix_start_ts,
        query_history=query_history,
        action_history=action_history,
        image_history=image_history
    )

    assert image_data.shape == (image_history + 1, 3, 3, 480, 640)
    assert qpos_data.shape == (query_history + 1, 14)
    assert action_history_data.shape == (action_history, 14)

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

        # Check that the data is the same
        assert np.allclose(start_ts_qpos, qpos_data[-1])

        # Check that the first and second data is from the data from 0 repeated
        for i in range(query_history):
            start_i_qpos = f['observations/qpos'][fix_start_ts - i - 1]
            start_i_qpos = (start_i_qpos - norm_stats['qpos_mean']) / norm_stats['qpos_std']
            assert np.allclose(start_i_qpos, qpos_data[query_history - i - 1])

        # Read the action history data from the HDF5 file and compare it with the data from the dataloader
        for i in range(action_history):
            start_i_action = f['action'][fix_start_ts - i - 1]
            start_i_action = (start_i_action - norm_stats['action_mean']) / norm_stats['action_std']
            assert np.allclose(start_i_action, action_history_data[action_history - i - 1])

        hdf5_image_data = np.stack([
            f[f'observations/images/{camera_name}'][fix_start_ts] for camera_name in camera_names
        ], axis=0)
        # Convert to torch
        hdf5_image_data = torch.tensor(hdf5_image_data)

        hdf5_image_data = torch.einsum('k h w c -> k c h w', hdf5_image_data)
        hdf5_image_data = hdf5_image_data / 255.0

        # Compare data from dataloader and HDF5 file
        assert np.allclose(image_data[-1], hdf5_image_data)

        # Read the image history data from the HDF5 file and compare it with the data from the dataloader
        for i in range(image_history):
            hdf5_image_data = np.stack([
            f[f'observations/images/{camera_name}'][fix_start_ts - i - 1] for camera_name in camera_names
            ], axis=0)
            # Convert to torch
            hdf5_image_data = torch.tensor(hdf5_image_data)

            hdf5_image_data = torch.einsum('k h w c -> k c h w', hdf5_image_data)
            hdf5_image_data = hdf5_image_data / 255.0

            # Compare data from dataloader and HDF5 file
            assert np.allclose(image_data[image_history - i - 1], hdf5_image_data)

def test_episodic_dataset_with_history_repeat():
    """
    Test the Episodic dataset and that it can retrieve the data from where we want with a repeat of the first data.
    """
    fix_start_ts = 1
    query_history = 2
    action_history = 2
    image_history = 2

    image_data, qpos_data, action_data, action_history_data = get_dataset_data(
        fix_start_ts=fix_start_ts,
        query_history=query_history,
        action_history=action_history,
        image_history=image_history
    )

    assert image_data.shape == (3, 3, 3, 480, 640)
    assert qpos_data.shape == (3, 14)
    assert action_history_data.shape == (2, 14)

    # Check that qpos_data first data is the same as the second and third data is not the same
    assert np.all(qpos_data[0].numpy() == qpos_data[1].numpy())
    assert np.any(qpos_data[0].numpy() != qpos_data[2].numpy())

    # Check that action_history_data first data is the same as the second and third data is not the same
    assert np.all(action_history_data[0].numpy() == action_history_data[1].numpy())

    # Check that image_data first data is the same as the second and third data is not the same
    assert np.all(image_data[0].numpy() == image_data[1].numpy())
    assert np.any(image_data[0].numpy() != image_data[2].numpy())

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

        # Check that the data is the same
        assert np.allclose(start_ts_qpos, qpos_data[-1])

        # Check that the first and second data is from the data from 0 repeated
        start_0_qpos = f['observations/qpos'][0]
        start_0_qpos = (start_0_qpos - norm_stats['qpos_mean']) / norm_stats['qpos_std']
        assert np.allclose(start_0_qpos, qpos_data[0])
        assert np.allclose(start_0_qpos, qpos_data[1])

        hdf5_image_data = np.stack([
            f[f'observations/images/{camera_name}'][fix_start_ts] for camera_name in camera_names
        ], axis=0)
        # Convert to torch
        hdf5_image_data = torch.tensor(hdf5_image_data)

        hdf5_image_data = torch.einsum('k h w c -> k c h w', hdf5_image_data)
        hdf5_image_data = hdf5_image_data / 255.0

        # Compare data from dataloader and HDF5 file
        assert np.allclose(image_data[-1], hdf5_image_data)

