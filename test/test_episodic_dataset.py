import numpy as np
import torch

from utils import EpisodicDataset, get_norm_stats


def get_dataset_data(fix_start_ts=None, query_history=0, action_history=0):
    """
    Retrieves dataset data (sim_insertion_scripted) for a specified number of episodes (10).

    Args:
        query_history (int, optional): The history length for querying the dataset. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - image_data (ndarray): The image data from the dataset.
            - qpos_data (ndarray): The position data from the dataset.
            - action_data (ndarray): The action data from the dataset.
    """
    num_episodes = 10
    episode_ids = list(zip([0] * num_episodes, np.arange(0, num_episodes)))
    dataset_dirs = 'data/sim_insertion_scripted/'
    camera_names = ['top']
    norm_stats = get_norm_stats([dataset_dirs], [num_episodes])

    dataset = EpisodicDataset(episode_ids, [dataset_dirs], camera_names, norm_stats, fix_start_ts=fix_start_ts,
                               query_history=query_history, action_history=action_history)
    data = dataset[0]

    image_data, qpos_data, action_data, task_name, is_pad, action_history_data = data
    return image_data, qpos_data, action_data, action_history_data

def test_episodic_dataset():
    """
    Test the Episodic dataset and that it can retrieve the data from where we want.
    """
    image_data, qpos_data, action_data, action_history_data = get_dataset_data()

    assert image_data.shape == (1, 3, 480, 640)
    assert qpos_data.shape == (1, 14)
    assert action_data.shape[1] == 14
    
    
    # Check that action_history_data is empty
    assert action_history_data.shape == torch.Size([0])
def test_episodic_dataset_with_history():
    """
    Test the Episodic dataset and that it can retrieve the data from where we want.
    """
    image_data, qpos_data, action_data, action_history_data = get_dataset_data(fix_start_ts=1, query_history=2, action_history=2)

    assert image_data.shape == (1, 3, 480, 640)
    assert qpos_data.shape == (3, 14)
    assert action_history_data.shape == (3, 14)
    
    # Check that qpos_data first data is the same as the second and third data is not the same
    assert np.all(qpos_data[0].numpy() == qpos_data[1].numpy())
    assert np.any(qpos_data[0].numpy() != qpos_data[2].numpy())
    
    # Check that action_history_data first data is the same as the second and third data is not the same
    assert np.all(action_history_data[0].numpy() == action_history_data[1].numpy())
    assert np.any(action_history_data[0].numpy() != action_history_data[2].numpy())