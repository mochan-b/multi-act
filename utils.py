import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dirs, camera_names, norm_stats,
                fix_start_ts=None, query_history=0, action_history=0, image_history=0):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dirs = dataset_dirs
        self.camera_names = camera_names
        self.n_cameras = len(camera_names)
        self.norm_stats = norm_stats
        self.fix_start_ts = fix_start_ts
        self.query_history = query_history
        self.action_history = action_history
        self.image_history = image_history
        self.is_sim = None
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        dataset_id, episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dirs[dataset_id], f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            task_name = root.attrs['task_name']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]

            # Get the observation at a random timestep
            if sample_full_episode:
                start_ts = 0
            else:
                if self.fix_start_ts is not None:
                    start_ts = self.fix_start_ts
                else:
                    start_ts = np.random.choice(episode_len)

            # ------------------------------------------------------------------
            # 1. Load qpos with history
            #    We'll gather `query_history + 1` frames: the current frame at 
            #    `start_ts` plus `query_history` preceding frames (if available).
            # ------------------------------------------------------------------
            start_index = start_ts - self.query_history
            if start_index < 0:
                # We don't have enough past frames, so replicate the earliest 
                # qpos if the requested history extends before index 0
                needed_frames = -start_index  # how many frames we are missing
                # slice_data is from time 0 up to and including start_ts
                slice_data = root['/observations/qpos'][0 : start_ts + 1]
                # replicate the first row `needed_frames` times
                pad = np.repeat(slice_data[0:1], needed_frames, axis=0)
                qpos = np.concatenate([pad, slice_data], axis=0)
            else:
                # We can directly slice from start_index to start_ts (inclusive)
                qpos = root['/observations/qpos'][start_index : (start_ts + 1)]

            # ------------------------------------------------------------------
            # 2. Load action with history
            #    We'll gather `action_history + 1` frames: the current frame at
            #    `start_ts` plus `action_history` preceding frames (if available).
            # ------------------------------------------------------------------
            if self.action_history == 0:
                action_history_data = np.array([])
            else:
                start_index_action = start_ts - self.action_history
                if start_index_action < 0:
                    # We don't have enough past frames, so replicate the earliest
                    # action if the requested history extends before index 0
                    needed_frames = -start_index_action  # how many frames we are missing
                    # slice_data is from time 0 up to and including start_ts
                    slice_data = root['/action'][0: start_ts]
                    # replicate the first row `needed_frames` times
                    pad = np.repeat(slice_data[0:1], needed_frames, axis=0)
                    action_history_data = np.concatenate([pad, slice_data], axis=0)
                else:
                    # We can directly slice from start_index to start_ts (inclusive)
                    action_history_data = root['/action'][start_index_action: (start_ts)]

            # ------------------------------------------------------------------
            # 3. Load image with history
            # ------------------------------------------------------------------
            all_cam_images = []
            for i in range(self.image_history + 1):
                # Get the timestamp for the image and replicate the first frame if needed
                ts = max(0, start_ts - i)
                cam_images = []
                for cam_name in self.camera_names:
                    cam_images.append(root[f'/observations/images/{cam_name}'][ts])
                all_cam_images.append(np.stack(cam_images, axis=0)) # reverse so latest is last
            # Stack the images and reverse the order so that the latest frame is first
            all_cam_images = np.stack(all_cam_images[::-1], axis=0)
            
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned
    
        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1
    
        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        action_history_data = torch.from_numpy(action_history_data).float()

        # channel last
        image_data = torch.einsum('p k h w c -> p k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        if self.action_history > 0:
            action_history_data = (action_history_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        return image_data, qpos_data, action_data, task_name, is_pad, action_history_data


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for i, each_num_episodes in enumerate(num_episodes):
        for episode_idx in range(each_num_episodes):
            dataset_path = os.path.join(dataset_dir[i], f'episode_{episode_idx}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                action = root['/action'][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8

    num_datasets = len(dataset_dir)
    total_num_episodes = sum(num_episodes)

    episodes = [(i, j) for i in range(num_datasets) for j in range(num_episodes[i])]
    shuffled_indices = np.random.permutation(total_num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * total_num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * total_num_episodes):]
    train_episodes = [episodes[i] for i in train_indices]
    val_episodes = [episodes[i] for i in val_indices]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_episodes, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_episodes, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=0)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
