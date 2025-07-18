import glob
import torch
import numpy as np
import os
import pickle
import argparse
import yaml
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from text_encoders import ClipLanguageConditioned
from utils import load_data  # data functions
from utils import sample_box_pose, sample_insertion_pose  # robot functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos
from sim_utils import QposHistoryManager  # Import QposHistoryManager

from sim_env import BOX_POSE


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    resume = args.get('resume', False)

    # get task parameters
    is_sim = task_name[0][:4] == 'sim_'

    if is_sim:
        pass
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]

    # For each task, get the yaml files
    task_yaml_file = []
    task_yaml_data = []
    for task in task_name:
        task_yaml_file.append(os.path.join(args['task_config_dir'], task + '.yaml'))
        with open(task_yaml_file[-1], 'r') as f:
            task_yaml_data.append(yaml.load(f, Loader=yaml.FullLoader))

    # Check that the camera names are the same for all of the tasks
    camera_names = task_yaml_data[0]['camera_names']
    for task in task_yaml_data:
        assert task['camera_names'] == camera_names

    # Get the various parameters for each task
    dataset_dir = [task['dataset_dir'] for task in task_yaml_data]
    num_episodes = [task['num_episodes'] for task in task_yaml_data]
    episode_len = [task['episode_len'] for task in task_yaml_data]

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'eval_num_queries': args.get('eval_chunk_size', None),
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
        # Add all the args to the policy config
        for k, v in args.items():
            policy_config[k] = v

    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone': backbone, 'num_queries': 1,
                         'camera_names': camera_names, }
    else:
        raise NotImplementedError
    
    # Get the qpos_history, action_history, and image_history from the config
    multi_history = args.get('multi_history', False)
    qpos_history_minmax = args.get('qpos_history', [0, 0])
    action_history_minmax = args.get('action_history', [0, 0])
    image_history_minmax = args.get('image_history', [0, 0])

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'resume': resume,
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    # Set the data loaders to use the min or max values for the qpos, action, and image history
    if not multi_history:
        # Use the min values
        qpos_history = qpos_history_minmax[0]
        action_history = action_history_minmax[0]
        image_history = image_history_minmax[0]
    else:
        # Use the max values
        qpos_history = qpos_history_minmax[1]
        action_history = action_history_minmax[1]
        image_history = image_history_minmax[1]

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train,
                                                           batch_size_val, qpos_history=qpos_history, action_history=action_history, image_history=image_history)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)

    task_name = config['policy_config']['eval_task']
    task_index = config['task_name'].index(task_name)

    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len'][task_index]
    temporal_agg = config['temporal_agg']
    eval_task = config['policy_config']['eval_task']
    camera_names_eval = config['policy_config']['camera_names_eval']
    onscreen_cam = 'angle'

    # Initialize QposHistoryManager to maintain qpos history
    qpos_history_config = {'qpos_history_min': 0}  # Current + 2 historical values
    qpos_manager = QposHistoryManager(qpos_history_config)

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    clip = ClipLanguageConditioned()
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['eval_num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset

        ts = env.reset()
        
        # Reset qpos manager for each new rollout
        qpos_manager = QposHistoryManager(qpos_history_config)

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                
                # Get current qpos
                qpos_numpy = np.array(obs['qpos'])
                
                # Update the qpos history manager with current qpos
                qpos_manager.update(qpos_numpy)
                
                # Get padded history (current + historical values)
                qpos_history_padded = qpos_manager.get_padded_history()
                
                # Process current and historical qpos
                qpos_processed = np.stack(qpos_history_padded, axis=0)
                qpos_processed = pre_process(qpos_processed)
                qpos_processed = torch.from_numpy(qpos_processed).float().cuda().unsqueeze(0)
                
                # Store only the current qpos in qpos_history for visualization
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos

                # Add an extra dimension after the batch dimension for the qpos
                # qpos = qpos.unsqueeze(1)
                
                curr_image = get_image(ts, camera_names)

                # Add a dimension for curr_image after the batch dimension for the image_history
                # curr_image = curr_image.unsqueeze(1)

                task_embeddings = clip.get_text_feature(eval_task).unsqueeze(dim=0).cuda()

                ### query policy - use processed qpos history instead of just current qpos
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        camera_indices = [camera_names.index(cam_name) for cam_name in camera_names_eval]
                        print(qpos)
                        all_actions = policy(qpos, curr_image, task_embeddings=task_embeddings,
                                             camera_indices=camera_indices)
                    if temporal_agg:
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos_processed, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                          move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward == env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate * 100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy, clip, camera_names, multi_options={}):
    image_data, qpos_data, action_data, task_names, is_pad, action_history_data = data
    image_data, qpos_data, action_data, is_pad, action_history_data = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), action_history_data.cuda()

    # Make the task embeddings
    task_embeddings = [clip.get_text_feature(task_name) for task_name in task_names]
    task_embeddings = torch.stack(task_embeddings).cuda()

    # --- Multi options ---

    # --- Multi camera ---
    # Get the multi-camera option
    multi_camera = multi_options.get('multi_camera', False)

    # Get the camera indices
    n_cameras = len(camera_names)
    if multi_camera:
        # Make the choice of sensors that we will be using
        # Get a number between 1 and num_cameras+1 and set the cameras to sample from in the dataset
        sample_cameras = np.random.randint(0, len(camera_names)) + 1
        camera_indices = np.random.choice(n_cameras, sample_cameras, replace=False)
    else:
        camera_indices = np.arange(n_cameras)

    # --- Multi horizon ---
    # Get the multi horizon option
    multi_horizon = multi_options.get('multi_horizon', False)

    num_queries = policy.model.num_queries    
    if multi_horizon:
        # Choose a random value between 1 and self.model.num_queries
        rand_num_queries = torch.randint(1, num_queries, (1,))

        action_data = action_data[:, :rand_num_queries]
        is_pad = is_pad[:, :rand_num_queries]
    else:
        # Choose all of the num queries
        action_data = action_data[:, :num_queries]
        is_pad = is_pad[:, :num_queries]

    # Get the policy output
    return policy(qpos_data, image_data, action_data, task_embeddings, camera_indices, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    resume = config['resume']

    multi_options = {
        'multi_camera': policy_config.get('multi_camera', True),
        'multi_horizon': policy_config.get('multi_horizon', True),
        'multi_history': policy_config.get('multi_history', True),
    }

    set_seed(seed)

    clip = ClipLanguageConditioned()

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    # Check if we want to resume training from a previous checkpoint
    start_epoch = 0
    if resume:
        # Find the latest ckpt file to resume from
        ckpt_files = os.listdir(ckpt_dir)
        ckpt_files = glob.glob(os.path.join(ckpt_dir, f'policy_epoch_*_seed_{seed}.ckpt'))

        # Check that ckpt_files is not empty
        if len(ckpt_files) > 0:
            latest_ckpt = max(ckpt_files, key=os.path.getmtime)

            # Get the epoch number from the ckpt file
            start_epoch = int(latest_ckpt.split('_')[2])

            # Load the policy from the ckpt file to resume from
            policy.load_state_dict(torch.load(latest_ckpt))

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy, clip, camera_names)
                epoch_dicts.append(forward_dict)

            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, clip, camera_names, multi_options)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))

        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * (epoch - start_epoch):(batch_idx + 1) * ((epoch - start_epoch) + 1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', action='store', type=str, help='Configuration yaml file')
    yaml_file = parser.parse_args().config
    print(f'Using config: {yaml_file}')

    # Read the contents of the config file and convert to dictionary
    with open(yaml_file, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--onscreen_render', action='store_true')
    # parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    # parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    # parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    # parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    # parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    # parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    # parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    #
    # # for ACT
    # parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    # parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    # parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    # parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    # parser.add_argument('--temporal_agg', action='store_true')

    main(args)
