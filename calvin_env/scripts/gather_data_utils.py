import numpy as np
import os
import cv2
from tqdm import tqdm

def gather_trajectory(
        dataset_path,
        start, end,
        env,
        store_images=False, show_images=False, max_steps=None, extract_quat=False
):
    traj = dict(
        obs=dict(),
        actions=[],
        states=[],
    )

    num_steps = 0
    num_euler_warnings = 0

    for timestep in tqdm(range(start, end)):
        data = np.load(os.path.join(dataset_path, 'episode_{}.npz'.format(f'{timestep:07d}')))

        if extract_quat:
            env_obs = env.reset(state=np.concatenate((
                data['robot_obs'],
                data['scene_obs'],
            )))

            if np.linalg.norm(env_obs['robot0_eef_euler'] - data['robot_obs'][3:6]) > 0.1:
                num_euler_warnings += 1

            eef_euler = env_obs['robot0_eef_euler']
            eef_quat = env_obs['robot0_eef_quat']
        else:
            eef_euler = data['robot_obs'][3:6]
            eef_quat = None

        for k in data:
            if 'rgb' in k or 'depth' in k:
                if store_images and k in ['rgb_static', 'rgb_gripper']:
                    pass
                else:
                    continue

            if k not in traj['obs']:
                traj['obs'][k] = []
            traj['obs'][k].append(data[k])

        scene_obs = data['scene_obs']
        scene_obs_dict = dict(
            non_blocks=scene_obs[:6],
            block_red=scene_obs[6:12],
            block_blue=scene_obs[12:18],
            block_pink=scene_obs[18:24],
        )
        for k in scene_obs_dict:
            if k not in traj['obs']:
                traj['obs'][k] = []
            traj['obs'][k].append(scene_obs_dict[k])

        robot_obs = data['robot_obs']

        proprio_dict = dict(
            eef_pos=robot_obs[0:3],
            eef_euler=eef_euler,
            gripper_qpos=robot_obs[6:7],
            joint_qpos=robot_obs[7:14],
            prev_gripper_action=robot_obs[14:],
        )
        if extract_quat:
            assert eef_quat is not None
            proprio_dict['eef_quat'] = eef_quat
        for k in proprio_dict.keys():
            robot_key = 'robot0_{}'.format(k)
            if robot_key not in traj['obs']:
                traj['obs'][robot_key] = []
            traj['obs'][robot_key].append(proprio_dict[k])

        traj['states'].append(np.concatenate([data['robot_obs'], data['scene_obs']]))

        traj['actions'].append(data['rel_actions'])

        if show_images:
            im = data['rgb_static'][:,:,::-1]
            cv2.imshow('play data', im)
            cv2.waitKey(1)

        num_steps += 1
        if (max_steps is not None) and num_steps >= max_steps:
            break

    info = dict(
        num_steps=num_steps,
        num_euler_warnings=num_euler_warnings,
    )
    return traj, info
