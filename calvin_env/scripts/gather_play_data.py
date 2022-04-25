import numpy as np
import os
import h5py
import json

import robomimic.envs.env_base as EB
from robomimic.envs.env_calvin import EnvCalvin
import robomimic.utils.obs_utils as ObsUtils

from calvin_env.scripts.gather_data_utils import gather_trajectory

def main():
    # dataset_path = '/home/soroushn/research/calvin/dataset/task_D_D/training'
    # scene = 'calvin_scene_D'

    dataset_path = '/home/soroushn/research/calvin/dataset/task_ABC_D/training'
    scene = 'calvin_scene_B'

    output_dataset = '/home/soroushn/tmp/play_B_ld.hdf5'
    # output_dataset = '/home/soroushn/research/mtil/datasets/calvin/play_A.hdf5'

    max_total_steps = 1000000 #1000000 #5000
    traj_size = 1000
    store_images = False
    show_images = False

    # if store_images:
    #     assert max_total_steps <= 50000

    f_out = h5py.File(output_dataset, "w")
    data_grp = f_out.create_group("data")

    # ep_nums = np.array([int(ep_name[8:-4]) for ep_name in os.listdir(dataset_path) if ep_name.startswith('episode')])
    ep_start_end_ids = np.load(os.path.join(dataset_path, 'ep_start_end_ids.npy'))
    scene_info = np.load(os.path.join(dataset_path, 'scene_info.npy'), allow_pickle=True)[()]
    # post processing: switch A and D
    D_tmp = scene_info.get('calvin_scene_A', None)
    A_tmp = scene_info.get('calvin_scene_D', None)
    scene_info.pop("calvin_scene_A", None)
    scene_info.pop("calvin_scene_D", None)
    if A_tmp is not None:
        scene_info['calvin_scene_A'] = A_tmp
    if D_tmp is not None:
        scene_info['calvin_scene_D'] = D_tmp


    env = EnvCalvin('play', render=False, scene=scene)
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot_obs", "scene_obs"],
            rgb=[],
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    total_num_steps = 0
    total_euler_warnings = 0
    ep_num = 1
    for (start, end) in ep_start_end_ids:
        if scene is not None:
            if start < scene_info[scene][0] or end > scene_info[scene][1]:
                print("skipping traj from {} to {}".format(start, end))
                continue
        for curr in range(start, end, traj_size):
            print("Writing episode # {} ...".format(ep_num))
            ep_data_grp = data_grp.create_group('demo_{}'.format(ep_num))

            traj, traj_info = gather_trajectory(
                dataset_path, curr, min(curr+traj_size, end + 1), env,
                store_images=store_images, show_images=show_images,
                max_steps=max_total_steps-total_num_steps,
            )

            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("timestep_ids", data=np.array(traj["timestep_ids"]))
            for k in traj["obs"]:
                ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            ep_data_grp.attrs["num_samples"] = len(traj["actions"])

            total_num_steps += traj_info['num_steps']
            total_euler_warnings += traj_info['num_euler_warnings']

            ep_num += 1
            print("Euler warning percent: {:.2f} %".format(total_euler_warnings / total_num_steps * 100))

            if total_num_steps >= max_total_steps:
                break

        if total_num_steps >= max_total_steps:
            break

    env_meta = dict(
        type=EB.EnvType.CALVIN_TYPE,
        env_name="play",
        env_kwargs=dict(
            scene=scene,
        ),
    )
    data_grp.attrs["env_args"] = json.dumps(env_meta, indent=4)

    data_grp.attrs["total"] = total_num_steps
    f_out.close()
    print("Saved {} steps to {}".format(total_num_steps, output_dataset))

    os._exit(0)

if __name__ == "__main__":
    main()
