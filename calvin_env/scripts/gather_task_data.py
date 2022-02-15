import numpy as np
import os
import h5py
import json

import robomimic.envs.env_base as EB
from robomimic.envs.env_calvin import EnvCalvin
import robomimic.utils.obs_utils as ObsUtils

from calvin_env.scripts.gather_data_utils import gather_trajectory

def main():
    dataset_path = '/home/soroushn/research/calvin/dataset/task_D_D/training'
    target_task = (
        # # easy tasks
        # 'turn_on_led' # good
        # 'turn_on_lightbulb' # nice
        # 'open_drawer' # possible
        # 'move_slider_left' # good
        #
        # # medium tasks
        # 'push_red_block_right' # see other
        #
        # # hard tasks
        # 'stack_block' # nice, and multi-modal
        'place_in_slider' # nice
    )
    output_dataset = os.path.join('/home/soroushn/research/robomimic-dev/datasets/calvin', target_task + '_D_D_ld.hdf5')
    # output_dataset = '/home/soroushn/tmp/test.hdf5'

    env = EnvCalvin(target_task, render=False)
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot_obs", "scene_obs"],
            rgb=[],
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    store_images = False
    show_images = False

    f_out = h5py.File(output_dataset, "w")
    data_grp = f_out.create_group("data")

    lang_anotations = np.load(os.path.join(dataset_path, 'lang_annotations/auto_lang_ann.npy'), allow_pickle=True)[()]
    start_end_ids = lang_anotations['info']['indx']
    task_annotations = lang_anotations['language']['task']

    total_num_steps = 0
    total_euler_warnings = 0
    ep_num = 1
    for i, task_ann in enumerate(task_annotations):
        if task_ann != target_task:
            continue

        print("Writing episode # {} ...".format(ep_num))
        ep_data_grp = data_grp.create_group('demo_{}'.format(ep_num))

        start, end = start_end_ids[i]

        traj, traj_info = gather_trajectory(
            dataset_path, start, end, env,
            store_images=store_images, show_images=show_images,
        )

        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        for k in traj["obs"]:
            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
        ep_data_grp.attrs["num_samples"] = len(traj["actions"])

        total_num_steps += traj_info['num_steps']
        total_euler_warnings += traj_info['num_euler_warnings']

        ep_num += 1
        print("Euler warning percent: {:.2f} %".format(total_euler_warnings / total_num_steps * 100))

    env_meta = dict(
        type=EB.EnvType.CALVIN_TYPE,
        env_name=target_task,
        env_kwargs=dict(),
    )
    data_grp.attrs["env_args"] = json.dumps(env_meta, indent=4)

    data_grp.attrs["total"] = total_num_steps
    f_out.close()
    print("Saved {} steps to {}".format(total_num_steps, output_dataset))

    os._exit(0)

if __name__ == "__main__":
    main()