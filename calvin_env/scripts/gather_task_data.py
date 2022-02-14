import numpy as np
import os
import cv2
import h5py
import json
import robomimic.envs.env_base as EB
from robomimic.envs.env_calvin import EnvCalvin
import robomimic.utils.obs_utils as ObsUtils
from robosuite.utils.transform_utils import euler2mat, mat2quat

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

    # env = EnvCalvin(target_task, render=True)
    # dummy_spec = dict(
    #     obs=dict(
    #         low_dim=["robot_obs", "scene_obs"],
    #         rgb=[],
    #     ),
    # )
    # ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    store_images = False
    show_images = False

    f_out = h5py.File(output_dataset, "w")
    data_grp = f_out.create_group("data")

    lang_anotations = np.load(os.path.join(dataset_path, 'lang_annotations/auto_lang_ann.npy'), allow_pickle=True)[()]
    start_end_ids = lang_anotations['info']['indx']
    task_annotations = lang_anotations['language']['task']

    num_steps = 0
    ep_num = 1
    for i, task_ann in enumerate(task_annotations):
        if task_ann != target_task:
            continue

        print("Writing episode # {} ...".format(ep_num))
        ep_data_grp = data_grp.create_group('demo_{}'.format(ep_num))

        traj = dict(
            obs=dict(),
            actions=[],
            states=[],
        )

        start, end = start_end_ids[i]
        for timestep in range(start, end + 1):
            data = np.load(os.path.join(dataset_path, 'episode_{}.npz'.format(f'{timestep:07d}')))

            # env.reset(state=dict(
            #     robot_obs=data['robot_obs'],
            #     scene_obs=data['scene_obs'],
            # ))

            for k in data:
                if not store_images:
                    if 'rgb' in k or 'depth' in k:
                        continue

                if k not in traj['obs']:
                    traj['obs'][k] = []
                traj['obs'][k].append(data[k])

            robot_obs = data['robot_obs']
            proprio_dict = dict(
                eef_pos=robot_obs[0:3],
                eef_euler=robot_obs[3:6],
                eef_quat=mat2quat(euler2mat(robot_obs[3:6])),
                gripper_qpos=robot_obs[6:7],
                joint_qpos=robot_obs[7:14],
                prev_gripper_action=robot_obs[14:],
            )
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
                cv2.waitKey(10)

            num_steps += 1

        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        for k in traj["obs"]:
            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
        ep_data_grp.attrs["num_samples"] = len(traj["actions"])

        ep_num += 1

    env_meta = dict(
        type=EB.EnvType.CALVIN_TYPE,
        env_name=target_task,
        env_kwargs=dict(),
    )
    data_grp.attrs["env_args"] = json.dumps(env_meta, indent=4)

    data_grp.attrs["total"] = num_steps
    print("Saved {} steps to {}".format(num_steps, output_dataset))

    f_out.close()

if __name__ == "__main__":
    main()