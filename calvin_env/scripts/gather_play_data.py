import numpy as np
import os
import cv2
import h5py
import json
import robomimic.envs.env_base as EB

def main():
    dataset_path = '/home/soroushn/research/calvin/dataset/task_D_D/training'
    # output_dataset = '/home/soroushn/tmp/test.hdf5'
    # output_dataset = '/home/soroushn/research/robomimic-dev/datasets/calvin/task_D_D_ld.hdf5'
    output_dataset = '/home/soroushn/research/robomimic-dev/datasets/calvin/test.hdf5'

    max_steps = 1000000 #25000
    store_images = False
    show_images = False

    if show_images:
        assert max_steps < 50000

    f_out = h5py.File(output_dataset, "w")
    data_grp = f_out.create_group("data")

    # ep_nums = np.array([int(ep_name[8:-4]) for ep_name in os.listdir(dataset_path) if ep_name.startswith('episode')])
    ep_start_end_ids = np.load(os.path.join(dataset_path, 'ep_start_end_ids.npy'))

    num_steps = 0
    ep_num = 1
    for (start, end) in ep_start_end_ids:
        print("Writing episode # {} ...".format(ep_num))
        ep_data_grp = data_grp.create_group('demo_{}'.format(ep_num))

        traj = dict(
            obs=dict(
                robot_obs=[],
                scene_obs=[],
            ),
            actions=[],
        )

        if store_images:
            traj['obs']['rgb_static'] = []
            traj['obs']['rgb_gripper'] = []

        for timestep in range(start, end + 1):
            data = np.load(os.path.join(dataset_path, 'episode_{}.npz'.format(f'{timestep:07d}')))

            for k in traj['obs']:
                traj['obs'][k].append(data[k])
            traj['actions'].append(data['rel_actions'])

            if show_images:
                im = data['rgb_static'][:,:,::-1]
                cv2.imshow('play data', im)
                cv2.waitKey(1)

            num_steps += 1
            if num_steps >= max_steps:
                break

        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        for k in traj["obs"]:
            ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
        ep_data_grp.attrs["num_samples"] = len(traj["actions"])

        ep_num += 1
        if num_steps >= max_steps:
            break

    env_meta = dict(
        type=EB.EnvType.CALVIN_TYPE,
        env_name="play",
        env_kwargs=dict(),
    )
    data_grp.attrs["env_args"] = json.dumps(env_meta, indent=4)

    data_grp.attrs["total"] = num_steps
    print("Saved {} steps".format(num_steps))

    f_out.close()

if __name__ == "__main__":
    main()