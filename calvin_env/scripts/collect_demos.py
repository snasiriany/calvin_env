#!/usr/bin/python3
from copy import deepcopy
import logging
import os
import sys
import numpy as np
from calvin_env.utils.input_utils import input2action

import hydra
import pybullet as p
import quaternion  # noqa

from calvin_env.io_utils.data_recorder import DataRecorder

from robomimic.envs.env_calvin import EnvCalvin
import robomimic.utils.obs_utils as ObsUtils

from robosuite.devices import SpaceMouse

from calvin_env.envs.play_table_env import get_env
import calvin_env

from calvin_env.io_utils.data_collection_wrapper import DataCollectionWrapper

import time
import datetime
from glob import glob
import h5py
import json
import argparse

# # A logger for this file
# log = logging.getLogger(__name__)


def collect_human_trajectory(env, device):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    imsize = 250

    env.reset()

    # ID = 2 always corresponds to agentview
    env.render(height=imsize, width=imsize)

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    while True:
        # Get the newest action
        action, grasp = input2action(device)

        # If action is none, then this a reset so we should break
        if action is None:
            break

        action = np.clip(action, -1, 1)

        # Run environment step
        env.step(action)
        env.render(height=imsize, width=imsize)
        time.sleep(0.05)

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env.is_success()["task"]:
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()

def gather_demonstrations_as_hdf5(directory, out_dir, env_info, excluded_episodes=None, meta_data=None):
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    print("Saving hdf5 to", hdf5_path)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print("Processing {} ...".format(ep_directory))
        if (excluded_episodes is not None) and (ep_directory in excluded_episodes):
            # print("\tExcluding this episode!")
            continue

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        action_modes = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                action_modes.append(ai["action_modes"])

        if len(states) == 0:
            continue

        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        ep_data_grp.create_dataset("action_modes", data=np.array(action_modes))

    if num_eps == 0:
        f.close()
        return

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = calvin_env.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    if meta_data is not None:
        for (k, v) in meta_data.items():
            grp.attrs[k] = v

    f.close()

def collect_demos(args):
    config = {
        "env_name": args.task,
    }
    env = EnvCalvin(**config, render=False)
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot_obs", "scene_obs"],
            rgb=[],
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # calvin_env_base_path = os.path.abspath(os.path.join(os.path.dirname(calvin_env.__file__), os.pardir))
    # env = get_env(os.path.join(calvin_env_base_path, '../dataset/task_D_D/training'), show_gui=False)

    device = SpaceMouse(pos_sensitivity=1.0, rot_sensitivity=4.0)

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join('/home/soroushn/tmp', "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    env_info = json.dumps(config)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        type=str,
    )

    args = parser.parse_args()
    collect_demos(args)
