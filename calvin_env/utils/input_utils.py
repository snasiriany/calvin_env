from robosuite.devices import *
import numpy as np

def input2action(device):
    state = device.get_controller_state()
    # Note: Devices output rotation with x and z flipped to account for robots starting with gripper facing down
    #       Also note that the outputted rotation is an absolute rotation, while outputted dpos is delta pos
    #       Raw delta rotations from neutral user input is captured in raw_drotation (roll, pitch, yaw)
    dpos, rotation, raw_drotation, grasp, reset = (
        state["dpos"],
        state["rotation"],
        state["raw_drotation"],
        state["grasp"],
        state["reset"],
    )

    # If we're resetting, immediately return None
    if reset:
        return None, None

    gripper_dof = 1

    # First process the raw drotation
    drotation = raw_drotation[[1, 0, 2]]

    dpos = dpos[[1, 0, 2]]
    dpos[1] = -dpos[1]

    # Flip z
    drotation[2] = -drotation[2]

    drotation[0] = -drotation[0]
    drotation[1] = -drotation[1]

    # Scale rotation for teleoperation (tuned for OSC) -- gains tuned for each device
    drotation = drotation * 1.5 if isinstance(device, Keyboard) else drotation * 50
    dpos = dpos * 75 if isinstance(device, Keyboard) else dpos * 125

    # map 0 to -1 (open) and map 1 to 1 (closed)
    grasp = -1 if grasp else 1

    action = np.concatenate([dpos, drotation, [grasp] * gripper_dof])

    # Return the action and grasp
    return action, grasp