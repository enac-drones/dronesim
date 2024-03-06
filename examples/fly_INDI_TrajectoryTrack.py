"""Script demonstrating the quadrotor following a MinSnap trajectory generated from 3 gate positions
"""
import argparse
import math
import os
import pdb
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from dronesim.control.INDIControl import INDIControl
from dronesim.envs.BaseAviary import DroneModel, Physics
from dronesim.envs.CtrlAviary import CtrlAviary
from dronesim.utils.Logger import Logger
from dronesim.utils.trajGen import *
from dronesim.utils.utils import str2bool, sync

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl"
    )
    parser.add_argument(
        "--drone",
        default=["robobee"],
        type=list,
        help="Drone model (default: CF2X)",
        metavar="",
        choices=[DroneModel],
    )
    parser.add_argument(
        "--num_drones",
        default=1,
        type=int,
        help="Number of drones (default: 3)",
        metavar="",
    )
    parser.add_argument(
        "--physics",
        default="pyb",
        type=Physics,
        help="Physics updates (default: PYB)",
        metavar="",
        choices=Physics,
    )
    parser.add_argument(
        "--vision",
        default=False,
        type=str2bool,
        help="Whether to use VisionAviary (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--gui",
        default=True,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=False,
        type=str2bool,
        help="Whether to record a video (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--plot",
        default=True,
        type=str2bool,
        help="Whether to plot the simulation results (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--user_debug_gui",
        default=False,
        type=str2bool,
        help="Whether to add debug lines and parameters to the GUI (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--aggregate",
        default=True,
        type=str2bool,
        help="Whether to aggregate physics steps (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--obstacles",
        default=False,
        type=str2bool,
        help="Whether to add obstacles to the environment (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--simulation_freq_hz",
        default=240,
        type=int,
        help="Simulation frequency in Hz (default: 240)",
        metavar="",
    )
    parser.add_argument(
        "--control_freq_hz",
        default=96,
        type=int,
        help="Control frequency in Hz (default: 48)",
        metavar="",
    )
    parser.add_argument(
        "--duration_sec",
        default=10,
        type=int,
        help="Duration of the simulation in seconds (default: 5)",
        metavar="",
    )
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = 0.50
    H_STEP = 0.05
    R = 0.6

    initial_gate = np.array([[-3.0, 0, 2]])
    mid_gate = np.array([0.5, 1, 5])
    final_gate = np.array([3, 0, 2])

    gates = np.vstack((initial_gate, mid_gate, final_gate))

    traj = trajGenerator(gates, max_vel=0.7, gamma=1e6)
    control_frequency = ARGS.control_freq_hz
    t0 = 0
    tf = traj.TS[-1]
    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []
    ax = []
    ay = []
    az = []
    t2 = []
    yaw = []
    ts = np.arange(t0, tf, 1 / control_frequency)
    for ti in ts:
        state = traj.get_des_state(ti)
        x.append(state.pos[0])
        y.append(state.pos[1])
        z.append(state.pos[2])
        vx.append(state.vel[0])
        vy.append(state.vel[1])
        vz.append(state.vel[2])
        ax.append(state.acc[0])
        ay.append(state.acc[1])
        az.append(state.acc[2])
        yaw.append(state.yaw)

    AGGR_PHY_STEPS = (
        int(ARGS.simulation_freq_hz / ARGS.control_freq_hz) if ARGS.aggregate else 1
    )
    INIT_XYZS = initial_gate
    INIT_RPYS = np.array([[0.0, 0.0, 0.0]])

    # plt.plot(ts, vx, label = 'Vx')
    # plt.plot(ts, vy, label = 'Vy')
    # plt.plot(ts, vz, label = 'Vz')
    # plt.legend(loc = 'upper right')
    # plt.show()

    #### Initialize a circular trajectory ######################
    PERIOD = 15
    NUM_WP = len(x)  # ARGS.control_freq_hz*PERIOD

    TARGET_POS = np.zeros((NUM_WP, 3))
    TARGET_VEL = np.zeros((NUM_WP, 3))
    TARGET_RPYS = np.zeros((NUM_WP, 3))
    TARGET_ACC = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = x[i], y[i], z[i]
        TARGET_RPYS[i, :] = 0, 0, yaw[i]
        TARGET_VEL[i, :] = vx[i], vy[i], vz[i]
        TARGET_ACC[i, :] = ax[i], ay[i], az[i]
    wp_counters = np.array(
        [int((i * NUM_WP / 6) % NUM_WP) for i in range(ARGS.num_drones)]
    )

    #### Create the environment
    env = CtrlAviary(
        drone_model=ARGS.drone,
        num_drones=ARGS.num_drones,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=ARGS.physics,
        neighbourhood_radius=10,
        freq=ARGS.simulation_freq_hz,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        gui=ARGS.gui,
        record=ARGS.record_video,
        obstacles=ARGS.obstacles,
        user_debug_gui=ARGS.user_debug_gui,
    )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=int(ARGS.simulation_freq_hz / AGGR_PHY_STEPS),
        num_drones=ARGS.num_drones,
    )

    #### Initialize the controllers ############################
    ctrl = [INDIControl(drone_model=drone) for drone in ARGS.drone]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / ARGS.control_freq_hz))
    action = {str(i): np.array([0.4, 0.4, 0.4, 0.4]) for i in range(ARGS.num_drones)}
    START = time.time()
    for i in range(0, int(ARGS.duration_sec * env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            for j in range(ARGS.num_drones):
                action[str(j)], _, _ = ctrl[j].computeControlFromState(
                    control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                    state=obs[str(j)]["state"],
                    target_pos=TARGET_POS[wp_counters[j]],
                    target_rpy=TARGET_RPYS[wp_counters[j]],
                    target_vel=TARGET_VEL[wp_counters[j]],
                    target_acc=TARGET_ACC[wp_counters[j]],
                )

            # Finish the simulation if arrived to final gate position
            if np.linalg.norm(obs[str(0)]["state"][:3] - final_gate) < 0.3:
                break

            #### Go to the next way point and loop #####################
            for j in range(ARGS.num_drones):
                wp_counters[j] = (
                    wp_counters[j] + 1 if wp_counters[j] < (NUM_WP - 1) else 0
                )

        #### Log the simulation ####################################
        for j in range(ARGS.num_drones):
            logger.log(
                drone=j,
                timestamp=i / env.SIM_FREQ,
                state=obs[str(j)]["state"],
                control=np.hstack(
                    [
                        TARGET_POS[wp_counters[j], 0:2],
                        INIT_XYZS[j, 2],
                        INIT_RPYS[j, :],
                        np.zeros(6),
                    ]
                ),
            )

        #### Printout ##############################################
        if i % env.SIM_FREQ == 0:
            env.render()
            #### Print matrices with the images captured by each drone #
            if ARGS.vision:
                for j in range(ARGS.num_drones):
                    print(
                        obs[str(j)]["rgb"].shape,
                        np.average(obs[str(j)]["rgb"]),
                        obs[str(j)]["dep"].shape,
                        np.average(obs[str(j)]["dep"]),
                        obs[str(j)]["seg"].shape,
                        np.average(obs[str(j)]["seg"]),
                    )

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    # logger.save()

    #### Plot the simulation results ###########################
    # if ARGS.plot:
    logger.plot()

x_flown = logger.states[0][0]
y_flown = logger.states[0][1]
z_flown = logger.states[0][2]


fig = plt.figure()
ax = plt.axes(projection="3d")
ax.set_title("Real and planned trajectory")
ax.plot3D(x, y, z, "gray", label="Minimum snap")
ax.plot3D(x_flown, y_flown, z_flown, "red", label="Flown trajectory")
for gate in gates:
    ax.plot3D([gate[0]], [gate[1]], [gate[2]], "o")
ax.legend(loc="upper right")
plt.show()
