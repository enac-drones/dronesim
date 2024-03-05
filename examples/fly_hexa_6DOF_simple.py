"""Script demonstrating the 6DOF hexacopter navigating
   with zero roll-pitch-yaw attitude using only the
   lateral forces.

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
from dronesim.utils.utils import str2bool, sync

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl"
    )
    parser.add_argument(
        "--drone",
        default=["hexa_6DOF_simple"],
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
        default=12,
        type=int,
        help="Duration of the simulation in seconds (default: 5)",
        metavar="",
    )
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = 0.50
    H_STEP = 0.05
    R = 1.2
    INIT_XYZS = np.array(
        [
            [
                R * np.cos((i / 6) * 2 * np.pi + np.pi / 2),
                R * np.sin((i / 6) * 2 * np.pi + np.pi / 2) - R,
                H + i * H_STEP,
            ]
            for i in range(ARGS.num_drones)
        ]
    )
    INIT_RPYS = np.array(
        [[0, 0, i * (np.pi / 20.0) / ARGS.num_drones] for i in range(ARGS.num_drones)]
    )
    AGGR_PHY_STEPS = (
        int(ARGS.simulation_freq_hz / ARGS.control_freq_hz) if ARGS.aggregate else 1
    )

    INIT_RPYS = np.array(
        [[0, 0, 0 * i * (np.pi / 2) / ARGS.num_drones] for i in range(ARGS.num_drones)]
    )
    INIT_XYZS = np.array([[0.0, 0.0, 0.5]])

    ## Hover ###
    INIT_XYZS = np.array([[0.0, 0.0, 0.6]])
    INIT_RPYS = np.array([[0.0, 0.0 * 3.14 / 180.0, 0.0 * 3.14 / 180.0]])
    INIT_VELS = np.array([[0.0, 0.0, 0.0]])

    #### Initialize a circular trajectory ######################
    PERIOD = 15
    NUM_WP = ARGS.control_freq_hz * PERIOD

    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = (
            R * np.cos((i / NUM_WP) * (4 * np.pi) + np.pi / 2) + INIT_XYZS[0, 0],
            R * np.sin((i / NUM_WP) * (4 * np.pi) + np.pi / 2) - R + INIT_XYZS[0, 1],
            0,
        )
    wp_counters = np.array(
        [int((i * NUM_WP / 6) % NUM_WP) for i in range(ARGS.num_drones)]
    )

    TARGET_RPYS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        TARGET_RPYS[i, :] = [0.0, 0.0, 0.0]  # 0.4+(i*1./200)]

    #### Create the environment ##
    env = CtrlAviary(
        drone_model=ARGS.drone,
        num_drones=ARGS.num_drones,
        initial_xyzs=INIT_XYZS,
        initial_vels=INIT_VELS,
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
    action = {
        str(i): np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) for i in range(ARGS.num_drones)
    }
    # action = {'0': np.array([0.5,0.5,0.5,0.5,0.5,0.5])}# , '1': np.array([0.5,0.5,0.5,0.5])}
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
                    target_pos=np.hstack(
                        [TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]
                    ),
                    # target_rpy=INIT_RPYS[j, :],
                    # target_pos=np.array([0.0, 0.0, 1.0]),#INIT_XYZS[j,:],
                    # target_rpy=np.array([0., 0., 0.]),
                    target_rpy=TARGET_RPYS[wp_counters[j]],
                    # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                )

            #### Go to the next way point and loop #####################
            for j in range(ARGS.num_drones):
                wp_counters[j] = (
                    wp_counters[j] + 1 if wp_counters[j] < (NUM_WP - 1) else 0
                )

        #### Camera View follows the vehicle #######################
        # if i%(CTRL_EVERY_N_STEPS*1) == 0:
        #     x,y,z, = obs[str(0)]["state"][0:3]
        #     p.resetDebugVisualizerCamera(cameraDistance=2,
        #                                  cameraYaw=-30,
        #                                  cameraPitch=-30,
        #                                  cameraTargetPosition=[x, y, z],
        #                                  physicsClientId=PYB_CLIENT
        #                                  )

        #### Log the simulation ####################################
        for j in range(ARGS.num_drones):
            logger.log(
                drone=j,
                timestamp=i / env.SIM_FREQ,
                state=obs[str(j)]["state"][:20],
                control=np.hstack(
                    [
                        TARGET_POS[wp_counters[j], 0:2],
                        INIT_XYZS[j, 2],
                        INIT_RPYS[j, :],
                        np.zeros(6),
                    ]
                )
                # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
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

        ### Break the simulation if we are close to ground
        # if z < 0.3:
        #     break
    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    # logger.save()

    #### Plot the simulation results ###########################
    if ARGS.plot:
        logger.plot()
