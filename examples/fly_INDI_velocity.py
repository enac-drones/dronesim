"""Script demonstrating the quadrotors following a desired velocity vector (unit vector + magnitude normalized by max vehicle speed in Km/h)
"""
import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from dronesim.control.INDIControl import INDIControl
from dronesim.envs.BaseAviary import DroneModel, Physics

# from dronesim.envs.CtrlAviary import CtrlAviary
from dronesim.envs.VelocityAviary import VelocityAviary
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
        default=5,
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
        default=20,
        type=int,
        help="Duration of the simulation in seconds (default: 5)",
        metavar="",
    )
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = 0.50
    H_STEP = 0.05
    R = 1.5

    AGGR_PHY_STEPS = (
        int(ARGS.simulation_freq_hz / ARGS.control_freq_hz) if ARGS.aggregate else 1
    )
    INIT_XYZS = np.array(
        [
            [
                R * np.cos((i / 6) * 2 * np.pi + np.pi / 2),
                R * np.sin((i / 6) * 2 * np.pi + np.pi / 2),
                H + i * H_STEP,
            ]
            for i in range(ARGS.num_drones)
        ]
    )
    INIT_RPYS = np.array([[0.0, 0.0, 0.0] for i in range(ARGS.num_drones)])

    #### Create the environment
    env = VelocityAviary(
        drone_model=ARGS.num_drones * ARGS.drone,
        num_drones=ARGS.num_drones,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=Physics.PYB,
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

            #### Compute control for the current state #############
            for j in range(ARGS.num_drones):

                # Write your Guidance Vector Field HERE !
                # If you need : obs[str(j)]["state"] include jth vehicle's
                # position [0:3]  quaternion [3:7]   Attitude  VelocityInertialFrame     qpr         motors
                #   X  Y  Z       Q1   Q2   Q3  Q4   R  P   Y    VX     VY    VZ      WX WY WZ       P0 P1 P2 P3
                V_des_unit = np.ones(3) * 0.2
                magnitude = 0.02  # When 1 , the vehicle flies at its max speed in Km/h ! Max speed is in vehicle's .urdf file inside assets folder.
                action[str(j)] = np.array(
                    [V_des_unit[0], V_des_unit[1], V_des_unit[2], magnitude]
                )

        #### Log the simulation ####################################
        # for j in range(ARGS.num_drones):
        #     logger.log(drone=j,
        #                timestamp=i/env.SIM_FREQ,
        #                state= obs[str(j)]["state"],
        #                control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
        #                )

        #### Printout ##############################################
        if i % env.SIM_FREQ == 0:
            env.render()

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    # logger.save()

    #### Plot the simulation results ###########################
    # if ARGS.plot:
    # logger.plot()

# EOF
