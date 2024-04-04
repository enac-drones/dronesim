import collections
import os
import time
import xml.etree.ElementTree as etxml
from datetime import datetime
from enum import Enum
from sys import platform

import gym

# import pkgutil
# egl = pkgutil.get_loader('eglRenderer')
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

from dronesim.database.propeller_database import *

# for fixed-wing vehicle's physics
from dronesim.utils.utils import (
    MultiDimensionalContinuousPerlinNoise,
    R_aero_to_body,
    calculate_propeller_forces_moments,
)


class DroneModel(Enum):
    """Drone models enumeration class."""

    CF2X = "cf2x"  # Bitcraze Craziflie 2.0 in the X configuration
    CF2P = "cf2p"  # Bitcraze Craziflie 2.0 in the + configuration
    HB = "hb"  # Generic quadrotor (with AscTec Hummingbird inertial properties)
    TELLO = "tello"  # Tello quadrotor
    TAILSITTER = "darkO"


################################################################################


class Physics(Enum):
    """Physics implementations enumeration class."""

    PYB = "pyb"  # Base PyBullet physics update
    DYN = "dyn"  # Update with an explicit model of the dynamics
    PYB_GND = "pyb_gnd"  # PyBullet physics update with ground effect
    PYB_DRAG = "pyb_drag"  # PyBullet physics update with drag
    PYB_DW = "pyb_dw"  # PyBullet physics update with downwash
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw"  # PyBullet physics update with ground effect, drag, and downwash


################################################################################


class ImageType(Enum):
    """Camera capture image type enumeration class."""

    RGB = 0  # Red, green, blue (and alpha)
    DEP = 1  # Depth
    SEG = 2  # Segmentation by object id
    BW = 3  # Black and white


################################################################################

from dataclasses import dataclass


@dataclass
class Drone:
    TYPE: str
    M: float  # int - str - tuple
    L: float
    THRUST2WEIGHT_RATIO: float
    J: float
    J_INV: float
    KF: float
    KM: float
    COLLISION_H: float
    COLLISION_R: float
    COLLISION_Z_OFFSET: float
    MAX_SPEED_KMH: float
    GND_EFF_COEFF: float
    PROP_RADIUS: float
    DRAG_COEFF: float
    DW_COEFF_1: float
    DW_COEFF_2: float
    DW_COEFF_3: float
    PWM2RPM_SCALE: float  # list[floats] #FIXME ?? These are list of floats right now...
    PWM2RPM_CONST: float  # list[floats] #FIXME ??
    INDI_ACTUATOR_NR: int
    INDI_OUTPUT_NR: int
    G1: float  # list[floats] #FIXME
    MIN_PWM: float
    MAX_PWM: float


# def __post_init__(self):
#   self.M = float(self.M) # int - str - tuple
#   self.L = float(self.L)
#   self.THRUST2WEIGHT_RATIO = float(self.THRUST2WEIGHT_RATIO)
#   self.J = float(self.J)
#   self.J_INV = float(self.J_INV)
#   self.KF = float(self.KF)
#   self.KM = float(self.KM)
#   self.COLLISION_H = float(self.COLLISION_H)
#   self.COLLISION_R = float(self.)
#   self.COLLISION_Z_OFFSET = float(self.)
#   self.MAX_SPEED_KMH = float(self.)
#   self.GND_EFF_COEFF = float(self.)
#   self.PROP_RADIUS = float(self.)
#   self.DRAG_COEFF = float(self.)
#   self.DW_COEFF_1 = float(self.)
#   self.DW_COEFF_2 = float(self.)
#   self.DW_COEFF_3 = float(self.)
#   # self.name = str(self.name, 'utf-8')
#   # self.linkName = str(self.linkName, 'utf-8')

################################################################################


class BaseAviary(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    metadata = {"render.modes": ["human"]}

    ################################################################################

    def __init__(
        self,
        drone_model: list = ["tello"],  # DroneModel=DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_vels=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        freq: int = 240,
        aggregate_phy_steps: int = 1,
        gui=True,
        record=False,
        obstacles=False,
        user_debug_gui=True,
        vision_attributes=False,
        dynamics_attributes=False,
    ):
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        vision_attributes : bool, optional
            Whether to allocate the attributes needed by vision-based aviary subclasses.
        dynamics_attributes : bool, optional
            Whether to allocate the attributes needed by subclasses accepting thrust and torques inputs.

        """
        #### Constants #############################################
        self.G = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        self.SIM_FREQ = freq
        self.TIMESTEP = 1.0 / self.SIM_FREQ
        self.AGGR_PHY_STEPS = aggregate_phy_steps
        #### Parameters ############################################
        self.NUM_DRONES = num_drones
        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = physics
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.URDF = [drone + ".urdf" for drone in drone_model]
        #### Load the drone properties from the .urdf file #########
        # self.M, \
        # self.L, \
        # self.THRUST2WEIGHT_RATIO, \
        # self.J, \
        # self.J_INV, \
        # self.KF, \
        # self.KM, \
        # self.COLLISION_H,\
        # self.COLLISION_R, \
        # self.COLLISION_Z_OFFSET, \
        # self.MAX_SPEED_KMH, \
        # self.GND_EFF_COEFF, \
        # self.PROP_RADIUS, \
        # self.DRAG_COEFF, \
        # self.DW_COEFF_1, \
        # self.DW_COEFF_2, \
        # self.DW_COEFF_3 = self._parseURDFParameters(self.URDF)
        # print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
        #     self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        self.drones = [Drone(*self._parseURDFParameters(drone)) for drone in self.URDF]

        for drone, urdf in zip(self.drones, self.URDF):
            if "fixed_wing" in drone.TYPE:
                self._parseURDFFixedwingParameters(drone, urdf)

        #### Compute constants #####################################
        #        self.GRAVITY = self.G*self.M
        # self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        #        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        # self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        # if self.DRONE_MODEL == DroneModel.CF2X:
        #     self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        # elif self.DRONE_MODEL in [DroneModel.CF2P, DroneModel.HB]:
        #     self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        # self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        # self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        #### Create attributes for vision tasks ####################
        self.VISION_ATTR = vision_attributes
        if self.VISION_ATTR:
            self.IMG_RES = np.array([64, 48])
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.SIM_FREQ / self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros(
                ((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4))
            )
            self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ % self.AGGR_PHY_STEPS != 0:
                print(
                    "[ERROR] in BaseAviary.__init__(), aggregate_phy_steps incompatible with the desired video capture frame rate ({:f}Hz)".format(
                        self.IMG_FRAME_PER_SEC
                    )
                )
                exit()
            if self.RECORD:
                self.ONBOARD_IMG_PATH = (
                    os.path.dirname(os.path.abspath(__file__))
                    + "/../../files/videos/onboard-"
                    + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
                    + "/"
                )
                os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
        #### Create attributes for dynamics control inputs #########
        self.DYNAMICS_ATTR = dynamics_attributes
        if self.DYNAMICS_ATTR:
            if self.DRONE_MODEL == DroneModel.CF2X:
                self.A = np.array(
                    [
                        [1, 1, 1, 1],
                        [
                            1 / np.sqrt(2),
                            1 / np.sqrt(2),
                            -1 / np.sqrt(2),
                            -1 / np.sqrt(2),
                        ],
                        [
                            -1 / np.sqrt(2),
                            1 / np.sqrt(2),
                            1 / np.sqrt(2),
                            -1 / np.sqrt(2),
                        ],
                        [-1, 1, -1, 1],
                    ]
                )
            elif self.DRONE_MODEL in [DroneModel.CF2P, DroneModel.HB]:
                self.A = np.array(
                    [[1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1]]
                )
            self.INV_A = np.linalg.inv(self.A)
            self.B_COEFF = np.array(
                [
                    1 / self.KF,
                    1 / (self.KF * self.L),
                    1 / (self.KF * self.L),
                    1 / self.KM,
                ]
            )
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = p.connect(p.GUI)  # p.connect(p.GUI, options="--opengl2")
            for i in [
                p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
            ]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(
                cameraDistance=6,
                cameraYaw=-30,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self.CLIENT,
            )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)

            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

            if self.USER_DEBUG:
                #### Add input sliders to the GUI ##########################
                self.SLIDERS = -1 * np.ones(4)
                for i in range(4):
                    self.SLIDERS[i] = p.addUserDebugParameter(
                        "Propeller " + str(i) + " RPM",
                        0,
                        self.MAX_RPM,
                        self.HOVER_RPM,
                        physicsClientId=self.CLIENT,
                    )
                self.INPUT_SWITCH = p.addUserDebugParameter(
                    "Use GUI RPM", 9999, -1, 0, physicsClientId=self.CLIENT
                )
        else:
            #### Without debug GUI #####################################
            self.CLIENT = p.connect(p.DIRECT)
            #### Uncomment the following line to use EGL Render Plugin #
            #### Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)
            if self.RECORD:
                #### Set the camera parameters to save frames in DIRECT mode
                self.VID_WIDTH = int(640)
                self.VID_HEIGHT = int(480)
                self.FRAME_PER_SEC = 24
                self.CAPTURE_FREQ = int(self.SIM_FREQ / self.FRAME_PER_SEC)
                self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(
                    distance=3,
                    yaw=-30,
                    pitch=-30,
                    roll=0,
                    cameraTargetPosition=[0, 0, 0],
                    upAxisIndex=2,
                    physicsClientId=self.CLIENT,
                )
                self.CAM_PRO = p.computeProjectionMatrixFOV(
                    fov=60.0,
                    aspect=self.VID_WIDTH / self.VID_HEIGHT,
                    nearVal=0.1,
                    farVal=1000.0,
                )
        #### Set initial poses #####################################
        if initial_xyzs is None:
            self.INIT_XYZS = (
                np.vstack(
                    [
                        np.array([x * 4 * self.L for x in range(self.NUM_DRONES)]),
                        np.array([y * 4 * self.L for y in range(self.NUM_DRONES)]),
                        np.ones(self.NUM_DRONES)
                        * (self.COLLISION_H / 2 - self.COLLISION_Z_OFFSET + 0.1),
                    ]
                )
                .transpose()
                .reshape(self.NUM_DRONES, 3)
            )
        elif np.array(initial_xyzs).shape == (self.NUM_DRONES, 3):
            self.INIT_XYZS = initial_xyzs
        else:
            print(
                "[ERROR] invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)"
            )

        self.INIT_VELS = initial_vels
        print("INITTTT", self.INIT_VELS)

        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
        elif np.array(initial_rpys).shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            print(
                "[ERROR] invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)"
            )
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()

        self.perlin_noise = MultiDimensionalContinuousPerlinNoise(
            dimensions=3, period=100
        )

    ################################################################################

    def reset(self):
        """Resets the environment.

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.

        """
        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        return self._computeObs()

    ################################################################################

    def step(self, action):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current epoisode is over, check the specific implementation of `_computeDone()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter % self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(
                width=self.VID_WIDTH,
                height=self.VID_HEIGHT,
                shadow=1,
                viewMatrix=self.CAM_VIEW,
                projectionMatrix=self.CAM_PRO,
                renderer=p.ER_TINY_RENDERER,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=self.CLIENT,
            )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), "RGBA")).save(
                self.IMG_PATH + "frame_" + str(self.FRAME_NUM) + ".png"
            )
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(
                self.INPUT_SWITCH, physicsClientId=self.CLIENT
            )
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(
                    int(self.SLIDERS[i]), physicsClientId=self.CLIENT
                )
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter % (self.SIM_FREQ / 2) == 0:
                self.GUI_INPUT_TEXT = [
                    p.addUserDebugText(
                        "Using GUI RPM",
                        textPosition=[0, 0, 0],
                        textColorRGB=[1, 0, 0],
                        lifeTime=1,
                        textSize=2,
                        parentObjectUniqueId=self.DRONE_IDS[i],
                        parentLinkIndex=-1,
                        replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                        physicsClientId=self.CLIENT,
                    )
                    for i in range(self.NUM_DRONES)
                ]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            # self._saveLastAction(action) # FIXME
            # clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
            clipped_action = self._preprocessAction(action)

        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.AGGR_PHY_STEPS):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.AGGR_PHY_STEPS > 1 and self.PHYSICS in [
                Physics.DYN,
                Physics.PYB_GND,
                Physics.PYB_DRAG,
                Physics.PYB_DW,
                Physics.PYB_GND_DRAG_DW,
            ]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range(self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[str(i)], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward()
        done = self._computeDone()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.AGGR_PHY_STEPS)
        return obs, reward, done, info

    ################################################################################

    def render(self, mode="human", close=False):
        """Prints a textual output of the environment.

        Parameters
        ----------
        mode : str, optional
            Unused.
        close : bool, optional
            Unused.

        """
        # self.CLIENT = p.connect(p.GUI)
        if self.first_render_call and not self.GUI:
            print(
                "[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface"
            )
            self.first_render_call = False
        print(
            "\n[INFO] BaseAviary.render() ——— it {:04d}".format(self.step_counter),
            "——— wall-clock time {:.1f}s,".format(time.time() - self.RESET_TIME),
            "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(
                self.step_counter * self.TIMESTEP,
                self.SIM_FREQ,
                (self.step_counter * self.TIMESTEP) / (time.time() - self.RESET_TIME),
            ),
        )
        for i in range(self.NUM_DRONES):
            print(
                "[INFO] BaseAviary.render() ——— drone {:d}".format(i),
                "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(
                    self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]
                ),
                "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(
                    self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]
                ),
                "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(
                    self.rpy[i, 0] * self.RAD2DEG,
                    self.rpy[i, 1] * self.RAD2DEG,
                    self.rpy[i, 2] * self.RAD2DEG,
                ),
                "——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".format(
                    self.ang_v[i, 0], self.ang_v[i, 1], self.ang_v[i, 2]
                ),
            )

    ################################################################################

    def close(self):
        """Terminates the environment."""
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.CLIENT)
        p.disconnect(physicsClientId=self.CLIENT)

    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.

        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT

    ################################################################################

    def getDroneIds(self):
        """Return the Drone Ids.

        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.

        """
        return self.DRONE_IDS

    ################################################################################

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1 * np.ones(self.NUM_DRONES)
        self.Y_AX = -1 * np.ones(self.NUM_DRONES)
        self.Z_AX = -1 * np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1 * np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM = False
        self.last_input_switch = 0
        self.last_action = -1 * np.ones((self.NUM_DRONES, 4))

        self.last_clipped_action = {
            str(i): np.zeros(self.drones[i].INDI_ACTUATOR_NR)
            for i in range(self.NUM_DRONES)
        }  # Action is a dictionary in order to keep heterogeneous control action of multi-vehicle scenarios. np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.CLIENT
        )
        #### Load ground plane, drone and obstacles models #########
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        self.DRONE_IDS = np.array(
            [
                p.loadURDF(
                    os.path.dirname(os.path.abspath(__file__))
                    + "/../assets/"
                    + self.URDF[i],
                    self.INIT_XYZS[i, :],
                    p.getQuaternionFromEuler(self.INIT_RPYS[i, :]),
                    flags=p.URDF_USE_INERTIA_FROM_FILE,
                    physicsClientId=self.CLIENT,
                )
                for i in range(self.NUM_DRONES)
            ]
        )
        for i in range(self.NUM_DRONES):
            #### If the vehicle initialized with flight velocity : e.g. Fixed-wing configuration
            if self.INIT_VELS is not None:
                if (
                    self.INIT_VELS[i] is not None
                ):  # FIXME Ugly :( use something else ? instance
                    p.resetBaseVelocity(
                        self.DRONE_IDS[i],
                        linearVelocity=self.INIT_VELS[i],
                        physicsClientId=self.CLIENT,
                    )
            #### Show the frame of reference of the drone, note that ###
            #### It severly slows down the GUI #########################
            if self.GUI and self.USER_DEBUG:
                self._showDroneLocalAxes(i)
            #### Disable collisions between drones' and the ground plane
            #### E.g., to start a drone at [0,0,0] #####################
            # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES:
            self._addObstacles()

    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range(self.NUM_DRONES):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(
                self.DRONE_IDS[i], physicsClientId=self.CLIENT
            )

    ################################################################################

    def _startVideoRecording(self):
        """Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.
        The video is saved under folder `files/videos`.

        """
        if self.RECORD and self.GUI:
            self.VIDEO_ID = p.startStateLogging(
                loggingType=p.STATE_LOGGING_VIDEO_MP4,
                fileName=os.path.dirname(os.path.abspath(__file__))
                + "/../../files/videos/video-"
                + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
                + ".mp4",
                physicsClientId=self.CLIENT,
            )
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = (
                os.path.dirname(os.path.abspath(__file__))
                + "/../../files/videos/video-"
                + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
                + "/"
            )
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)

    ################################################################################

    def _getDroneStateVector(self, nth_drone):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.

        """
        state = np.hstack(
            [
                self.pos[nth_drone, :],
                self.quat[nth_drone, :],
                self.rpy[nth_drone, :],
                self.vel[nth_drone, :],
                self.ang_v[nth_drone, :],
                self.last_clipped_action[str(nth_drone)],
            ]
        )
        return state  # state.reshape(20,) FIXME

    ################################################################################

    def _getDroneImages(self, nth_drone, segmentation: bool = True):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        segmentation : bool, optional
            Whehter to compute the compute the segmentation mask.
            It affects performance.

        Returns
        -------
        ndarray
            (h, w, 4)-shaped array of uint8's containing the RBG(A) image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the depth image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the segmentation image captured from the n-th drone's POV.

        """
        if self.IMG_RES is None:
            print(
                "[ERROR] in BaseAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])"
            )
            exit()
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(
            3, 3
        )
        #### Set target point, camera view and projection matrices #
        target = np.dot(rot_mat, np.array([1000, 0, 0])) + np.array(
            self.pos[nth_drone, :]
        )
        DRONE_CAM_VIEW = p.computeViewMatrix(
            cameraEyePosition=self.pos[nth_drone, :] + np.array([0, 0, self.L]),
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
            physicsClientId=self.CLIENT,
        )
        DRONE_CAM_PRO = p.computeProjectionMatrixFOV(
            fov=60.0, aspect=1.0, nearVal=self.L, farVal=1000.0
        )
        SEG_FLAG = (
            p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            if segmentation
            else p.ER_NO_SEGMENTATION_MASK
        )
        [w, h, rgb, dep, seg] = p.getCameraImage(
            width=self.IMG_RES[0],
            height=self.IMG_RES[1],
            shadow=1,
            viewMatrix=DRONE_CAM_VIEW,
            projectionMatrix=DRONE_CAM_PRO,
            flags=SEG_FLAG,
            physicsClientId=self.CLIENT,
        )
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg

    ################################################################################

    def _exportImage(
        self, img_type: ImageType, img_input, path: str, frame_num: int = 0
    ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        img_type : ImageType
            The image type: RGB(A), depth, segmentation, or B&W (from RGB).
        img_input : ndarray
            (h, w, 4)-shaped array of uint8's for RBG(A) or B&W images.
            (h, w)-shaped array of uint8's for depth or segmentation images.
        path : str
            Path where to save the output as PNG.
        fram_num: int, optional
            Frame number to append to the PNG's filename.

        """
        if img_type == ImageType.RGB:
            (Image.fromarray(img_input.astype("uint8"), "RGBA")).save(
                path + "frame_" + str(frame_num) + ".png"
            )
        elif img_type == ImageType.DEP:
            temp = (
                (img_input - np.min(img_input))
                * 255
                / (np.max(img_input) - np.min(img_input))
            ).astype("uint8")
        elif img_type == ImageType.SEG:
            temp = (
                (img_input - np.min(img_input))
                * 255
                / (np.max(img_input) - np.min(img_input))
            ).astype("uint8")
        elif img_type == ImageType.BW:
            temp = (np.sum(img_input[:, :, 0:2], axis=2) / 3).astype("uint8")
        else:
            print("[ERROR] in BaseAviary._exportImage(), unknown ImageType")
            exit()
        if img_type != ImageType.RGB:
            (Image.fromarray(temp)).save(path + "frame_" + str(frame_num) + ".png")

    ################################################################################

    def _getAdjacencyMatrix(self):
        """Computes the adjacency matrix of a multi-drone system.

        Attribute NEIGHBOURHOOD_RADIUS is used to determine neighboring relationships.

        Returns
        -------
        ndarray
            (NUM_DRONES, NUM_DRONES)-shaped array of 0's and 1's representing the adjacency matrix
            of the system: adj_mat[i,j] == 1 if (i, j) are neighbors; == 0 otherwise.

        """
        adjacency_mat = np.identity(self.NUM_DRONES)
        for i in range(self.NUM_DRONES - 1):
            for j in range(self.NUM_DRONES - i - 1):
                if (
                    np.linalg.norm(self.pos[i, :] - self.pos[j + i + 1, :])
                    < self.NEIGHBOURHOOD_RADIUS
                ):
                    adjacency_mat[i, j + i + 1] = adjacency_mat[j + i + 1, i] = 1
        return adjacency_mat

    ################################################################################

    def _physics(self, cmd, nth_drone):  # was rpm
        """Base PyBullet physics implementation.

        Parameters
        ----------
        cmd : ndarray
            (4)-shaped array of ints containing the command values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        """
        if "quad" in self.drones[nth_drone].TYPE:
            self._quad_copter_physics(cmd, nth_drone)
        elif "morphing_hexa" in self.drones[nth_drone].TYPE:
            self._morphing_hexa_physics(cmd, nth_drone)
        elif "fixed_wing" in self.drones[nth_drone].TYPE:
            self._fixed_wing_physics(cmd, nth_drone)
        elif "tail_sitter" in self.drones[nth_drone].TYPE:
            self._tail_sitter_physics(cmd, nth_drone)
        elif "coaxial_birotor" in self.drones[nth_drone].TYPE:
            self._coaxial_birotor_physics(cmd, nth_drone)
        else:
            rpm = (
                self.drones[nth_drone].PWM2RPM_SCALE * cmd
                + self.drones[nth_drone].PWM2RPM_CONST
            )
            # rpm = 20000.*cmd
            forces = np.array(rpm**2) * self.drones[nth_drone].KF
            torques = np.array(rpm**2) * self.drones[nth_drone].KM
            z_torque = -torques[0] + torques[1] - torques[2] + torques[3]
            for i in range(4):
                p.applyExternalForce(
                    self.DRONE_IDS[nth_drone],
                    i,
                    forceObj=[0, 0, forces[i]],
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                    physicsClientId=self.CLIENT,
                )
            p.applyExternalTorque(
                self.DRONE_IDS[nth_drone],
                4,  # FIXME This should be -1 to define the body, here it only works as there is an additional link called center of mass in the URDF file! will not work for tohers !
                torqueObj=[0, 0, z_torque],
                flags=p.LINK_FRAME,
                physicsClientId=self.CLIENT,
            )

    ################################################################################

    def _fixed_wing_physics(self, cmd, nth_drone):
        """Fixed wing aerodynamics"""

        #### Current state in Inertial Frame #############################
        pos = self.pos[nth_drone, :]
        quat = self.quat[nth_drone, :]
        rpy = self.rpy[nth_drone, :]
        vel = self.vel[nth_drone, :]
        rvel = self.ang_v[nth_drone, :]

        # Rotate inertial velocity to body frame
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        vel_b = R.T.dot(vel)
        rvel_b = R.T.dot(rvel)
        gamma = np.arcsin(vel[2] / np.linalg.norm(vel))
        alpha = (
            -rpy[1] - gamma
        )  # Assuming no WIND ! and using pitch angle as Angle of Attack
        beta = np.arctan(vel_b[1] / vel_b[0])
        V_air = (
            vel_b[0] if vel_b[0] > 0.0 else 0.0
        )  # np.linalg.norm(vel) # FIXME get real airspeed along x axis with wind
        rho = 1.225  # Get this as a function of altitude... pos[2]
        Pdyn = 0.5 * rho * V_air * V_air

        def get_f_aero_coef(alpha, beta, rvel, Uctrl, drone):
            """
            return aero coefficients for forces
            """
            d_alpha = alpha - drone.alpha0
            nrvel = (
                rvel * np.array([drone.Bref, drone.Cref, drone.Bref]) / 2 / drone.Vref
            )  # FIXME va??
            CL = (
                drone.CL0
                + drone.CL_alpha * d_alpha
                + drone.CL_beta * beta
                + np.dot(drone.CL_omega, nrvel)
                + np.dot(drone.CL_ctrl, Uctrl)
            )
            CD = (
                drone.CD0
                + drone.CD_k1 * CL
                + drone.CD_k2 * (CL**2)
                + np.dot(drone.CD_ctrl, Uctrl)
            )
            CY = (
                drone.CY_alpha * d_alpha
                + drone.CY_beta * beta
                + np.dot(drone.CY_omega, nrvel)
                + np.dot(drone.CY_ctrl, Uctrl)
            )
            return [CL, CY, CD]

        def get_f_aero_body(va, alpha, beta, rvel, Uctrl, drone, Pdyn):
            """
            return aerodynamic forces in body frame
            """
            CL, CY, CD = get_f_aero_coef(alpha, beta, rvel, Uctrl, drone)
            F_aero_body = (
                Pdyn * drone.Sref * np.array([-CD, -CY, CL])
            )  # np.dot(R_aero_to_body(alpha, beta), [-CD, -CY, CL])#[-CD, CY, -CL]) # [-CD, CY, CL] #
            return F_aero_body

        def get_m_aero_coef(alpha, beta, rvel, Uctrl, drone):
            d_alpha = alpha - drone.alpha0
            nrvel = (
                rvel * np.array([drone.Bref, drone.Cref, drone.Bref]) / 2 / drone.Vref
            )
            Cl = (
                drone.Cl_alpha * d_alpha
                + drone.Cl_beta * beta
                + np.dot(drone.Cl_omega, nrvel)
                + np.dot(drone.Cl_ctrl, Uctrl)
            )
            Cm = (
                drone.Cm0
                + drone.Cm_alpha * d_alpha
                + drone.Cm_beta * beta
                + np.dot(drone.Cm_omega, nrvel)
                + np.dot(drone.Cm_ctrl, Uctrl)
            )
            Cn = (
                drone.Cn_alpha * d_alpha
                + drone.Cn_beta * beta
                + np.dot(drone.Cn_omega, nrvel)
                + np.dot(drone.Cn_ctrl, Uctrl)
            )
            return Cl, Cm, Cn

        def get_m_aero_body(va, alpha, beta, rvel, Uctrl, drone, Pdyn):
            Cl, Cm, Cn = get_m_aero_coef(alpha, beta, rvel, Uctrl, drone)
            return (
                Pdyn
                * drone.Sref
                * np.array([-Cl * drone.Bref, Cm * drone.Cref, -Cn * drone.Bref])
            )

        F_aero_body = get_f_aero_body(
            V_air, alpha, beta, rvel_b, cmd, self.drones[nth_drone], Pdyn
        )
        M_aero_body = get_m_aero_body(
            V_air, alpha, beta, rvel_b, cmd, self.drones[nth_drone], Pdyn
        )

        rpm = (
            self.drones[nth_drone].PWM2RPM_SCALE * cmd
            + self.drones[nth_drone].PWM2RPM_CONST
        )
        prop_force = np.array(rpm**2) * self.drones[nth_drone].KF

        # F_aero_body = np.zeros(3)
        # M_aero_body = np.zeros(3)
        print(F_aero_body, M_aero_body)
        print(
            f"Airspeed :{V_air:.2f}, VX : {vel[0]:.2f}, VZ : {vel[2]:.2f}, Alpha : {alpha:.3f}, Beta : {beta:.3f} Gamma : {gamma:.3f} Prop Foce : {prop_force[0]:.2f} , {prop_force[1]:.2f}"
        )

        p.applyExternalForce(
            self.DRONE_IDS[nth_drone],
            -1,  # to base link : fuselage
            forceObj=[F_aero_body[0], F_aero_body[1], F_aero_body[2]],
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self.CLIENT,
        )
        for i in range(2):
            p.applyExternalForce(
                self.DRONE_IDS[nth_drone],
                i,  # to propeller axis
                forceObj=[prop_force[i], 0, 0],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.CLIENT,
            )

        p.applyExternalTorque(
            self.DRONE_IDS[nth_drone],
            2,  # to center of mass - Check this for different configurations - Will not work generically :(
            torqueObj=[M_aero_body[0], M_aero_body[1], M_aero_body[2]],
            flags=p.LINK_FRAME,  # FIXME : see https://github.com/bulletphysics/bullet3/issues/1949
            physicsClientId=self.CLIENT,
        )

    ################################################################################
    def _tail_sitter_physics(self, cmd, nth_drone):
        """%% AERO aerodynamic forces and moments computation (in a wing section!)
        %
        %  this function computes the aerodynamic forces and moments in body axis
        %    in view of Phi-theory in a wing section. it includes propwash effects
        %    due to thrust.
        %
        %  INPUTS:
        %    state def. : x = (vl wb q)     in R(10x1)
        %    thrust     : T (N)             in R( 3x1)
        %    elevon def.: delta (rad)       in R( 1x1)
        %    motor speed: w (rad/s)         in R( 1x1)
        %    drone specs: drone             struct
        %    + required drone specs:
        %    |---- PHI                      in R( 6x6)
        %    |---- RHO (kg/m^3)             in R( 1x1)
        %    |---- WET_SURFACE (m^2)        in R( 1x1)
        %    |---- DRY_SURFACE (m^2)        in R( 1x1)
        %    |---- PHI_n                    in R( 1x1)
        %    |---- CHORD (m)                in R( 1x1)
        %    |---- WINGSPAN (m)             in R( 1x1)
        %    |       (of the wing section, half of full drone)
        %    |---- PROP_RADIUS (m)          in R( 1x1)
        %    |---- ELEVON_MEFFICIENCY       in R( 3x1)
        %    |---- ELEVON_FEFFICIENCY       in R( 3x1)
        %
        %  OUTPUTS:
        %    Aero force : Fb (body axis)    in R( 3x1)
        %    Aero moment: Mb (body axis)    in R( 3x1)
        %
        %  vl: vehicle velocity in NED axis (m/s) [3x1 Real]
        %  wb: vehicle angular velocity in body axis (rad/s) [3x1 Real]
        %  q:  quaternion attitude (according to MATLAB convention) [4x1 Real]
        %
        %  NOTE1: notice that w's sign depend on which section we are due to
        %    counter-rotating propellers;
        %  NOTE2: elevon sign convention is positive pictch-up deflections.
        %
        %  refer to [1] for further information.
        %
        %  REFERENCES
        %    [1] Lustosa L.R., Defay F., Moschetta J.-M., "The Phi-theory
        %    approach to flight control design of tail-sitter vehicles"
        %    @ http://lustosa-leandro.github.io"""

        # data extraction from drone struct
        # PHI_fv = drone.PHI[0:3,0:3]
        # PHI_mv = drone.PHI[3:6,0:3]
        # PHI_mw = drone.PHI[3:6,3:6]
        # phi_n  = drone.PHI_n
        # RHO    = drone.RHO
        # Swet   = drone.WET_SURFACE
        # Sdry   = drone.DRY_SURFACE
        # chord  = drone.CHORD
        # ws     = drone.WINGSPAN
        # Prop_R = drone.PROP_RADIUS
        # Thetam = drone.ELEVON_MEFFICIENCY
        # Thetaf = drone.ELEVON_FEFFICIENCY
        # PHI = np.array([[0., 0., 0., 0., 0., 0.],
        #                 [0., 0., 0., 0., 0., 0.],
        #                 [0., 0., 0., 0., 0., 0.],
        #                 [0., 0., 0., 0., 0., 0.],
        #                 [0., 0., 0., 0., 0., 0.],
        #                 [0., 0., 0., 0., 0., 0.]])
        # PHI_fv = PHI[0:3,0:3]
        # PHI_mv = PHI[3:6,0:3]
        # PHI_mw = PHI[3:6,3:6]

        Cd0 = 0.025
        Cy0 = 0.1
        phi_n = 0.0
        RHO = 1.225
        Swet = 0.0743
        Sdry = 0.0
        chord = 0.13
        dR = -0.1 * chord
        ws = 0.55
        Prop_R = 0.125
        Thetam = np.array([0.0, 0.93, 0.0])
        Thetaf = np.array([0.0, 0.48, 0.0])

        PHI_fv = np.diag([Cd0, Cy0, (2 * np.pi + Cd0)])
        PHI_mv = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -1 / chord * dR * (2 * np.pi + Cd0)],
                [0.0, 1 / ws * dR * Cy0, 0.0],
            ]
        )
        PHI_mw = 0.5 * np.diag([0.47, 0.54, 0.52])
        # derivative data
        Sp = np.pi * Prop_R**2

        #### Current state in Inertial Frame #############################
        pos = self.pos[nth_drone, :]
        quat = self.quat[nth_drone, :]
        rpy = self.rpy[nth_drone, :]
        vl = self.vel[nth_drone, :]
        rvel = self.ang_v[nth_drone, :]

        # Rotate inertial velocity to body frame
        R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        vel_b = R.T.dot(vl)
        wb = R.T.dot(rvel)
        # gamma = np.arcsin(vel[2]/np.linalg.norm(vel))
        # alpha = -rpy[1]-gamma # Assuming no WIND ! and using pitch angle as Angle of Attack
        # beta  = np.arctan(vel_b[1]/vel_b[0])
        # V_air = vel_b[0] if vel_b[0] > 0. else 0. #np.linalg.norm(vel) # FIXME get real airspeed along x axis with wind
        # rho   = 1.225 # Get this as a function of altitude... pos[2]
        # Pdyn  = 0.5*rho*V_air*V_air

        # state demultiplexing
        # vl = x[0:3]
        # wb = x[3:6]
        # q  = x[6:10]

        # DCM computation
        # D = q2dcm(q.T) # q2dcm(q')

        # freestream velocity computation in body frame
        # vinf = D*(vl-w)
        # Also pretent like the vehicle is horizontal like in phi-theory paper body frame definition.
        vinf = np.array([vel_b[2], -vel_b[1], vel_b[0]])  # FIXME
        wb = np.array([wb[2], -wb[1], wb[0]])
        wb = np.zeros(3)
        print("Vinf in phi-body : ", vinf)

        # computation of total wing section area
        S = Swet + Sdry

        # computation of chord matrix
        B = np.diag([ws, chord, ws])

        # eta computation
        eta = np.sqrt(
            np.linalg.norm(vinf) ** 2 + phi_n * np.linalg.norm(B.dot(wb)) ** 2
        )

        rpm = (
            self.drones[nth_drone].PWM2RPM_SCALE * cmd
            + self.drones[nth_drone].PWM2RPM_CONST
        )
        prop_force = np.array(rpm**2) * self.drones[nth_drone].KF
        # Apply propeller thrust : specific to this vehicle at 0 and 1 for left and right prop.

        for i in range(2):
            p.applyExternalForce(
                self.DRONE_IDS[nth_drone],
                i,
                forceObj=[0.0, 0.0, prop_force[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.CLIENT,
            )

        # F = np.zeros(3)
        # M = np.zeros(3)
        i = 3  # left wing link id (4 is right wing)
        for T, delta in zip(
            prop_force[:2], cmd[2:]
        ):  # FIXME : using the motors whicha re the first 2 of prop_force, and elevon which are the last 2 of cmd... Make it more generic
            T = np.array([T, 0.0, 0.0])
            delta = delta * np.deg2rad(
                30.0
            )  # cmd is -1/+1, representing +-30deg flap deflection
            # force computation
            # airfoil contribution
            Fb = (
                -1 / 2 * RHO * S * eta * PHI_fv.dot(vinf)
                - 1 / 2 * RHO * S * eta * PHI_mv.dot(B.dot(wb))
                - 1 / 2 * Swet / Sp * PHI_fv.dot(T)
            )
            # elevon contribution
            Fb = (
                Fb
                + 1 / 2 * RHO * S * eta * PHI_fv.dot(np.cross(delta * Thetaf, vinf))
                + 1
                / 2
                * RHO
                * S
                * eta
                * PHI_mv.dot(B.dot(np.cross(delta * Thetaf, wb)))
                + 1 / 2 * Swet / Sp * PHI_fv.dot(np.cross(delta * Thetaf, T))
            )
            # moment computation
            # airfoil contribution
            Mb = (
                -1 / 2 * RHO * S * eta * B.dot(PHI_mv.dot(vinf))
                - 1 / 2 * RHO * S * eta * B.dot(PHI_mw.dot(B.dot(wb)))
                - 1 / 2 * Swet / Sp * B.dot(PHI_mv.dot(T))
            )
            # elevon contribution
            Mb = (
                Mb
                + 1
                / 2
                * RHO
                * S
                * eta
                * B.dot(PHI_mv.dot(np.cross(delta * Thetam, vinf)))
                + 1
                / 2
                * RHO
                * S
                * eta
                * B.dot(PHI_mw.dot(B.dot(np.cross(delta * Thetam, wb))))
                + 1 / 2 * Swet / Sp * B.dot(PHI_mv.dot(np.cross(delta * Thetam, T)))
            )
            print(f" {i}- Fb : {Fb} Mb : {Mb} CMD : {cmd}")

            # Directly apply force and moments to each wing links (left and right)
            p.applyExternalForce(
                self.DRONE_IDS[nth_drone],
                i,
                forceObj=[Fb[2], 0.0, 0.0],  # [Fb[2],-Fb[1],Fb[0]],# [0., 0., 0.],#
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.CLIENT,
            )
            p.applyExternalTorque(
                self.DRONE_IDS[nth_drone],
                i,  # to base link : fuselage
                torqueObj=[0.0, -Mb[1], 0.0],  # [M[0],M[1],M[2]],
                flags=p.LINK_FRAME,  # FIXME FIXME
                physicsClientId=self.CLIENT,
            )
            i += 1

        # print([F, M])
        # return [Fb, Mb]

    ################################################################################

    def _coaxial_birotor_physics(self, cmd, nth_drone):
        rpm = (
            self.drones[nth_drone].PWM2RPM_SCALE * cmd
            + self.drones[nth_drone].PWM2RPM_CONST
        )
        forces = np.array(rpm**2) * self.drones[nth_drone].KF
        torques = np.array(rpm**2) * self.drones[nth_drone].KM
        # print(f'Forces : {forces}')
        # print(cmd)

        i = 0
        for _cmd in cmd[2:]:
            deflection = _cmd * np.deg2rad(10.0)  # cmd is in -1/+1 for radians
            p.resetJointState(nth_drone + 1, i, deflection)
            # print(f' {i}- deflection : {deflection}')
            i += 1

        s = [-1, 1]
        for i in range(2):
            p.applyExternalForce(
                self.DRONE_IDS[nth_drone],
                i + 1,
                forceObj=[0.0, 0.0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.CLIENT,
            )

            p.applyExternalTorque(
                self.DRONE_IDS[nth_drone],
                i + 1,  # to base link : fuselage
                torqueObj=[0.0, 0.0, s[i] * torques[i]],
                flags=p.LINK_FRAME,  # FIXME FIXME
                physicsClientId=self.CLIENT,
            )

    ################################################################################

    def _morphing_hexa_physics(self, cmd, nth_drone):
        """Morphing hexa rotor physics implementation.
        Parameters
        ----------
        cmd : ndarray
            (6)-shaped array of ints containing the command values of the 6 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        """
        rpm = (
            self.drones[nth_drone].PWM2RPM_SCALE * cmd
            + self.drones[nth_drone].PWM2RPM_CONST
        )
        forces = np.array(rpm**2) * self.drones[nth_drone].KF
        torques = np.array(rpm**2) * self.drones[nth_drone].KM

        # Checking the configuration shape :
        conf = abs(self.drones[nth_drone].conf)
        # print(f'self.drones[nth_drone].conf : {conf}')

        # Top propellers thrust reduction due to structural blockage
        for i in [0, 2, 4]:
            forces[i] = (1 - self.drones[nth_drone].top_prop_blockage * 2.2) * forces[i]

        # Top propeller thrust reduction due to bottom propeller blockage
        for i in [0, 2, 4]:
            forces[i] = (
                1 - self.drones[nth_drone].bottom_prop_blockage * 0.1
            ) * forces[i]

        # Bottom propellers have lower pitch, less thrust ?
        for i in [1, 3, 5]:
            forces[i] = 0.9 * forces[i]

        # Bottom propellers thrust reduction due to top propeller inflow
        for i in [1, 3, 5]:
            forces[i] = (
                1 - self.drones[nth_drone].bottom_prop_blockage * 0.1
            ) * forces[i]

        f_noise = np.random.normal(0, 0.01, self.drones[nth_drone].INDI_ACTUATOR_NR)
        m_noise = np.random.normal(0, 0.001, self.drones[nth_drone].INDI_ACTUATOR_NR)
        forces += f_noise
        torques += m_noise

        # forces[1] = 0.*forces[1]
        # torques[1] = 0.*torques[1]

        # z_torque = (-torques[0] + torques[1] - torques[2] + torques[3] - torques[4] + torques[5])

        for i in [0, 2, 4]:
            torques[i] *= -1.0

        for i, j in zip(range(1, 12, 2), range(6)):
            p.applyExternalForce(
                self.DRONE_IDS[nth_drone],
                i,
                forceObj=[0.0, 0.0, forces[j]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.CLIENT,
            )
            p.applyExternalTorque(
                self.DRONE_IDS[nth_drone],
                i,  # to base link : fuselage
                torqueObj=[0.0, 0.0, torques[j]],
                flags=p.LINK_FRAME,  # FIX ME FIXME
                physicsClientId=self.CLIENT,
            )
        # Perlin noise applied directly to main body
        px, py, pz = self.perlin_noise.next_value()
        p.applyExternalForce(
            self.DRONE_IDS[nth_drone],
            -1,
            forceObj=[0.1 * px + 2.0, 0.1 * py, 0.02 * pz],
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self.CLIENT,
        )
        # p.applyExternalTorque(self.DRONE_IDS[nth_drone],
        #                           -1, # to base link : fuselage
        #                           torqueObj=[0., 0.1, 0.],
        #                           flags=p.LINK_FRAME,
        #                           physicsClientId=self.CLIENT
        #                           )

    ################################################################################

    def _quad_copter_physics(self, cmd, nth_drone):  # was rpm
        """Base PyBullet quad rotor physics implementation.
        Parameters
        ----------
        cmd : ndarray
            (4)-shaped array of ints containing the command values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        """

        rpm = (
            self.drones[nth_drone].PWM2RPM_SCALE * cmd
            + self.drones[nth_drone].PWM2RPM_CONST
        )
        # rpm = 20000.*cmd

        if "advanced" in self.drones[nth_drone].TYPE:
            F_prop, M_prop = self._get_prop_FMs(rpm, nth_drone)
            direction = np.array([-1.0, 1.0, -1.0, 1.0])
            # print('########### F-M prop : ',F_prop, M_prop)
            for i in range(self.drones[nth_drone].INDI_ACTUATOR_NR):
                p.applyExternalForce(
                    self.DRONE_IDS[nth_drone],
                    i,  # Applied to the baselink of the vehicle
                    forceObj=[F_prop[i][0], F_prop[i][1], F_prop[i][2]],
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                    physicsClientId=self.CLIENT,
                )
                p.applyExternalTorque(
                    self.DRONE_IDS[nth_drone],
                    i,
                    torqueObj=[0, 0, M_prop[i][2] * direction[i]],
                    flags=p.LINK_FRAME,
                    physicsClientId=self.CLIENT,
                )

        else:
            forces = np.array(rpm**2) * self.drones[nth_drone].KF
            torques = np.array(rpm**2) * self.drones[nth_drone].KM

            f_noise = np.random.normal(0, 0.01, self.drones[nth_drone].INDI_ACTUATOR_NR)
            m_noise = np.random.normal(
                0, 0.001, self.drones[nth_drone].INDI_ACTUATOR_NR
            )
            # f_noise = np.zeros(4)
            # m_noise = np.zeros(4)
            forces += f_noise
            torques += m_noise

            z_torque = -torques[0] + torques[1] - torques[2] + torques[3]
            for i in range(self.drones[nth_drone].INDI_ACTUATOR_NR):
                p.applyExternalForce(
                    self.DRONE_IDS[nth_drone],
                    i,
                    forceObj=[f_noise[0], f_noise[1], forces[i]],
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                    physicsClientId=self.CLIENT,
                )
            p.applyExternalTorque(
                self.DRONE_IDS[nth_drone],
                -1,  # FIXME : this is not correct , use center of mass !!!
                torqueObj=[m_noise[0], m_noise[1], z_torque],
                flags=p.LINK_FRAME,
                physicsClientId=self.CLIENT,
            )
        wind = 0
        wind_vector = np.array([0.0, -10.5, 0.0])  # FIXME : make this parametric
        drag_coeff = (
            -0.0438
        )  # This is only for Tello (modeled infront of ENAC's Windshape)

        if wind:
            base_rot = np.array(
                p.getMatrixFromQuaternion(self.quat[nth_drone, :])
            ).reshape(3, 3)
            #### Simple draft model applied to the base/center of mass #
            # drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2*np.pi*rpm/60))
            drag = np.dot(
                base_rot, drag_coeff * (np.array(self.vel[nth_drone, :] - wind_vector))
            )
            p.applyExternalForce(
                self.DRONE_IDS[nth_drone],
                4,
                forceObj=drag,
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self.CLIENT,
            )

    ################################################################################

    def _get_prop_FMs(self, rpm, nth_drone):
        """
        Calculates inclined propeller forces and moments
        """

        # Get orientation of the quad, and generate rotation matrix
        # self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
        # self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
        R = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone])).reshape(3, 3)

        # Get quad's inertial speed vector
        # self.vel[i], self.ang_v[i] = p.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
        # print('Speed vector [inertial] : ', self.vel[nth_drone] )

        # Project inertial speed to quad frame and prepare normalised vector
        V_i = (
            self.vel[nth_drone]
            if np.linalg.norm(self.vel[nth_drone]) > 0.1
            else np.array([0.1, 0.0, 0.0])
        )  # Not the best :(
        V_b = R.dot(V_i)
        V_b_normed = V_b / np.linalg.norm(V_b)
        # V_b_normed_x = np.array([V_b_normed[0], 0., 0.])
        # print('Speed vector [  body  ] : ', V_b )

        # Obtain largest angle and orientation
        T_b = np.array([0.0, 0.0, 1.0])  # Thrust vector in body frame
        # T_b_x= np.array([1., 0., 0.]) # Body x direction

        # So, if v1 and v2 are normalised so that |v1|=|v2|=1, then,
        beta = np.arccos(
            V_b_normed.dot(T_b)
        )  # T_b is already normalized, thats wy we can do this !
        psi = (
            np.arctan(V_b[1] / V_b[0]) if V_b[0] > 0.1 else 0.0
        )  # If the psi is close to 90deg, this is not correct... also for low speed...
        # print(f'Largest angle : {beta} Heading difference : {np.rad2deg(psi)}')
        # angle = acos(v1.dot(v2))
        # axis = norm(np.cross(v1,v2))

        # Obtain the Force and Moment vectors from propeller database fitted method...
        # Fake force and moment vectors for the moment
        FM = np.zeros((len(rpm), 6))
        # M = np.zeros(3)
        # V = 12
        # beta = 0.45
        # Omega = 1050
        # V = np.linalg.norm(self.vel[nth_drone]) if np.linalg.norm(self.vel[nth_drone])>0.1 else 0.1
        propeller = "mamr-8x4.5"
        for i, _rpm in enumerate(rpm):
            # print(f'BETA : {beta}')
            # FM[i] = calculate_propeller_forces_moments(propeller, np.linalg.norm(self.vel[nth_drone]), beta, _rpm/60.*2*np.pi, Data_section3_ObliqueFlow, method = 1)
            FM[i] = calculate_propeller_forces_moments(
                propeller,
                np.linalg.norm(self.vel[nth_drone]),
                beta,
                _rpm / 60.0 * 2 * np.pi,
                Data_section5_ObliqueFlow,
                method=2,
            )
            # print(f' Vel : {np.linalg.norm(self.vel[nth_drone])}  --  RPM : {rpm} --- FM : {FM}')
            # method2 = calculate_propeller_forces_moments(propeller, self.vel[nth_drone], beta, rpm, Data_section5_ObliqueFlow,method = 2)

        # Rotate the F,M to Body axis F_b, M_b:
        R_z = np.array(
            [
                [np.cos(psi), -np.sin(psi), 0.0],
                [np.sin(psi), np.cos(psi), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        F_b = [R_z.dot(FM[i, :3]) for i in range(len(rpm))]
        M_b = [R_z.dot(FM[i, 3:]) for i in range(len(rpm))]

        return F_b, M_b

    ################################################################################

    def _groundEffect(self, rpm, nth_drone):
        """PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Kin. info of all links (propellers and center of mass)
        link_states = np.array(
            p.getLinkStates(
                self.DRONE_IDS[nth_drone],
                linkIndices=[0, 1, 2, 3, 4],
                computeLinkVelocity=1,
                computeForwardKinematics=1,
                physicsClientId=self.CLIENT,
            )
        )
        #### Simple, per-propeller ground effects ##################
        prop_heights = np.array(
            [
                link_states[0, 0][2],
                link_states[1, 0][2],
                link_states[2, 0][2],
                link_states[3, 0][2],
            ]
        )
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = (
            np.array(rpm**2)
            * self.KF
            * self.GND_EFF_COEFF
            * (self.PROP_RADIUS / (4 * prop_heights)) ** 2
        )
        if (
            np.abs(self.rpy[nth_drone, 0]) < np.pi / 2
            and np.abs(self.rpy[nth_drone, 1]) < np.pi / 2
        ):
            for i in range(4):
                p.applyExternalForce(
                    self.DRONE_IDS[nth_drone],
                    i,
                    forceObj=[0, 0, gnd_effects[i]],
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                    physicsClientId=self.CLIENT,
                )
        #### TODO: a more realistic model accounting for the drone's
        #### Attitude and its z-axis velocity in the world frame ###

    ################################################################################

    def _drag(self, rpm, nth_drone):
        """PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Rotation matrix of the base ###########################
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(
            3, 3
        )
        #### Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2 * np.pi * rpm / 60))
        drag = np.dot(base_rot, drag_factors * np.array(self.vel[nth_drone, :]))
        p.applyExternalForce(
            self.DRONE_IDS[nth_drone],
            4,
            forceObj=drag,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self.CLIENT,
        )

    ################################################################################

    def _downwash(self, nth_drone):
        """PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        for i in range(self.NUM_DRONES):
            delta_z = self.pos[i, 2] - self.pos[nth_drone, 2]
            delta_xy = np.linalg.norm(
                np.array(self.pos[i, 0:2]) - np.array(self.pos[nth_drone, 0:2])
            )
            if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
                alpha = self.DW_COEFF_1 * (self.PROP_RADIUS / (4 * delta_z)) ** 2
                beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
                downwash = [0, 0, -alpha * np.exp(-0.5 * (delta_xy / beta) ** 2)]
                p.applyExternalForce(
                    self.DRONE_IDS[nth_drone],
                    4,
                    forceObj=downwash,
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                    physicsClientId=self.CLIENT,
                )

    ################################################################################

    def _dynamics(self, rpm, nth_drone):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Current state #########################################
        pos = self.pos[nth_drone, :]
        quat = self.quat[nth_drone, :]
        rpy = self.rpy[nth_drone, :]
        vel = self.vel[nth_drone, :]
        rpy_rates = self.rpy_rates[nth_drone, :]
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        #### Compute forces and torques ############################
        forces = np.array(rpm**2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        z_torques = np.array(rpm**2) * self.KM
        z_torque = -z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3]
        if self.DRONE_MODEL == DroneModel.CF2X:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (
                self.L / np.sqrt(2)
            )
            y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (
                self.L / np.sqrt(2)
            )
        elif self.DRONE_MODEL == DroneModel.CF2P or self.DRONE_MODEL == DroneModel.HB:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.M
        #### Update state ##########################################
        vel = vel + self.TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.TIMESTEP * rpy_rates_deriv
        pos = pos + self.TIMESTEP * vel
        rpy = rpy + self.TIMESTEP * rpy_rates
        #### Set PyBullet's state ##################################
        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[nth_drone],
            pos,
            p.getQuaternionFromEuler(rpy),
            physicsClientId=self.CLIENT,
        )
        #### Note: the base's velocity only stored and not used ####
        p.resetBaseVelocity(
            self.DRONE_IDS[nth_drone],
            vel,
            [-1, -1, -1],  # ang_vel not computed by DYN
            physicsClientId=self.CLIENT,
        )
        #### Store the roll, pitch, yaw rates for the next step ####
        self.rpy_rates[nth_drone, :] = rpy_rates

    ################################################################################

    def _normalizedActionToRPM(self, action):
        """De-normalizes the [-1, 1] range to the [0, MAX_RPM] range.

        Parameters
        ----------
        action : ndarray
            (4)-shaped array of ints containing an input in the [-1, 1] range.

        Returns
        -------
        ndarray
            (4)-shaped array of ints containing RPMs for the 4 motors in the [0, MAX_RPM] range.

        """
        if np.any(np.abs(action)) > 1:
            print(
                "\n[ERROR] it",
                self.step_counter,
                "in BaseAviary._normalizedActionToRPM(), out-of-bound action",
            )
        return np.where(
            action <= 0, (action + 1) * self.HOVER_RPM, action * self.MAX_RPM
        )  # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM

    ################################################################################

    def _saveLastAction(self, action):
        """Stores the most recent action into attribute `self.last_action`.

        The last action can be used to compute aerodynamic effects.
        The method disambiguates between array and dict inputs
        (for single or multi-agent aviaries, respectively).

        Parameters
        ----------
        action : ndarray | dict
            (4)-shaped array of ints (or dictionary of arrays) containing the current RPMs input.

        """
        if isinstance(action, collections.Mapping):
            for k, v in action.items():
                res_v = np.resize(
                    v, (1, 4)
                )  # Resize, possibly with repetition, to cope with different action spaces in RL subclasses
                self.last_action[int(k), :] = res_v
        else:
            res_action = np.resize(
                action, (1, 4)
            )  # Resize, possibly with repetition, to cope with different action spaces in RL subclasses
            self.last_action = np.reshape(res_action, (self.NUM_DRONES, 4))

    ################################################################################

    def _showDroneLocalAxes(self, nth_drone):
        """Draws the local frame of the n-th drone in PyBullet's GUI.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if self.GUI:
            AXIS_LENGTH = 2 * self.L
            self.X_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[AXIS_LENGTH, 0, 0],
                lineColorRGB=[1, 0, 0],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.X_AX[nth_drone]),
                physicsClientId=self.CLIENT,
            )

            self.Y_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, AXIS_LENGTH, 0],
                lineColorRGB=[0, 1, 0],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                physicsClientId=self.CLIENT,
            )
            self.Z_AX[nth_drone] = p.addUserDebugLine(
                lineFromXYZ=[0, 0, 0],
                lineToXYZ=[0, 0, AXIS_LENGTH],
                lineColorRGB=[0, 0, 1],
                parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                parentLinkIndex=-1,
                replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                physicsClientId=self.CLIENT,
            )
            import pdb

            print("client :", self.CLIENT)
            print("parentObjectUniqueId= ", self.DRONE_IDS[nth_drone])
            print("int(self.Z_AX[nth_drone] :", int(self.Z_AX[nth_drone]))
            pdb.set_trace()

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        print(
            "Adding obstacles from BaseAviary is not possible anymore. Please load them as URDF in the main script."
        )
        # p.loadURDF("samurai.urdf",
        #            physicsClientId=self.CLIENT
        #            )
        # p.loadSDF("stadium.sdf",
        #            physicsClientId=self.CLIENT
        #            )
        # p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../assets/voliere.urdf",
        #            physicsClientId=self.CLIENT
        #            )
        # p.loadURDF("duck_vhacd.urdf",
        #            [-.5, -.5, .05],
        #            p.getQuaternionFromEuler([0, 0, 0]),
        #            physicsClientId=self.CLIENT
        #            )
        # p.loadURDF("cube_no_rotation.urdf",
        #            [-.5, -2.5, .5],
        #            p.getQuaternionFromEuler([0, 0, 0]),
        #            physicsClientId=self.CLIENT
        #            )
        # p.loadURDF("sphere2.urdf",
        #            basePosition=[2.5, 2.5, 2.5],
        #            baseOrientation=p.getQuaternionFromEuler([2.5,2.5,2.5]),
        #            physicsClientId=self.CLIENT,
        #            useFixedBase=1
        #            )

    ################################################################################
    def _parseURDFFixedwingParameters(self, drone, URDF):
        """Loads Fixed-wing related coefficients from URDF file"""
        URDF_TREE = etxml.parse(
            os.path.dirname(os.path.abspath(__file__)) + "/../assets/" + URDF
        ).getroot()

        ref = URDF_TREE.find("fixed_wing_aero_coeffs/ref")
        drone.alpha0 = float(ref.attrib["alpha0"])
        drone.Bref = float(ref.attrib["Bref"])
        drone.Sref = float(ref.attrib["Sref"])
        drone.Cref = float(ref.attrib["Cref"])
        drone.Vref = float(ref.attrib["Vref"])

        cl = URDF_TREE.find("fixed_wing_aero_coeffs/CL")
        drone.CL0 = float(cl.attrib["CL0"])
        drone.CL_alpha = float(cl.attrib["CL_alpha"])
        drone.CL_beta = float(cl.attrib["CL_beta"])
        vals = str(cl.attrib["CL_omega"])
        drone.CL_omega = [float(s) for s in vals.split(" ") if s != ""]
        vals = str(cl.attrib["CL_ctrl"])
        drone.CL_ctrl = [float(s) for s in vals.split(" ") if s != ""]

        cd = URDF_TREE.find("fixed_wing_aero_coeffs/CD")
        drone.CD0 = float(cd.attrib["CD0"])
        drone.CD_k1 = float(cd.attrib["CD_k1"])
        drone.CD_k2 = float(cd.attrib["CD_k1"])
        vals = str(cd.attrib["CD_ctrl"])
        drone.CD_ctrl = [float(s) for s in vals.split(" ") if s != ""]

        cy = URDF_TREE.find("fixed_wing_aero_coeffs/CY")
        drone.CY_alpha = float(cy.attrib["CY_alpha"])
        drone.CY_beta = float(cy.attrib["CY_beta"])
        vals = str(cy.attrib["CY_omega"])
        drone.CY_omega = [float(s) for s in vals.split(" ") if s != ""]
        vals = str(cy.attrib["CY_ctrl"])
        drone.CY_ctrl = [float(s) for s in vals.split(" ") if s != ""]

        cl = URDF_TREE.find("fixed_wing_aero_coeffs/Cl")
        drone.Cl_alpha = float(cl.attrib["Cl_alpha"])
        drone.Cl_beta = float(cl.attrib["Cl_beta"])
        vals = str(cl.attrib["Cl_omega"])
        drone.Cl_omega = [float(s) for s in vals.split(" ") if s != ""]
        vals = str(cl.attrib["Cl_ctrl"])
        drone.Cl_ctrl = [float(s) for s in vals.split(" ") if s != ""]

        cm = URDF_TREE.find("fixed_wing_aero_coeffs/Cm")
        drone.Cm0 = float(cm.attrib["Cm0"])
        drone.Cm_alpha = float(cm.attrib["Cm_alpha"])
        drone.Cm_beta = float(cm.attrib["Cm_beta"])
        vals = str(cm.attrib["Cm_omega"])
        drone.Cm_omega = [float(s) for s in vals.split(" ") if s != ""]
        vals = str(cm.attrib["Cm_ctrl"])
        drone.Cm_ctrl = [float(s) for s in vals.split(" ") if s != ""]

        cn = URDF_TREE.find("fixed_wing_aero_coeffs/Cn")
        drone.Cn_alpha = float(cn.attrib["Cn_alpha"])
        drone.Cn_beta = float(cn.attrib["Cn_beta"])
        vals = str(cn.attrib["Cn_omega"])
        drone.Cn_omega = [float(s) for s in vals.split(" ") if s != ""]
        vals = str(cn.attrib["Cn_ctrl"])
        drone.Cn_ctrl = [float(s) for s in vals.split(" ") if s != ""]

    ################################################################################
    def _parseURDFParameters(self, URDF):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(
            os.path.dirname(os.path.abspath(__file__)) + "/../assets/" + URDF
        ).getroot()

        conf = URDF_TREE.find("configuration")
        TYPE = str(conf.attrib["type"])

        mass = URDF_TREE.find("link/inertial/mass")
        M = float(mass.attrib["value"])

        prop = URDF_TREE.find("properties")
        L = float(prop.attrib["arm"])
        THRUST2WEIGHT_RATIO = float(prop.attrib["thrust2weight"])
        KF = float(prop.attrib["kf"])
        KM = float(prop.attrib["km"])

        inertia = URDF_TREE.find("link/inertial/inertia")
        IXX = float(inertia.attrib["ixx"])
        IYY = float(inertia.attrib["iyy"])
        IZZ = float(inertia.attrib["izz"])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)

        coll = URDF_TREE.find("link/collision/geometry/cylinder")
        COLLISION_H = float(coll.attrib["length"])
        COLLISION_R = float(coll.attrib["radius"])

        coll_offset = URDF_TREE.find("link/collision/origin")
        COLLISION_SHAPE_OFFSETS = [
            float(s) for s in coll_offset.attrib["xyz"].split(" ")
        ]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]

        MAX_SPEED_KMH = float(prop.attrib["max_speed_kmh"])
        GND_EFF_COEFF = float(prop.attrib["gnd_eff_coeff"])
        PROP_RADIUS = float(prop.attrib["prop_radius"])
        DRAG_COEFF_XY = float(prop.attrib["drag_coeff_xy"])
        DRAG_COEFF_Z = float(prop.attrib["drag_coeff_z"])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])

        DW_COEFF_1 = float(prop.attrib["dw_coeff_1"])
        DW_COEFF_2 = float(prop.attrib["dw_coeff_2"])
        DW_COEFF_3 = float(prop.attrib["dw_coeff_3"])

        indi = URDF_TREE.find("control/indi")
        indi_actuator_nr = int(indi.attrib["actuator_nr"])
        indi_output_nr = int(indi.attrib["output_nr"])
        G1 = np.zeros((indi_output_nr, indi_actuator_nr))

        indi = URDF_TREE.find("control")
        for i in range(indi_output_nr):
            vals = [str(k) for k in indi[i + 1].attrib.values()]
            G1[i] = [float(s) for s in vals[0].split(" ") if s != ""]

        pwm2rpm = URDF_TREE.find("control/pwm/pwm2rpm")
        # PWM2RPM_SCALE = float(pwm2rpm.attrib['scale'])
        # PWM2RPM_CONST = float(pwm2rpm.attrib['const'])
        vals = [str(k) for k in pwm2rpm.attrib.values()]
        PWM2RPM_SCALE = [float(s) for s in vals[0].split(" ") if s != ""]
        PWM2RPM_CONST = [float(s) for s in vals[1].split(" ") if s != ""]

        pwmlimit = URDF_TREE.find("control/pwm/limit")
        vals = [str(k) for k in pwmlimit.attrib.values()]
        MIN_PWM = [float(s) for s in vals[0].split(" ") if s != ""]
        MAX_PWM = [float(s) for s in vals[1].split(" ") if s != ""]

        return (
            TYPE,
            M,
            L,
            THRUST2WEIGHT_RATIO,
            J,
            J_INV,
            KF,
            KM,
            COLLISION_H,
            COLLISION_R,
            COLLISION_Z_OFFSET,
            MAX_SPEED_KMH,
            GND_EFF_COEFF,
            PROP_RADIUS,
            DRAG_COEFF,
            DW_COEFF_1,
            DW_COEFF_2,
            DW_COEFF_3,
            PWM2RPM_SCALE,
            PWM2RPM_CONST,
            indi_actuator_nr,
            indi_output_nr,
            G1,
            MIN_PWM,
            MAX_PWM,
        )

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _preprocessAction(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Must be implemented in a subclass.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, to be translated into RPMs.

        """
        raise NotImplementedError

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeDone(self):
        """Computes the current done value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError
