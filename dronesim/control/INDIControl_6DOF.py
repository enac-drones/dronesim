import math
import os
import pdb

# Active set library from : https://github.com/JimVaranelli/ActiveSet
import sys
import xml.etree.ElementTree as etxml
from dataclasses import dataclass

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from dronesim.control.BaseControl import BaseControl

# from dronesim.control.ActiveSet import ActiveSet, ConstrainedLS
from dronesim.control.wls_alloc import wls_alloc
from dronesim.envs.BaseAviary import BaseAviary, DroneModel

# @dataclass
# class PlayingCard:
#     rank: str
#     suit: str


@dataclass
class Rate:
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0


@dataclass
class Gains:
    att = Rate()
    rate = Rate()


def quat_comp(a2b, b2c):
    # qi,qx,qy,qz = 0,1,2,3
    qi, qx, qy, qz = 3, 0, 1, 2
    a2c = np.zeros(4)
    a2c[qi] = (
        a2b[qi] * b2c[qi] - a2b[qx] * b2c[qx] - a2b[qy] * b2c[qy] - a2b[qz] * b2c[qz]
    )
    a2c[qx] = (
        a2b[qi] * b2c[qx] + a2b[qx] * b2c[qi] + a2b[qy] * b2c[qz] - a2b[qz] * b2c[qy]
    )
    a2c[qy] = (
        a2b[qi] * b2c[qy] - a2b[qx] * b2c[qz] + a2b[qy] * b2c[qi] + a2b[qz] * b2c[qx]
    )
    a2c[qz] = (
        a2b[qi] * b2c[qz] + a2b[qx] * b2c[qy] - a2b[qy] * b2c[qx] + a2b[qz] * b2c[qi]
    )
    return a2c


def quat_inv_comp(q1, q2):
    # i,x,y,z = 0,1,2,3
    i, x, y, z = 3, 0, 1, 2
    qerr = np.zeros(4)
    qerr[i] = q1[i] * q2[i] + q1[x] * q2[x] + q1[y] * q2[y] + q1[z] * q2[z]
    qerr[x] = q1[i] * q2[x] - q1[x] * q2[i] - q1[y] * q2[z] + q1[z] * q2[y]
    qerr[y] = q1[i] * q2[y] + q1[x] * q2[z] - q1[y] * q2[i] - q1[z] * q2[x]
    qerr[z] = q1[i] * q2[z] - q1[x] * q2[y] + q1[y] * q2[x] - q1[z] * q2[i]
    return qerr


def quat_norm(q):
    return np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


def quat_normalize(q):
    n = quat_norm(q)
    if n > 0.0:
        for i in range(4):
            q[i] = q[i] / n
    return q


def quat_wrap_shortest(q):
    w = 3  # 0 or 3 according to quaternion definition.
    if q[w] < 0:
        for i in range(4):  # QUAT_EXPLEMENTARY(q)
            q[i] = -q[i]
    return q


def thrust_from_rpm(rpm):
    """input is the array of actuator rpms"""
    thrust = 0.0
    for _rpm in rpm:
        thrust += _rpm**2.0 * 3.16e-10
    return thrust


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[1], 0.0]])


def jac_vec_quat(vec, q):
    w = q[3]
    v = q[:3]
    I = np.eye(3)
    p1 = w * vec + np.cross(v, vec)
    p2 = np.dot(np.dot(v.T, vec), I) + v.dot(vec.T) - vec.dot(v.T) - w * skew(vec)
    return np.hstack([p1.reshape(3, 1), p2]) * 2  # p1, p2


class INDIControl(BaseControl):
    """INDI control class for Crazyflies.

    by Murat Bronz based on work conducted at TUDelft by Ewoud Smeur.

    """

    ################################################################################

    def __init__(
        self,
        drone_model: list = ["tello"],
        g: float = 9.8,
    ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        self.DRONE_MODEL = drone_model
        self.URDF = self.DRONE_MODEL + ".urdf"

        # this is being called from the init of BaseControl !
        # self._parseURDFControlParameters()
        # ========
        # self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        # self.I_COEFF_FOR = np.array([.05, .05, .05])
        # self.D_COEFF_FOR = np.array([.2, .2, .5])
        # self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        # self.I_COEFF_TOR = np.array([.0, .0, 500.])
        # self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        # self.PWM2RPM_SCALE = 0.2685
        # self.PWM2RPM_CONST = 4070.3
        # self.MIN_PWM = 20000
        # self.MAX_PWM = 65535
        # if self.DRONE_MODEL == DroneModel.CF2X:
        #     self.MIXER_MATRIX = np.array([ [.5, -.5,  -1], [.5, .5, 1], [-.5,  .5,  -1], [-.5, -.5, 1] ])
        # elif self.DRONE_MODEL == DroneModel.CF2P:
        #     self.MIXER_MATRIX = np.array([ [0, -1,  -1], [+1, 0, 1], [0,  1,  -1], [-1, 0, 1] ])
        # ========
        self.reset()

    ################################################################################
    def _parseURDFControlParameters(self):
        """Loads Control parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(
            os.path.dirname(os.path.abspath(__file__)) + "/../assets/" + self.URDF
        ).getroot()

        mass = URDF_TREE.find("link/inertial/mass")
        self.m = float(mass.attrib["value"])

        indi = URDF_TREE.find("control/indi")
        self.indi_actuator_nr = int(indi.attrib["actuator_nr"])
        self.indi_output_nr = int(indi.attrib["output_nr"])
        self.G1 = np.zeros((self.indi_output_nr, self.indi_actuator_nr))
        print("*************")

        indi = URDF_TREE.find("control")
        for i in range(self.indi_output_nr):
            vals = [str(k) for k in indi[i + 1].attrib.values()]
            self.G1[i] = [float(s) for s in vals[0].split(" ") if s != ""]

        self.indi_gains = Gains()
        guidance_gains = URDF_TREE.find("control/indi_guidance_gains/pos")
        self.guidance_indi_pos_gain = float(guidance_gains.attrib["kp"])
        self.guidance_indi_speed_gain = float(guidance_gains.attrib["kd"])

        att_att_gains = URDF_TREE.find("control/indi_att_gains/att")
        att_rate_gains = URDF_TREE.find("control/indi_att_gains/rate")

        self.indi_gains.att.p = float(att_att_gains.attrib["p"])
        self.indi_gains.att.q = float(att_att_gains.attrib["q"])
        self.indi_gains.att.r = float(att_att_gains.attrib["r"])
        self.indi_gains.rate.p = float(att_rate_gains.attrib["p"])
        self.indi_gains.rate.q = float(att_rate_gains.attrib["q"])
        self.indi_gains.rate.r = float(att_rate_gains.attrib["r"])

        pwm2rpm = URDF_TREE.find("control/pwm/pwm2rpm")
        # self.PWM2RPM_SCALE = float(pwm2rpm.attrib['scale'])
        # self.PWM2RPM_CONST = float(pwm2rpm.attrib['const'])
        vals = [str(k) for k in pwm2rpm.attrib.values()]
        self.PWM2RPM_SCALE = [float(s) for s in vals[0].split(" ") if s != ""]
        self.PWM2RPM_CONST = [float(s) for s in vals[1].split(" ") if s != ""]

        pwmlimit = URDF_TREE.find("control/pwm/limit")
        # self.MIN_PWM = float(pwmlimit.attrib['min'])
        # self.MAX_PWM = float(pwmlimit.attrib['max'])
        vals = [str(k) for k in pwmlimit.attrib.values()]
        self.MIN_PWM = [float(s) for s in vals[0].split(" ") if s != ""]
        self.MAX_PWM = [float(s) for s in vals[1].split(" ") if s != ""]

    ################################################################################
    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        self.diffed_cur_ang_vel = np.zeros(3)  # ERASE
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

        self.last_rates = np.zeros(3)  # p,q,r
        # self.last_pwm = np.ones(self.indi_actuator_nr)*1. # initial pwm
        self.last_thrust = 0.3
        # self.indi_increment = np.zeros(4)
        self.cmd = np.ones(self.indi_actuator_nr) * 0.5
        self.last_vel = np.zeros(3)
        self.last_torque = np.zeros(3)  # For SU2 controller

        self.xax = -1
        self.yax = -1
        self.zax = -1
        self.xax1 = -2
        self.yax1 = -2
        self.zax1 = -2

        # for debugging logs...
        self.att_log = np.zeros((30 * 100, 20))
        self.guid_log = np.zeros((30 * 100, 20))
        self.att_log_inc = 0
        self.guid_log_inc = 0

        self.rpm = np.zeros(self.indi_actuator_nr)

    def rpm_of_pwm(self, pwm):
        self.rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        return self.rpm

    ################################################################################

    def computeControl(
        self,
        control_timestep,
        cur_pos,
        cur_quat,
        cur_vel,
        cur_ang_vel,
        target_pos,
        target_rpy=np.zeros(3),
        target_vel=np.zeros(3),
        target_rpy_rates=np.zeros(3),
        target_acc=np.zeros(3),
    ):
        """Computes the PID control action (as RPMs) for a single drone.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        (
            thrust,
            computed_target_rpy,
            pos_e,
            quat_,
            accel_error,
        ) = self._INDIPositionControl(
            control_timestep,
            cur_pos,
            cur_quat,
            cur_vel,
            target_pos,
            target_rpy,
            target_vel,
            target_acc,
        )

        rpm = self._INDIAttitudeControl(
            control_timestep,
            thrust,
            cur_quat,
            cur_ang_vel,
            computed_target_rpy,
            quat_,
            target_rpy_rates,
            accel_error,
        )

        cur_rpy = p.getEulerFromQuaternion(cur_quat)

        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    ################################################################################

    ################################################################################
    def _INDIPositionControl(
        self,
        control_timestep,
        cur_pos,
        cur_quat,
        cur_vel,
        target_pos,
        target_rpy,
        target_vel,
        target_acc=np.zeros(3),
        use_quaternion=False,
        nonlinear_increment=False,
    ):

        """ENAC generic INDI position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        debug_log = False

        # Linear controller to find the acceleration setpoint from position and velocity
        # pos_x_err  = guidance_h.ref.pos.x) - stateGetPositionNed_f()->x;
        # pos_y_err  = guidance_h.ref.pos.y) - stateGetPositionNed_f()->y;
        # pos_z_err  = guidance_v_z_ref - stateGetPositionNed_i()->z);
        pos_e = target_pos - cur_pos

        # Speed setpoint
        # speed_sp_y = pos_y_err * guidance_indi_pos_gain;
        # speed_sp_z = pos_z_err * guidance_indi_pos_gain;
        speed_sp = pos_e * self.guidance_indi_pos_gain

        # Not used for the momonet
        vel_e = speed_sp + target_vel - cur_vel

        # Set acceleration setpoint :

        # accel_sp = (speed_sp - cur_vel) * self.guidance_indi_speed_gain
        accel_sp = vel_e * self.guidance_indi_speed_gain

        # Calculate the acceleration via finite difference TODO : this is a rotated sensor output in real life, so ad sensor to the sim !
        cur_accel = (cur_vel - self.last_vel) / control_timestep
        # print(f'Cur Velocity : {cur_vel[2]}, Last Velocity : {self.last_vel[2]}')
        self.last_vel = cur_vel

        accel_e = accel_sp - cur_accel

        # Bound the acceleration error so that the linearization still holds
        accel_e = np.clip(accel_e, -6.0, 6.0)  # For Z : -9.0, 9.0 FIXME !
        # accel_e = np.clip(accel_e, -2.0, 2.0) # Trying to slow down crazy agility ! Nope it does not work ! Oscillates...

        # EULER VERSION
        # # Calculate matrix of partial derivatives
        # guidance_indi_calcG_yxz(&Ga, &eulers_yxz);
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        phi, theta, psi = cur_rpy[0], cur_rpy[1], cur_rpy[2]

        # print(f'Phi-Theta-Psi : {phi}, {theta}, {psi}')

        sph, sth, sps = np.sin(phi), np.sin(theta), np.sin(psi)
        cph, cth, cps = np.cos(phi), np.cos(theta), np.cos(psi)

        # theta = np.clip(theta,-np.pi,0) # FIXME
        lift = np.sin(theta) * -9.81  # FIXME
        liftd = 0.0
        T = np.cos(theta) * 9.81
        # get the derivative of the lift wrt to theta
        # liftd = guidance_indi_get_liftd(stateGetAirspeed_f(), eulers_zxy.theta);

        T = 9.81  # np.array([0., 0., 1.]) # Thrust guestimation

        # Calculate the matrix of partial derivatives of the roll, pitch and thrust.
        # w.r.t. the NED accelerations for ZYX eulers
        # ddx = G*[dtheta,dphi,dT]
        G = np.array(
            [
                [
                    (cph * sps - sph * cps * sth) * T,
                    (cph * cps * cth) * T,
                    sph * sps + cph * cps * sth,
                ],
                [
                    (-sph * sps * sth - cps * cph) * T,
                    (cph * sps * cth) * T,
                    cph * sps * sth - cps * sph,
                ],
                [-cth * sph * T, -sth * cph * T, cph * cth],
            ]
        )

        # Calculate the matrix of partial derivatives of the pitch, roll and thrust.
        # w.r.t. the NED accelerations for YXZ eulers
        # ddx = G*[dtheta,dphi,dT]
        # G = np.array([[cth * cph * T , -sth * sph * T, sth * cph],
        #             [0.,    -cph * T   ,  -sph ],
        #             [-sth * cph * T,   -cth * sph * T  ,  cth * cph] ])

        # Matrix of partial derivatives for Lift force
        GUIDANCE_INDI_PITCH_EFF_SCALING = 1.0

        # GL = np.array([[ cph*cth*sps*T + cph*sps*lift , (cth*cps - sph*sth*sps)*T*GUIDANCE_INDI_PITCH_EFF_SCALING + sph*sps*liftd , sth*cps + sph*cth*sps],
        #                [-cph*cth*cps*T - cph*cps*lift , (cth*sps + sph*sth*cps)*T*GUIDANCE_INDI_PITCH_EFF_SCALING - sph*cps*liftd , sth*sps - sph*cth*cps],
        #                [    -sph*cth*T -     sph*lift ,                -cph*sth*T*GUIDANCE_INDI_PITCH_EFF_SCALING +     cph*liftd ,        cph*cth       ] ])

        # GL = np.array([[(cth*cps - sph*sth*sps)*T*GUIDANCE_INDI_PITCH_EFF_SCALING + sph*sps*liftd ,  cph*cth*sps*T + cph*sps*lift , sth*cps + sph*cth*sps],
        #                [(cth*sps + sph*sth*cps)*T*GUIDANCE_INDI_PITCH_EFF_SCALING - sph*cps*liftd , -cph*cth*cps*T - cph*cps*lift , sth*sps - sph*cth*cps],
        #                [               -cph*sth*T*GUIDANCE_INDI_PITCH_EFF_SCALING +     cph*liftd ,     -sph*cth*T -     sph*lift ,        cph*cth       ] ])

        # Invert this matrix
        G_inv = np.linalg.pinv(G)  # FIXME

        # Calculate roll,pitch and thrust command
        control_increment = G_inv.dot(accel_e)

        # Rotate the phi theta : Need to correct this on the upper G ! FIXME !
        R_psi = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
        control_increment_rotated = R_psi.dot(control_increment[:2])
        control_increment[:2] = control_increment_rotated

        target_quat = np.array([0.0, 0.0, 0.0, 1.0])

        yaw_increment = target_rpy[2] - psi  # cur_rpy[2]
        target_euler = cur_rpy + np.array(
            [control_increment[0], control_increment[1], yaw_increment]
        )

        thrust = self.last_thrust + control_increment[2]  # for EULER version !!!! FIXME
        # thrust = self.last_thrust + thrust_increment # for Quaternion version

        # Over-wrtie the target euler for 6DOF -> This should be changed to an intelligent method !
        target_euler = np.zeros(3)
        return thrust, target_euler, pos_e, target_quat, accel_e  # quat_increment

    ################################################################################

    def _INDIAttitudeControl(
        self,
        control_timestep,
        thrust,
        cur_quat,
        cur_ang_vel,
        target_euler,
        target_quat,
        target_rpy_rates,
        accel_error,
    ):
        """INDI attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (actuator_nr,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        # calculate quat attitude error q_err = q2 (hamilton product) inverse(q1)
        # https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames

        # int32_quat_inv_comp(&att_err, att_quat, &quat_sp) # from Paparazzi !
        # XYZ - XZY - ...
        # target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        target_quat = np.array(p.getQuaternionFromEuler(target_euler))

        quat_err = quat_inv_comp(cur_quat, target_quat)

        # wrap it in the shortest direction
        # att_err = quat_wrap_shortest(quat_err);

        att_err = np.array(quat_err[:3])

        # Rotate the phi theta : Need to correct this on the upper G ! FIXME !
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        phi, theta, psi = cur_rpy[0], cur_rpy[1], cur_rpy[2]

        R_psi = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])

        R_psi = np.linalg.inv(R_psi)

        att_err_rotated = R_psi.dot(att_err[:2])
        # pdb.set_trace()
        att_err[:2] = att_err_rotated

        # local variable to compute rate setpoints based on attitude error
        rate_sp = Rate()

        rate_sp.p = self.indi_gains.att.p * att_err[0]
        rate_sp.q = self.indi_gains.att.q * att_err[1]
        rate_sp.r = self.indi_gains.att.r * att_err[2]

        # Rotate angular velocity to body frame
        R = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)

        cur_ang_vel = R.T.dot(cur_ang_vel)

        # Calculate the angular acceleration via finite difference
        angular_accel = (cur_ang_vel - self.last_rates) / (1.0 * control_timestep)

        # Filter the "noisy" angular velocities : Doing nothing for the moment... placeholder.
        rates_filt = Rate()
        rates_filt.p = cur_ang_vel[0]
        rates_filt.q = cur_ang_vel[1]
        rates_filt.r = cur_ang_vel[2]

        # Remember the last rates for differentiation on the next step
        self.last_rates = cur_ang_vel

        # Calculate the virtual control (reference acceleration) based on a PD controller
        angular_accel_ref = Rate()
        angular_accel_ref.p = (rate_sp.p - rates_filt.p) * self.indi_gains.rate.p
        angular_accel_ref.q = (rate_sp.q - rates_filt.q) * self.indi_gains.rate.q
        angular_accel_ref.r = (rate_sp.r - rates_filt.r) * self.indi_gains.rate.r

        accel_error_body = R.T.dot(accel_error)

        indi_v = np.zeros(6)  # roll-pitch-yaw-Fx-Fy-Fz
        indi_v[0] = angular_accel_ref.p - angular_accel[0]
        indi_v[1] = angular_accel_ref.q - angular_accel[1]
        indi_v[2] = angular_accel_ref.r - angular_accel[2]
        indi_v[3] = accel_error_body[0]  # thrust - self.last_thrust #* 0.
        indi_v[4] = accel_error_body[1]
        indi_v[5] = accel_error_body[2]
        self.last_thrust = thrust

        pseudo_inv = 0
        if pseudo_inv:
            indi_du = np.dot(np.linalg.pinv(self.G1 / 0.05), indi_v)  # *self.m
            # print(f'Command : {self.cmd}')
            # pdb.set_trace()
        else:
            # Use Active set for control allocation
            umin = np.asarray(
                [self.MIN_PWM[i] - self.cmd[i] for i in range(self.indi_actuator_nr)]
            )
            umax = np.asarray(
                [self.MAX_PWM[i] - self.cmd[i] for i in range(self.indi_actuator_nr)]
            )
            # umax = np.asarray([self.MAX_PWM for i in range(4)])
            # indi_v1 = [indi_v[i] for i in range(4)]

            # up = np.array([0., 0., 0., 0.])
            # Wv = np.array([1000, 1000, 0.1, 10])
            Wv = np.array([1000, 1000, 0.1, 10, 10, 100])  # This can be a decision...
            Wu = np.ones(self.indi_actuator_nr)  # np.array([1, 1, 1, 1, 1, 1]) #FIXME
            u_guess = None
            W_init = None
            up = None

            # import scipy.optimize
            # res = scipy.optimize.lsq_linear(A, v, bounds=(umin, umax), lsmr_tol='auto', verbose=1)
            indi_du, nit = wls_alloc(
                indi_v, umin, umax, self.G1 / 0.05, u_guess, W_init, Wv, Wu, up
            )

        self.cmd += indi_du
        self.cmd = np.clip(self.cmd, self.MIN_PWM, self.MAX_PWM)  # command in PWM

        # print(f'CMD : {self.cmd}  ---  RPM : {self.rpm_of_pwm(self.cmd)}')
        return self.cmd  # self.rpm_of_pwm(self.cmd)

    ################################################################################


# EOF
