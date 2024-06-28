import math
import os
import pdb

# Active set library from : https://github.com/JimVaranelli/ActiveSet
import sys
import xml.etree.ElementTree as etxml


import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from dronesim.control.BaseControl import BaseControl

# from dronesim.control.ActiveSet import ActiveSet, ConstrainedLS
from dronesim.control.wls_alloc import wls_alloc
from dronesim.envs.BaseAviary import BaseAviary, DroneModel


from dronesim.utils.math import quat_inv_comp, quat_wrap_shortest, norm_ang, quat_comp, quat_norm, quat_normalize
from dronesim.utils.utils import Rate, Gains


class INDIControl(BaseControl):
    """INDI control class

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
        self.last_thrust = 0.0
        # self.indi_increment = np.zeros(4)
        self.cmd = np.ones(self.indi_actuator_nr) * 0.0
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
        target_vel=np.zeros(3),
        target_acc=np.zeros(3),
        target_rpy=np.zeros(3),
        target_rpy_rates=np.zeros(3),
        
    ):
        """Computes the INDI control action (as RPMs) for a single drone.

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
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_acc : ndarray, optional
            (3,1)-shaped array of floats containing the desired accelerations.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
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
        thrust, computed_target_rpy, pos_e, quat_ = self._INDIPositionControl(
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
        # -------------------
        # Linear controller to find the acceleration setpoint from position and velocity
        pos_e = target_pos - cur_pos

        # Speed setpoint
        speed_sp = pos_e * self.guidance_indi_pos_gain

        vel_e = speed_sp + target_vel - cur_vel

        # Set acceleration setpoint :
        accel_sp = vel_e * self.guidance_indi_speed_gain

        # Calculate the acceleration via finite difference TODO : this is a rotated sensor output in real life, so ad sensor to the sim !
        cur_accel = (cur_vel - self.last_vel) / control_timestep
  
        self.last_vel = cur_vel # FIXME remove

        accel_e = accel_sp + target_acc - cur_accel

        # Bound the acceleration error so that the linearization still holds
        accel_e = np.clip(accel_e, -6.0, 6.0)  # For Z : -9.0, 9.0 FIXME !

        # EULER VERSION
        # # Calculate matrix of partial derivatives
        # guidance_indi_calcG_yxz(&Ga, &eulers_yxz);
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        phi, theta, psi = cur_rpy[0], cur_rpy[1], cur_rpy[2]

        sph, sth, sps = np.sin(phi), np.sin(theta), np.sin(psi)
        cph, cth, cps = np.cos(phi), np.cos(theta), np.cos(psi)

        # theta = np.clip(theta,-np.pi,0) # FIXME
        # lift = np.sin(theta)*-9.81 # FIXME
        # liftd = 0.
        # T = np.cos(theta)*9.81
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

        # Invert this matrix
        G_inv = np.linalg.pinv(G)  # FIXME

        # Calculate roll,pitch and thrust increments
        control_increment = G_inv.dot(accel_e)

        yaw_increment = norm_ang(target_rpy[2] - psi)

        # Add increments to the attitude and thrust
        target_euler = cur_rpy + np.array(
            [control_increment[0], control_increment[1], yaw_increment]
        )
        thrust = self.last_thrust + control_increment[2]

        # Placeholder FIXME
        target_quat = np.array([0.0, 0.0, 0.0, 1.0])
        return thrust, target_euler, pos_e, target_quat

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
        # int32_quat_inv_comp(&att_err, att_quat, &quat_sp) # from Paparazzi !
        # XYZ - XZY - ...
        target_quat = np.array(p.getQuaternionFromEuler(target_euler))

        quat_err = quat_inv_comp(cur_quat, target_quat)

        # wrap it in the shortest direction
        att_err = quat_wrap_shortest(quat_err)

        att_err = np.array(quat_err[:3])

        # local variable to compute rate setpoints based on attitude error
        rate_sp = Rate()

        rate_sp.p = self.indi_gains.att.p * att_err[0]
        rate_sp.q = self.indi_gains.att.q * att_err[1]
        rate_sp.r = self.indi_gains.att.r * att_err[2]

        self.cmd = self._INDIRateControl(
            control_timestep,
            thrust,
            cur_quat,
            cur_ang_vel,
            target_rpy_rates=np.array([rate_sp.p, rate_sp.q, rate_sp.r])
        )
        return self.cmd

    def _INDIRateControl(
            self,
            control_timestep,
            thrust,
            cur_quat,
            cur_ang_vel,
            target_rpy_rates ):

        # FIXME : rate set point, reference angular speed, rpy rates, FIND a correct unique name for all...
        rate_sp = Rate()
        rate_sp.p = target_rpy_rates[0]
        rate_sp.q = target_rpy_rates[1]
        rate_sp.r = target_rpy_rates[2]

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

        indi_v = np.zeros(4)  # roll-pitch-yaw-thrust
        indi_v[0] = angular_accel_ref.p - angular_accel[0]
        indi_v[1] = angular_accel_ref.q - angular_accel[1]
        indi_v[2] = angular_accel_ref.r - angular_accel[2]
        indi_v[3] = thrust - self.last_thrust  # * 0.
        self.last_thrust = thrust

        pseudo_inv = 1
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
            Wv = np.array([1000, 1000, 0.1, 10])
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
