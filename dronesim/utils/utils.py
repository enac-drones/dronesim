"""General use functions.
"""
import argparse
import time

# Related to propeller functions
import warnings
from math import cos, log, pi, sin

import numpy as np
from scipy.optimize import nnls
from dataclasses import dataclass

@dataclass
class Rate:
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0


@dataclass
class Gains:
    att = Rate()
    rate = Rate()


# from gym_pybullet_drones.database.propeller_database import *
class ContinuousPerlinNoise:
    def __init__(self, period=100):
        self.period = period
        self.gradients = np.random.rand(2) * 2 - 1  # Initial two gradients
        self.position = 0

    def interpolate(self, a, b, x):
        """Perform a smooth interpolation between a and b, given fraction x."""
        ft = x * np.pi
        f = (1 - np.cos(ft)) * 0.5
        return a * (1 - f) + b * f

    def next_value(self):
        """Generate the next Perlin noise value."""
        if self.position % self.period == 0 and self.position > 0:
            # Append a new gradient every period
            new_gradient = np.random.rand() * 2 - 1
            self.gradients = np.append(self.gradients, new_gradient)

        segment_index = self.position // self.period
        local_x = (self.position % self.period) / self.period

        left_grad = self.gradients[segment_index]
        right_grad = self.gradients[segment_index + 1]

        noise_value = self.interpolate(left_grad, right_grad, local_x)
        self.position += 1

        return noise_value


class MultiDimensionalContinuousPerlinNoise:
    def __init__(self, dimensions=1, period=100):
        self.period = period
        self.dimensions = dimensions
        self.gradients = [
            np.random.rand(2) * 2 - 1 for _ in range(dimensions)
        ]  # Initial gradients for each dimension
        self.positions = [0] * dimensions

    def interpolate(self, a, b, x):
        """Perform a smooth interpolation between a and b, given fraction x."""
        ft = x * np.pi
        f = (1 - np.cos(ft)) * 0.5
        return a * (1 - f) + b * f

    def next_value(self, dimension_index=None):
        """Generate the next Perlin noise value for a specific dimension or for all dimensions."""
        if dimension_index is not None:
            # Generate noise for the specified dimension
            return self._generate_noise_for_dimension(dimension_index)
        else:
            # Generate and return noise for all dimensions
            return tuple(
                self._generate_noise_for_dimension(i) for i in range(self.dimensions)
            )

    def _generate_noise_for_dimension(self, dimension_index):
        """Generate the next noise value for a specific dimension."""
        if (
            self.positions[dimension_index] % self.period == 0
            and self.positions[dimension_index] > 0
        ):
            # Append a new gradient every period
            new_gradient = np.random.rand() * 2 - 1
            self.gradients[dimension_index] = np.append(
                self.gradients[dimension_index], new_gradient
            )

        segment_index = self.positions[dimension_index] // self.period
        local_x = (self.positions[dimension_index] % self.period) / self.period

        left_grad = self.gradients[dimension_index][segment_index]
        right_grad = self.gradients[dimension_index][segment_index + 1]

        noise_value = self.interpolate(left_grad, right_grad, local_x)
        self.positions[dimension_index] += 1

        return noise_value


def R_aero_to_body(alpha, beta):
    """
    computes the aero to body rotation matrix
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    return np.array([[ca * cb, -ca * sb, -sa], [sb, cb, 0.0], [sa * cb, -sa * sb, ca]])


def clamp_vector(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Rescale the vector(s) such that the norm does not exceed the given argument.

    Parameters
    ----------
    v : np.ndarray
        Vector (or vector array, with coordinates of each vector on axis 0) to be clamped.

    max_norm: float
        Maximal accepted norm for the vector(s)

    Returns
    -------
    np.ndarray
        Clamped vector(s).

    """
    norm_v = np.linalg.norm(v, axis=0)
    normalized_v = v / norm_v
    clamped_norms = np.clip(norm_v, 0, max_norm)
    return normalized_v * clamped_norms


################################################################################
"""
The following scripts contains the reduced order methods presented on the paper ''Computationally Efficient Force and Moment Models
for Propellers in UAV Forward Flight Applications'' by Rajan Gill and Raffaello D’Andrea.

"""


def calculate_propeller_forces_moments(
    propeller, V, beta, omega, propeller_dictionary, method=1, Nb=2, rho=1.225
):
    """
    This function calls the function calculate_propeller_coefficients m1 ir m2 to obtain propellers coefficients and then
    calculates 'dimensionalizes' the obtained values using the relations:

    Force = Cforce * 0.5 * rho * A * (omega*R)**2

    Moment = Cmoment * 0.5 * rho * A * R*(omega*R)**2

    *We are using SI units

    arguments
    propeller: The name of the propeller
    Nb: The number of blades
    V: incoming wind speed
    beta: angle between the rotor plane and incoming wind
    omega: propeller’s rotation rate
    propeller_dictionary: dict with data from the paper. Section3 dicts only!
    rho: air density
    method: int equal to 1 (for higher order model) or 2 (for lower order model)
    """

    # initially, we need the propeller diameter
    prop_diameter = float(propeller.split("-")[1].split("x")[0])
    # Propeller radius in meters
    R = prop_diameter / 2 * 0.0254

    omega = omega if omega > 10.0 else 10.0

    if method == 1:
        cft, cfh, cmq, cmr, cmp = calculate_propeller_coefficients_m1(
            propeller, V, beta, omega, propeller_dictionary, R, Nb
        )
    elif method == 2:
        cft, cfh, cmq, cmr, cmp = calculate_propeller_coefficients_m2(
            propeller, V, beta, omega, propeller_dictionary, R
        )
    else:
        raise ValueError(
            "Method should be 1 for higher fidelity or 2 to lower fidelity"
        )

    dynPress = 0.5 * rho * (omega * R) ** 2
    average_coefficient = dynPress * pi * R**2

    ft = cft * average_coefficient
    fh = cfh * average_coefficient
    mq = cmq * average_coefficient * R
    mr = cmr * average_coefficient * R
    mp = cmp * average_coefficient * R

    return np.array([fh, 0.0, ft, mp, mq, mr])


################################################################################
## First method (Higher fidelity) ##
def calculate_propeller_coefficients_m1(
    propeller, V, beta, omega, propeller_dictionary, R, Nb
):
    """
    This function calculates the following coefficients of a propeller according to the paper:
    CFT - Equation 27
    CFH - Equation 33
    CMQ - Equation 37
    CMR - Equation 42
    CP - Equation 47

    *We are using SI units

    arguments
    propeller: The name of the propeller
    Nb: The number of blades
    V: incoming wind speed
    beta: angle between the rotor plane and incoming wind
    omega: propeller’s rotation rate
    propeller_dictionary: dict with data from the paper. Section3 dicts only!
    """

    (
        Cl0,
        Clalpha,
        Cd0,
        Cdalpha,
        Cm0,
        Cmalpha,
        delta,
        thetatip,
        ctip,
    ) = propeller_dictionary[propeller]

    if not propeller_dictionary:
        raise Exception("Could not import data fom the propeller")

    # mu and lambda_c according to eq 73
    mu = V * sin(beta) / (omega * R)
    lambda_c = V * cos(beta) / (omega * R)

    if mu >= 0.3:
        warnings.warn(
            "Attention, mu is bigger than 0.3 and the model might be no longer valid ! mu = "
            + str(mu)
        )

    # solidity
    sigma = Nb * ctip / (pi * R)

    # lambda_i according to eq 68
    lambda_i_1 = ((delta - 1) * sigma / delta) * (
        (-8 * Cl0 * delta * (1 + delta))
        + Clalpha
        * (
            Clalpha * (delta - 1) * delta * sigma
            - (8 * (2 * delta + mu**2) * thetatip)
        )
    )
    lambda_i_2 = (
        16 * lambda_c**2
        + 8 * Clalpha * (delta - 1) * lambda_c * sigma
        + lambda_i_1
        - 8 * Cl0 * sigma * log(delta) * mu**2
    )
    lambda_i = (1 / 8) * (
        -4 * lambda_c + Clalpha * sigma * (delta - 1) + lambda_i_2**0.5
    )

    # lambda_t calculation according to eq 9
    lambda_t = (
        lambda_i + lambda_c
    )  # the _t is used here to avoid confusion with lambda functions from python

    if lambda_t >= 0.3:
        warnings.warn(
            "Attention, lambda is bigger than 0.3 and the model might be no longer valid ! lambda = "
            + str(lambda_t)
        )

    # CFT from equation 27
    cft_1 = (1 - delta) * (
        (Cl0 * delta * (1 + delta))
        + (-2 * Clalpha * delta * (lambda_t - thetatip))
        + (Clalpha * thetatip * mu**2)
    )
    cft_2 = -Cl0 * delta * log(delta) * mu**2
    cft = sigma / (2 * delta) * (cft_1 + cft_2)

    # CFH from equation 33
    cfh = (mu * sigma / (2 * delta)) * (
        (1 - delta)
        * (
            2 * Cd0 * delta
            + thetatip * ((Clalpha - 2 * Cdalpha) * lambda_t + 2 * Cdalpha * thetatip)
        )
        - Cl0 * delta * lambda_t * log(delta)
    )

    # CMQ from equation 37
    cmq = (
        sigma
        * (1 - delta)
        / 6
        * (
            2 * Cd0 * (1 + delta + delta**2)
            + 3 * Cl0 * (delta + 1) * lambda_t
            + 6
            * (Cdalpha * (lambda_t - thetatip) - Clalpha * lambda_t)
            * (lambda_t - thetatip)
            + 3 * mu**2 * (Cd0 * delta + Cdalpha * thetatip**2) / delta
        )
    )

    # CMR from equation 42
    cmr = (sigma * mu * (1 - delta) / 2) * (
        Cl0 * (1 + delta) - Clalpha * (lambda_t - 2 * thetatip)
    )

    # CMP from equation 47
    cmp = (
        sigma
        * mu
        * ctip
        / (2 * delta * R)
        * (
            Cmalpha * (delta - 1) * (lambda_t - 2 * thetatip)
            - 2 * Cm0 * delta * log(delta)
        )
    )

    return cft, cfh, cmq, cmr, cmp


################################################################################
# Second method (lower fidelity) ##
def calculate_propeller_coefficients_m2(
    propeller, V, beta, omega, propeller_dictionary, R
):
    """
         This function calculates the following coefficients of a propeller according to the paper:
         CFT - Equation 95
         CFH - Equation 99
         CMQ - Equation 100
         CMR - Equation 101
         CP - Equation 102

         *We are using SI units

     arguments
    propeller: The name of the propeller
    V: incoming wind speed
    beta: angle between the rotor plane and incoming wind
    omega: propeller’s rotation rate
    propeller_dictionary: dict with data from the paper. Section5 dicts only!

    """

    (
        CstaticFT,
        k1,
        k2,
        k3,
        k4,
        k5,
        CstaticMQ,
        k6,
        k7,
        k8,
        k9,
        k10,
        k11,
        k12,
    ) = propeller_dictionary[propeller]

    if not propeller_dictionary:
        raise Exception("Could not import data fom the propeller")

    # mu and lambda_c according to eq 73
    mu = V * sin(beta) / (omega * R)
    lambda_c = V * cos(beta) / (omega * R)

    if mu >= 0.3:
        warnings.warn(
            "Attention, mu is bigger than 0.3 and the model might be no longer valid ! mu = "
            + str(mu)
        )

    if lambda_c >= 0.3:
        warnings.warn(
            "Attention, lambda is bigger than 0.3 and the model might be no longer valid ! lambda = "
            + str(lambda_c)
        )

    # CFT from equation 95
    cft = CstaticFT + k1 * lambda_c + k2 * mu**2 + k3 * lambda_c**2

    # CFH from equation 99
    cfh = k4 * mu + k5 * lambda_c * mu

    # CMQ from equation 100
    cmq = CstaticMQ + k6 * lambda_c + k7 * mu**2 + k8 * lambda_c**2

    # CMR from equation 101
    cmr = k9 * mu + k10 * lambda_c * mu

    # CMP from equation 102
    cmp = k11 * mu + k12 * lambda_c * mu

    return cft, cfh, cmq, cmr, cmp


################################################################################


def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > 0.04 or i % (int(1 / (24 * timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i * timestep):
            time.sleep(timestep * i - elapsed)


################################################################################


def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif val.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "[ERROR] in str2bool(), a Boolean value is expected"
        )


################################################################################


def nnlsRPM(
    thrust,
    x_torque,
    y_torque,
    z_torque,
    counter,
    max_thrust,
    max_xy_torque,
    max_z_torque,
    a,
    inv_a,
    b_coeff,
    gui=False,
):
    """Non-negative Least Squares (NNLS) RPMs from desired thrust and torques.

    This function uses the NNLS implementation in `scipy.optimize`.

    Parameters
    ----------
    thrust : float
        Desired thrust along the drone's z-axis.
    x_torque : float
        Desired drone's x-axis torque.
    y_torque : float
        Desired drone's y-axis torque.
    z_torque : float
        Desired drone's z-axis torque.
    counter : int
        Simulation or control iteration, only used for printouts.
    max_thrust : float
        Maximum thrust of the quadcopter.
    max_xy_torque : float
        Maximum torque around the x and y axes of the quadcopter.
    max_z_torque : float
        Maximum torque around the z axis of the quadcopter.
    a : ndarray
        (4, 4)-shaped array of floats containing the motors configuration.
    inv_a : ndarray
        (4, 4)-shaped array of floats, inverse of a.
    b_coeff : ndarray
        (4,1)-shaped array of floats containing the coefficients to re-scale thrust and torques.
    gui : boolean, optional
        Whether a GUI is active or not, only used for printouts.

    Returns
    -------
    ndarray
        (4,)-shaped array of ints containing the desired RPMs of each propeller.

    """
    #### Check the feasibility of thrust and torques ###########
    if gui and thrust < 0 or thrust > max_thrust:
        print(
            "[WARNING] iter",
            counter,
            "in utils.nnlsRPM(), unfeasible thrust {:.2f} outside range [0, {:.2f}]".format(
                thrust, max_thrust
            ),
        )
    if gui and np.abs(x_torque) > max_xy_torque:
        print(
            "[WARNING] iter",
            counter,
            "in utils.nnlsRPM(), unfeasible roll torque {:.2f} outside range [{:.2f}, {:.2f}]".format(
                x_torque, -max_xy_torque, max_xy_torque
            ),
        )
    if gui and np.abs(y_torque) > max_xy_torque:
        print(
            "[WARNING] iter",
            counter,
            "in utils.nnlsRPM(), unfeasible pitch torque {:.2f} outside range [{:.2f}, {:.2f}]".format(
                y_torque, -max_xy_torque, max_xy_torque
            ),
        )
    if gui and np.abs(z_torque) > max_z_torque:
        print(
            "[WARNING] iter",
            counter,
            "in utils.nnlsRPM(), unfeasible yaw torque {:.2f} outside range [{:.2f}, {:.2f}]".format(
                z_torque, -max_z_torque, max_z_torque
            ),
        )
    B = np.multiply(np.array([thrust, x_torque, y_torque, z_torque]), b_coeff)
    sq_rpm = np.dot(inv_a, B)
    #### NNLS if any of the desired ang vel is negative ########
    if np.min(sq_rpm) < 0:
        sol, res = nnls(a, B, maxiter=3 * a.shape[1])
        if gui:
            print(
                "[WARNING] iter",
                counter,
                "in utils.nnlsRPM(), unfeasible squared rotor speeds, using NNLS",
            )
            print(
                "Negative sq. rotor speeds:\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(
                    sq_rpm[0], sq_rpm[1], sq_rpm[2], sq_rpm[3]
                ),
                "\t\tNormalized: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(
                    sq_rpm[0] / np.linalg.norm(sq_rpm),
                    sq_rpm[1] / np.linalg.norm(sq_rpm),
                    sq_rpm[2] / np.linalg.norm(sq_rpm),
                    sq_rpm[3] / np.linalg.norm(sq_rpm),
                ),
            )
            print(
                "NNLS:\t\t\t\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(
                    sol[0], sol[1], sol[2], sol[3]
                ),
                "\t\t\tNormalized: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(
                    sol[0] / np.linalg.norm(sol),
                    sol[1] / np.linalg.norm(sol),
                    sol[2] / np.linalg.norm(sol),
                    sol[3] / np.linalg.norm(sol),
                ),
                "\t\tResidual: {:.2f}".format(res),
            )
        sq_rpm = sol
    return np.sqrt(sq_rpm)
