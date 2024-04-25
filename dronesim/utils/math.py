import numpy as np


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


def norm_ang(x):
    while x > np.pi:
        x -= 2 * np.pi
    while x < -np.pi:
        x += 2 * np.pi
    return x