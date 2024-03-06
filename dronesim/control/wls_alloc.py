# /*
#  * Copyright (C) Anton Naruta && Daniel Hoppener
#  * MAVLab Delft University of Technology
#  *
#  * This file is part of paparazzi.
#  *
#  * paparazzi is free software; you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation; either version 2, or (at your option)
#  * any later version.
#  *
#  * paparazzi is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with paparazzi; see the file COPYING.  If not, write to
#  * the Free Software Foundation, 59 Temple Place - Suite 330,
#  * Boston, MA 02111-1307, USA.
#  */

"""
@file wls_alloc.c
@brief This is an active set algorithm for WLS control allocation

This algorithm will find the optimal inputs to produce the least error wrt
the control objective, taking into account the weighting matrices on the
control objective and the control effort.

The algorithm is described in:
Prioritized Control Allocation for Quadrotors Subject to Saturation -
E.J.J. Smeur, D.C. Höppener, C. de Wagter. In IMAV 2017
written by Anton Naruta && Daniel Hoppener 2016
MAVLab Delft University of Technology
"""

# include "wls_alloc.h"
# include <stdio.h>
# include "std.h"

# include <string.h>
# include <math.h>
# include <float.h>
# include "math/qr_solve/qr_solve.h"
# include "math/qr_solve/r8lib_min.h"


# Problem size needs to be predefined to avoid having to use VLAs
# ifndef CA_N_V
# error CA_N_V needs to be defined!
# endif

# ifndef CA_N_U
# error CA_N_U needs to be defined!
# endif

# define CA_N_C  (CA_N_U+CA_N_V)

# /**
#  * @brief Wrapper for qr solve
#  *
#  * Possible to use a different solver if needed.
#  * Solves a system of the form Ax = b for x.
#  *
#  * @param m number of rows
#  * @param n number of columns
#  */
# // void qr_solve_wrapper(int m, int n, float** A, float* b, float* x) {
# //   float in[m * n];
# //   // convert A to 1d array
# //   int k = 0;
# //   for (int j = 0; j < n; j++) {
# //     for (int i = 0; i < m; i++) {
# //       in[k++] = A[i][j];
# //     }
# //   }
# //   // use solver
# //   qr_solve(m, n, in, b, x);
# // }

import numpy as np

# CA_N_U = 6
# CA_N_V = 4
# CA_N_C = CA_N_U + CA_N_V
FLT_EPSILON = 1e-7
INFINITY = 1e32


def qr_solve(A, b):
    """Solves a system of the form Ax = b for x."""
    q, r = np.linalg.qr(A)
    p = np.dot(q.T, b)
    return np.dot(np.linalg.pinv(r), p)


# /**
#  * @brief active set algorithm for control allocation
#  *
#  * Takes the control objective and max and min inputs from pprz and calculates
#  * the inputs that will satisfy most of the control objective, subject to the
#  * weighting matrices Wv and Wu
#  *
#  * @param u The control output vector
#  * @param v The control objective
#  * @param umin The minimum u vector
#  * @param umax The maximum u vector
#  * @param B The control effectiveness matrix
#  * @param n_u Length of u
#  * @param n_v Lenght of v
#  * @param u_guess Initial value for u
#  * @param W_init Initial working set, if known
#  * @param Wv Weighting on different control objectives
#  * @param Wu Weighting on different controls
#  * @param up Preferred control vector
#  * @param gamma_sq Preference of satisfying control objective over desired
#  * control vector (sqare root of gamma)
#  * @param imax Max number of iterations
#  *
#  * @return Number of iterations, -1 upon failure
#  */


def wls_alloc(v, umin, umax, B, u_guess, W_init, Wv, Wu, up, gamma_sq=100000, imax=100):
    # Allocate variables, use defaults where parameters are set to 0
    # if gamma_sq==None : gamma_sq = 100000
    # if imax == None   : imax     = 100
    CA_N_U = len(umin)
    CA_N_V = len(v)
    CA_N_C = CA_N_U + CA_N_V

    n_c = CA_N_C
    n_u = CA_N_U
    n_v = CA_N_V

    A = np.zeros((CA_N_C, CA_N_U))
    A_free = np.zeros((CA_N_C, CA_N_U))

    # Create a pointer array to the rows of A_free
    # such that we can pass it to a function
    # A_free_ptr = np.zeros((CA_N_C, CA_N_U), dtype=int)

    # for i in range(n_c):
    # A_free_ptr[i,:] = A_free[i,:]

    b = np.zeros(CA_N_C)
    d = np.zeros(CA_N_C)

    free_index = np.zeros(CA_N_U, dtype=int)
    free_index_lookup = np.zeros(CA_N_U, dtype=int)
    n_free = 0
    free_chk = -1

    iter = 0
    p_free = np.zeros(CA_N_U)
    p = np.zeros(CA_N_U)
    u = np.zeros(CA_N_U)
    u_opt = np.zeros(CA_N_U)
    infeasible_index = np.zeros(CA_N_U, dtype=int)  # UNUSED
    n_infeasible = 0
    Lambda = np.zeros(CA_N_U)
    W = np.zeros(CA_N_U)

    # Initialize u and the working set, if provided from input
    if u_guess is None:
        for i in range(n_u):
            u[i] = (umax[i] + umin[i]) * 0.5
    else:
        # for i in range(n_u):
        #   u[i] = u_guess[i]
        u = u_guess.copy()

    if W_init is not None:
        W = W_init.copy()
    else:
        W = np.zeros(n_u)

    free_index_lookup = np.ones(n_u, dtype=int) * -1

    # Find free indices
    for i in range(n_u):
        if W[i] == 0:
            free_index_lookup[i] = n_free
            free_index[n_free] = i  # WHAT IS THIS ???
            n_free += 1  # $$$$$######@@@@$$$$%%%%
            # print(f'n_free : {n_free}')

    # Fill up A, A_free, b and d
    for i in range(n_v):
        # If Wv is a NULL pointer, use Wv = identity
        if Wv is not None:
            b[i] = gamma_sq * Wv[i] * v[i]
        else:
            b[i] = gamma_sq * v[i]
        d[i] = b[i]
        for j in range(n_u):
            # If Wv is a NULL pointer, use Wv = identity
            if Wv is not None:
                A[i][j] = gamma_sq * Wv[i] * B[i][j]
            else:
                A[i][j] = gamma_sq * B[i][j]
            d[i] -= A[i][j] * u[j]

    for i in range(n_v, n_c):
        A[i, :] = 0  # , n_u * sizeof(float));
        if Wu is not None:
            A[i][i - n_v] = Wu[i - n_v]
        else:
            A[i][i - n_v] = 1.0

        if up is not None:
            if Wu is not None:
                b[i] = Wu[i - n_v] * up[i - n_v]
            else:
                b[i] = up[i - n_v]
        else:
            b[i] = 0
        d[i] = b[i] - A[i][i - n_v] * u[i - n_v]

    # -------------- Start loop ------------
    while iter < imax:
        iter += 1
        # clear p, copy u to u_opt
        p = np.zeros(n_u)  # * sizeof(float));
        u_opt = u.copy()  # , n_u * sizeof(float));
        # print(f'u : {u}')
        # print(f'u_opt : {u_opt}')
        # print(f'free_index : {free_index}')
        # print(f'n_free : {n_free}')

        # Construct a matrix with the free columns of A
        if free_chk != n_free:
            for i in range(n_c):
                for j in range(n_free):
                    # print(f'Free_index[j] : {free_index[j]}')
                    A_free[i][j] = A[i][free_index[j]]
            free_chk = n_free

        # print('A_free : ', A_free_ptr)
        # print('d : ', d)

        if n_free:
            # Still free variables left, calculate corresponding solution
            # Use a solver to find the solution to A_free*p_free = d
            # print('A_free : ', A_free_ptr)
            # print('A : ', A)
            # print('d : ', d)
            # print(f'n_c : {n_c}, n_free : {n_free}, A_free_ptr.shape : {A_free_ptr.shape} ')

            # p_free = qr_solve(A_free[:n_c,:n_free], d)
            p_free = np.linalg.lstsq(A_free[:n_c, :n_free], d, rcond=None)[0]
            # print(f'p_free : {p_free}')
            # p_free = np.linalg.solve(A_free_ptr[:], d)

        # Set the nonzero values of p and add to u_opt
        for i in range(n_free):
            p[free_index[i]] = p_free[i]
            u_opt[free_index[i]] += p_free[i]

        # check limits
        n_infeasible = 0
        for i in range(n_u):
            if u_opt[i] >= (umax[i] + 1.0) or u_opt[i] <= (umin[i] - 1.0):
                infeasible_index[n_infeasible] = i
                n_infeasible += 1

        # Check feasibility of the solution
        if n_infeasible == 0:
            # all variables are within limits
            u = u_opt.copy()
            Lambda = np.zeros(n_u)

            # d = d + A_free*p_free; lambda = A*d;
            for i in range(n_c):
                for k in range(n_free):
                    d[i] -= A_free[i][k] * p_free[k]

                for k in range(n_u):
                    Lambda[k] += A[i][k] * d[i]

            break_flag = True

            # lambda = lambda x W;
            for i in range(n_u):
                Lambda[i] *= W[i]
                # if any lambdas are negative, keep looking for solution
                if Lambda[i] < -FLT_EPSILON:
                    break_flag = False
                    W[i] = 0
                    # add a free index
                    if free_index_lookup[i] < 0:
                        free_index_lookup[i] = n_free
                        free_index[n_free] = i
                        n_free += 1
            if break_flag:
                # if solution is found, return number of iterations
                return u, iter
        else:
            alpha = INFINITY  # ???
            alpha_tmp = 0.0
            id_alpha = 0

        # Find the lowest distance from the limit among the free variables
        for i in range(n_free):
            id = free_index[i]
            if np.abs(p[id]) > FLT_EPSILON:
                if p[id] < 0:
                    alpha_tmp = (umin[id] - u[id]) / p[id]
                else:
                    alpha_tmp = (umax[id] - u[id]) / p[id]
            else:
                alpha_tmp = INFINITY

            if alpha_tmp < alpha:
                alpha = alpha_tmp
                id_alpha = id

        # update input u = u + alpha*p
        for i in range(n_u):
            u[i] += alpha * p[i]

        # update d = d-alpha*A*p_free
        for i in range(n_c):
            k_len = min(
                n_free, len(p_free)
            )  # FIXME : Pfff this should be fixed! Somehow k is becoming bigger than the p_free length...
            for k in range(k_len):  # Normally should be range(n_free)
                # print(f'Dangerous place ! i : {i} , k : {k}, A shape : {A_free.shape} , p_free shape : {p_free.shape}')
                d[i] -= (
                    A_free[i][k] * alpha * p_free[k]
                )  # having problem here with i:0 k:1, A(8,4) p_free:(1,) : IndexError: index 1 is out of bounds for axis 0 with size 1

        # get rid of a free index
        if p[id_alpha] > 0:
            W[id_alpha] = 1.0
        else:
            W[id_alpha] = -1.0

        # print(n_free, id_alpha)
        # print(n_free, free_index[n_free], id_alpha, free_index_lookup[id_alpha] )
        n_free -= 1
        free_index[free_index_lookup[id_alpha]] = free_index[n_free]
        free_index_lookup[free_index[free_index_lookup[id_alpha]]] = free_index_lookup[
            id_alpha
        ]
        free_index_lookup[id_alpha] = -1

    # solution failed, return negative one to indicate failure
    return None, iter


if __name__ == "__main__":
    # v = np.array([0.5, 0.3, 20.2, 0.7])
    # umin = np.array([-107, -19093, 0, -95])
    # umax = np.array([9093, 107, 4600, 4505])
    # A = np.array([
    #   [      0,         0,  -0.0105,  0.0107016],
    #   [ -0.0030044, 0.0030044, 0.035, 0.035],
    #   [ -0.004856, -0.004856, 0, 0],
    #   [       0,         0,   -0.0011,   -0.0011] ])

    # up = np.array([1000., 1000., 1000., 1000.])
    # Wv = np.array([100, 1000, 0.1, 10])
    # Wu = np.array([1, 1, 1, 1])
    # # B = np.array([
    # #   [ 15.0,  15.0 , -15.0 , -15.0],
    # #   [-15.0,  15.0 ,  15.0 , -15.0],
    # #   [-5.0,  5.0 , -5.0 ,  5.0],
    # #   [ 0.7,  0.7 ,  0.7 ,  0.7]  ])
    # u_guess = None
    # W_init = None

    # # import scipy.optimize
    # # res = scipy.optimize.lsq_linear(A, v, bounds=(umin, umax), lsmr_tol='auto', verbose=1)
    # # print(f'LSQ_Lin : {res}')

    # du, it = wls_alloc(v, umin, umax, A/1000., u_guess, W_init, Wv, Wu, up)
    # print(f'Control increment : {du} and iteration : {it}')

    umin = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    umax = np.array([9600, 9600, 9600, 9600, 9600, 9600])
    uc = np.array([4614, 4210, 4210, 4614, 4210, 4210])
    dumin = umin - uc
    dumax = umax - uc
    up = dumin.copy()

    v = np.array([240, -240.5658, 600.0, 1.8532])
    Wv = np.array([100, 100, 1, 10])
    # Wu = np.array([1, 1, 1, 1, 1, 1, 1])
    Wu = None
    A = np.array(
        [
            [0.0, -0.015, 0.015, 0.0, -0.015, 0.015],
            [0.015, -0.010, -0.010, 0.015, -0.010, -0.010],
            [0.103, 0.103, 0.103, -0.103, -0.103, -0.103],
            [-0.0009, -0.0009, -0.0009, -0.0009, -0.0009, -0.0009],
        ]
    )

    u_guess = None
    W_init = None

    du, it = wls_alloc(v, dumin, dumax, A, u_guess, W_init, Wv, Wu, up)
    print(
        "Matlab lsqlin output : -4614.0, 426.064612091305, 5390.0, -4614.0, -4210.0, 5390.0 "
    )
    print(f"Control increment : {du} and iteration : {it}")
