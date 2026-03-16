import numpy as np


# ---------------------------------------------------------------------
# Equality-constrained least squares via QR / nullspace method
# ---------------------------------------------------------------------
def solve_eq_ls_qr(A, b, Cw=None, dw=None, tol=1e-12):
    """
    Solve the equality-constrained least-squares problem

        minimize   0.5 * ||A x - b||_2^2
        subject to Cw x = dw

    using a QR-based nullspace method:

        Cw^T = Q [R11; 0],   Q = [Q1 Q2]

    Then any feasible x can be written as
        x = Q1 y + Q2 z
    with y chosen to satisfy the constraints and z solving a reduced
    unconstrained least-squares problem.

    Parameters
    ----------
    A : (m, n) ndarray
    b : (m,) ndarray
    Cw : (k, n) ndarray or None
        Active-constraint matrix.
    dw : (k,) ndarray or None
        Active-constraint RHS.
    tol : float
        Rank tolerance for the active-constraint matrix.

    Returns
    -------
    x : (n,) ndarray
        Equality-constrained least-squares solution.
    mu : (k,) ndarray
        Multipliers for the active constraints in the convention:
            grad f(x) - Cw.T @ mu = 0,
        where f(x) = 0.5 * ||A x - b||^2.
        Empty if no active constraints.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape

    if Cw is None or len(Cw) == 0:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x, np.zeros(0, dtype=float)

    Cw = np.asarray(Cw, dtype=float)
    dw = np.asarray(dw, dtype=float).reshape(-1)
    k = Cw.shape[0]

    if k > n:
        raise ValueError("Too many active constraints: k > n.")

    # QR factorization of Cw^T
    # Cw^T = Q R, with Q in R^{n x n}, R in R^{n x k}
    Q, R = np.linalg.qr(Cw.T, mode="complete")
    Q1 = Q[:, :k]
    Q2 = Q[:, k:]
    R11 = R[:k, :]   # upper triangular if rows of Cw are independent

    # Check rank of active constraints
    diag = np.abs(np.diag(R11)) if k > 0 else np.array([])
    if k > 0 and np.any(diag < tol):
        raise np.linalg.LinAlgError(
            "Active constraints appear linearly dependent (rank deficient)."
        )

    # Solve for y from feasibility:
    #   Cw x = dw
    #   => R11^T y = dw
    try:
        y = np.linalg.solve(R11.T, dw)
    except np.linalg.LinAlgError:
        y, *_ = np.linalg.lstsq(R11.T, dw, rcond=None)

    x_part = Q1 @ y

    # Solve reduced unconstrained LS in the nullspace component z:
    #   minimize || A(Q1 y + Q2 z) - b ||_2
    if Q2.shape[1] == 0:
        z = np.zeros(0, dtype=float)
        x = x_part
    else:
        A_red = A @ Q2
        b_red = b - A @ x_part
        z, *_ = np.linalg.lstsq(A_red, b_red, rcond=None)
        x = x_part + Q2 @ z

    # Recover multipliers mu from stationarity:
    #   grad f(x) - Cw^T mu = 0
    #   g = A^T (A x - b) = Cw^T mu = Q1 R11 mu
    r = A @ x - b
    g = A.T @ r

    if k == 0:
        mu = np.zeros(0, dtype=float)
    else:
        rhs_mu = Q1.T @ g
        try:
            mu = np.linalg.solve(R11, rhs_mu)
        except np.linalg.LinAlgError:
            mu, *_ = np.linalg.lstsq(R11, rhs_mu, rcond=None)

    return x, mu


# ---------------------------------------------------------------------
# Core active-set solver: requires a feasible starting point x0
# ---------------------------------------------------------------------
def _active_set_lsi_feasible(
    A:np.ndarray, b:np.ndarray, C:np.ndarray, d:np.ndarray,
    x0:np.ndarray,
    W0:np.ndarray|None=None,
    tol:float=1e-10,
    max_iter:int=100,
    verbose:bool=False
) -> tuple[np.ndarray, dict]:
    """
    Active-set method for

        minimize   0.5 * ||A x - b||_2^2
        subject to C x >= d

    assuming x0 is already feasible.
    
    Return:
        x : (n,) ndarray
            Approximate optimizer.
        info : dict
            Diagnostics, including:
            - "status": string describing the termination status
            - "objective": final objective value
            - "working_set": list of indices of active constraints at termination
            - "slack": (p,) ndarray of final constraint slacks C x - d
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)
    x = np.asarray(x0, dtype=float).reshape(-1).copy()

    p = C.shape[0]

    # Check feasibility of provided start
    slack0 = C @ x - d
    if np.any(slack0 < -tol):
        raise ValueError("x0 must be feasible for _active_set_lsi_feasible().")

    if W0 is None:
        W = [i for i in range(p) if abs((C[i] @ x) - d[i]) <= tol]
    else:
        W = sorted(set(W0))

    def objective(x_):
        r = A @ x_ - b
        return 0.5 * np.dot(r, r)

    status = "max_iter_exceeded"

    for it in range(1, max_iter + 1):
        Cw = C[W, :] if len(W) > 0 else None
        dw = d[W] if len(W) > 0 else None

        # Solve equality-constrained LS for current working set
        x_trial, mu = solve_eq_ls_qr(A, b, Cw, dw, tol=tol)

        slack_trial = C @ x_trial - d
        feasible_trial = np.all(slack_trial >= -tol)

        if feasible_trial:
            # If trial point is feasible and all active multipliers are nonnegative,
            # KKT conditions are satisfied => optimal.
            x = x_trial

            if verbose:
                min_mu = np.min(mu) if len(mu) > 0 else np.nan
                print(
                    f"[it={it}] feasible trial, "
                    f"obj={objective(x):.6e}, |W|={len(W)}, min(mu)={min_mu}"
                )

            if len(W) == 0 or np.all(mu >= -tol):
                status = "optimal"
                break

            # Otherwise drop the most negative active multiplier
            drop_local = int(np.argmin(mu))
            drop_global = W[drop_local]

            if verbose:
                print(f"  dropping constraint {drop_global} (mu={mu[drop_local]:.3e})")

            W.pop(drop_local)
            continue

        # Infeasible trial:
        # Move from the current feasible x toward x_trial until a new constraint blocks.
        pdir = x_trial - x
        inactive = [i for i in range(p) if i not in W]

        alphas = []
        blockers = []

        for i in inactive:
            ci_p = C[i] @ pdir
            # Constraint may tighten only if ci_p < 0
            if ci_p < -tol:
                alpha_i = (d[i] - C[i] @ x) / ci_p
                if tol < alpha_i <= 1.0 + tol:
                    alphas.append(alpha_i)
                    blockers.append(i)

        if len(alphas) == 0:
            status = "stalled_no_blocker"
            break

        alpha = min(alphas)
        blocking_constraints = [
            blockers[j] for j, a in enumerate(alphas)
            if abs(a - alpha) <= 10 * tol
        ]

        x = x + alpha * pdir

        for idx in blocking_constraints:
            if idx not in W:
                W.append(idx)
        W = sorted(set(W))

        if verbose:
            print(
                f"[it={it}] infeasible trial -> step alpha={alpha:.6e}, "
                f"add blockers={blocking_constraints}, |W|={len(W)}"
            )

    info = {
        "status": status,
        "objective": objective(x),
        "working_set": W,
        "slack": C @ x - d,
    }
    return x, info


# ---------------------------------------------------------------------
# Phase I: find a feasible point automatically if x0 is None
# ---------------------------------------------------------------------
def phase1_find_feasible_point(C, d, tol=1e-10, max_iter=100, verbose=False):
    """
    Phase I feasibility problem:

        minimize   0.5 * t^2
        subject to C x + t * 1 >= d
                   t >= 0

    Variables are y = [x; t].

    If the optimal t <= tol, then x is feasible for Cx >= d.
    If the optimal t > tol, then the original problem is infeasible.

    Returns
    -------
    x_feas : (n,) ndarray
        Feasible point for the original constraints.
    info_phase1 : dict
        Diagnostics from the Phase I solve.
    """
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)
    p, n = C.shape

    # Objective: minimize 0.5 * t^2 = 0.5 * ||A1 y - b1||^2
    # with y = [x; t], A1 extracts the t component
    A1 = np.zeros((1, n + 1), dtype=float)
    A1[0, -1] = 1.0
    b1 = np.zeros(1, dtype=float)

    # Constraints:
    #   C x + t * 1 >= d
    #   t >= 0
    C_phase = np.hstack([C, np.ones((p, 1), dtype=float)])
    d_phase = d.copy()

    t_row = np.zeros((1, n + 1), dtype=float)
    t_row[0, -1] = 1.0

    C_phase = np.vstack([C_phase, t_row])
    d_phase = np.concatenate([d_phase, np.array([0.0])])

    # Feasible start for Phase I:
    # x = 0, t0 large enough so that Cx + t0 >= d and t0 >= 0
    t0 = max(0.0, np.max(d)) + 1.0
    y0 = np.zeros(n + 1, dtype=float)
    y0[-1] = t0

    y_star, info = _active_set_lsi_feasible(
        A1, b1, C_phase, d_phase,
        x0=y0,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose
    )

    t_star = y_star[-1]
    x_star = y_star[:-1]

    info_phase1 = dict(info)
    info_phase1["t_star"] = t_star

    if t_star > 10 * tol:
        raise ValueError(
            f"Phase I failed: original inequality system appears infeasible "
            f"(minimum t = {t_star:.3e} > 0)."
        )

    return x_star, info_phase1


# ---------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------
def active_set_lsi(
    A:np.ndarray, b:np.ndarray, C:np.ndarray, d:np.ndarray,
    x0:np.ndarray|None=None,
    W0:np.ndarray|None=None,
    tol:float=1e-10,
    max_iter:int=100,
    verbose:bool=False
) -> tuple[np.ndarray, dict]:
    """
    Solve

        minimize   0.5 * ||A x - b||_2^2
        subject to C x >= d

    using an active-set method.

    If x0 is None, a feasible starting point is obtained automatically
    using a Phase I solve.

    Parameters
    ----------
    A : (m, n) ndarray
    b : (m,) ndarray
    C : (p, n) ndarray
    d : (p,) ndarray
    x0 : (n,) ndarray or None
        Feasible initial point. If None, Phase I is used.
    W0 : list[int] or None
        Initial working set; ignored if x0 is None and Phase I is used.
    tol : float
        Tolerance for feasibility and multiplier tests.
    max_iter : int
        Maximum iterations for the main solve (and also Phase I).
    verbose : bool
        Print diagnostics if True.

    Returns
    -------
    x : (n,) ndarray
        Approximate optimizer.
    info : dict
        Diagnostics, including:
        - "status": string describing the termination status
        - "objective": final objective value
        - "working_set": list of indices of active constraints at termination
        - "slack": (p,) ndarray of final constraint slacks C x - d
        - "x0": initial point used for the main solve (result of Phase I if x0 was None)
        - "phase1": diagnostics from Phase I if x0 was None, absent otherwise
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).reshape(-1)

    if x0 is None:
        if verbose:
            print("No x0 supplied -> running Phase I to construct a feasible start.")
        x0, phase1_info = phase1_find_feasible_point(
            C, d, tol=tol, max_iter=max_iter, verbose=verbose
        )
    else:
        phase1_info = None
        x0 = np.asarray(x0, dtype=float).reshape(-1)

    x, info = _active_set_lsi_feasible(
        A, b, C, d,
        x0=x0,
        W0=W0,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose
    )

    info["x0"] = x0
    info["phase1"] = phase1_info
    return x, info


def indi_lsi_wrapper(v, umin, umax, B, u_guess, W_init, Wv, Wu, up, gamma_sq:float=100000, imax=100):
    Av = gamma_sq * np.diag(Wv) @ B
    Au = np.diag(Wu)
    A = np.vstack((Av, Au))
        
    bv = gamma_sq * np.diag(Wv) @ v
    bu = np.diag(Wu) @ up
    b = np.hstack((bv, bu))

    n_u = len(umin)
    n_c = 2*n_u
    C = np.zeros((n_c, n_u))
    d = np.zeros(n_c)
    for i in range(n_u):
        C[i, i] = 1.0  # u >= u_min
        d[i] = umin[i]
        C[n_u + i, i] = -1.0  # -u >= -u_max that is u <= u_max
        d[n_u + i] = -umax[i]
        

    return active_set_lsi(A, b, C, d, x0=u_guess, W0=W_init, tol=1e-10, max_iter=imax, verbose=False)

# ---------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from wls_alloc import wls_alloc
    
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

    du_wls, it = wls_alloc(v, dumin, dumax, A, u_guess, W_init, Wv, Wu, up)
    du_lsi, info = indi_lsi_wrapper(v, dumin, dumax, A, u_guess, W_init, Wv, Wu, up)
    
    print(f"Control increment from WLS : {du_wls}")
    print(f"Control increment from LSI : {du_lsi}")