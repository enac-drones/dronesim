from dronesim.control.pywls import wls_alloc as _wls_alloc
import numpy as np


def _wls_score(v, B, Wv, Wu, u_pref, gamma_sq, u_sol):
    Wu_mat = np.diag(Wu) if Wu is not None else np.eye(len(u_sol))
    Wv_mat = np.diag(Wv) if Wv is not None else np.eye(len(v))
    
    return gamma_sq * np.linalg.norm(Wv_mat @ (B @ u_sol - v))**2 + np.linalg.norm(Wu_mat @ (u_sol - u_pref))**2

def scipy_wls_alloc(
    v:np.ndarray,umin:np.ndarray, umax:np.ndarray,
    B:np.ndarray, 
    u_guess:np.ndarray|None=None, W_init:np.ndarray|None=None, 
    Wv:np.ndarray|None=None, Wu:np.ndarray|None=None,
    u_pref:np.ndarray|None=None, gamma_sq:float=100000.0, imax:int=100):
    
    from scipy import optimize as opt
    if u_pref is None:
        u_pref = np.zeros_like(umin)
    Wu_mat = np.diag(Wu) if Wu is not None else np.eye(len(umin))
    Wv_mat = np.diag(Wv) if Wv is not None else np.eye(len(v))
    A = np.vstack((np.sqrt(gamma_sq) * Wv_mat @ B,  Wu_mat))
    b = np.hstack((np.sqrt(gamma_sq) * Wv_mat @ v, Wu_mat @ u_pref))
    
    sol = opt.lsq_linear(A, b, bounds=(umin, umax),
                method='bvls', tol=1e-6)
    
    if sol.success:
        return sol.x, sol.nit
    else:
        return u_pref, imax+1
    


def wls_alloc(
    v:np.ndarray, u_min:np.ndarray, u_max:np.ndarray, B:np.ndarray,
    u_guess:np.ndarray|None=None, W_init:np.ndarray|None=None,
    Wv:np.ndarray|None=None, Wu:np.ndarray|None=None, u_pref:np.ndarray|None=None,
    gamma_sq:float=100000.0, imax:int=100,
):
    """
    Python wrapper for the C-bound wls_alloc.
    
    Solve the following optimal control allocation problem:
    minimize_u gamma_sq * || Wv * (B u - v) ||^2 + || Wu * (u - u_pref) ||^2

    Parameters:
        v: (nv,) Target force vector
        u_min: (nu,) Min input values
        u_max: (nu,) Max input values
        B: (nv, nu) Control effectiveness matrix
        u_guess: (nu,) Initial guess for u (optional)
        W_init: (nu,) Initial weights for u (optional)
        Wv: (nv,) vector of weights for v (optional)
        Wu: (nu,) vector of weights for u_pref (optional)
        u_pref: (nu,) vector of preferred u values (optional)
        gamma_sq: Squared weight for prefering control allocation to u_pref (optional, default: 100000.0)
        imax: maximum number of iterations (optional, default: 100)
    Returns:
        u: (nu,) vector of allocated controls
        n_iter: number of iterations taken
    """
    nv, nu = B.shape
    if Wv is None:
        Wv = np.ones(nv, dtype=np.float32)
    if Wu is None:
        Wu = np.ones(nu, dtype=np.float32)
    if u_pref is None:
        u_pref = np.zeros(nu, dtype=np.float32)

    u_sol,it = _wls_alloc(
        B.astype(np.float32),
        v.astype(np.float32),
        u_min.astype(np.float32),
        u_max.astype(np.float32),
        u_guess.astype(np.float32) if u_guess is not None else None,
        W_init.astype(np.float32) if W_init is not None else None,
        Wv.astype(np.float32),
        Wu.astype(np.float32),
        u_pref.astype(np.float32),
        gamma_sq,
        imax,
    )
    
    scipy_sol, scipy_it = scipy_wls_alloc(
        v, u_min, u_max, B, u_guess, W_init, Wv, Wu, u_pref, gamma_sq, imax
    )
    
    wls_score = _wls_score(v, B, Wv, Wu, u_pref, gamma_sq, u_sol)
    scipy_score = _wls_score(v, B, Wv, Wu, u_pref, gamma_sq, scipy_sol)
    print(f"WLS Score: {wls_score:.6f} (it={it}), Scipy Score: {scipy_score:.6f} (it={scipy_it})")
    print(f"Score difference: {abs(wls_score - scipy_score):.6f}")
    print(f"WLS Solution: {u_sol}, Scipy Solution: {scipy_sol}")
    print(f"Solution difference: {u_sol - scipy_sol}")
    return u_sol, it
    



