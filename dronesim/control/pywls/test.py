#!/usr/bin/python3

import numpy as np
from pywls import wls_alloc

# Example sizes
nv = 4
nu = 6

B = np.random.randn(nv, nu).astype(np.float32)
v = np.array([0.1, -0.2, 0.05, 0.0], dtype=np.float32)

u_min = np.full(nu, -1.0, dtype=np.float32)
u_max = np.full(nu,  1.0, dtype=np.float32)

u_pref = np.zeros(nu, dtype=np.float32)
Wv = np.ones(nv, dtype=np.float32)
Wu = np.ones(nu, dtype=np.float32)

u, n_iter = wls_alloc(
    B, v, u_min, u_max,u_guess=np.zeros(nu, dtype=np.float32), W_init=None,
    Wv=Wv, Wu=Wu, u_pref=u_pref,
    gamma_sq=100000.0,
    imax=100,
)

print("u =", u)
print("iterations =", n_iter)
