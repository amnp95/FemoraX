"""Stabilized LF-LTS (Grote, Michel & Sauter, Math. Comp. 2021, arXiv:2005.13350), 1D setup of
their section 5: wave equation on (0,1), c = 1, Dirichlet ends, P1 with mass lumping, coarse
(0, 0.9) with h_c, fine [0.9, 1) with h_f = h_c / p. Prototype for FemoraX's LTS plan."""
import numpy as np
from numpy.polynomial import chebyshev as C

def mesh(hc, p):
    nc = int(round(0.9 / hc)); nf = int(round(0.1 / (hc / p)))
    x = np.concatenate([np.linspace(0, 0.9, nc + 1), np.linspace(0.9, 1.0, nf + 1)[1:]])
    h = np.diff(x); n = len(x)
    K = np.zeros((n, n)); m = np.zeros(n)
    for e, he in enumerate(h):
        K[e:e+2, e:e+2] += np.array([[1, -1], [-1, 1]]) / he
        m[e:e+2] += he / 2
    inner = slice(1, n - 1)                       # Dirichlet ends removed
    K = K[inner, inner]; m = m[inner]; xi = x[inner]
    fine = (xi >= 0.9 - 1e-12).astype(float)      # Sigma_f: nodes in the closed fine region
    return xi, K, m, fine

def T(k, x): return C.chebval(x, [0] * k + [1])
def dT(k, x): return C.chebval(x, C.chebder([0] * k + [1]))

def lts_step(A, fine, u_prev, u, dt, p, nu):
    """The algorithm of their section 3.3 (steps 1-4), with A = M^-1 K."""
    d = 1 + nu / p**2; om = 2 * dT(p, d) / T(p, d)
    w = A @ ((1 - fine) * u)
    z_prev, z = u, u - 0.5 * (dt / p)**2 * (2 * p**2 / (om * d)) * (w + A @ (fine * u))
    for k in range(1, p):
        bk = T(k - 1, d) / T(k + 1, d); bkh = T(k, d) / T(k + 1, d)
        z_prev, z = z, (1 + bk) * z - bk * z_prev - (dt / p)**2 * (2 * p**2 / om) * bkh * (w + A @ (fine * z))
    return -u_prev + 2 * z

def Apnu(A, fine, dt, p, nu):
    """A^{S,p,nu} column by column: u_next = 2u - u_prev - dt^2 A^{p,nu} u."""
    n = A.shape[0]; I = np.eye(n)
    return np.column_stack([(2 * I[:, j] - lts_step(A, fine, np.zeros(n), I[:, j], dt, p, nu)) / dt**2 for j in range(n)])

def spectrum(hc, p, nu, cfl):
    xi, K, m, fine = mesh(hc, p); A = K / m[:, None]; dt = cfl * hc
    ev = np.linalg.eigvals(dt**2 * Apnu(A, fine, dt, p, nu))
    return np.sort(ev.real), np.abs(ev.imag).max()

if __name__ == "__main__":
    import sys
    # 1. One level (p = 1) is leapfrog: A^{1,nu} = A.
    xi, K, m, fine = mesh(1/40, 1); A = K / m[:, None]
    print("p=1: |A^{1,nu} - A| / |A| =", np.abs(Apnu(A, fine, 0.9/40, 1, 0.05) - A).max() / np.abs(A).max())
    # 2. The first critical step of the unstabilized scheme, p = 2 (paper: 0.7104 h_c):
    #    the smallest dt/h_c where an eigenvalue of dt^2 A^{p,0} touches 0 or 4.
    for p, hc, paper in ((2, 1/40, "0.7104 h_c"), (3, 1/40, "0.5011 dt_opt (dt_opt ~ h_c)")):
        cfls = np.linspace(0.30, 0.99, 1381); dist = []
        for c in cfls:
            ev, _ = spectrum(hc, p, 0.0, c)
            dist.append(min(ev.min(), 4 - ev.max(), np.min(np.abs(ev[1:-1] - 4)) if len(ev) > 2 else 9))
        dist = np.array(dist)
        loc = [i for i in range(1, len(dist) - 1) if dist[i] < dist[i-1] and dist[i] < dist[i+1] and dist[i] < 2e-3]
        print(f"p={p}, nu=0: first near-critical dt/h_c = {cfls[loc[0]] if loc else None} (min distance "
              f"{dist[loc[0]] if loc else dist.min():.1e}); paper {paper}")
