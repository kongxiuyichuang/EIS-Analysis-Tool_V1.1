# DDT: Distribution of Diffusion Times
# ----------------------------------------------------------------------
# Impedance-space kernels and matrix construction for DRT-DDT joint fitting.
#
# The kernels implemented here are the impedance-space counterparts (Z = 1/Y)
# of the admittance kernels of the DDT model of
# J. Song and M. Z. Bazant, Phys. Rev. Lett. 120, 116001 (2018).
#
# For a single diffusion element with diffusion time tau and anomalous exponent
# p the dimensionless argument is u = (i * omega * tau)^p. The three supported
# kernel types are:
#     1: planar, transmissive (open) boundary   Y = u*tanh(u)  -> Z = coth(u)/u
#     2: cylindrical, blocking boundary         Y = u*I1(u)/I0(u) -> Z = I0(u)/(u*I1(u))
#     3: planar, transmissive (shorted)         Y = u*coth(u)  -> Z = tanh(u)/u
#
# With p = 0.5 the standard (Fickian) Warburg kernels are recovered. By allowing
# a grid of p values the DDT becomes a two-dimensional distribution q(tau_D, p)
# over diffusion times and anomalous exponents.
#
# Because Z depends only on the product omega*tau (and p), matrices can be built
# from a one-dimensional lookup over log(omega*tau) for efficiency.

import numpy as np
from scipy.special import ive

DDT_KERNEL_TYPES = (1, 2, 3)

# Default two-dimensional exponent grid: sub-diffusive to Fickian (p <= 0.5)
DEFAULT_DDT_P_GRID = np.linspace(0.1, 0.5, 17)


def get_ddt_impedance_func(kernel_type):
    """
    Get the single-element diffusion impedance function Z(u).

    :param int kernel_type: DDT kernel type (1, 2, or 3)
    :return: function u -> z, where u = (i*omega*tau)^p (complex array) and
        z is the complex impedance of a single diffusion element
    """
    if kernel_type not in DDT_KERNEL_TYPES:
        raise ValueError(f'Invalid DDT kernel type {kernel_type}. Options: {DDT_KERNEL_TYPES}')

    if kernel_type == 1:
        # Z = coth(u)/u
        def z(u):
            out = np.empty_like(u, dtype=complex)
            small = np.abs(u) < 0.3
            u2 = u[small] ** 2
            # coth(u)/u = 1/u^2 + 1/3 - u^2/45 + 2 u^4/945 - u^6/4725 + ...
            out[small] = 1 / u2 + 1 / 3 - u2 / 45 + 2 * u2 ** 2 / 945 - u2 ** 3 / 4725
            ur = u[~small]
            # coth(u) = (1 + e^{-2u}) / (1 - e^{-2u}); Re(u) > 0 for omega*tau > 0
            out_rest = np.empty_like(ur)
            large = np.abs(ur) >= 30
            out_rest[large] = 1 / ur[large]
            e2u = np.exp(-2 * ur[~large])
            out_rest[~large] = (1 + e2u) / ((1 - e2u) * ur[~large])
            out[~small] = out_rest
            return out
    elif kernel_type == 2:
        # Z = I0(u)/(u*I1(u)) = ive(0,u)/(u*ive(1,u))
        def z(u):
            out = np.empty_like(u, dtype=complex)
            tiny = np.abs(u) < 1e-4
            # Z ~ 2/u^2 + 1/4 for small u
            out[tiny] = 2 / u[tiny] ** 2
            large = np.abs(u) >= 1e7
            # For very large |u|, I0/I1 -> 1, so Z -> 1/u (semi-infinite Warburg).
            # This avoids overflow of the scaled Bessel functions.
            out[large] = 1 / u[large]
            mid = ~tiny & ~large
            ur = u[mid]
            out[mid] = ive(0, ur) / (ur * ive(1, ur))
            return out
    elif kernel_type == 3:
        # Z = tanh(u)/u
        def z(u):
            out = np.empty_like(u, dtype=complex)
            small = np.abs(u) < 0.3
            u2 = u[small] ** 2
            # tanh(u)/u = 1 - u^2/3 + 2 u^4/15 - 17 u^6/315 + 62 u^8/2835 + ...
            out[small] = 1 - u2 / 3 + 2 * u2 ** 2 / 15 - 17 * u2 ** 3 / 315 + 62 * u2 ** 4 / 2835
            large = np.abs(u) >= 30
            out[large] = 1 / u[large]
            mid = ~small & ~large
            out[mid] = np.tanh(u[mid]) / u[mid]
            return out

    return z


def _construct_ddt_block(omega, tau, kernel_type, basis_type, p, ddt_epsilon, integrate_points):
    """
    Construct the DDT impedance matrix for a single exponent p.

    :param ndarray omega: angular frequencies (rad/s)
    :param ndarray tau: diffusion time grid (s)
    :param int kernel_type: DDT kernel type
    :param str basis_type: 'delta' or 'gaussian'
    :param float p: anomalous exponent
    :return: complex matrix of shape (len(omega), len(tau))
    """
    z_func = get_ddt_impedance_func(kernel_type)

    if basis_type == 'delta':
        return z_func((1j * np.outer(omega, tau)) ** p)

    if ddt_epsilon is None:
        ddt_epsilon = 1 / np.mean(np.diff(np.log(tau)))

    # Integration range is set by the Gaussian envelope; kernel values outside
    # the envelope are negligible. Capped at 20 to avoid extreme arguments.
    y_lim = min(20.0, 5.0 / ddt_epsilon + 2.0)
    y = np.linspace(-y_lim, y_lim, integrate_points)
    phi = np.exp(-(ddt_epsilon * y) ** 2)

    zm = np.zeros((len(omega), len(tau)), dtype=complex)
    for m in range(len(tau)):
        wt = np.outer(omega, tau[m] * np.exp(y))
        zvals = z_func((1j * wt) ** p)
        zm[:, m] = np.trapezoid(zvals * phi, x=y, axis=1)

    return zm


# ----------------------------------------------------------------------
# 2D Gaussian basis (over ln(tau_D) and p)
# ----------------------------------------------------------------------
def _bilinear_interp(a_grid, p_grid, Z, a_pts, p_pts):
    """
    Bilinear interpolation of Z (n_a x n_p) at (a_pts, p_pts), vectorized over
    a_pts. p_pts may be scalar or an array broadcastable against a_pts.
    """
    ia = np.clip(np.searchsorted(a_grid, a_pts) - 1, 0, len(a_grid) - 2)
    ip = np.clip(np.searchsorted(p_grid, p_pts) - 1, 0, len(p_grid) - 2)
    a0 = a_grid[ia]
    a1 = a_grid[ia + 1]
    p0 = p_grid[ip]
    p1 = p_grid[ip + 1]
    wa = (a_pts - a0) / (a1 - a0)
    wp = (p_pts - p0) / (p1 - p0)
    Z00 = Z[ia, ip]
    Z10 = Z[ia + 1, ip]
    Z01 = Z[ia, ip + 1]
    Z11 = Z[ia + 1, ip + 1]
    return (Z00 * (1 - wa) * (1 - wp) + Z10 * wa * (1 - wp)
            + Z01 * (1 - wa) * wp + Z11 * wa * wp)


def _build_ddt_z_lookup(kernel_type, a_grid, p_grid):
    """
    Precompute the DDT kernel Z[k, l] = z((i*exp(a_k))^p_l) on a regular grid.

    The kernel depends only on a = ln(omega*tau) and p because
    u = (i*omega*tau)^p = exp(p*(a + i*pi/2)), so a two-dimensional lookup is
    complete for the 2D Gaussian basis integral.

    :param int kernel_type: DDT kernel type
    :param ndarray a_grid: ln(omega*tau) grid (real, ascending)
    :param ndarray p_grid: anomalous exponent grid (real, ascending)
    :return: complex array of shape (len(a_grid), len(p_grid))
    """
    z_func = get_ddt_impedance_func(kernel_type)
    A, P = np.meshgrid(a_grid, p_grid, indexing='ij')
    u = (1j * np.exp(A)) ** P
    return z_func(u)


def _construct_ddt_block_2d_gaussian(omega, tau, kernel_type, p_grid, ddt_epsilon, p_epsilon,
                                     n_gh=7, n_a=500, n_p=80):
    """
    Construct the 2D Gaussian DDT impedance matrix.

    Each column (tau_i, p_j) is
        B_ij(w) = int int z((i*w*tau_i*e^y)^(p_j+q)) G_tau(y) G_p(q) dy dq
    with G_tau(y) = exp(-(eps_tau y)^2) and G_p(q) = exp(-(eps_p q)^2).

    The double integral is evaluated with tensor-product Gauss-Hermite
    quadrature; the kernel is obtained by bilinear interpolation of a
    precomputed 2D lookup (z depends only on ln(w*tau) and p). The exponent
    p_j + q is clamped to the physical range [0, 0.5].

    :param ndarray omega: angular frequencies (rad/s)
    :param ndarray tau: diffusion time grid (s)
    :param int kernel_type: DDT kernel type
    :param ndarray p_grid: anomalous exponent grid
    :param float ddt_epsilon: inverse Gaussian width in ln(tau)
    :param float p_epsilon: inverse Gaussian width in p
    :return: complex matrix of shape (len(omega), len(tau) * len(p_grid))
    """
    if ddt_epsilon is None:
        ddt_epsilon = 1 / np.mean(np.diff(np.log(tau)))
    if p_epsilon is None:
        p_epsilon = 1 / np.mean(np.diff(np.asarray(p_grid, dtype=float)))

    # Lookup domain. a = ln(omega*tau) has no physical bound; the Gaussian
    # envelope (GH nodes |t| <= 3.75) extends the required a-range by 3.75/eps_tau.
    a_lo = np.min(np.log(np.outer(omega, tau))) - 3.75 / ddt_epsilon
    a_hi = np.max(np.log(np.outer(omega, tau))) + 3.75 / ddt_epsilon
    a_grid = np.linspace(a_lo, a_hi, n_a)
    # p is physical only in [0, 0.5]; no unphysical extension below 0.
    p_grid_lookup = np.linspace(0.0, 0.5, n_p)
    Z = _build_ddt_z_lookup(kernel_type, a_grid, p_grid_lookup)

    # Gauss-Hermite nodes/weights for int e^{-x^2} f(x) dx
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_gh)

    n_tau = len(tau)
    A = np.log(omega[None, :] * tau[:, None])          # (n_tau, M)

    zm = np.zeros((len(omega), n_tau * len(p_grid)), dtype=complex)
    for j, pj in enumerate(np.asarray(p_grid, dtype=float)):
        col = np.zeros((n_tau, len(omega)), dtype=complex)
        for k in range(n_gh):
            a_pts = A + gh_x[k] / ddt_epsilon          # (n_tau, M)
            a_pts = np.clip(a_pts, a_grid[0], a_grid[-1])
            for l in range(n_gh):
                p_pts = np.clip(pj + gh_x[l] / p_epsilon, 0.0, 0.5)
                col += gh_w[k] * gh_w[l] * _bilinear_interp(a_grid, p_grid_lookup, Z, a_pts, p_pts)
        zm[:, j * n_tau:(j + 1) * n_tau] = col.T / (ddt_epsilon * p_epsilon)

    return zm


def _construct_ddt_block_gaussian_gh(omega, tau, kernel_type, p, ddt_epsilon, n_gh=7, n_a=500):
    """
    Fast 1D Gaussian DDT basis over ln(tau) via Gauss-Hermite quadrature and a
    precomputed lookup along a = ln(omega*tau) at fixed exponent p.

    B_m(w) = int z((i*w*tau_m*e^y)^p) exp(-(eps*y)^2) dy

    :param ndarray omega: angular frequencies (rad/s)
    :param ndarray tau: diffusion time grid (s)
    :param int kernel_type: DDT kernel type
    :param float p: anomalous exponent
    :param float ddt_epsilon: inverse Gaussian width in ln(tau)
    :return: complex matrix of shape (len(omega), len(tau))
    """
    if ddt_epsilon is None:
        ddt_epsilon = 1 / np.mean(np.diff(np.log(tau)))

    a_lo = np.min(np.log(np.outer(omega, tau))) - 3.75 / ddt_epsilon
    a_hi = np.max(np.log(np.outer(omega, tau))) + 3.75 / ddt_epsilon
    a_grid = np.linspace(a_lo, a_hi, n_a)

    z_func = get_ddt_impedance_func(kernel_type)
    Z = z_func((1j * np.exp(a_grid)) ** p)          # 1D lookup at fixed p

    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_gh)
    A = np.log(omega[None, :] * tau[:, None])       # (n_tau, M)

    col = np.zeros((len(tau), len(omega)), dtype=complex)
    for k in range(n_gh):
        a_pts = np.clip(A + gh_x[k] / ddt_epsilon, a_grid[0], a_grid[-1])
        ia = np.clip(np.searchsorted(a_grid, a_pts) - 1, 0, len(a_grid) - 2)
        a0 = a_grid[ia]
        a1 = a_grid[ia + 1]
        wa = (a_pts - a0) / (a1 - a0)
        zvals = Z[ia] * (1 - wa) + Z[ia + 1] * wa
        col += gh_w[k] * zvals

    return col.T / ddt_epsilon


def get_ddt_impedance_matrix_components(frequencies, basis_tau_dd, kernel_type, p=0.5):
    """
    Evaluate the DDT impedance kernel matrix (delta basis) at the given
    frequencies, diffusion-time grid, and anomalous exponent p.

    :param ndarray frequencies: measurement frequencies (Hz)
    :param ndarray basis_tau_dd: diffusion time grid (s)
    :param int kernel_type: DDT kernel type
    :param float p: anomalous exponent
    :return: complex matrix of shape (len(frequencies), len(basis_tau_dd))
    """
    omega = 2 * np.pi * np.asarray(frequencies, dtype=float)
    tau = np.asarray(basis_tau_dd, dtype=float)
    z_func = get_ddt_impedance_func(kernel_type)
    return z_func((1j * np.outer(omega, tau)) ** p)


def get_ddt_impedance_matrix(frequencies, basis_tau_dd, kernel_type, basis_type='delta',
                             ddt_epsilon=None, integrate_points=1000, p=0.5, p_grid=None,
                             p_epsilon=None):
    """
    Construct the DDT impedance matrix.

    For basis_type 'delta', each column is the impedance of a single diffusion
    element with diffusion time basis_tau_dd[n] and exponent p.

    For basis_type 'gaussian', each column is the impedance obtained by
    integrating the single-element kernel against a Gaussian RBF. In 1D mode
    (p_grid is None) the Gaussian is over y = ln(tau / tau_m); in 2D mode it is
    a two-dimensional Gaussian over both ln(tau / tau_m) and p.

    If p_grid is None, a one-dimensional DDT matrix (scalar p, default 0.5) is
    returned. If p_grid is an array, a two-dimensional DDT is returned with the
    (tau_D, p) basis flattened column-major over p (i.e. columns are ordered by
    p index, and within each p block by tau index).

    :param ndarray frequencies: measurement frequencies (Hz)
    :param ndarray basis_tau_dd: diffusion time grid (s)
    :param int kernel_type: DDT kernel type
    :param str basis_type: 'delta' or 'gaussian'
    :param float ddt_epsilon: inverse length scale for the Gaussian basis
    :param int integrate_points: number of integration points for Gaussian basis
    :param float p: anomalous exponent (used when p_grid is None)
    :param ndarray p_grid: anomalous exponent grid (enables 2D mode)
    :param float p_epsilon: inverse Gaussian width in p (2D gaussian basis)
    :return: complex matrix of shape (len(frequencies), len(basis_tau_dd)) or
        (len(frequencies), len(basis_tau_dd) * len(p_grid))
    """
    if basis_type not in ('delta', 'gaussian'):
        raise ValueError(f'Invalid basis_type {basis_type}. Options: delta, gaussian')

    omega = 2 * np.pi * np.asarray(frequencies, dtype=float)
    tau = np.asarray(basis_tau_dd, dtype=float)

    if p_grid is None:
        return _construct_ddt_block(omega, tau, kernel_type, basis_type, p,
                                    ddt_epsilon, integrate_points)

    p_grid = np.asarray(p_grid, dtype=float)

    if basis_type == 'delta':
        blocks = [
            _construct_ddt_block(omega, tau, kernel_type, 'delta', pj,
                                 ddt_epsilon, integrate_points)
            for pj in p_grid
        ]
        return np.hstack(blocks)

    # 2D Gaussian basis
    return _construct_ddt_block_2d_gaussian(omega, tau, kernel_type, p_grid,
                                            ddt_epsilon, p_epsilon)


def get_ddt_basis_area(basis_type, ddt_epsilon=None, basis_tau_dd=None, p_grid=None,
                       p_epsilon=None):
    """
    Get the total area (integral over ln tau and p) of the DDT basis functions.

    For the 'delta' basis the area per grid point is the grid spacing; for the
    'gaussian' basis it is the area of the Gaussian RBF. In 2D mode the p-grid
    spacing (delta) or p-Gaussian area multiplies the per-tau area.
    """
    if basis_type == 'delta':
        if basis_tau_dd is None:
            raise ValueError('basis_tau_dd must be provided for delta DDT basis')
        area = np.mean(np.diff(np.log(basis_tau_dd)))
    elif basis_type == 'gaussian':
        if ddt_epsilon is None:
            raise ValueError('ddt_epsilon must be provided for gaussian DDT basis')
        area = np.sqrt(np.pi) / ddt_epsilon
    else:
        raise ValueError(f'Invalid basis_type {basis_type}. Options: delta, gaussian')

    if p_grid is not None:
        p_grid = np.asarray(p_grid, dtype=float)
        if basis_type == 'delta':
            area = area * np.mean(np.diff(p_grid))
        else:
            if p_epsilon is None:
                p_epsilon = 1 / np.mean(np.diff(p_grid))
            area = area * (np.sqrt(np.pi) / p_epsilon)

    return area
