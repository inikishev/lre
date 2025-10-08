import random
from typing import Literal

import numpy as np
import torch

from .compile import allow_compile


def regularize_eigh(
    L: torch.Tensor,
    Q: torch.Tensor,
    truncate: int | None = None,
    tol: float | None = None,
    damping: float = 0,
    rdamping: float = 0,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Applies regularization to eigendecomposition. Returns ``(L, Q)``.

    Args:
        L (torch.Tensor): eigenvalues, shape ``(rank,)``.
        Q (torch.Tensor): eigenvectors, shape ``(n, rank)``.
        truncate (int | None, optional):
            keeps top ``truncate`` eigenvalues. Defaults to None.
        tol (float | None, optional):
            all eigenvalues smaller than largest eigenvalue times ``tol`` are removed. Defaults to None.
        damping (float | None, optional): scalar added to eigenvalues. Defaults to 0.
        rdamping (float | None, optional): scalar multiplied by largest eigenvalue and added to eigenvalues. Defaults to 0.
    """
    # remove non-finite eigenvalues
    finite = L.isfinite()
    if finite.any():
        L = L[finite]
        Q = Q[:, finite]
    else:
        return None, None

    # largest finite!!! eigval
    L_max = L[-1] # L is sorted in ascending order

    # remove small eigenvalues relative to largest
    if tol is not None:
        indices = L > tol * L_max
        L = L[indices]
        Q = Q[:, indices]

    # truncate to rank (L is ordered in ascending order)
    if truncate is not None:
        L = L[-truncate:]
        Q = Q[:, -truncate:]

    # damping
    d = damping + rdamping * L_max
    if d != 0:
        L += d

    return L, Q


def eigh_plus_uuT(
    L: torch.Tensor,
    Q: torch.Tensor,
    u: torch.Tensor,
    alpha: float = 1,
    tol: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """
    compute eigendecomposition of Q L Q^T + alpha * (u u^T) where Q is ``(m, rank)`` and L is ``(rank, )`` and u is ``(m, )``

    returns ``(L, Q)``
    """
    if tol is None: tol = torch.finfo(Q.dtype).eps
    Qtu = Q.T @ u  # (rank,)

    # component of u orthogonal to the column space of Q
    res = u - Q @ Qtu # (m,)
    beta = torch.linalg.vector_norm(res) # pylint:disable=not-callable

    if beta < tol:
        # u is already in the column space of Q
        B = L.diag_embed().add_(Qtu.outer(Qtu), alpha=alpha) # (rank, rank)
        L_prime, S = torch.linalg.eigh(B) # pylint:disable=not-callable
        Q_prime = Q @ S
        return L_prime, Q_prime

    # normalize the orthogonal component to get a new orthonormal vector
    v = res / beta # (m, )

    # project and compute new eigendecomposition
    D_diag = torch.cat([L, torch.tensor([0.0], device=Q.device, dtype=Q.dtype)])
    w = torch.cat([Qtu, beta.unsqueeze(0)]) # Shape: (rank+1,)
    B = D_diag.diag_embed().add_(w.outer(w), alpha=alpha)

    try:
        L_prime, S = torch.linalg.eigh(B) # pylint:disable=not-callable
    except torch.linalg.LinAlgError:
        return None, None

    # unproject and sort
    basis = torch.cat([Q, v.unsqueeze(-1)], dim=1) # (m, rank+1)
    Q_prime = basis @ S # (m, rank+1)

    idx = torch.argsort(L_prime)
    L_prime = L_prime[idx]
    Q_prime = Q_prime[:, idx]

    return L_prime, Q_prime

OrthoMethod = Literal["ns5", 'ns1', 'ns2', 'ns3', 'ns4', "svd", "qr"]

def eigh_plus_UUt(
    L: torch.Tensor,
    Q: torch.Tensor,
    U: torch.Tensor,
    alpha: float | torch.Tensor = 1,
    tol = None,
    ortho_method: OrthoMethod = 'qr',
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """
    compute eigendecomposition of Q L Q^T + alpha * (U U^T), where Q is ``(m, rank)`` and L is ``(rank, )``,
    U is ``(m, k)`` where k is rank of correction

    returns ``(L, Q)``
    """
    if U.size(1) == 1:
        return eigh_plus_uuT(L, Q, U[:,0], alpha=float(alpha), tol=tol)

    # make alpha shape (k, )
    k = U.size(1)
    if isinstance(alpha, torch.Tensor):
        alpha = torch.broadcast_to(alpha, (k, ))
    else:
        alpha = torch.full((k,), float(alpha), device=U.device, dtype=U.dtype)

    if tol is None: tol = torch.finfo(Q.dtype).eps
    m, r = Q.shape
    QtU = Q.T @ U  # (r, k)
    U_res = U - Q @ QtU  # (m, k)

    # find cols of U not in col space of Q
    res_norms = torch.linalg.vector_norm(U_res, dim=0) # pylint:disable=not-callable
    new_indices = torch.where(res_norms > tol)[0]
    k_prime = len(new_indices)

    if k_prime == 0:
        # all cols are in Q
        B = Q
        C = QtU # (r x k)
        r_new = r
    else:
        # orthonormalize directions that aren't in Q
        U_new = U_res[:, new_indices]
        Q_u = orthogonalize(U_new, method=ortho_method) # pylint:disable=not-callable
        B = torch.hstack([Q, Q_u])
        C = torch.vstack([QtU, Q_u.T @ U_res])
        r_new = r + k_prime

    # project and compute new eigendecomposition
    A_proj = torch.zeros((r_new, r_new), device=Q.device, dtype=Q.dtype)
    A_proj[:r, :r] = L.diag_embed()
    # A_proj += (C @ C.T).mul_(alpha)
    A_proj.addmm_(C * alpha, C.T)

    try:
        L_prime, S = torch.linalg.eigh(A_proj) # pylint:disable=not-callable
    except torch.linalg.LinAlgError:
        return None, None

    # unproject and sort
    Q_prime = B @ S
    idx = torch.argsort(L_prime)
    L_prime = L_prime[idx]
    Q_prime = Q_prime[:, idx]

    return L_prime, Q_prime

def eigh_plus_UVt_symmetrize(
    L: torch.Tensor,
    Q: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    alpha: float | torch.Tensor = 1,
    ortho_method: OrthoMethod = 'qr',
):
    """
    Q is ``(m, rank)``; L is ``(rank, )``; U and V are the low rank correction such that U V^T is ``(m, m)``.

    This computes eigendecomposition of A, where

    ``M = Q diag(L) Q^T + alpha * (U V^T)``;

    ``A = (M + M^T) / 2``

    returns ``(L, Q)``

    so apparrenly just symmetrizing U and V as Y=(U+Y)/2 works much better
    """
    m, rank = Q.shape
    _, k = V.shape

    # project U and V out of the Q subspace via Gram-schmidt
    QtU = Q.T @ U
    U_perp = U - Q @ QtU

    QtV = Q.T @ V
    V_perp = V - Q @ QtV

    R = torch.hstack([U_perp, V_perp])
    Q_perp = orthogonalize(R, method=ortho_method) # pylint:disable=not-callable

    Q_B = torch.hstack([Q, Q_perp])
    r_B = Q_B.shape[1]

    # project, symmetrize and compute new eigendecomposition
    A_proj = torch.zeros((r_B, r_B), device=Q.device, dtype=Q.dtype)
    A_proj[:rank, :rank] = L.diag_embed()

    QptU = Q_perp.T @ U
    QBTU = torch.vstack([QtU, QptU])

    QptV = Q_perp.T @ V
    QBTV = torch.vstack([QtV, QptV])

    update_proj = QBTU @ QBTV.T + QBTV @ QBTU.T
    A_proj += update_proj * (alpha/2)

    L_prime, S = torch.linalg.eigh(A_proj) # pylint:disable=not-callable

    # unproject and sort
    Q_prime = Q_B @ S

    idx = torch.argsort(L_prime)
    L_prime = L_prime[idx]
    Q_prime = Q_prime[:, idx]

    return L_prime, Q_prime



def eigh_plus_UUt_mm(
    # A1 = Q @ diag(L) @ Q.T
    L: torch.Tensor,
    Q: torch.Tensor,

    # A2 = U @ U.T
    U: torch.Tensor,

    # rhs
    B: torch.Tensor,

    # weights
    w1: float,
    w2: float | torch.Tensor,

) -> torch.Tensor:
    """
    Computes ``(w1 * (Q L Q^T) + (U diag(w2) U^T) @ B``,

    Q is ``(m, rank)``, L is ``(rank, rank)``, U is ``(m, z)``, B is ``(m, k)``.

    Returns ``(m, k)``
    """
    # sketch Q L Q^T
    QtB = Q.T @ B # (rank, k)
    LQtB = L.unsqueeze(1) * QtB  # (rank, k)
    sketch1 = Q @ LQtB  # (m, k)

    # skecth U U^T
    UtB = U.T @ B # (z, k)
    if isinstance(w2, torch.Tensor) and w2.numel() > 1: w2UtB = w2.unsqueeze(-1) * UtB
    else:  w2UtB = w2 * UtB
    sketch2 = U @ w2UtB # (m, k)

    return w1 * sketch1 + sketch2


def randomized_eigh_plus_UUt(
    L1: torch.Tensor,
    Q1: torch.Tensor,
    U: torch.Tensor,
    w1: float,
    w2: float | torch.Tensor,
    oversampling_p: int,
    rank: int,
    eig_tol: float,
    damping: float,
    rdamping: float,
    ortho_method: OrthoMethod = 'qr',
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    compute randomized eigendecomposition of w1 * Q L Q^T + w2 * (U U^T),
    where Q is ``(m, rank)`` and L is ``(rank, )``,
    U is ``(m, k)`` where k is rank of correction, returns ``(L, Q)``
    """
    n = Q1.shape[0]
    device = Q1.device
    dtype = Q1.dtype
    l = rank + oversampling_p

    # gaussian test matrix
    Omega = torch.randn(n, l, device=device, dtype=dtype)

    # sketch
    AOmega = eigh_plus_UUt_mm(L1, Q1, U, Omega, w1, w2)
    Q = orthogonalize(AOmega, ortho_method)

    AQ = eigh_plus_UUt_mm(L1, Q1, U, Q, w1, w2)
    QtAQ = Q.T @ AQ

    W = (QtAQ + QtAQ.T) / 2.0

    # compute new L and Q
    try:
        L_prime, S = torch.linalg.eigh(W) # pylint:disable=not-callable
    except torch.linalg.LinAlgError:
        return L1, Q1

    L_prime, S = regularize_eigh(L=L_prime, Q=S, truncate=rank, tol=eig_tol, damping=damping, rdamping=rdamping)

    if L_prime is None or S is None:
        return L1, Q1

    return L_prime, Q @ S



def tall_reduced_svd_via_eigh(A: torch.Tensor, tol: float = 0, retry_float64:bool=False):
    """
    A is (m,n), computes U and S from the reduced SVD(A) using the eigendecomposition of (n, n)
    """
    # if m < n, A.T A will be low rank and svd is faster
    m, n = A.size()
    if m < n:
        U, S, V = torch.linalg.svd(A, full_matrices=False) # pylint:disable=not-callable
        return U, S

    M = A.mH @ A # n,n

    try:
        L, Q = torch.linalg.eigh(M) # pylint:disable=not-callable
    except torch.linalg.LinAlgError:
        U, S, V = torch.linalg.svd(A, full_matrices=False) # pylint:disable=not-callable
        return U, S

    L = torch.flip(L, dims=[-1])
    Q = torch.flip(Q, dims=[-1])

    indices = L > tol * L[0] # L[0] is the max eigenvalue
    L = L[indices]
    Q = Q[:, indices]

    S = L.sqrt()
    U = (A @ Q) / S

    return U, S

# https://arxiv.org/pdf/2110.02820
def nystrom_approximation(
    Omega: torch.Tensor,
    AOmega: torch.Tensor,
    eigv_tol: float = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Nyström approximation to positive-semidefinite A factored as Q L Q^T (truncatd eigenvalue decomp),
    returns ``(L, Q)``.

    A is ``(m,m)``, O and AO are ``(m, rank)``, then Q is ``(m, rank)``;
    L is a ``(rank, )`` vector - diagonal of ``(rank, rank)``

    to get Omega and AOmega:
    ```py
    Omega = torch.randn((ndim, rank), device=device, dtype=dtype, generator=generator)
    Omega = orthogonalize(Omega, method=orthogonalize_method) # Thin QR decomposition
    AOmega = A @ Omega # change to A_mm
    ```
    """
    v = torch.finfo(Omega.dtype).eps * torch.linalg.matrix_norm(AOmega, ord='fro') # Compute shift # pylint:disable=not-callable
    Yv = AOmega + v*Omega # Shift for stability
    C, info = torch.linalg.cholesky_ex(Omega.mT @ Yv) # pylint:disable=not-callable
    B = torch.linalg.solve_triangular(C, Yv.mT, upper=False, unitriangular=False).mT # pylint:disable=not-callable

    # Q, S, _ = torch_linalg.svd(B, full_matrices=False) # pylint:disable=not-callable
    # B is (ndim, rank) so we can use eigendecomp of (rank, rank)
    Q, S = tall_reduced_svd_via_eigh(B, tol=eigv_tol, retry_float64=True)

    L = S.pow(2) - v
    return L, Q


def nystrom_sketch_and_solve(
    L: torch.Tensor,
    Q: torch.Tensor,
    b: torch.Tensor,
    reg: float = 1e-3,
) -> torch.Tensor:
    """Solves ``(Q diag(L) Q.T + reg*I)x = b``. Becomes super unstable with reg smaller than like 1e-5.
    (this is from nystrom PCG paper)

    Args:
        L (torch.Tensor): eigenvalues
        Q (torch.Tensor): eigenvectors
        b (torch.Tensor): right hand side
        reg (float, optional): regularization. Defaults to 1e-3.
    """

    b = b.unsqueeze(-1)
    L += reg
    # x = (A + μI)⁻¹ b
    # (A + μI)⁻¹ = Q(L + μI)⁻¹Qᵀ + (1/μ)(b - QQᵀ)
    # x = Q(L + μI)⁻¹Qᵀb + (1/μ)(b - QQᵀb)
    Qᵀb = Q.T @ b
    term1 = Q @ ((1/L).unsqueeze(-1) * Qᵀb)
    term2 = (1.0 / reg) * (b - Q @ Qᵀb)
    return (term1 + term2).squeeze(-1)


def rank1_eigh(v: torch.Tensor):
    """returns ``(L, Q)`` of ``(v v^T)``"""
    vv = v.dot(v)
    norm = vv.sqrt().clip(min=torch.finfo(vv.dtype).tiny * 2)

    L = vv.unsqueeze(0) # (rank, )
    Q = v.unsqueeze(-1) / norm # (m, rank)

    return L, Q

def low_rank_eigh(U: torch.Tensor):
    """returns ``(L, Q)`` of ``alpha * (U U^T)`` (from GGT)"""
    M = U.T @ U
    L, S = torch.linalg.eigh(M) # pylint:disable=not-callable

    Q = U @ S
    Q /= torch.sqrt(L).clip(min=torch.finfo(L.dtype).tiny * 2)

    return L, Q




# zeropower_via_newtonschulz5 from:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
# and
# https://github.com/HomebrewML/HeavyBall/blob/main/heavyball/utils.py#L452
_NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012)
)
@allow_compile
def zeropower_via_newtonschulz5(G: torch.Tensor, coeffs=_NS_COEFFS) -> torch.Tensor:
    """
    Applies to last 2 dims - so usually reverse_dims should be applied to G before and after.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng

    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True).clip(min=torch.finfo(X.dtype).tiny * 2))

    # Perform the NS iterations
    for a,b,c in coeffs:
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X.to(G.dtype)

def zeropower_via_svd(A: torch.Tensor) -> torch.Tensor:
    """
    Applies to first 2 dims and isn't batched - rest of dimensions are flattened.
    """
    try:
        U, S, Vt = torch.linalg.svd(A, full_matrices=False) # pylint:disable=not-callable
    except torch.linalg.LinAlgError:
        U, S, Vt = torch.svd_lowrank(A, q=1, M=1e-4 * A.mean() * torch.rand_like(A))

    return  U @ Vt

def orthogonalize_via_qr(A: torch.Tensor):
    *_, m, n = A.shape
    T = False
    if m < n:
        T = True
        m,n = n,m
        A = A.mH

    Q = torch.linalg.qr(A, mode='reduced').Q # pylint:disable=not-callable

    if T:
        Q = Q.mH

    return Q

# since we keep Q, we can (probably) perform a single NS step per iteration so I use the fixed-coefficient newton-schulz
# this is from from https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py
# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
# github permlink: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5

@allow_compile
def zeropower_via_newtonschulz1(
    grad: torch.Tensor,
    ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
    ns_steps: int = 1,
) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Implementation reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
    with suggestions by @jxbz, @leloykun, and @YouJiacheng.
    """
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm is at most 1
    ortho_grad.div_(ortho_grad.norm().clip(min=torch.finfo(grad.dtype).tiny * 2))
    # Perform the NS iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad.to(grad.dtype)


def orthogonalize(A: torch.Tensor, method: OrthoMethod) -> torch.Tensor:
    if method == "ns5": return zeropower_via_newtonschulz5(A)
    if method == "ns1": return zeropower_via_newtonschulz1(A)
    if method == "ns2": return zeropower_via_newtonschulz1(A, ns_steps=2)
    if method == "ns3": return zeropower_via_newtonschulz1(A, ns_steps=3)
    if method == "ns4": return zeropower_via_newtonschulz1(A, ns_steps=4)
    if method == "svd": return zeropower_via_svd(A)
    if method == "qr": return orthogonalize_via_qr(A)
    raise ValueError(method)






def sanger_(Q:torch.Tensor, Y: torch.Tensor, lr:float, value_clip:float|None, norm_clip:float|None):
    """Q is ``(ndim, rank)``, Y is ``(ndim, batch_size)``

    https://en.wikipedia.org/wiki/Generalized_Hebbian_algorithm"""
    P = Q.T @ Y # (rank, batch_size)

    P = P.T # (batch_size, rank)
    Y = Y.T # (batch_size, ndim)
    #update = y.outer(p) - Q @ torch.tril(p.outer(p))
    g = (Y.unsqueeze(-1) * P.unsqueeze(-2) - Q @ torch.tril(P.unsqueeze(-1) @ P.unsqueeze(-2)))
    g = g.mean(0)

    if value_clip is not None:g = g.clip(-value_clip, value_clip)
    if norm_clip is not None:
        g_norm = g.norm()
        if g_norm > norm_clip:
            g = g * (norm_clip / g_norm.clip(min=torch.finfo(g.dtype).tiny * 2))

    Q.add_(g, alpha=lr)
    return Q # set reortho_freq=1

def sanger_stiefel_(Q: torch.Tensor, Y: torch.Tensor, lr, value_clip:float|None, norm_clip:float|None):
    """sanger but on stiefel manifold"""
    QtY = Q.T @ Y # (rank, batch_size)
    P = QtY.T # (batch_size, rank)
    Y = Y.T # (batch_size, ndim)

    # euclidian grad
    g_e = (Y.unsqueeze(-1) * P.unsqueeze(-2) - Q @ torch.tril(P.unsqueeze(-1) @ P.unsqueeze(-2)))
    g_e = g_e.mean(0)
    if value_clip is not None: g_e = g_e.clip(-value_clip, value_clip)

    A = Q.T @ g_e
    A = (A + A.T) / 2.0
    g_r = g_e - Q @ A # riemmanian grad

    # clip norm
    if norm_clip is not None:
        g_norm = g_r.norm()
        if g_norm > norm_clip:
            g_r = g_r * (norm_clip / g_norm.clip(min=torch.finfo(g_r.dtype).tiny * 2))

    Q.add_(g_r, alpha=lr)

    # HebbianLearning retracts (set reortho_freq=1)
    return Q

# ----------------------------------- AdamG ---------------------------------- #
# from https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform/blob/master/stiefel_optimizer.py
def norm(v, dim=1):
    assert len(v.size())==2
    return v.norm(p=2, dim=dim, keepdim=True)

def unit(v, dim=1, eps=1e-8):
    vnorm = norm(v, dim) # pylint:disable=not-callable
    return v/vnorm.add(eps), vnorm

def qr_retraction(tan_vec): # tan_vec, p-by-n, p <= n
    tan_vec.t_()
    q,r = torch.linalg.qr(tan_vec) # pylint:disable=not-callable
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    q.t_()
    return q

def matrix_norm_one(W):
    out = torch.abs(W)
    out = torch.sum(out, dim=0)
    out = torch.max(out)
    return out

def Cayley_loop(X, W, tan_vec, t): #
    [n, p] = X.size()
    Y = X + t * tan_vec
    for i in range(5):
        Y = X + t * torch.matmul(W, 0.5*(X+Y))

    return Y.t()

def adamg(p: torch.Tensor, g:torch.Tensor, state: dict, lr:float=1e-3, beta1:float=0.9, beta2:float=0.99, eps:float=1e-8, norm_clip:float|None=None, value_clip:float|None=None, cautious:bool=False):
    """https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform/blob/master/stiefel_optimizer.py"""
    unity,_ = unit(p.data.view(p.size()[0],-1))
    rand_num = random.randint(1,101)
    if rand_num==1:
        unity = qr_retraction(unity)

    g = g.view(g.size(0), -1)
    if value_clip is not None:
        g.clip_(-value_clip, value_clip)

    if norm_clip is not None:
        g_norm = g.norm()
        if g_norm > norm_clip:
            g = g * (norm_clip / g_norm.clip(min=torch.finfo(g.dtype).tiny * 2))

    if 'm_buffer' not in state:
        size=p.size()
        state['m_buffer'] = torch.zeros([int(np.prod(size[1:])), size[0]])
        state['v_buffer'] = torch.zeros([1])
        if p.is_cuda:
            state['m_buffer'] = state['m_buffer'].cuda()
            state['v_buffer'] = state['v_buffer'].cuda()

        state['beta1_power'] = beta1
        state['beta2_power'] = beta2

    m = state['m_buffer']
    if cautious:
        m = m * ((m * g.T) > 0)
    v = state['v_buffer']
    beta1_power = state['beta1_power']
    beta2_power = state['beta2_power']

    mnew = beta1*m  + (1.0-beta1)*g.t() # p by n
    vnew = beta2*v  + (1.0-beta2)*(torch.linalg.vector_norm(g)**2) # pylint:disable=not-callable

    mnew_hat = mnew / (1 - beta1_power)
    vnew_hat = vnew / (1 - beta2_power)

    MX = torch.matmul(mnew_hat, unity)
    XMX = torch.matmul(unity, MX)
    XXMX = torch.matmul(unity.t(), XMX)
    W_hat = MX - 0.5 * XXMX
    W = (W_hat - W_hat.t())/vnew_hat.add(eps).sqrt()

    t = 0.5 * 2 / (matrix_norm_one(W) + eps)
    alpha = min(t, lr)

    p_new = Cayley_loop(unity.t(), W, mnew, -alpha)

    mnew = torch.matmul(W, unity.t()) * vnew_hat.add(eps).sqrt() * (1 - beta1)
    m.copy_(mnew)
    v.copy_(vnew)

    state['beta1_power']*=beta1
    state['beta2_power']*=beta2

    return p_new

def sanger_adamg(Q:torch.Tensor, Y: torch.Tensor, state: dict, lr:float=1e-3, beta1:float=0.9, beta2:float=0.99, eps:float=1e-8, norm_clip:float|None=None, value_clip:float|None=None, cautious:bool=False):
    """Q is ``(ndim, rank)``, Y is ``(ndim, batch_size)``

    https://en.wikipedia.org/wiki/Generalized_Hebbian_algorithm"""
    P = Q.T @ Y # (rank, batch_size)

    P = P.T # (batch_size, rank)
    Y = Y.T # (batch_size, ndim)
    #update = y.outer(p) - Q @ torch.tril(p.outer(p))
    g = (Y.unsqueeze(-1) * P.unsqueeze(-2) - Q @ torch.tril(P.unsqueeze(-1) @ P.unsqueeze(-2)))
    g = g.mean(0)

    return adamg(Q, g, state=state, lr=lr, beta1=beta1, beta2=beta2, eps=eps, norm_clip=norm_clip, value_clip=value_clip, cautious=cautious)
