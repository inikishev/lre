import math
import random
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Literal

import torch

from .linalg import low_rank_eigh, rank1_eigh


def _cat_grads(params):
    return torch.cat([p.grad.ravel() if p.grad is not None else torch.zeros_like(p).ravel() for p in params])

def _cat_tensors(tensors):
    return torch.cat([t.ravel() for t in tensors])

def vec_to_tensors_(vec: torch.Tensor, tensors_: Iterable[torch.Tensor]):
    cur = 0
    for t in tensors_:
        numel = t.numel()
        t.set_(vec[cur:cur+numel].view_as(t)) # pyright: ignore[reportArgumentType]
        cur += numel


class Curvature(ABC):
    """computes low rank correction to curvature"""

    @abstractmethod
    def initialize(self, grad: torch.Tensor, state:dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """returns initial ``(Q, L)``"""

    @abstractmethod
    def update(
        self,
        loss: torch.Tensor | None,
        grad: torch.Tensor | None,
        closure,
        params_list: list[torch.Tensor],
        should_update: bool,
        Q: torch.Tensor | None,
        L: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """returns ``(loss, grad)``. This is because we might need hessian-vector products
        so this has to manually handle autograd"""

    @abstractmethod
    def compute_correction(self, Q:torch.Tensor, L: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """returns one ``(Y, Z, alpha)`` tuple, where Z may be None if correction is symmetric,
        so Y Y^T or Y Z^T is the correction"""

class Covariance(Curvature):
    """Covariance matrix. The correction is ``g g^T``, where ``g`` is the gradient.

    If in LRE ``update_freq`` is set to ``k > 1``, adds rank-k correction ``G G^T``,
    where ``G`` is last ``k`` gradients stacked into ``(ndim, k)``.

    Therefore rank of correction is ``update_freq``.
    """
    def __init__(self):
        self.history = []

    def initialize(self, grad, state):
        Y = torch.stack(self.history, 1)
        L, Q = low_rank_eigh(Y)
        return Q, L

    def update(self, loss, grad, closure, params_list, should_update, Q, L):
        if grad is None:
            with torch.enable_grad():
                if loss is None:
                    loss = closure()
                else:
                    for p in params_list: p.grad = None
                    loss.backward()
            grad = _cat_grads(params_list)

        self.history.append(grad)
        assert loss is not None
        return loss, grad, {}

    def compute_correction(self, Q, L):
        Y = torch.stack(self.history, 1)
        self.history.clear()
        return Y, None, torch.tensor(1., device=Y.device, dtype=Y.dtype)

class CovarianceTimesLoss(Curvature):
    """Covariance matrix. The correction is ``g' g'^T``, where ``g'`` is gradient times loss.

    If in LRE ``update_freq`` is set to ``k > 1``, adds rank-k correction ``G G^T``,
    where ``G`` is last ``k`` gradients stacked into ``(ndim, k)``.

    Therefore rank of correction is ``update_freq``.
    """
    def __init__(self, adaptive:bool=True):
        self.history = []
        self.adaptive = adaptive
        self.lowest_loss = None

    def initialize(self, grad, state):
        Y = torch.stack(self.history, 1)
        L, Q = low_rank_eigh(Y)
        return Q, L

    def update(self, loss, grad, closure, params_list, should_update, Q, L):
        assert grad is None
        with torch.enable_grad():
            if loss is None:
                loss = closure()
            else:
                for p in params_list: p.grad = None
                loss.backward()

        grad = _cat_grads(params_list)

        loss_hat = loss
        if self.adaptive and self.lowest_loss is not None:
            loss_hat = loss - self.lowest_loss

        self.history.append(grad * loss_hat)

        if self.lowest_loss is None or loss < self.lowest_loss:
            self.lowest_loss = loss.item()

        return loss, grad, {}

    def compute_correction(self, Q, L):
        Y = torch.stack(self.history, 1)
        self.history.clear()
        return Y, None, torch.tensor(1., device=Y.device, dtype=Y.dtype)

def _flatten_jacobian(jacs: Sequence[torch.Tensor]) -> torch.Tensor:
    """cats jacs to ``(output.size(0), wrt.numel())``"""
    if not jacs:
        return torch.empty(0, 0)

    n_out = jacs[0].shape[0]
    return torch.cat([j.reshape(n_out, -1) for j in jacs], dim=1)

class EmpiricalFisher(Curvature):
    """Empirical fisher matrix is the sum of outer products of per-sample gradients,
    and the total gradient is sum of per-sample gradients.

    Rank of correction is ``batch_size * update_freq``.

    Since we use an exponential moving average, this becomes a mix of empirical fisher and empirical gradient covariance.

    This requires per-sample losses vector to be passed to ``step`` in LRE, or that closure returns them.

    Per-sample gradients are only evaluated once every ``update_freq``. Otherwise it uses their sum.
    """
    def __init__(self):
        self.history = []

    def initialize(self, grad, state):
        Y = torch.stack(self.history, 1)
        L, Q = low_rank_eigh(Y)
        return Q, L

    def update(self, loss, grad, closure, params_list, should_update, Q, L):
        assert grad is None

        if should_update:
            with torch.enable_grad():
                if loss is None: loss = closure(False)

                I = torch.eye(loss.numel(), device=loss.device, dtype=loss.dtype)
                jac = _flatten_jacobian(torch.autograd.grad(loss.ravel(), params_list, I, is_grads_batched=True))

            grad = jac.sum(0)
            loss = loss.sum()

            self.history.extend(jac)
            return loss, grad, {"jac": jac}

        with torch.enable_grad():
            if loss is None: loss = closure(False)
            loss.sum().backward()
        grad = _cat_grads(params_list)
        return loss, grad, {}

    def compute_correction(self, Q, L):
        Y = torch.stack(self.history, 1)
        self.history.clear()
        return Y, None, torch.tensor(1., device=Y.device, dtype=Y.dtype)

class GaussNewton(Curvature):
    """Gauss-Newton matrix is the sum of outer products of per-residual (per-sample) gradients,
    and the total gradient is J^T r where J is per-residual gradients and r is per-residual losses.

    So the matrix is the same as Empirical fisher, but total gradient is different.

    Rank of correction is ``batch_size * update_freq``.

    This requires per-sample losses vector to be passed to ``step`` in LRE, or that closure returns them.

    Per-sample gradients are only evaluated once every ``update_freq``. Otherwise it uses sum of their squares.
    """
    def __init__(self):
        self.history = []

    def initialize(self, grad, state):
        Y = torch.stack(self.history, 1)
        L, Q = low_rank_eigh(Y)
        return Q, L

    def update(self, loss, grad, closure, params_list, should_update, Q, L):
        assert grad is None

        if should_update:
            with torch.enable_grad():
                if loss is None: loss = closure(False)
                loss = loss.ravel()

                I = torch.eye(loss.numel(), device=loss.device, dtype=loss.dtype)
                jac = _flatten_jacobian(torch.autograd.grad(loss, params_list, I, is_grads_batched=True))

            grad = jac.T @ loss
            loss = loss.sum()

            self.history.extend(jac)
            return loss, grad, {"jac": jac}

        with torch.enable_grad():
            if loss is None: loss = closure(False)
            loss.pow(2).sum().backward()
        grad = _cat_grads(params_list)
        return loss, grad, {}

    def compute_correction(self, Q, L):
        Y = torch.stack(self.history, 1)
        self.history.clear()
        return Y, None, torch.tensor(1., device=Y.device, dtype=Y.dtype)


class UnrolledGaussNewton(Curvature):
    """This keeps last ``history_size`` losses and gradients and assumes them to be residuals.

    The curvature is just covariance matrix, so correction is ``g g^T`` where ``g`` is the gradient,
    and rank of correction is ``update_freq``.

    However the gradient is computed as ``J^T @ r``, where ``J`` is past ``history_size`` gradients
    and ``r`` is past ``history_size`` losses.

    So the matrix is the same as empirical fisher, but total gradient is different.
    """
    def __init__(self, history_size: int = 10, adaptive:bool=False):
        self.loss_history = []
        self.grad_history = []
        self.history_size = history_size
        self.adaptive = adaptive
        self.lowest_loss = None
        self.last_correction_idx = 0

    def initialize(self, grad, state):
        Y = torch.stack(self.grad_history, 1)
        L, Q = low_rank_eigh(Y)
        return Q, L

    def update(self, loss, grad, closure, params_list, should_update, Q, L):
        assert grad is None
        with torch.enable_grad():
            if loss is None:
                loss = closure()
            else:
                for p in params_list: p.grad = None
                loss.backward()

        grad = _cat_grads(params_list)

        loss_hat = loss
        if self.adaptive and self.lowest_loss is not None:
            loss_hat = loss - self.lowest_loss

        self.grad_history.append(grad.detach())
        self.loss_history.append(loss_hat.detach())
        self.last_correction_idx += 1

        if self.lowest_loss is None or loss < self.lowest_loss:
            self.lowest_loss = loss.item()

        # now we compute unrolled GN grad
        J = torch.stack(self.grad_history[-self.history_size:], -1) # ndim, history_size
        r = torch.stack(self.loss_history[-self.history_size:]) # history_size
        gn_grad = J @ r

        return loss, gn_grad, {}

    def compute_correction(self, Q, L):
        Y = torch.stack(self.grad_history[-self.last_correction_idx:], 1)
        self.last_correction_idx = 0
        self.grad_history = self.grad_history[-self.history_size:]
        self.loss_history = self.loss_history[-self.history_size:]
        return Y, None, torch.tensor(1., device=Y.device, dtype=Y.dtype)


_z_type = Literal["normal", "rademacher", "s", "g", "mixed"]


@torch.no_grad
def _hvp_fd_central(
    closure,
    params: Iterable[torch.Tensor],
    x: Iterable[torch.Tensor],
    h=1e-3,
    normalize=True,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:

    params = list(params)
    x = list(x)

    vec_norm = None
    if normalize:
        vec_norm = torch.linalg.vector_norm(torch.cat([t.view(-1) for t in x])) # pylint:disable=not-callable
        if vec_norm == 0: return None, [torch.zeros_like(p) for p in params]
        x = torch._foreach_div(x, vec_norm)

    xh = torch._foreach_mul(x, h)
    torch._foreach_add_(params, xh)
    with torch.enable_grad(): loss = closure()
    g_plus = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    torch._foreach_sub_(params, xh)
    torch._foreach_sub_(params, xh)
    with torch.enable_grad(): loss = closure()
    g_minus = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    torch._foreach_add_(params, xh)
    for p in params: p.grad = None

    hx = g_plus
    torch._foreach_sub_(hx, g_minus)
    torch._foreach_div_(hx, 2*h)

    if normalize: torch._foreach_mul_(hx, vec_norm)
    return loss, hx

@torch.no_grad
def _hvp_fd_forward(
    closure,
    params: Iterable[torch.Tensor],
    x: Iterable[torch.Tensor],
    h=1e-3,
    g_0=None,
    normalize=True,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    params = list(params)
    x = list(x)
    loss = None

    vec_norm = None
    if normalize:
        vec_norm = torch.linalg.vector_norm(torch.cat([t.ravel() for t in x])) # pylint:disable=not-callable
        if vec_norm == 0: return None, [torch.zeros_like(p) for p in params]
        x = torch._foreach_div(x, vec_norm)

    xh = torch._foreach_mul(x, h)

    if g_0 is None:
        with torch.enable_grad(): loss = closure()
        g_0 = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    else:
        g_0 = list(g_0)

    torch._foreach_add_(params, xh)
    with torch.enable_grad():
        l = closure()
        if loss is None: loss = l
    g_plus = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]

    torch._foreach_sub_(params, xh)
    for p in params: p.grad = None

    hx = g_plus
    torch._foreach_sub_(hx, g_0)
    torch._foreach_div_(hx, h)

    if normalize: torch._foreach_mul_(hx, vec_norm)
    return loss, hx

class FullHutchinsonHessian(Curvature):
    """randomized Rank-1 approximations to hessian matrix.

    The approximation is for a random vector ``z`` compute hessian-vector product ``Hz``,
    then the approximation is ``Hz z^T`` (outer product of ``Hz`` with ``z``), as described in https://arxiv.org/pdf/1206.6464.

    The hessian-vector product is computed once every ``update_freq`` in LRE. Therefore the correction is always rank 1.

    Args:
        z (str, optional): distribution for random vectors, "normal" or "rademacher". Defaults to "rademacher".
        hvp_method (str, optional):
            how to compute hessian-vector products.
            - ``"autograd"`` - via autograd;
            - ``"forward"`` - finite difference with forward formula via one extra backward pass.
            - ``"central"`` - finite difference with central formula via two extra backward passes.

            Defaults to 'autograd'.
        h (float, optional): finite difference step size for finite-difference hessian-vector product. Defaults to 1e-3.

    """
    def __init__(self, z: _z_type = "rademacher", orthogonalize:bool=False, hvp_method:Literal['autograd','forward','central']='autograd', h=1e-3):
        self.z = z
        self.hvp_method = hvp_method
        self.orthogonalize = orthogonalize
        self.h = h

        self.z_history = []
        self.Hz_history = []

        self.p_prev = None

    def initialize(self, grad, state):
        z = _cat_tensors(state["z_list"])
        Hz = _cat_tensors(state["Hz_list"])
        v = (z + Hz) / 2

        L, Q = rank1_eigh(v)
        return Q, L

    def update(self, loss, grad, closure, params_list, should_update, Q, L):
        assert grad is None

        if should_update:

            # compute grads possibly with graph
            create_graph = self.hvp_method == "autograd"
            with torch.enable_grad():
                if loss is None: loss = closure(False)
                grad_list = torch.autograd.grad(loss, params_list, create_graph=create_graph, materialize_grads=True)

            # generate z for Hz
            if self.z == 'mixed':
                z_type = random.choice(["normal", "rademacher", "s", "g"])
            else:
                z_type = self.z

            if z_type == 'normal' or self.p_prev is None:
                z_list = [torch.randn_like(p) for p in params_list]

            elif z_type == 'rademacher':
                z_list = [torch.randint_like(p,0,2) for p in params_list]
                torch._foreach_mul_(z_list, 2)
                torch._foreach_sub_(z_list, 1)

            elif z_type == 's':
                z_list = torch._foreach_sub(params_list, self.p_prev)
                torch._foreach_copy_(self.p_prev, params_list)

            elif z_type == 'g':
                z_list = grad_list

            else:
                raise RuntimeError(z_type)

            if z_type in ('s', 'g'):
                # make sure z isn't too small
                if sum(t.sum() for t in torch._foreach_mul(z_list, z_list)) < torch.finfo(z_list[0].dtype).tiny * 2:
                    z_list = [torch.randint_like(p,0,2) for p in params_list]
                    torch._foreach_mul_(z_list, 2)
                    torch._foreach_sub_(z_list, 1)

            if self.orthogonalize:
                z = _cat_tensors(z_list)
                if Q is not None: z = z - Q @ (Q.T @ z)
                z = z / z.norm().clip(min=torch.finfo(z.dtype).tiny * 2)
                vec_to_tensors_(z, z_list)

            # compute Hz
            if self.hvp_method == "autograd":
                with torch.enable_grad():
                    Hz_list = torch.autograd.grad(grad_list, params_list, z_list, materialize_grads=True)

            elif self.hvp_method == "forward":
                _, Hz_list = _hvp_fd_forward(closure, params_list, z_list, self.h, g_0=grad_list)

            elif self.hvp_method == "central":
                _, Hz_list = _hvp_fd_central(closure, params_list, z_list, self.h)

            else: raise ValueError(self.hvp_method)

            self.z_history.append(_cat_tensors(z_list))
            self.Hz_history.append(_cat_tensors(Hz_list))

            return loss, _cat_tensors(grad_list), {"z_list":z_list, "Hz_list":Hz_list}

        with torch.enable_grad(): loss = closure()
        return loss, _cat_grads(params_list), {}

    def compute_correction(self, Q, L) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        Y = torch.stack(self.Hz_history, 1)
        Z = torch.stack(self.z_history, 1)
        assert Y.size(1) == 1

        self.z_history.clear()
        self.Hz_history.clear()

        return Y, Z, torch.tensor(1., device=Y.device, dtype=Y.dtype)



def sr1_u(L: torch.Tensor, Q: torch.Tensor, S:torch.Tensor, Y: torch.Tensor, tol:float):
    """SR1 adds a rank one correction vv^T to B, this computes v and the sign since it might be subtracted.

    This is batched, S and Y have last batch dimensions.
    """
    BS = Q @ (L.unsqueeze(-1) * (Q.T @ S))
    R = Y - BS  # (ndim, b)

    rs = (R * S).sum(0) # (b, )

    R_norm = torch.linalg.vector_norm(R, dim=0) # pylint:disable=not-callable
    S_norm = torch.linalg.vector_norm(S, dim=0) # pylint:disable=not-callable
    mask = rs.abs() > tol * R_norm * S_norm
    if not mask.any():
        return None, torch.tensor(1., device=Y.device, dtype=Y.dtype)

    R = R[:, mask]
    rs = rs[mask]

    V = R / rs.abs().sqrt()

    return V, rs.sign()

class SR1(FullHutchinsonHessian):
    """Symmetric Rank-1 (SR1) hessian approximation.

    SR1 is a Quasi-Newton method which updates hessian by adding or subtracting a rank-1 update (u u^T).
    So this is perfect for LRE framework as we can easily compute rank 1 correction to eigendecomposition.

    Instead of relying on gradient differences this computes ``y_k`` as hessian-vector product with ``s_k`` (or random vector).

    The hessian-vector product is computed once every ``update_freq`` in LRE. Therefore the correction is always rank 1.

    Args:
        z (str, optional): what vectors to use as ``y_k``.
            - ``"s"`` - parameter differences;
            - ``"g"`` - gradient vector (for testing).
            - ``"normal"`` - random normally-distributed vectors.
            - ``"rademacher"`` - random Rademacher-distributed vectors.
            - ``"mixed"`` - randomly picks one of the above (for testing).

            Defaults to ``"s"``.
        inverse (bool, optional):
            if True, estimates hessian inverse, if False, estimates hessian.
            The formulas are almost the same, only two variables are swapped.
        positive_only (bool, optional):
            if True, disards updates with negative sign. This is useful for GGT to avoid QR due to different alphas.
        hvp_method (str, optional):
            how to compute hessian-vector products.
            - ``"autograd"`` - via autograd;
            - ``"forward"`` - finite difference with forward formula via one extra backward pass.
            - ``"central"`` - finite difference with central formula via two extra backward passes.

            Defaults to 'autograd'.
        h (float, optional): finite difference step size for finite-difference hessian-vector product. Defaults to 1e-3.

    """
    def __init__(self, z: _z_type = "s", orthogonalize: bool=True, tol=1e-16, inverse:bool=False, positive_only:bool=False, hvp_method:Literal['autograd','forward','central']='autograd', h=1e-3):
        super().__init__(z, hvp_method=hvp_method, h=h, orthogonalize=orthogonalize)
        self.tol = tol
        self.inverse=inverse
        self.positive_only = positive_only

    def compute_correction(self, Q, L):
        assert L is not None
        S = torch.stack(self.z_history, 1)
        Y = torch.stack(self.Hz_history, 1)

        if self.inverse: S, Y = Y, S
        U, sign = sr1_u(L=L, Q=Q, S=S, Y=Y, tol=self.tol)

        self.z_history.clear()
        self.Hz_history.clear()

        if self.positive_only and sign < 0: U = None

        return U, None, sign


def bfgs_uv(L: torch.Tensor, Q: torch.Tensor, S: torch.Tensor, Y: torch.Tensor, c: float | None = 0.2, tol: float = 1e-12):
    """computes two BFGS rank 1 correction vectors"""
    BS = Q @ (L.unsqueeze(-1) * (Q.T @ S))

    SBS = (S * BS).sum(0)
    SY = (S * Y).sum(0)

    # Y damping
    if c is not None:
        theta = torch.ones_like(SY)
        mask = SY < c * SBS

        if mask.any():
            SBS_d = SBS[mask]
            SY_d = SY[mask]
            denom = SBS_d - SY_d
            theta[mask] = ((1 - c) * SBS_d / (denom + tol)).clamp(0, 1)

        theta = theta.unsqueeze(0)
        Y = theta * Y + (1 - theta) * BS

    SY = (S * Y).sum(0)

    mask = SY > tol
    if not mask.any():
        return None, None

    S = S[..., mask]
    BS = BS[..., mask]
    Y = Y[..., mask]
    SY = SY[mask]
    SBS = SBS[mask]

    U = Y / SY.sqrt()
    V = BS / (SBS + tol).sqrt()

    C = torch.cat([U, V], dim=1)

    signs = torch.ones(C.size(1), dtype=C.dtype, device=C.device)
    signs[signs.numel() // 2:] = -1

    return C, signs

class BFGS(FullHutchinsonHessian):
    """Broyden–Fletcher–Goldfarb–Shanno (BFGS) hessian approximation.

    BFGS is a Quasi-Newton method which updates hessian by adding one rank 1 matrix and subtracting another rank 1 matrix.

    Instead of relying on gradient differences we compute ``y_k`` as hessian-vector product with ``s_k`` (or random vector).

    The hessian-vector product is computed once every ``update_freq`` in LRE. Therefore the correction is always rank-2.

    Args:
        z (str, optional): what vectors to use as ``y_k``.
            - ``"s"`` - parameter differences;
            - ``"g"`` - gradient vector (for testing).
            - ``"normal"`` - random normally-distributed vectors.
            - ``"rademacher"`` - random Rademacher-distributed vectors.
            - ``"mixed"`` - randomly picks one of the above (for testing).

            Defaults to ``"s"``.
        c (float | None, optional): damping threshold, None disables damping. Defaults to 0.2.
        hvp_method (str, optional):
            how to compute hessian-vector products.
            - ``"autograd"`` - via autograd;
            - ``"forward"`` - finite difference with forward formula via one extra backward pass.
            - ``"central"`` - finite difference with central formula via two extra backward passes.

            Defaults to 'autograd'.
        h (float, optional): finite difference step size for finite-difference hessian-vector product. Defaults to 1e-3.

    """
    def __init__(self, z: _z_type = "s", orthogonalize: bool=True, c: float | None = 0.2, tol=1e-12, hvp_method:Literal['autograd','forward','central']='autograd', h=1e-3):
        super().__init__(z, hvp_method=hvp_method, h=h, orthogonalize=orthogonalize)
        self.c = c
        self.tol = tol

    def compute_correction(self, Q, L):
        assert L is not None
        S = torch.stack(self.z_history, 1)
        Y = torch.stack(self.Hz_history, 1)

        C, signs = bfgs_uv(L=L, Q=Q, S=S, Y=Y, tol=self.tol, c=self.c)

        self.z_history.clear()
        self.Hz_history.clear()

        if C is None or signs is None: return None, None, torch.tensor(1., device=Y.device, dtype=Y.dtype)
        return C, None, signs


# reference for this formula
# [Ansari, Zafar A. Limited Memory Space Dilation and Reduction Algorithms. Diss. Virginia Tech, 1998.](https://camo.ici.ro/books/thesis/th.pdf)
# def shor_r_(H:torch.Tensor, y:torch.Tensor, alpha:float):
#     Hy = H @ y
#     yHy = safe_clip(y.dot(Hy))
#     term = Hy.outer(Hy).div_(yHy)
#     H.sub_(term, alpha=(1-alpha**2))
#     return H

def shor_r(Q: torch.Tensor, L:torch.Tensor, Y: torch.Tensor, alpha: float):
    HY = Q @ (L.unsqueeze(-1) * (Q.T @ Y))

    YHY = (Y * HY).sum(0)
    tiny = torch.finfo(YHY.dtype).tiny * 2
    YHY = torch.where(YHY.abs() < tiny, torch.full_like(YHY, tiny).copysign_(YHY), YHY)

    sign = -math.copysign(1.0, 1.0 - alpha**2)

    magnitude = (abs((1.0 - alpha**2) / YHY)).sqrt()

    U = magnitude * HY

    return U, sign

class ShorR(FullHutchinsonHessian):
    """Experimental ShorR"""
    def __init__(self, alpha=0.5, z: _z_type = "s", orthogonalize: bool=False, hvp_method:Literal['autograd','forward','central']='autograd', h=1e-3):
        super().__init__(z, hvp_method=hvp_method, h=h, orthogonalize=orthogonalize)
        self.alpha = alpha

    def compute_correction(self, Q, L):
        assert L is not None
        Y = torch.stack(self.Hz_history, 1)
        # so what we can do

        C, sign = shor_r(L=L, Q=Q, Y=Y, alpha=self.alpha)

        self.z_history.clear()
        self.Hz_history.clear()

        return C, None, torch.tensor(sign, device=C.device, dtype=C.dtype)


class ShorRGradient(Curvature):
    """MUST SET UPDATE_FREQ TO LARGER NUMBER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    def __init__(self, alpha=0.5, filter_signs:bool=False):
        self.history = []
        self.g_prev = None
        self.alpha = alpha

        self.filter_signs = filter_signs

    def initialize(self, grad, state):
        Y = torch.randn(grad.numel(), 1, device=grad.device, dtype=grad.dtype)
        L, Q = low_rank_eigh(Y)
        return Q, L

    def update(self, loss, grad, closure, params_list, should_update, Q, L):
        if grad is None:
            with torch.enable_grad():
                if loss is None:
                    loss = closure()
                else:
                    for p in params_list: p.grad = None
                    loss.backward()
            grad = _cat_grads(params_list)

        self.history.append(grad)
        assert loss is not None
        return loss, grad, {}

    def compute_correction(self, Q, L):
        A = torch.stack(self.history, 1)
        if self.g_prev is None:
            self.g_prev = A.mean(1)
            self.history.clear()
            return torch.randn_like(self.g_prev).unsqueeze(1), None, torch.tensor(1., device=A.device, dtype=A.dtype)

        g = A.mean(1)
        y = g = self.g_prev
        self.g_prev = g
        self.history.clear()

        if self.filter_signs:
            y[g * self.g_prev > 0] = 1

        assert L is not None
        C, sign = shor_r(Q, L, y.unsqueeze(1), alpha=self.alpha)

        return C, None, torch.tensor(sign, device=C.device, dtype=C.dtype)



class EMA(Curvature):
    """EMA of gradients, same as momentum into preconditioner
    """
    def __init__(self, beta=0.9):
        self.history = []
        self.exp_avg = None
        self.beta = beta

    def initialize(self, grad, state):
        Y = torch.stack(self.history, 1)
        L, Q = low_rank_eigh(Y)
        return Q, L

    def update(self, loss, grad, closure, params_list, should_update, Q, L):
        if grad is None:
            with torch.enable_grad():
                if loss is None:
                    loss = closure()
                else:
                    for p in params_list: p.grad = None
                    loss.backward()
            grad = _cat_grads(params_list)

        if self.exp_avg is None: self.exp_avg = grad
        else: self.exp_avg.lerp_(grad, 1-self.beta)

        self.history.append(self.exp_avg.clone())
        assert loss is not None
        return loss, grad, {}

    def compute_correction(self, Q, L):
        Y = torch.stack(self.history, 1)
        self.history.clear()
        return Y, None, torch.tensor(1., device=Y.device, dtype=Y.dtype)



class Parameters(Curvature):
    """returns parameters. This corresponds to empirical covariance of parameters
    """
    def __init__(self):
        self.history = []

    def initialize(self, grad, state):
        Y = torch.stack(self.history, 1)
        L, Q = low_rank_eigh(Y)
        return Q, L

    def update(self, loss, grad, closure, params_list, should_update, Q, L):
        if grad is None:
            with torch.enable_grad():
                if loss is None:
                    loss = closure()
                else:
                    for p in params_list: p.grad = None
                    loss.backward()
            grad = _cat_grads(params_list)

        self.history.append(_cat_tensors(params_list))
        assert loss is not None
        return loss, grad, {}

    def compute_correction(self, Q, L):
        Y = torch.stack(self.history, 1)
        self.history.clear()
        return Y, None, torch.tensor(1., device=Y.device, dtype=Y.dtype)

class Update(Curvature):
    """returns previous update. This corresponds to empirical covariance of past (already preconditioned) updates
    """
    def __init__(self):
        self.history = []
        self.p_prev = None

    def initialize(self, grad, state):
        Y = torch.stack(self.history, 1)
        L, Q = low_rank_eigh(Y)
        return Q, L

    def update(self, loss, grad, closure, params_list, should_update, Q, L):
        if grad is None:
            with torch.enable_grad():
                if loss is None:
                    loss = closure()
                else:
                    for p in params_list: p.grad = None
                    loss.backward()
            grad = _cat_grads(params_list)

        p = _cat_tensors(params_list)
        if self.p_prev is None:
            self.p_prev = p
        else:
            s = p - self.p_prev
            self.p_prev = p
            self.history.append(s)

        assert loss is not None
        return loss, grad, {}

    def compute_correction(self, Q, L):
        Y = torch.stack(self.history, 1)
        self.history.clear()
        return Y, None, torch.tensor(1., device=Y.device, dtype=Y.dtype)
