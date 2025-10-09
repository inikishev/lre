from abc import ABC, abstractmethod
from typing import Literal
import torch
from .curvature import _cat_tensors
from .linalg import regularize_eigh

class BasisOptimizer(ABC):
    """reproject is always called first"""
    def __init__(self):
        self.Q: torch.Tensor | None = None
        self.L: torch.Tensor | None = None

    def reproject(self, Q: torch.Tensor, L: torch.Tensor | None) -> None:
        """reproject this optimizer to new basis"""
        self.Q = Q
        self.L = L

    def project(self, g: torch.Tensor) -> torch.Tensor:
        """project g to basis"""
        assert self.Q is not None
        return self.Q.T @ g

    def unproject(self, u: torch.Tensor) -> torch.Tensor:
        """unproject u"""
        assert self.Q is not None
        return self.Q @ u

    @abstractmethod
    def step(self, u: torch.Tensor, g: torch.Tensor, state:dict) -> torch.Tensor:
        """step with u in projected space, g is for cautioning, state might hold hessian-vector product"""



class Chain(BasisOptimizer):
    """Chain basis optimizers together, e.g. ``Chain(Adam(), Cautious())``."""
    def __init__(self, *opts: BasisOptimizer):
        super().__init__()
        self.opts = opts

    def reproject(self, Q, L):
        for opt in self.opts:
            opt.reproject(Q, L)

    def project(self, g):
        return self.opts[0].project(g)

    def unproject(self, u):
        return self.opts[-1].unproject(u)

    def step(self, u, g, state:dict):
        for opt in self.opts:
            u = opt.step(u, g, state)
        return u

class Cautious(BasisOptimizer):
    """Negates updates where update sign doesn't match with gradient sign. Chain with another optimizer."""
    def step(self, u, g, state:dict):
        return u * ((g * u) > 0)


class EMA(BasisOptimizer):
    """exponential moving average."""

    def __init__(self, beta:float=0.9):
        super().__init__()
        self.beta = beta
        self.exp_avg = None

    def step(self, u, g, state:dict):
        # initialize
        if self.exp_avg is None:
            self.exp_avg = torch.zeros_like(u)

        return self.exp_avg.lerp_(u, 1-self.beta).clone()

    def reproject(self, Q: torch.Tensor, L: torch.Tensor | None) -> None:
        if self.Q is None:
            self.Q = Q
            return

        if self.exp_avg is not None:
            C = Q.T @ self.Q
            self.exp_avg = C @ self.exp_avg

        self.Q = Q

def _clip(x, min=None,max=None):
    if min is not None and x < min:return min
    if max is not None and x > max:return max
    return x

class ClipNormByEMA(BasisOptimizer):
    def __init__(self, beta=0.99, max_growth: float | None = 1.2, normalize:bool=False, min_norm=1e-2, init=1e-1):
        super().__init__()
        self.beta = beta
        self.max_growth = max_growth
        self.min_norm = min_norm
        self.normalize = normalize
        self.init = init

        self.exp_avg = None
        self.prev_norm = None

    def step(self, u, g, state):
        # initialize
        if self.exp_avg is None:
            self.exp_avg = u*self.init

        # update buffer
        new_exp_avg = self.exp_avg.lerp(u, 1-self.beta)
        new_norm = new_exp_avg.norm()

        # clip max ema growth
        if self.max_growth is not None and self.prev_norm is not None:
            allowed_norm = _clip(self.prev_norm * self.max_growth, min=self.min_norm)
            ema_denom = (new_norm / allowed_norm).clip(min=1)
            new_exp_avg.div_(ema_denom)
            new_norm.div_(ema_denom)
            self.prev_norm = new_norm

        # clip or normalize by norm of buffers
        u_norm = u.norm()
        tiny = torch.finfo(u_norm.dtype).tiny * 2
        denom = u_norm / new_norm.clip(min=tiny)
        if self.normalize: denom.clip_(min=tiny)
        else: denom.clip_(min=1)

        self.exp_avg.set_(new_exp_avg) # type:ignore
        return u / denom

    def reproject(self, Q: torch.Tensor, L: torch.Tensor | None) -> None:
        if self.Q is None:
            self.Q = Q
            return

        if self.exp_avg is not None:
            C = Q.T @ self.Q
            self.exp_avg = C @ self.exp_avg
            self.prev_norm = self.exp_avg.norm()

            # make sure buffer doesn't collapse
            if self.prev_norm < torch.finfo(self.prev_norm.dtype).tiny * 2:
                self.exp_avg = torch.randn_like(self.exp_avg)
                self.prev_norm = self.exp_avg.norm()

            # make sure buffer stays above min norm
            if self.prev_norm < self.min_norm:
                self.exp_avg *= self.min_norm / self.prev_norm
                self.prev_norm = self.min_norm
        self.Q = Q


class CopyGradSign(BasisOptimizer):
    """Copies gradient sign to update"""
    def step(self, u, g, state:dict):
        return u.copysign(g)

class Normalize(BasisOptimizer):
    """normalizes update to have unit norm"""
    def step(self, u, g, state:dict):
        return u / u.norm().clip(min=torch.finfo(u.dtype).tiny * 2)

class NanToZero(BasisOptimizer):
    """normalizes update to have unit norm"""
    def step(self, u, g, state:dict):
        return u.nan_to_num(0,0,0)

class ClipNorm(BasisOptimizer):
    """clips update norm"""
    def __init__(self, max_norm=1):
        super().__init__()
        self.max_norm = max_norm

    def step(self, u, g, state:dict):
        norm = u.norm()
        if norm > self.max_norm:
            u = u * self.max_norm / norm
        return u

class ClipValue(BasisOptimizer):
    """clips update values"""
    def __init__(self, max_value=0.2):
        super().__init__()
        self.max_value = max_value

    def step(self, u, g, state:dict):
        return u.clip(-self.max_value, self.max_value)

class Sign(BasisOptimizer):
    """takes sign of the update"""
    def step(self, u, g, state:dict):
        return u.sign()

class LInv(BasisOptimizer):
    """Scales ``g`` by reciprocal of ``L``. Only use for hessian approximations."""
    def step(self, u, g, state:dict):
        assert self.L is not None
        return u / self.L



class LInvSqrt(BasisOptimizer):
    """Scales ``g`` by inverse square root of ``L``. Only use for covariance."""
    def step(self, u, g, state:dict):
        assert self.L is not None
        return u * self.L.rsqrt()


class Adam(BasisOptimizer):
    """Adam. If ``amsgrad_beta`` is not None, enables AMSgrad, except the accumulator is decayed
    by ``amsgrad_beta`` on each step (set to 1 for no decay)."""
    def __init__(self, beta1=0.9, beta2=0.99, eps=1e-8, amsgrad_beta:float|None=None):# 1 no decay
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad_decay = amsgrad_beta

        self.exp_avg = None
        self.exp_avg_sq = None
        self.max_exp_avg_sq = None

        self.current_step = 1

    def reproject(self, Q, L):
        if self.Q is None:
            self.Q = Q
            return

        C = Q.T @ self.Q
        if self.exp_avg is not None:
            self.exp_avg = C @ self.exp_avg

        if self.exp_avg_sq is not None:
            self.exp_avg_sq = (C @ self.exp_avg_sq.diag_embed() @ C.T).diagonal()

        if self.max_exp_avg_sq is not None:
            self.max_exp_avg_sq = (C @ self.max_exp_avg_sq.diag_embed() @ C.T).diagonal()

        self.Q = Q


    def step(self, u, g, state:dict):
        # initialize
        if self.exp_avg is None or self.exp_avg_sq is None:
            self.exp_avg = torch.zeros_like(u)
            self.exp_avg_sq = torch.zeros_like(u)

        # update accumulators
        self.exp_avg.lerp_(u, weight=1-self.beta1)
        self.exp_avg_sq.mul_(self.beta2).addcmul_(u, u, value=1-self.beta2)
        exp_avg_sq = self.exp_avg_sq

        # amsgrad
        if self.amsgrad_decay is not None:
            if self.max_exp_avg_sq is None:
                self.max_exp_avg_sq = torch.zeros_like(u)

            self.max_exp_avg_sq.mul_(self.amsgrad_decay).clip_(min=self.exp_avg_sq)
            exp_avg_sq = self.max_exp_avg_sq

        # debias
        exp_avg = self.exp_avg / (1 - self.beta1 ** self.current_step)
        exp_avg_sq = exp_avg_sq / (1 - self.beta2 ** self.current_step)

        # compute update
        denom = exp_avg_sq.sqrt().add_(self.eps)
        u = exp_avg / denom

        self.current_step += 1
        return u

def signed_cbrt(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * x.abs().pow(1/3)

def _clip_min_magnitude(x: torch.Tensor, eps: float):
    return x.sign() * x.abs().clamp(min=eps)

_cubic_adam_mode = Literal["signed_cbrt", "unsigned_cbrt", "halve"]

def _cubic_minimize(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, eps):
    """minimizes (A/3)x^3 + (A/2)x^2 + Cx"""
    discriminant = B**2 - 4 * A * C

    denom = _clip_min_magnitude(2 * A, eps)
    root = discriminant.clamp(min=0).sqrt_()

    x0 = (-B + root) / denom
    x1 = (-B - root) / denom

    f0 = (A/3)*x0**3 + (B/2)*x0**2 + C*x0
    f1 = (A/3)*x1**3 + (B/2)*x1**2 + C*x1

    x_star = x0.where(f0 < f1, x1)

    adam = -C / (B + eps)
    return adam.where(discriminant < 0, x_star)

def cubic_adam_(
    tensors: torch.Tensor,
    exp_avg_: torch.Tensor,
    exp_avg_sq_: torch.Tensor,
    exp_avg_cu_: torch.Tensor,
    alpha: float,
    beta1: float,
    beta2: float,
    beta3: float,
    eps: float,
    debiased: bool,
    step: int,

    mode: _cubic_adam_mode = 'signed_cbrt'
):
    exp_avg_.lerp_(tensors, 1-beta1)
    exp_avg_sq_.lerp_(tensors**2, 1-beta2)
    exp_avg_cu_.lerp_(tensors**3, 1-beta3)

    if debiased:
        m1 = exp_avg_ / (1 - beta1 ** step)
        m2 = exp_avg_sq_ / (1 - beta2 ** step)
        m3 = exp_avg_cu_ / (1 - beta3 ** step)
    else:
        m1, m2, m3 = exp_avg_, exp_avg_sq_, exp_avg_cu_

    # adam minimizes ax^2 + bx
    # we are going to minimize ax^3 + bx^2 + cx

    if mode == "signed_cbrt": A = signed_cbrt(m3)
    elif mode == "unsigned_cbrt": A = m3.abs().pow(1/3)
    elif mode == 'halve': A = 0.5 * m3
    else: raise ValueError(mode)

    B = m2.sqrt()
    C = m1
    x_star = _cubic_minimize(A, B, C, eps)
    return x_star.mul_(-alpha)

def _cubic_reproject(C: torch.Tensor, cu: torch.Tensor, approx:bool):
    if approx: return C.pow(3) @ cu

    n = cu.numel()
    T = torch.zeros([n,n,n], device=cu.device, dtype=cu.dtype)
    T[range(n),range(n),range(n)] = cu
    T = torch.einsum('ai,bj,ck,ijk->abc', C, C, C, T)
    n2 = T.size(0)
    return T[range(n2), range(n2), range(n2)]

class CubicAdam(BasisOptimizer):
    r"""diagonal Adam which also maintains third moments. So Adam actually minimizes a second order polynomial $\sqrt{m_2}x^2 + m_1x$, where $v_2$ is second moments and $m_1$ is first moments. By adding third moments $m_3$ (cubed gradients), we can minimize $\sqrt[3]{m_3} + \sqrt{m_2}x^2 + m_1x$ since it has an exact solution.,"""
    def __init__(self, beta1=0.9, beta2=0.99, beta3=0.99, eps=1e-8, approx_reproject:bool=False):# 1 no decay
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.eps = eps
        self.approx_reproject = approx_reproject

        self.exp_avg = None
        self.exp_avg_sq = None
        self.exp_avg_cu = None

        self.current_step = 1

    def reproject(self, Q, L):
        if self.Q is None:
            self.Q = Q
            return

        C = Q.T @ self.Q
        if self.exp_avg is not None:
            self.exp_avg = C @ self.exp_avg

        if self.exp_avg_sq is not None:
            self.exp_avg_sq = (C @ self.exp_avg_sq.diag_embed() @ C.T).diagonal()

        if self.exp_avg_cu is not None:
            self.exp_avg_cu = _cubic_reproject(C, self.exp_avg_cu, self.approx_reproject)

        self.Q = Q


    def step(self, u, g, state:dict):
        # initialize
        if self.exp_avg is None or self.exp_avg_sq is None or self.exp_avg_cu is None:
            self.exp_avg = torch.zeros_like(u)
            self.exp_avg_sq = torch.zeros_like(u)
            self.exp_avg_cu = torch.zeros_like(u)

        u = cubic_adam_(
            tensors=u,
            exp_avg_=self.exp_avg,
            exp_avg_sq_=self.exp_avg_sq,
            exp_avg_cu_=self.exp_avg_cu,
            beta1=self.beta1,
            beta2=self.beta2,
            beta3=self.beta3,
            eps=self.eps,
            step=self.current_step,
            debiased=True,
            alpha=1
        )
        self.current_step += 1
        return u

class FullMatrixAdam(BasisOptimizer):
    """Full-matrix Adam. If ``amsgrad_beta`` is not None, enables AMSgrad on diagonal of the covariance matrix accumulator,
    the accumulator is decayed by ``amsgrad_beta`` on each step (set to 1 for no decay)."""
    def __init__(self, beta1=0.9, beta2=0.99, eps=1e-8, amsgrad_beta:float|None=None, eig_tol=1e-12, damping=1e-8, rdamping=1e-8):# 1 no decay
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad_decay = amsgrad_beta

        self.eig_tol = eig_tol
        self.damping = damping
        self.rdamping = rdamping

        self.exp_avg = None
        self.covariance = None
        self.preconditioner = None
        self.max_covariance_diag = None
        self.reprojected = True

        self.current_step = 1

    def reproject(self, Q, L):
        if self.Q is None:
            self.Q = Q
            return

        self.reprojected = True

        C = Q.T @ self.Q
        if self.exp_avg is not None:
            self.exp_avg = C @ self.exp_avg

        if self.max_covariance_diag is not None:
            assert self.covariance is not None
            amsgrad_covariance = self.covariance.diagonal_scatter(self.max_covariance_diag)
            self.max_covariance_diag = (C @ amsgrad_covariance @ C.T).diagonal()

        if self.covariance is not None:
            self.covariance = C @ self.covariance @ C.T

        self.Q = Q


    def step(self, u, g, state:dict):
        # initialize
        if self.exp_avg is None or self.covariance is None:
            self.exp_avg = torch.zeros_like(u)
            self.covariance = torch.eye(u.numel(), dtype=u.dtype, device=u.device)

        # update accumulators
        self.exp_avg.lerp_(u, weight=1-self.beta1)
        self.covariance.lerp_(u.outer(u), weight=1-self.beta2)

        # amsgrad (on covariance diagonal)
        if self.amsgrad_decay is not None:
            if self.max_covariance_diag is None:
                self.max_covariance_diag = torch.zeros_like(u)

            self.max_covariance_diag.mul_(self.amsgrad_decay).clip_(min=self.covariance.diagonal())

        # debias exp avg
        exp_avg = self.exp_avg / (1 - self.beta1 ** self.current_step)

        # update after reprojecting
        if self.preconditioner is None or self.reprojected:
            self.reprojected = False

            # set diagonal to amsgrad
            covariance = self.covariance
            if self.max_covariance_diag is not None:
                covariance = self.covariance.diagonal_scatter(self.max_covariance_diag)

            # debias covairance
            covariance = covariance / (1 - self.beta2 ** self.current_step)

            # regularize
            reg = torch.eye(u.numel(), dtype=u.dtype, device=u.device) * self.eps
            covariance = covariance + reg

            # inverse square root
            try:
                L, Q = torch.linalg.eigh(covariance) # pylint:disable=not-callable
                L, Q = regularize_eigh(L, Q, tol=self.eig_tol, damping=self.damping, rdamping=self.rdamping)
                if L is not None and Q is not None:
                    self.preconditioner = Q @ L.rsqrt().diag_embed() @ Q.T
                else:
                    self.preconditioner = covariance.diagonal().rsqrt().diag_embed()
            except torch.linalg.LinAlgError:
                self.preconditioner = covariance.diagonal().rsqrt().diag_embed()

        # compute update
        u = self.preconditioner @ exp_avg
        self.current_step += 1
        return u


class HutchinsonAdam(BasisOptimizer):
    """Adam with squared exponential average replaced by exponential average of hutchinson hessian estimates. So this is SophiaH if ``square=False`` or AdaHessian if ``square=True``. This can only be used with Curvatures that use hessian-vector products."""
    def __init__(self, beta1=0.9, beta2=0.99, eps=1e-8, amsgrad_beta=None, square:bool=True, zHz: bool = True):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad_decay = amsgrad_beta
        self.square = square
        self.zHz = zHz

        self.exp_avg = None
        self.D_avg = None
        self.max_D_avg = None

        self.current_step = 1
        self.num_Ds = 0

        super().__init__()


    def reproject(self, Q, L):
        if self.Q is None:
            self.Q = Q
            return

        C = Q.T @ self.Q
        if self.exp_avg is not None:
            self.exp_avg = C @ self.exp_avg

        if self.D_avg is not None:
            if self.square: self.D_avg = (C @ self.D_avg.diag_embed() @ C.T).diagonal()
            else: self.D_avg = C @ self.D_avg

        if self.max_D_avg is not None:
            if self.square: self.max_D_avg = (C @ self.max_D_avg.diag_embed() @ C.T).diagonal()
            else: self.D_avg = C @ self.max_D_avg

        self.Q = Q

    def step(self, u, g, state:dict):
        # initialize
        assert self.Q is not None
        if self.exp_avg is None or self.D_avg is None:
            if "z_list" not in state:
                raise RuntimeError("HutchinsonAdam can only be used with HutchinsonHessian, SR1 and BFGS curvatures")
            self.exp_avg = torch.zeros_like(u)
            self.D_avg = torch.zeros_like(u)

        # update exp avg
        self.exp_avg.lerp_(u, weight=1-self.beta1)

        # update hessian diag if Hvp was calculated this step
        D_avg = self.D_avg
        if "z_list" in state:
            self.num_Ds += 1
            Hz = self.Q.T @ _cat_tensors(state["Hz_list"])
            if self.zHz:
                z = self.Q.T @ _cat_tensors(state["z_list"])
                v = z * Hz
            else:
                v = Hz

            if self.square:
                self.D_avg.mul_(self.beta2).addcmul_(v, v, value=1-self.beta2)

            else:
                self.D_avg.lerp_(v, weight=1-self.beta2)


            if self.amsgrad_decay is not None:
                if self.max_D_avg is None:
                    self.max_D_avg = torch.zeros_like(u)

                self.max_D_avg.mul_(self.amsgrad_decay).clip_(min=self.D_avg)

        if self.max_D_avg is not None:
            D_avg = self.max_D_avg

        # debias
        exp_avg = self.exp_avg / (1 - self.beta1 ** self.current_step)
        D_avg = D_avg / (1 - self.beta2 ** self.num_Ds)

        # compute update
        if self.square:
            denom = D_avg.sqrt().add_(self.eps)

        else:
            denom = D_avg.abs().add_(self.eps)

        u = exp_avg / denom

        self.current_step += 1
        return u


class FullMatrixHutchinsonAdam(BasisOptimizer):
    """Adam with squared exponential average replaced by exponential average of full-matrix hutchinson hessian estimates
    Hz z^T. So this is full-matrix SophiaH if ``square=False`` or full-matrix AdaHessian if ``square=True``. This can only be used with Curvatures that use hessian-vector products."""

    def __init__(self, beta1=0.9, beta2=0.99, eps=1e-6, amsgrad_beta:float|None=None, eig_tol=1e-12, damping=1e-8, rdamping=1e-8, abs:bool=True):# 1 no decay
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad_decay = amsgrad_beta

        self.eig_tol = eig_tol
        self.damping = damping
        self.rdamping = rdamping
        self.abs = abs

        self.exp_avg = None
        self.H_avg = None
        self.H_inv = None
        self.max_H_diag = None
        self.reprojected = True

        self.current_step = 1
        self.num_Hs = 0

    def reproject(self, Q, L):
        if self.Q is None:
            self.Q = Q
            return

        self.reprojected = True

        C = Q.T @ self.Q
        if self.exp_avg is not None:
            self.exp_avg = C @ self.exp_avg

        if self.max_H_diag is not None:
            assert self.H_avg is not None
            amsgrad_H = self.H_avg.diagonal_scatter(self.max_H_diag)
            self.max_H_diag = (C @ amsgrad_H @ C.T).diagonal()

        if self.H_avg is not None:
            self.H_avg = C @ self.H_avg @ C.T

        self.Q = Q


    def step(self, u, g, state:dict):
        # initialize
        if self.exp_avg is None or self.H_avg is None:
            if "z_list" not in state:
                raise RuntimeError("HutchinsonAdam can only be used with HutchinsonHessian, SR1 and BFGS curvatures")
            self.exp_avg = torch.zeros_like(u)
            self.H_avg = torch.eye(u.numel(), dtype=u.dtype, device=u.device)

        # update exp avg
        self.exp_avg.lerp_(u, weight=1-self.beta1)

        # update hessian if Hvp was calculated this step
        H_avg = self.H_avg
        if "z_list" in state:
            self.num_Hs += 1
            assert self.Q is not None
            z = self.Q.T @ _cat_tensors(state["z_list"])
            Hz = self.Q.T @ _cat_tensors(state["Hz_list"])

            H = Hz.outer(z)
            H = (H + H.T) / 2

            self.H_avg.lerp_(H, weight=1-self.beta2)

            # amsgrad on diagonal
            if self.amsgrad_decay is not None:
                if self.max_H_diag is None:
                    self.max_H_diag = torch.zeros_like(u)

                self.max_H_diag.mul_(self.amsgrad_decay).clip_(min=self.H_avg.diagonal())
                H_avg = H_avg.diagonal_scatter(self.max_H_diag)

        # debias exp avg
        exp_avg = self.exp_avg / (1 - self.beta1 ** self.current_step)

        # update after reprojecting
        if self.H_inv is None or self.reprojected:
            self.reprojected = False

            # debias hessian
            H_avg = H_avg / (1 - self.beta2 ** self.current_step)

            # regularize hessian
            reg = torch.eye(u.numel(), dtype=u.dtype, device=u.device) * self.eps
            H_avg = H_avg + reg

            # we could use inverse but using absolute value of eigenvalues is so much better
            try:
                L, Q = torch.linalg.eigh(H_avg) # pylint:disable=not-callable
                if self.abs: L = L.abs()
                L, Q = regularize_eigh(L, Q, tol=self.eig_tol, damping=self.damping, rdamping=self.rdamping)
                if L is not None and Q is not None:
                    self.H_inv = Q @ L.reciprocal().diag_embed() @ Q.T
                else:
                    self.H_inv = H_avg.diagonal().reciprocal().abs().diag_embed()
            except torch.linalg.LinAlgError:
                self.H_inv = H_avg.diagonal().reciprocal().abs().diag_embed()

        # compute update
        u = self.H_inv @ exp_avg
        self.current_step += 1
        return u


def safe_clip(x: torch.Tensor, min=None):
    """makes sure absolute value of scalar tensor x is not smaller than min"""
    assert x.numel() == 1, x.shape
    if min is None: min = torch.finfo(x.dtype).tiny * 2

    if x.abs() < min: return x.new_full(x.size(), min).copysign(x)
    return x

# if we use formula for H we would have to invert C when reprojecting
def bfgs_B(B:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    sy = s.dot(y)
    if sy < tol: return B

    Bs = B@s
    sBs = safe_clip(s.dot(Bs))

    term1 = y.outer(y).div_(sy)
    term2 = (Bs.outer(s) @ B.T).div_(sBs)
    return B + term1.sub_(term2)

class FullMatrixBFGSAdam(BasisOptimizer):
    """Adam with squared exponential average replaced by BFGS hessian estimate. This can only be used with Curvatures that use hessian-vector products."""

    def __init__(self, beta1=0.9, beta2: float | None = 0.99, eps=1e-6, amsgrad_beta:float|None=None, eig_tol=1e-12, damping=1e-8, rdamping=1e-8, tol=1e-16, use_eigh=False):# 1 no decay
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad_decay = amsgrad_beta
        self.tol = tol

        self.eig_tol = eig_tol
        self.damping = damping
        self.rdamping = rdamping
        self.use_eigh = use_eigh

        self.exp_avg = None
        self.B_avg = None
        self.B_inv = None
        self.max_B_diag = None
        self.reprojected = True

        self.current_step = 1
        self.num_Hs = 0

    def reproject(self, Q, L):
        if self.Q is None:
            self.Q = Q
            return

        self.reprojected = True

        C = Q.T @ self.Q
        if self.exp_avg is not None:
            self.exp_avg = C @ self.exp_avg

        if self.max_B_diag is not None:
            assert self.B_avg is not None
            amsgrad_B = self.B_avg.diagonal_scatter(self.max_B_diag)
            self.max_B_diag = (C @ amsgrad_B @ C.T).diagonal()

        if self.B_avg is not None:
            self.B_avg = C @ self.B_avg @ C.T

        self.Q = Q


    def step(self, u, g, state:dict):
        # initialize
        if self.exp_avg is None or self.B_avg is None:
            if "z_list" not in state:
                raise RuntimeError("BFGSAdam can only be used with HutchinsonHessian, SR1 and BFGS curvatures")

            self.exp_avg = torch.zeros_like(u)

            # could also use yy/ys
            self.B_avg = torch.eye(u.numel(), dtype=u.dtype, device=u.device)

        # update exp avg
        self.exp_avg.lerp_(u, weight=1-self.beta1)

        # update hessian if Hvp was calculated this step
        B_avg = self.B_avg
        if "z_list" in state:
            self.num_Hs += 1
            assert self.Q is not None
            s = self.Q.T @ _cat_tensors(state["z_list"])
            y = self.Q.T @ _cat_tensors(state["Hz_list"])

            B_new = bfgs_B(self.B_avg, s, y, self.tol)
            if self.beta2 is None: self.B_avg = B_new
            else: self.B_avg.lerp_(B_new, weight=1-self.beta2)

            # amsgrad on hessian diagonal
            if self.amsgrad_decay is not None:
                if self.max_B_diag is None:
                    self.max_B_diag = torch.zeros_like(u)

                self.max_B_diag.mul_(self.amsgrad_decay).clip_(min=self.B_avg.diagonal())
                B_avg = B_avg.diagonal_scatter(self.max_B_diag)

        # debias exp avg
        exp_avg = self.exp_avg / (1 - self.beta1 ** self.current_step)

        # update after reprojecting
        if self.B_inv is None or self.reprojected:
            self.reprojected = False

            # debias B
            if self.beta2 is not None:
                B_avg = B_avg / (1 - self.beta2 ** self.current_step)

            # regularuze
            reg = torch.eye(u.numel(), dtype=u.dtype, device=u.device) * self.eps
            B_avg = B_avg + reg

            # here maybe we can try inverse
            if self.use_eigh:
                try:
                    L, Q = torch.linalg.eigh(B_avg) # pylint:disable=not-callable
                    L, Q = regularize_eigh(L, Q, tol=self.eig_tol, damping=self.damping, rdamping=self.rdamping)
                    if L is not None and Q is not None:
                        self.B_inv = Q @ L.reciprocal().diag_embed() @ Q.T
                    else:
                        self.B_inv = B_avg.diagonal().reciprocal().abs().diag_embed()
                except torch.linalg.LinAlgError:
                    self.B_inv = B_avg.diagonal().reciprocal().abs().diag_embed()

            else:
                B_inv, info = torch.linalg.inv_ex(B_avg) # pylint:disable=not-callable
                if info == 0: self.B_inv = B_inv
                else: self.B_inv = B_avg.diagonal().reciprocal().abs().diag_embed()

        u = self.B_inv @ exp_avg

        self.current_step += 1
        return u


def lion_(tensors: torch.Tensor, exp_avg_: torch.Tensor, beta1, beta2,):
    update = exp_avg_.lerp(tensors, 1-beta1).sign_()
    exp_avg_.lerp_(tensors, 1-beta2)
    return update

class Lion(BasisOptimizer):
    """Runs Lion in the low rank basis."""
    def __init__(self, beta1=0.9, beta2=0.99, cautious:bool=False):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.cautious = cautious

        self.exp_avg = None

    def step(self, u, g, state):

        if self.exp_avg is None:
            self.exp_avg = torch.zeros_like(g)

        dir = lion_(g, self.exp_avg, beta1=self.beta1, beta2=self.beta2)

        if self.cautious:
            mask = (g * dir) > 0
            dir *= mask

        return dir

    def reproject(self, Q: torch.Tensor, L: torch.Tensor | None) -> None:
        if self.Q is None:
            self.Q = Q
            return

        if self.exp_avg is not None:
            C = Q.T @ self.Q
            self.exp_avg = C @ self.exp_avg

        self.Q = Q
        self.L = L


class NaturalGradient(BasisOptimizer):
    def __init__(self, reg=1e-8):
        super().__init__()
        self.jac = None
        self.reg = reg

    def step(self, u, g, state):
        assert self.Q is not None
        if "jac" in state:
            # jac is (batch_size, ndim)
            self.jac = self.Q.T @ state["jac"]

        jac = self.jac
        assert jac is not None
        GTG = jac.T @ jac
        reg = torch.eye(GTG.size(0), device=GTG.device, dtype=GTG.dtype) * self.reg
        return torch.linalg.lstsq(GTG.add_(reg), u).solution # pylint:disable=not-callable

    def reproject(self, Q: torch.Tensor, L: torch.Tensor | None) -> None:
        if self.Q is None:
            self.Q = Q
            return

        if self.jac is not None:
            C = Q.T @ self.Q
            self.jac = C @ self.jac

        self.Q = Q
        self.L = L


class GaussNewton(BasisOptimizer):
    def __init__(self, reg=1e-8):
        super().__init__()
        self.jac = None
        self.reg = reg

    def step(self, u, g, state):
        assert self.Q is not None
        if "jac" in state:
            # jac is (batch_size, ndim)
            self.jac = self.Q.T @ state["jac"]

        jac = self.jac
        assert jac is not None
        # well g is already gauss-newton grad
        # so this is basically identical to NaturalGradient
        # not sure if reprojection will work
        JTJ = jac.T @ jac
        reg = torch.eye(JTJ.size(0), device=JTJ.device, dtype=JTJ.dtype) * self.reg
        return torch.linalg.lstsq(JTJ.add_(reg), u).solution # pylint:disable=not-callable

    def reproject(self, Q: torch.Tensor, L: torch.Tensor | None) -> None:
        if self.Q is None:
            self.Q = Q
            return

        if self.jac is not None:
            C = Q.T @ self.Q
            self.jac = C @ self.jac

        self.Q = Q
        self.L = L
