"""how to update basis given a correction"""
import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
import numpy as np
import torch

from .linalg import (
    OrthoMethod,
    eigh_plus_UUt,
    eigh_plus_UUt_mm,
    eigh_plus_UVt_symmetrize,
    low_rank_eigh,
    nystrom_approximation,
    orthogonalize,
    randomized_eigh_plus_UUt,
    rank1_eigh,
    regularize_eigh,
    sanger_,
    sanger_adamg,
    sanger_stiefel_,
)


class Basis(ABC):

    @abstractmethod
    def update(self, Y: torch.Tensor, Z: torch.Tensor | None, alpha: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """updates self with low rank correction Y Z^T.

        Y and Z are ``(ndim, k)``, Z can be None.

        if Z is None, assumes symmetric correction Y Y^T

        returns ``(Q, Q_reg, L, L_reg)``.

        Both L and L_reg must be debiased if debiasing is applied.
        """

class GGT(Basis):
    """GGT from https://arxiv.org/pdf/1806.02958

    On each update it computes eigendecomposition of ``rank x rank`` matrix.

    However if update is asymmetrical or uses alphas, it has to orthogonalize an ``ndim x rank`` matrix each update,
    which typically is quite much slower than normal GGT update.

    Args:
        history_size (int, optional): number of gradients to store and upper bound on rank. Defaults to 100.
        truncate (int | None, optional):
            truncates eigendecomposition to top ``truncate`` eigenvalues (set to less than ``history_size``). Defaults to None.
        eig_tol (float, optional):
            removes eigenvalues smaller than largest eigenvalue times this tolerance value. Defaults to 1e-7.
        damping (float, optional): scale of identity matrix added to G^T G. Defaults to 1e-4.
        rdamping (float, optional): value multiplied by largest eigenvalue and added to eigenvalues. Defaults to 0.
        abs (bool, optional): whether to take absolute value of eigenvalues. Defaults to False.
        ortho_method (OrthoMethod, optional):
            how to orthogonalize . Defaults to 'qr'.
    """

    def __init__(
        self,
        history_size: int = 100,
        truncate: int | None = None,
        eig_tol: float = 1e-5,
        damping: float = 0,
        rdamping: float = 0,
        abs: bool = False,

        ortho_method: OrthoMethod = 'qr',
    ):
        # unlike GGT we hold two histories in case of assymetric corrections
        self.Y_history = deque(maxlen=history_size)
        self.Z_history = deque(maxlen=history_size)
        self.alpha_history = deque(maxlen=history_size)

        self.truncate = truncate
        self.eig_tol = eig_tol
        self.damping = damping
        self.rdamping = rdamping
        self.abs = abs
        self.ortho_method: OrthoMethod = ortho_method

    def update(self, Y, Z, alpha):
        self.Y_history.extend(Y.unbind(1))
        if Z is not None: self.Z_history.extend(Z.unbind(1))
        self.alpha_history.extend(torch.broadcast_to(alpha, (Y.size(1), )))

        symmetric = len(self.Z_history) == 0
        alphas = torch.stack(tuple(self.alpha_history))
        unit_alphas = (alphas == 1).all()

        try:
            if symmetric and unit_alphas:
                M = torch.stack(tuple(self.Y_history), dim=1)# / len(history)
                MtM = M.T @ M

                if self.damping != 0:
                    MtM.add_(torch.eye(MtM.size(0), device=MtM.device, dtype=MtM.dtype).mul_(self.damping))

                L, Q = torch.linalg.eigh(MtM) # pylint:disable=not-callable
                if self.abs: L = L.abs()

                # damping is already added to MTM, rdamping is added afterwards
                L, Q = regularize_eigh(L, Q, truncate=self.truncate, tol=self.eig_tol, damping=0, rdamping=0)

                if L is None or Q is None: # this means there are no finite eigenvalues
                    return None, None, None, None

                U = (M @ Q) * L.rsqrt()

                if self.rdamping != 0:
                    L.add_(self.rdamping * L[-1]) # L is sorted in ascending order

                # since `history` is not affected by regularization
                # we only use the first round of regularization (so Q = Q_reg)
                return U, U, L, L

            # assymetric or alphas case
            M1 = torch.stack(tuple(self.Y_history), dim=1)
            if len(self.Z_history) > 0:
                M2 = torch.stack(tuple(self.Z_history), dim=1)
            else: M2 = M1

            M = torch.hstack([M1, M2])
            try:
                Q_M = orthogonalize(M, method=self.ortho_method) # pylint:disable=not-callable
            except torch.linalg.LinAlgError:
                return None, None, None, None

            M1_proj = Q_M.T @ M1
            M2_proj = Q_M.T @ M2

            D = alphas.diag_embed()
            A_proj = (M1_proj @ D @ M2_proj.T + M2_proj @ D @ M1_proj.T) / 2

            if self.damping != 0:
                A_proj.add_(torch.eye(A_proj.size(0), device=A_proj.device, dtype=A_proj.dtype).mul_(self.damping))

            L, S = torch.linalg.eigh(A_proj) # pylint:disable=not-callable
            if self.abs: L = L.abs()

            L, S = regularize_eigh(L, S, truncate=self.truncate, tol=self.eig_tol, damping=0, rdamping=0)

            if L is None or S is None:
                return None, None, None, None

            U = Q_M @ S

            if self.rdamping != 0:
                L.add_(self.rdamping * L[-1])

            return U, U, L, L

        except torch.linalg.LinAlgError:
            return None, None, None, None


class Eigen(Basis):
    """stores curvature matrix directly as eigendecomposition.

    On a rank 1 update it computes eigendecomposition of ``rank x rank`` matrix.

    However if update more than rank 1, it has to orthogonalize a ``ndim x update_rank`` matrix.
    That happens for example if ``update_freq`` is not 1. With ``Covariance`` curvature it
    is very feasible to keep ``update_freq`` at 1 to avoid QR and just compute one cheap eigendecomposition.

    Note that there are two rounds of regularization. First round (``eig_tol``, ``damping``, ``rdamping``, ``abs``)
    affects the accumulator, i.e. further accumulator updates will be with regularized accumulator.

    The second round of regularization (every argument starting with ``update_``) is computed after the first
    and doesn't affect the accumulator, it is used to reproject inner optimizer and compute the update.

    Args:
        rank (int, optional): maximum rank of approximation. Defaults to 100.
        beta (float, optional): coefficient for exponenital moving average of the curvature matrix. Defaults to 0.99.
        eig_tol (float, optional):
            On each curvature update removes eigenvalues smaller than largest eigenvalue times this tolerance value.
            Defaults to 1e-4.
        damping (float, optional):
            On each curvature update adds this value to eigenvalues of the accumulator,
            this affects the curvature accumulator. Defaults to 0.
        rdamping (float, optional):
            On each curvature update adds this value times largest eigenvalue to eigenvalues of the accumulator,
            this affects the curvature accumulator. Defaults to 0.
        abs (bool, optional):
            if True, forces the accumulator to have positive eigenvalues (by taking their absolute values after each update). Defaults to False.

        update_truncate (int | None, optional):
            truncates eigendecomposition to top ``truncate`` eigenvalues (set to less than ``rank``). Defaults to None.
        update_eig_tol (float | None, optional):
            removes eigenvalues smaller than largest eigenvalue times this tolerance value, does not affect accumulator.
            Defaults to None.
        update_damping (float, optional):
            constant value added to eigenvalues, does not affect accumulator. Defaults to 1e-4.
        update_rdamping (float, optional):
            constant value times largest eigenvalue added to eigenvalues, does not affect accumulator. Defaults to 0.
        update_abs (bool, optional):
            whether to take absolute value of the eigenvalues, doesn't affect the accumulator. Defaults to False.
        ortho_method (OrthoMethod, optional):
            how to orthogonalize 2+ rank updates. Defaults to "qr".
        reortho_method (OrthoMethod, optional):
            how to re-orthogonalize if Q stops being orthonormal
            (generally that shouldn't happen unless there is some severe instability). Defaults to "qr".
        reortho_threshold (float, optional):
            threshold on MSE(Q^T Q, I) to trigger re-orthogonalization. Defaults to 1.
    """
    def __init__(
        self,
        rank: int = 100,
        beta: float | None = 0.99,

        eig_tol: float = 1e-4,
        damping: float = 0,
        rdamping: float = 0,
        abs: bool = False,

        update_truncate: int | None = None,
        update_eig_tol: float | None = None,
        update_damping: float = 1e-4,
        update_rdamping: float = 0,
        update_abs: bool = False,

        ortho_method: OrthoMethod = "qr",
        reortho_method: OrthoMethod = "qr",
        reortho_threshold: float = 1,

    ):
        self.Q = None
        self.L = None

        self.beta = beta
        self.rank = rank
        self.eig_tol = eig_tol
        self.damping = damping
        self.rdamping = rdamping
        self.abs = abs

        self.update_truncate = update_truncate
        self.update_eig_tol = update_eig_tol
        self.update_damping = update_damping
        self.update_rdamping = update_rdamping
        self.update_abs = update_abs
        self.ortho_method: OrthoMethod = ortho_method
        self.reortho_method: OrthoMethod = reortho_method
        self.reortho_threshold = reortho_threshold

        self.current_step = 0

    def update(self, Y, Z, alpha):
        self.current_step += 1

        # initialize on 1st step
        if self.Q is None or self.L is None:
            L, self.Q = low_rank_eigh(Y)
            if self.beta is None:
                self.L = L * alpha
                return self.Q, self.Q, L, L

            # debias
            self.L = L * (1 - self.beta) * alpha

            bias_correction = (1 - self.beta ** self.current_step)
            L = self.L / bias_correction

            return self.Q, self.Q, L, L

        # lerp accumulator
        if self.beta is None:
            w1 = w2 = 1
        else:
            w1 = self.beta
            w2 = 1 - self.beta

        if Z is None:
            L, Q = eigh_plus_UUt(self.L * w1, self.Q, Y, alpha=w2*alpha, ortho_method=self.ortho_method)
        else:
            L, Q = eigh_plus_UVt_symmetrize(self.L * w1, self.Q, Y, Z, alpha=w2*alpha, ortho_method=self.ortho_method)


        # regularize accumulator
        if L is not None and Q is not None:
            L, Q = regularize_eigh(L, Q, truncate=self.rank, tol=self.eig_tol, damping=self.damping, rdamping=self.rdamping)


            if L is not None and Q is not None:
                QtQ = Q.T @ Q
                I = torch.eye(QtQ.size(0), device=QtQ.device, dtype=QtQ.dtype)
                if torch.nn.functional.mse_loss(QtQ, I) > self.reortho_threshold:
                    Q = orthogonalize(Q, method=self.reortho_method)

                if self.abs: L = L.abs()
                self.L = L
                self.Q = Q

                # second round of regularization for computing the update
                # this doesn't affect the accumulator

                if self.update_abs: L = L.abs()
                L_reg, Q_reg = regularize_eigh(L, Q, truncate=self.update_truncate, tol=self.update_eig_tol,
                                            damping=self.update_damping, rdamping=self.update_rdamping)

                # debias
                L = self.L
                if self.beta is not None:
                    bias_correction = (1 - self.beta ** self.current_step)
                    L = self.L / bias_correction
                    if L_reg is not None:
                        L_reg = L_reg / bias_correction

                return self.Q, Q_reg, L, L_reg

        return None, None, None, None




class RandomizedEigen(Basis):
    """stores curvature matrix directly as eigendecomposition and updates it via randomized eigendecomposition.

    On each update it orthogonalizes ``ndim x rank`` matrix and computes eigendecomposition of ``rank x rank`` matrix.

    Note that there are two rounds of regularization. First round (``eig_tol``, ``damping``, ``rdamping``, ``abs``)
    affects the accumulator, i.e. further accumulator updates will be with regularized accumulator.

    The second round of regularization (every argument starting with ``update_``) is computed after the first
    and doesn't affect the accumulator, it is used to reproject inner optimizer and compute the update.

    Args:
        rank (int, optional): maximum rank of approximation. Defaults to 100.
        beta (float, optional): coefficient for exponenital moving average of the curvature matrix. Defaults to 0.99.
        oversampling (int, optional):
            computes ``rank+oversampling``-rank approximation and truncates to ``rank``. Defaults to 10.
        eig_tol (float, optional):
            On each curvature update removes eigenvalues smaller than largest eigenvalue times this tolerance value.
            Defaults to 1e-12.
        damping (float, optional):
            On each curvature update adds this value to eigenvalues of the accumulator,
            this affects the curvature accumulator. Defaults to 0.
        rdamping (float, optional):
            On each curvature update adds this value times largest eigenvalue to eigenvalues of the accumulator,
            this affects the curvature accumulator. Defaults to 0.
        abs (bool, optional):
            if True, forces the accumulator to have positive eigenvalues (by taking their absolute values after each update). Defaults to False.

        update_truncate (int | None, optional):
            truncates eigendecomposition to top ``truncate`` eigenvalues (set to less than ``rank``). Defaults to None.
        update_eig_tol (float | None, optional):
            removes eigenvalues smaller than largest eigenvalue times this tolerance value, does not affect accumulator.
            Defaults to None.
        update_damping (float, optional):
            constant value added to eigenvalues, does not affect accumulator. Defaults to 1e-12.
        update_rdamping (float, optional):
            constant value times largest eigenvalue added to eigenvalues, does not affect accumulator. Defaults to 0.
        update_abs (bool, optional):
            whether to take absolute value of the eigenvalues, doesn't affect the accumulator. Defaults to False.
        ortho_method (OrthoMethod, optional):
            how to orthogonalize Q Omega which is ``ndim x rank``. Defaults to "qr".
    """
    def __init__(
        self,
        rank: int = 100,
        beta: float | None = 0.99,
        oversampling: int = 10,

        eig_tol: float = 1e-12,
        damping: float = 0,
        rdamping: float = 0,
        abs: bool = False,

        update_truncate: int | None = None,
        update_eig_tol: float | None = None,
        update_damping: float = 1e-12,
        update_rdamping: float = 0,
        update_abs: bool = False,

        ortho_method: OrthoMethod = "qr",

    ):
        self.Q = None
        self.L = None

        self.beta = beta
        self.rank = rank
        self.oversampling = oversampling

        self.eig_tol = eig_tol
        self.damping = damping
        self.rdamping = rdamping
        self.abs = abs

        self.update_truncate = update_truncate
        self.update_eig_tol = update_eig_tol
        self.update_damping = update_damping
        self.update_rdamping = update_rdamping
        self.update_abs = update_abs
        self.ortho_method: OrthoMethod = ortho_method

        self.current_step = 0

    def update(self, Y, Z, alpha):
        assert Z is None, "Randomized eigen only works on symmetric matrices, set `symmetrize=True` in LRE"

        self.current_step += 1

        # initialize on 1st step
        if self.Q is None or self.L is None:
            L, self.Q = low_rank_eigh(Y)
            if self.beta is None:
                self.L = L * alpha
                return self.Q, self.Q, L, L

            # debias
            self.L = L * (1 - self.beta) * alpha

            bias_correction = (1 - self.beta ** self.current_step)
            L = self.L / bias_correction

            return self.Q, self.Q, L, L

        # lerp accumulator
        if self.beta is None:
            w1 = w2 = 1
        else:
            w1 = self.beta
            w2 = 1 - self.beta

        L, Q = randomized_eigh_plus_UUt(
            L1=self.L,
            Q1=self.Q,
            U=Y,
            w1=w1,
            w2=w2*alpha,
            oversampling_p=self.oversampling,
            rank=self.rank,
            eig_tol=self.eig_tol,
            damping=self.damping,
            rdamping=self.rdamping,
            ortho_method=self.ortho_method,
        )


        if L is not None and Q is not None:
            # QtQ = Q.T @ Q
            # I = torch.eye(QtQ.size(0), device=QtQ.device, dtype=QtQ.dtype)
            # print(f'{torch.nn.functional.mse_loss(QtQ, I) = }')

            if self.abs: L = L.abs()
            self.L = L
            self.Q = Q

            # second round of regularization for computing the update
            if self.update_abs: L = L.abs()
            L_reg, Q_reg = regularize_eigh(L, Q, truncate=self.update_truncate, tol=self.update_eig_tol,
                                           damping=self.update_damping, rdamping=self.update_rdamping)

            # debias
            L = self.L
            if self.beta is not None:
                bias_correction = (1 - self.beta ** self.current_step)
                L = L / bias_correction
                if L_reg is not None:
                    L_reg = L_reg / bias_correction

            return self.Q, Q_reg, L, L_reg

        return None, None, None, None




class Nystrom(Basis):
    """stores curvature matrix directly as eigendecomposition and updates it via Nyström approximation.

    On each update the most expensive operation is SVD of ``ndim x rank`` matrix, but we compute it by
    eigendecomposition of ``rank x rank``. It also computes cholesky+triangular solve of ``rank x rank``.

    It also requires an orthonormal test matrix which is computed via QR of ``ndim x rank`` random matrix
    so it shouldn't suffer from freezing, can also re-generate it every ``omega_update_freq`` steps.

    Note that there are two rounds of regularization. First round (``eig_tol``, ``damping``, ``rdamping``, ``abs``)
    affects the accumulator, i.e. further accumulator updates will be with regularized accumulator.

    The second round of regularization (every argument starting with ``update_``) is computed after the first
    and doesn't affect the accumulator, it is used to reproject inner optimizer and compute the update.

    Args:
        rank (int, optional): maximum rank of approximation. Defaults to 100.
        beta (float, optional): coefficient for exponenital moving average of the curvature matrix. Defaults to 0.99.
        oversampling (int, optional):
            computes ``rank+oversampling``-rank approximation and truncates to ``rank``. Defaults to 10.
        omega_update_freq (int, optional):
            generates new orthonormal matrix via QR every this many steps. Defaults to 1.
        eig_tol (float, optional):
            On each curvature update removes eigenvalues smaller than largest eigenvalue times this tolerance value.
            Defaults to 1e-12.
        damping (float, optional):
            On each curvature update adds this value to eigenvalues of the accumulator,
            this affects the curvature accumulator. Defaults to 0.
        rdamping (float, optional):
            On each curvature update adds this value times largest eigenvalue to eigenvalues of the accumulator,
            this affects the curvature accumulator. Defaults to 0.
        abs (bool, optional):
            if True, forces the accumulator to have positive eigenvalues (by taking their absolute values after each update). Defaults to False.

        update_truncate (int | None, optional):
            truncates eigendecomposition to top ``truncate`` eigenvalues (set to less than ``rank``). Defaults to None.
        update_eig_tol (float | None, optional):
            removes eigenvalues smaller than largest eigenvalue times this tolerance value, does not affect accumulator.
            Defaults to None.
        update_damping (float, optional):
            constant value added to eigenvalues, does not affect accumulator. Defaults to 1e-12.
        update_rdamping (float, optional):
            constant value times largest eigenvalue added to eigenvalues, does not affect accumulator. Defaults to 0.
        update_abs (bool, optional):
            whether to take absolute value of the eigenvalues, doesn't affect the accumulator. Defaults to False.

        ortho_method (OrthoMethod, optional):
            how to orthogonalize random test matrix which is ``ndim x rank``. Defaults to "qr".
    """
    def __init__(
        self,
        rank: int = 100,
        beta: float | None = 0.99,
        oversampling: int = 10,
        omega_update_freq: int = 1,

        eig_tol: float = 1e-12,
        damping: float = 0,
        rdamping: float = 0,
        abs: bool = False,

        update_truncate: int | None = None,
        update_eig_tol: float | None = None,
        update_damping: float = 1e-12,
        update_rdamping: float = 0,
        update_abs: bool = False,

        ortho_method: OrthoMethod = "qr",

    ):
        self.Q = None
        self.L = None
        self.Omega = None

        self.beta = beta
        self.rank = rank
        self.oversampling = oversampling
        self.omega_update_freq = omega_update_freq

        self.eig_tol = eig_tol
        self.damping = damping
        self.rdamping = rdamping
        self.abs = abs

        self.update_truncate = update_truncate
        self.update_eig_tol = update_eig_tol
        self.update_damping = update_damping
        self.update_rdamping = update_rdamping
        self.update_abs = update_abs
        self.ortho_method: OrthoMethod = ortho_method

        self.current_step = 0

    def update(self, Y, Z, alpha):
        assert Z is None, "Randomized eigen only works on symmetric matrices, set `symmetrize=True` in LRE"

        self.current_step += 1

        # initialize on 1st step
        if self.Q is None or self.L is None:
            L, self.Q = low_rank_eigh(Y)
            if self.beta is None:
                self.L = L * alpha
                return self.Q, self.Q, L, L

            # bias correction
            self.L = L * (1 - self.beta) * alpha

            bias_correction = (1 - self.beta ** self.current_step)
            L = self.L / bias_correction

            return self.Q, self.Q, L, L

        # Gaussian test matrix
        if self.Omega is None or self.current_step % self.omega_update_freq == 0:
            Omega = torch.randn((Y.size(0), self.rank + self.oversampling), device=Y.device, dtype=Y.dtype)
            self.Omega = orthogonalize(Omega, method=self.ortho_method)

        # lerp accumulator
        if self.beta is None:
            w1 = w2 = 1
        else:
            w1 = self.beta
            w2 = 1 - self.beta

        AOmega = eigh_plus_UUt_mm(L=self.L, Q=self.Q, U=Y, B=self.Omega, w1=w1, w2=(w2*alpha))
        L, Q = nystrom_approximation(self.Omega, AOmega, 0)

        L, Q = regularize_eigh(L, Q, truncate=self.rank, tol=self.eig_tol, damping=self.damping, rdamping=self.rdamping)

        if L is not None and Q is not None:
            # QtQ = Q.T @ Q
            # I = torch.eye(QtQ.size(0), device=QtQ.device, dtype=QtQ.dtype)
            # print(f'{torch.nn.functional.mse_loss(QtQ, I) = }')

            if self.abs: L = L.abs()
            self.L = L
            self.Q = Q

            # second round of regularization for computing the update
            if self.update_abs: L = L.abs()
            L_reg, Q_reg = regularize_eigh(L, Q, truncate=self.update_truncate, tol=self.update_eig_tol,
                                            damping=self.update_damping, rdamping=self.update_rdamping)

            # debias
            L = self.L
            if self.beta is not None:
                bias_correction = (1 - self.beta ** self.current_step)
                L = L / bias_correction
                if L_reg is not None:
                    L_reg = L_reg / bias_correction

            return self.Q, Q_reg, L, L_reg

        return None, None, None, None



class HebbianLearning(Basis):
    """Uses generalized Hebbian algorithm to find the highest principal component vectors.

    On each step it performs one step of gradient descent on principal component vectors,
    except we want Q to stay orthonormal, so this performs optimization on Stiefel manifold
    (meaning Q is forced to stay orthonormal) using AdamG algorithm from
    https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform

    AdamG on each step with 1% probability computes QR of the gradient, and there are no other decompositions involved.

    This does not approximate L. So you can only run a basis optimizer.

    Args:
        rank (int, optional): rank of approximation. Defaults to 100.
        lr (float, optional): learning rate for updating the basis. Defaults to 1e-2.
        value_clip (float | None, optional): gradient value clipping for updating basis. Defaults to None.
        norm_clip (float | None, optional): gradient norm clipping for updating basis. Defaults to None.
        opt (str, optional): optimizer.
            - "adamg" - runs AdamG, no reorthogonalization is needed
            - "gd" - just runs gradient descent, so you have to reorthogonalize after every step by setting ``reortho_freq=1``.
            - "stiefel_gd" - runs GD on stiefel manifold, except you still have to retract by setting ``reortho_freq=1``.
            Defaults to "adamg".
        opt_kwargs (_type_, optional):
            kwargs to pass to AdamG. Defaults:
            ```python
            dict(beta1:float=0.9, beta2:float=0.99, eps:float=1e-8, cautious:bool=False)
            ```
        reortho_freq (int | None, optional):
            frequency of reorthogonalization, only needed for "gd" and "stiefel_gd". Defaults to None.
        reortho_method (OrthoMethod, optional):
            how to reorthogonalize. Defaults to "ns2".

    """
    def __init__(
        self,
        rank: int = 100,
        lr=1e-2,
        value_clip:float|None=None,
        norm_clip:float|None=None,

        opt="adamg",
        opt_kwargs = None,

        reortho_freq: int | None = None,
        reortho_method: OrthoMethod = "ns2",
    ):
        self.Q = None
        self.L = None

        self.rank = rank
        self.lr = lr
        self.value_clip = value_clip
        self.norm_clip = norm_clip
        self.opt = opt
        self.opt_state = {}
        self.opt_kwargs = opt_kwargs if opt_kwargs is not None else {}

        self.current_step = 0

        self.reortho_freq = reortho_freq
        self.reortho_method: OrthoMethod = reortho_method

    def update(self, Y, Z, alpha):
        assert Z is None, "HebbianLearning only works on symmetric matrices, set `symmetrize=True` in LRE"
        self.current_step += 1

        # initialize Q on 1st step
        if self.Q is None:
            Q = torch.randn(Y.size(0), self.rank, device=Y.device, dtype=Y.dtype) * 1e-4
        else:
            Q = self.Q

        # lerp accumulator
        if self.opt == 'gd':
            Q = sanger_(Q, Y, lr=self.lr, norm_clip=self.norm_clip, value_clip=self.value_clip)
        elif self.opt == 'stiefel_gd':
            Q = sanger_stiefel_(Q, Y, lr=self.lr, norm_clip=self.norm_clip, value_clip=self.value_clip)
        elif self.opt == "adamg":
            Q = sanger_adamg(Q, Y, state=self.opt_state, lr=self.lr, norm_clip=self.norm_clip, value_clip=self.value_clip, **self.opt_kwargs)

        # reorthogonalize
        if self.reortho_freq is not None and self.current_step % self.reortho_freq == 0:
            Q = orthogonalize(Q, method=self.reortho_method)

        self.Q = Q

        return Q, Q, None, None


class Random(Basis):
    """random basis"""
    def __init__(self, rank=100, generate_freq:int=1):
        super().__init__()
        self.rank = rank
        self.generate_freq = generate_freq
        self.current_step = 0
        self.Q = None

    def update(self, Y, Z, alpha):
        if self.Q is None or self.current_step % self.generate_freq == 0:
            self.Q = orthogonalize(torch.randn([Y.size(0), self.rank], device=Y.device, dtype=Y.dtype), 'qr')

        return self.Q, self.Q, None, None

class FullMatrix(Basis):
    """Full-matrix basis. Stores full matrix and computes eigendecomposition on each update"""

    def __init__(
        self,
        beta: float | None = 0.99,
        truncate: int | None = None,
        eig_tol: float | None = 1e-12,
        damping: float = 0,
        rdamping: float = 0,
        abs: bool = False,
    ):
        super().__init__()
        self.M = None
        self.beta = beta

        self.truncate = truncate
        self.eig_tol = eig_tol
        self.damping = damping
        self.rdamping = rdamping
        self.abs = abs

    def update(self, Y, Z, alpha):
        if self.M is None:
            self.M = torch.eye(Y.size(0), Y.size(0), device=Y.device, dtype=Y.dtype)

        if Z is None: Z = Y
        correction = (Y * alpha) @ Z.T

        if self.beta is None:
            self.M.add_(correction)
        else:
            self.M.lerp_(correction, weight=1-self.beta)

        L, Q = torch.linalg.eigh(self.M) # pylint:disable=not-callable

        if self.abs: L = L.abs()
        L, Q = regularize_eigh(L, Q, truncate=self.truncate, tol=self.eig_tol, damping=self.damping, rdamping=self.rdamping)

        return Q, Q, L, L


class TopK(Basis):
    """returns a (truncated) permutation matrix with top rank elements in the correction, disregarding all other info. If ``beta`` is used, it has to be re-orthogonalized via QR"""
    def __init__(
        self,
        topk: int = 100,
        beta: float | None = 0.99,
        stable: bool = False,
        ortho_method: OrthoMethod = 'qr',
        track_ortho: bool = False,
    ):
        super().__init__()
        self.topk = topk
        self.beta = beta
        self.stable = stable
        self.ortho_method: OrthoMethod = ortho_method
        self.track_ortho = track_ortho
        self.P = None

    def update(self, Y, Z, alpha):
        assert Z is None, "TopKPermutationBasis only works on symmetric matrices, set `symmetrize=True` in LRE"

        v = Y.mean(1)
        topk = min(self.topk, v.numel())

        if self.stable:
            vals, argsort = torch.sort(v.abs(), descending=True, stable=True)
            vals = vals[:topk]
            argsort = argsort[:topk]
        else:
            vals, argsort = torch.topk(v.abs(), topk, sorted=True)

        P = torch.zeros((topk, v.numel()), dtype=v.dtype, device=v.device)
        P[range(topk), argsort] = 1.0
        P = P.T

        if self.P is None or self.beta is None or self.beta == 0:
            self.P = P
            self.vals = vals
        else:
            self.P.lerp_(P, 1-self.beta)
            vals = self.vals.lerp_(vals, 1-self.beta)

            P = orthogonalize(self.P, method=self.ortho_method)
            if self.track_ortho: self.P = P

        return P, P, vals, vals

class ImportanceSampling(Basis):
    """same as topk but doesn't just select top k elements, it selects random elements with probability based on their magnitude"""
    def __init__(
        self,
        rank: int = 100,
        beta: float | None = 0.99,
        importance_fn = torch.abs,
        ortho_method: OrthoMethod = 'qr',
        track_ortho: bool = False
    ):
        super().__init__()
        self.rank = rank
        self.beta = beta
        self.importance_fn = importance_fn
        self.ortho_method: OrthoMethod = ortho_method
        self.track_ortho = track_ortho
        self.P = None

    def update(self, Y, Z, alpha):
        assert Z is None, "TopKPermutationBasis only works on symmetric matrices, set `symmetrize=True` in LRE"

        v = Y.mean(1)
        m = v.numel()
        rank = min(self.rank, m)

        importance = self.importance_fn(v) + torch.finfo(v.dtype).tiny * 2

        # probabilities
        # stupid numpy keeps whining that they don't sum to 1
        # due to float precision so have to convert to numpy here
        p = importance.numpy(force=True)
        p = p / p.sum()

        # select random indices based on probabilities (note abs is needed because apparently minus 0 is negative)
        indices = torch.as_tensor(np.random.choice(m, size=rank, replace=False, p=p), device=v.device)
        indices, _ = torch.sort(indices)

        P = torch.zeros((rank, v.numel()), dtype=v.dtype, device=v.device)
        P[range(rank), indices] = 1.0
        P = P.T

        if self.P is None or self.beta is None or self.beta == 0:
            self.P = P

        else:
            self.P.lerp_(P, 1-self.beta)

            P = orthogonalize(self.P, method=self.ortho_method)
            if self.track_ortho: self.P = P

        return P, P, None, None



class HistogramSketching(Basis):
    """this tries to "clump" similar values together in the subspace"""
    def __init__(
        self,
        rank: int = 100,
        beta: float | None = 0.99,
        ortho_method: OrthoMethod = 'qr',
        track_ortho: bool = False,
    ):
        super().__init__()
        self.rank = rank
        self.beta = beta
        self.track_ortho = track_ortho
        self.Q = None
        self.ortho_method: OrthoMethod = ortho_method

    def update(self, Y, Z, alpha):
        assert Z is None, "TopKPermutationBasis only works on symmetric matrices, set `symmetrize=True` in LRE"

        v = Y.mean(1)
        m = v.shape[0]
        rank = min(self.rank, m)

        vmin = v.min()
        vmax = v.max()
        vmin, vmax = torch.min(v), torch.max(v)

        bin_width = (vmax - vmin) / rank
        bin_indices = ((v - vmin) / bin_width).floor().long()

        bin_indices = torch.clamp(bin_indices, 0, rank - 1)

        S = torch.nn.functional.one_hot(bin_indices, num_classes=rank).float() # pylint:disable=not-callable

        norms = torch.linalg.vector_norm(S, dim=0) # pylint:disable=not-callable
        norms[norms == 0] = 1.0
        Q = S / norms

        if self.Q is None or self.beta is None or self.beta == 0:
            self.Q = Q

        else:
            self.Q.lerp_(Q, 1-self.beta)

            Q = orthogonalize(self.Q, method=self.ortho_method)
            if self.track_ortho: self.Q = Q

        return Q, Q, None, None