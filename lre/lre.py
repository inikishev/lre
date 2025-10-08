"""LRE optimizer"""
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Protocol

import torch

from .basis import Basis
from .opt import BasisOptimizer, Chain
from .curvature import Curvature


def vec_to_tensors(vec: torch.Tensor, reference: list[torch.Tensor]) -> list[torch.Tensor]:
    tensors = []
    cur = 0
    for r in reference:
        numel = r.numel()
        tensors.append(vec[cur:cur+numel].view_as(r))
        cur += numel
    return tensors

class LRE(torch.optim.Optimizer):
    """Low-rank orthonormal basis optimizer.

    Args:
        params (Iterablep[torch.Tensor]):
            iterable of parameters or named_parameters to optimize or iterable of dicts defining parameter groups
        lr (float): learning rate.
        curvature (Curvature): Curvature matrix type.
        basis (Basis): Basis type.
        basis_optimizer (BasisOptimizer): Inner optimizer to run in basis.
        update_freq (int, optional):
            frequency of updating or recomputing the basis. For hessian-vector product based curvatures,
            this is also frequency of hessian-vector products.
        symmetrize (bool, optional):
            whether to symmetrize the correction, only has effect for FullHutchinsonHessian which is asymmetrical.
            Defaults to True.
        init_steps (int, optional):
            Number of first steps to not perform an update for and only update the basis,
            useful for hessian-vector product based curvatures to kick start hessian approximation. Defaults to 0.
    """
    def __init__(
        self,
        params,
        lr: float,

        curvature: Curvature,
        basis: Basis,
        basis_optimizer: BasisOptimizer | Sequence[BasisOptimizer],

        update_freq: int = 1,
        symmetrize:bool = True,

        init_steps: int = 0,
    ):
        self.curvature = curvature
        self.basis = basis

        if not isinstance(basis_optimizer, BasisOptimizer):
            basis_optimizer = Chain(*basis_optimizer)
        self.basis_optimizer = basis_optimizer

        self.update_freq = update_freq

        self.current_step = 0
        self.current_curvature_update = 0
        self.symmetrize = symmetrize
        self.init_steps = init_steps

        # those are unregularized factors needed to compute corrections for some curvatures like BFGS and SR1
        self.L = None
        self.Q = None

        defaults = dict(lr=lr) # to support schedulers
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None, loss=None): # pyright:ignore[reportIncompatibleMethodOverride]

        # concatenate grads to a vector
        params_list = [p for g in self.param_groups for p in g['params'] if p.requires_grad]

        if closure is None:
            grad_list = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params_list]
            grad = torch.cat([g.ravel() for g in grad_list])

        else:
            grad = None

        # update the curvature, for covariance this adds gradient to history every step;
        # for rank 1 hessian this computes Hz and stores z and Hz to history every update_freq steps;
        # state usually holds z_list and Hz_list so that we can run HutchinsonAdam, etc
        should_update = self.current_step % self.update_freq == 0 or self.current_step < self.init_steps

        loss, grad, state = self.curvature.update(
            loss=loss, grad=grad, closure=closure, params_list=params_list, should_update=should_update, Q=self.Q, L=self.L
        )

        if should_update:

            # compute the low rank correction Y Y^T or  Y Z^T, e.g. stacked gradients in case of covariance matrix,
            # or z and Hz in case of randomized hessian estimates.
            # Some methods like BFGS/SR1 requires current (unregularized) L and Q,
            if self.Q is None:
                self.Q, self.L = self.curvature.initialize(grad, state)

            Q = self.Q; L = self.L

            Y, Z, alpha = self.curvature.compute_correction(Q=Q, L=L)

            if self.symmetrize and Y is not None:
                # so this works much better than using assymetric corrections
                if Z is not None:
                    Y = (Y + Z) / 2
                    Z = None

            if Y is not None:
                # add correction to the curvature accumulator
                # and compute new factors Q and L (L may be None)
                Q, Q_reg, L, L_reg = self.basis.update(Y, Z, alpha)

                # store unregularized (but debiased) factors for future curvature updates
                if Q is not None:
                    self.Q = Q
                    self.L = L

                # reproject inner optimizer to new regularized basis
                if Q_reg is not None:
                    self.basis_optimizer.reproject(Q=Q_reg, L=L_reg)

                self.current_curvature_update += 1

        # for Hvp-based curvatures, use first few steps to kick-start the hessian
        if self.current_step < self.init_steps:
            self.current_step += 1
            return loss

        # step with optimizer in basis
        lr = self.param_groups[0]["lr"]
        assert grad is not None

        # project
        g_proj = self.basis_optimizer.project(grad)
        u_proj = g_proj.clone()

        # step
        u_proj = self.basis_optimizer.step(u_proj, g_proj, state)

        # unproject
        u = self.basis_optimizer.unproject(u_proj) * lr

        # update parameters
        updates_list = vec_to_tensors(u, reference=params_list)
        torch._foreach_sub_(params_list, updates_list)

        self.current_step += 1
        return loss

