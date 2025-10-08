# <h1 align='center'>Low-rank eigenbasis</h1>

## Preface

In whitening-based optimizers on each step a rank 1 correction $g g^T$ is computed to the covariance matrix.

Now what if we store covariance matrix as truncated eigendecomposition - $Q L Q^T$ where $Q$ is $ndim \times rank$ and $L$ is a diagonal matrix $rank \times rank$? A rank-1 correction to eigendecomposition can be computed by eigendecomposition of $rank \times rank$ matrix, making it cheap to update. We get inverse square root for free and can stabilize it easily by removing small eigenvalues and shifting them.

Furthermore, this method naturally extends to other curvature matrices, for example quasi-newton methods also add rank-1 or rank-2 correction to the hessian approximation, so we use the same exact algorithm.

So the algorithm maintains some curvature matrix as a low rank eigenbasis $Q L Q^T$. Specifically here $Q$ is an orthogonal ($Q^T = Q^{-1}$) rotation matrix which rotates curvature (e.g. covariance, hessian) to be diagonal. Now similar to how SOAP runs Adam in Shampoo's eigenbases, we can run some optimizer in our low rank eigebasis, except our basis is $rank$-dimensional making it possible to run expensive methods like full-matrix Adam or BFGS. Although since the basis is diagonalized, it is not clear if full-matrix methods are particularly useful here. Diagonal methods get rid of most of the variance as it is already mostly diagonal.

## The algorithm

We have: current curvature matrix (e.g. covariance, hessian) stored as truncated eigendecomposition $Q L Q^T$; some algorithm to compute low rank correction to truncated eigendecomposition; an inner optimizer we run in the basis.

1. Obtain gradients $g$

2. Obtain a correction to the curvature matrix. For covariance it is $g g^T$ (outer product of gradient with itself). For BFGS the update can be expressed in terms of two vectors $u u^T - v v^T$. There are many curvature matrices implemented in LRE detailed below. We never compute the outer product, all we need is the vectors.

3. Apply correction to truncated eigendecomposition. Several methods for this are implemented, as detailed below.

4. Regularize eigendecomposition - remove small eigenvalues, add constant value to eigenvalues, etc. We get new regularized $L$ and $Q$ factors with new correction.

5. Reproject inner optimizer to new basis. Define $C = Q^T_{new} Q_{old}$, then exponential moving average $m$ is reprojected as $C m$; Exponential moving average of squared gradients $v$ is reprojected as $\text{diagonal}(C \text{ diag}(v) C^T)$.

5. Project gradient $g_{proj} = Q^T g$, run some inner optimizer on $g_{proj}$ to get update $u_{proj}$, unproject via $u = Q u_{proj}$, and update parameters. We can also not run an inner optimizer and just precondition the gradient by computing $Q L^{-1/2} Q^T g$ in case of covariance or $Q L^{-1} Q^T g$ in case of hessian.

### Curvatures

1. Empirical covariance of gradients - the correction is $gg^T$.

2. Rank-1 hessian estimates - generate random $z$ and compute hessian-vector product $Hz$, then $Hz z^T$ (outer product of $Hz$ and $z$) approximates hessian (this is described here in section 3 <https://arxiv.org/pdf/1206.6464>). This is called ``FullHutchinsonHessian`` in LRE because it's similar to Hutchinson's diagonal estimator.

3. BFGS - given hessian approximation $B$, vectors $s_t = x_t - x_{t-1}$ (difference in consecutive parameters) and $y_t = g_{t} - g_{t-1}$ (difference in consecutive gradients), new $B$ is computed as $B_{t+1} = B_t + u u^T - v v^T$, where $u = \frac{1}{\sqrt{y_k^T s_k}} y_k$ and $v = \frac{1}{\sqrt{s_k^T B s_k}} B_k s_k$. But I tried to make it better for stochastic optimization, where $y_k$ is too noisy because gradients are sampled from different mini-batches, and BFGS doesn't work. But $y_k$ actually estimates hessian-vector product with $s_k$ using the finite difference formula $Hv_{est} = \frac{g(x + \epsilon v) - g(x)}{\epsilon}$. If we set $\epsilon$ to 1 and replace $v$ with $s_t = x_t - x_{t-1}$, we get $g(x_t + x_t - x_{t-1}) - g(x_t) = g(x_t + s_t) - g(x_t)$ - so it approximates hessian-vector product with $s_t$. You can actually use hessian-vector product with any vector, this is what greedy quasi-newton methods use (<https://arxiv.org/pdf/2002.00657>). Anyway all that to say that I replaced gradient differences with hessian-vector products.

4. SR1 - another quasi-newton method which apparently converges to hessian faster than BFGS. Unlike BFGS computes a rank 1 correction. The correction is $v v^T$ where $v = \frac{y_k - B_k s_k}{\sqrt{|(y_k - B_k s_k)^T s_k|}} $, note that correction is subtracted if $(y_k - B_k s_k)^T s_k$ is negative. We also use hessian-vector products instead of gradient differences.

5. Empirical fisher - this requires computing per-sample gradients, stacking them to $G \in \mathbb{R^{k \times n}}$ where $k$ is batch size and $n$ is number of parameters, and then the correction is $G^TG$. The total gradient is sum of per-sample gradients. In natural gradient you use $(G^TG)^{-1}$, there are no buffers. Since we maintain an exponential moving average, it becomes a mix of empirical covariance and empirical fisher.

6. Gauss-Newton - same correction as empirical fisher - $(G^T G)$. What is different is that the total gradient computed from per-sample gradients is $G^T r$ where $r$ is per-sample losses.

### Basis algorithms

1. GGT from <https://arxiv.org/pdf/1806.02958> - this explicitly stores a history of last ``history_size`` corrections and computes $Q$ and $L$ via eigendecomposition of ``history_size x history_size`` matrix. In the paper they only used gradients (for covariance), but other curvatures can be used too by maintainting history of corrections.

2. Direct eigendecomposition - as I said low rank correction to truncated eigendecomposition can be computed by eigendecomposition of $rank \times rank$ matrix, so same computational complexity as GGT. Unlike GGT this can be used to maintain exponential moving average of curvature matrix which may be useful.

3. Randomized eigendecomposition - this can approximate rank-$k$ truncated eigendecomposition of $A \in \mathbb{R}^{m \times m}$ by computing $A \Omega$ where $\Omega$ is a random test matrix of size $m \times k$, orthogonalizing resulting matrix, computing another product with $A$ and then computing eigendecomposition of a $k \times k$ matrix. We have our curvature as $Q L Q^T$, after adding correction it becomes $Q L Q^T + c c^T$ where $c$ is the correction. It's easy to compute $(Q L Q^T + c c^T) \Omega$ without forming $c c^T$, and then compute new factors via randomized eigendecomposition. This performs QR of $m \times k$ matrix on every step so its more expensive than previous ones. So why would we want this if direct eigendecomposition is cheaper? Well for some reason this randomized eigendecomposition is more stable, for example the tolerance for removing small eigenvalues can be much smaller.

4. Nyström approximation - another type of randomized eigendecomposition. Since I already have code for Nyström approximation in my another library I added it too but haven't tested much.

5. Hebbian learning - performs gradient descent on $Q$ which makes it converge to principal component vectors of whatever curvature corrections we put into it. It does not give you $L$ (singular values), but we don't need it if we run inner optimizer in $Q$. However it doesn't force $Q$ to be orthogonal, so we have to either reorthogonalize it after each step or optimize on Stiefel manifold. So I used AdamG from <https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform> to reduce the number of QRs. If this, works it will be the cheapest method because it only computes QR with 1% probability on each step, and no other decompositions. However this has the most hyperparameters and so far they have been very hard to tune.

6. Full matrix basis - literally stores $ndim \times ndim$ matrix, adds corrections to it, and returns its eigendecomposition. Only useful for testing on 2D functions.

7. Random orthonormal basis - just random Q, ignores curvature corrections.

### Inner optimizers

I made a chainable API so you can apply cautious updates to any optimizer, etc.

- `Adam` - Adam. You can get RMSprop by setting beta1 to 0.
- `FullMatrixAdam` - Adam with full covariance matrix (keep in mind it runs in `rank`-dimensional basis so its fine).
- `HutchinsonAdam` - replaces squared gradients with hutchinson hessian diagonal estimates, so basically SophiaH. The estimates can also be squared and it becomes AdaHessian. This reuses hessian-vector products from hessian-based curvatures.
- `FullMatrixHutchinsonAdam` - HutchinsonAdam but full matrix ($Hz z^T$ hessian estimates)
- `FullMatrixBFGSAdam` - replaces squared gradients with BFGS hessian estimate, but a debiased exponential moving average of it. This also reuses hessian-vector products from hessian-based curvatures.
- `CubicAdam` - diagonal Adam which also maintains third moments. So Adam actually minimizes a second order polynomial $\sqrt{m_2}x^2 + m_1x$, where $v_2$ is second moments and $m_1$ is first moments. By adding third moments $m_3$ (cubed gradients), we can minimize $\sqrt[3]{m_3} + \sqrt{m_2}x^2 + m_1x$ since it has an exact solution, does this have a theoretical meaning, I don't know.
- `Lion` - so for some reason, and I don't know why, it behaves like a second order optimizer in the basis despite not doing whitening.
- `EMA`, `Cautious`, `ClipNorm`, `ClipValue`, `NormalizeByEMA`, `CopyGradSign` - to chain with other inner optimizers.
- `LInvSqrt` - directly whitens gradients with covariance curvature, can also be chained and applied for example to momentum.
- `LInv` - same thing but for hessian-based curvatures.

So you can directly whiten momentum:

```py
[lre.opt.EMA() lre.opt.LInvSqrt()]
```

or track momentum of whitened gradients:

```py
[lre.opt.LInvSqrt(), lre.opt.EMA()]
```

or use any optimizer for whitening such as cautious Adam:

```py
[lre.opt.Adam(), lre.opt.Cautious()]
```

# How to use

Pick one of the curvatures from `lre.curvature`, one of the bases from `lre.basis`, and some optimizers to run in basis from `lre.opt`, and then use as any other pytorch optimizer. Here is a good one:

```python
optimizer = lre.LRE(
    model.parameters(),
    lr = 1e-2,
    curvature = lre.curvature.Covariance(),
    basis = lre.basis.GGT(),
    basis_optimizer = [lre.opt.Adam(), lre.opt.Cautious(), lre.opt.ClipNormByEMA(normalize=True)],
)
```

If you use any hessian-vector product curvatures or any per-sample gradient based ones, do this

```python
for inputs, targets in dataloader:
    preds = model(inputs)

    # for hessian-vector product
    loss = criterion(preds, targets)

    # for per-sample gradients loss should be vector of per-sample losses
    loss = criterion(preds, targets, reduction="none")

    # in both cases backward is handled by the optimizer
    optimizer.step(loss=loss)
```

Alternatively use closure with backward argument, this is also compatible with any pytorch optimizer.

```py
for inputs, targets in dataloader:
    def closure(backward=True):
        preds = model(inputs)
        loss = criterion(preds, targets)

        # If backward=True, closure should call optimizer.zero_grad() and loss.backward()
        # for per-sample gradients just remove this section or keep it because backward is always False.
        if backward:
            optimizer.zero_grad()
            loss.backward()

        return loss

    loss = optimizer.step(closure)
```

### Notes

- By default all bases except GGT maintain exponential moving average of curvature matrix. That can be changed to accumulating a sum by setting `beta=None`. BFGS and SR1 curvatures expect `beta=None` although maybe EMAs can work too.

- The default hyperparameters are somewhat tuned for covariance and GGT. So LRE might be unstable with other curvatures, biggest thing to increase in case of instability is `eig_tol` which is for removing small eigenvalues. I am doing experiments and tuning and stuff but its taking a while.

- There are a bunch of experimental curvatures. Like UnrolledGaussNewton and CovarianceTimesLoss. To be tested once I test other stuff, and to be removed if it doesn't work.

## Future directions

1. Do higher order optimization in the basis utilizing the small dimensionality. The 3rd moments (third order gradient outer product) and third derivatives are both a third-order tensor with 1,000,000-elements in rank-100 basis. To reproject it define $C = Q^T_{new} Q_{old}$, and reproject like this `torch.einsum('ai,bj,ck,ijk->abc', C, C, C, T)` where `T` is the tensor. The derivatives case defines a third order polynomial which you need to a add a fourth order penalty to and minimize with some optimizer, which I want to try. I don't really know what to do with 3rd moments. Maybe find a basis where they are diagonal and run cubic Adam?

2. There are algorithms specifically for tracking a subspace where most variance resides from noisy online observations, like [PETRELS](https://arxiv.org/abs/1207.6353). I think it would be good to run full-matrix Adam in it.

3. finding good default hyperparameters and running large scale benchmarks. I have to use my GPU for other stuff like my job and other projects, so this takes some time.

## License

MIT license

### Citing

```BibTeX
@Misc{lre2025,
  title        = {LRE: Low-rank eigenbasis},
  howpublished = {\url{https://github.com/inikishev/lre}},
  year         = {2025}
}
```
