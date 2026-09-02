# Mathematical architecture

## Dependency flow

`environment` → `dgp` → `initialization` → `primitives` → `algorithms` →
`callbacks` / `performance`; separately, population-law samplers and the same
`primitives` feed `asymptotics`.

## Quenched finite model

`QuenchedEnvironment(mu, Y, Delta)` in `src/environment.py` is the canonical
finite object.  It clones and validates the realised vectors and derives
`d,n,delta=n/d,I_L,I_U,N,M,rho=N/n`.  It is held fixed while GD runs.
`FourCellSampleTypeLaw` samples a general categorical law on
`(+1,1),(+1,0),(-1,1),(-1,0)`; `DataConfig(label_prior, supervision_ratio)`
remains the product/MCAR convenience model.

`IsotropicGaussian` in `src/dgp.py` generates the conditional design

$$
X_i=Y_i\mu/\sqrt d+\sigma U_i,qquad U_i\stackrel{\rm iid}{\sim}N(0,I_d),
$$

so `Delta` controls observation, not feature generation.

## Finite self-training

`SelfTrainingInitialization(b_init,w_init,Y_init)` is full indexed and
requires `Y_init[I_L]=Y[I_L]`.  At a finite update `t`,

$$
\widehat Y^0=Y^{init},\qquad \widehat Y^t=\operatorname{sign}(r^t) (t\ge1),
\qquad \operatorname{sign}(0)=1.
$$

`SelfTrainedGradientDescent.fit_full(X, environment, initialization)` is the
single mathematical implementation.  It freezes pseudo-labels, confidence
mask, and `omega` before forming

$$
g_i^t=-\eta\left\{\frac{\Delta_i}{\rho}\ell'(Y_i,r_i^t)+
\frac{(1-\Delta_i)\pi}{1-\rho}\frac{S_i^t}{\omega^t}
\ell'(\widehat Y_i^t,r_i^t)\right\},
$$

with a zero unlabeled term when `omega=0`.  The weight-vector update always
uses this residual with pseudo-label coefficient `pseudo_label_param`.  The
bias uses the companion residual

$$
g_{b,i}^t=-\eta\left\{\frac{\Delta_i}{\rho}\ell'(Y_i,r_i^t)+
\frac{(1-\Delta_i)\pi_b}{1-\rho}\frac{S_i^t}{\omega^t}
\ell'(\widehat Y_i^t,r_i^t)\right\},
$$

whose labeled contribution is unchanged.  If `bias_pseudo_label_param` is
omitted, $\pi_b$ follows the ordinary pseudo-label weight (including an
experimental schedule).  The updates are

$$
b^{t+1}=b^t+n^{-1}{\bf1}^Tg_b^t,\quad
w^{t+1}=w^t-\eta\lambda\nabla J(w^t)+\sqrt d\,X^Tg^t/n.
$$

`AlgorithmConfig.initial_bias` defaults to zero.  The value $b^0$ is included
in every logit regardless of `include_bias`; that flag controls only whether
the bias update is applied.  Thus `include_bias=False` defines a fixed
intercept, which may be nonzero.  Explicit initialization arguments in the
low-level finite and state-evolution APIs override the configuration default.

The legacy split `fit(X_lab,Y_lab,X_unl,...)` is only an adapter.  It never
silently uses `sign(r0)`: users can explicitly call
`compute_scores` and `compute_pseudo_labels_from_scores` for that
theorem-external experiment.

`FiniteStep` stores the exact frozen `g`, `omega`, labels, and observables used
in each update.  Callback residual metrics read this record rather than
reconstructing a different residual after the update.

## Effective dynamics

`src/asymptotics.py` keeps the projection-based orthogonal-coordinate core:
separate `K_w,K_g`, CGS2 bases with `B.T @ B/K=I`, transported `Q^[w],P^[g]`,
and small least-squares solves.  Its Gaussian convention is
`z_g` in sample space driving `q`, and `z_w` in parameter space driving `p`.
No production Gram inverse or autograd response is used.

`state_evolution_sample_base_sampler(law, ...)` adapts a joint sample-type law
to its existing low-level sampler.  `theorem_trajectory` presents
`(W_T,Q_T,R_T,G_{T-1},P_{T-1})`; internally retained `g^T,p^T` remain auxiliary
diagnostics.

Finite GD and state evolution also record a common, non-recursive oracle-bias
diagnostic.  For their shared test-score convention
$R^t=b^t+m^tY+\sigma\tau^tZ$, it evaluates

$$
b_{\rm oracle}^t=\frac{\sigma^2(\tau^t)^2}{2m^t}
\log\frac{p}{1-p}
$$

and the population error obtained by replacing only the evaluation intercept
$b^t$ with $b_{\rm oracle}^t$.  The actual $b^t$ remains in every finite and
effective-process update.  The oracle quantities are recorded as `NaN` when
$m^t$ is numerically zero.

## Reproducibility and extensions

The runner exposes `fixed_environment` (fixed `mu,Y,Delta,Y_init,w0`, fresh
design noise) and `resampled_environment` repetitions.  Metadata marks
endogenous `sign(r0)`, schedules, no-bias, smooth selection, and zero
initialisation as theorem-external.  Fixed scalar `pi` is canonical; legacy
ramps are represented by `LinearRampSchedule` / compatibility fields.
