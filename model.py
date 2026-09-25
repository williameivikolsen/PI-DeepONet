import itertools
from functools import partial
import jax.numpy as np
from jax import random, grad, jacrev, vmap, jit, lax
from jax.tree_util import tree_leaves
from jax import config
from jax.flatten_util import ravel_pytree
from jax.nn import relu, tanh, gelu, softplus, sigmoid, elu, swish
from numpy.polynomial.legendre import leggauss
import optax 
from torch.utils import data

ACTIVATIONS = {
    "relu": relu,
    "tanh": tanh,
    "gelu": gelu,
    "softplus": softplus,
    "sigmoid": sigmoid,
    "elu": elu,
}
_ACT_TO_NAME = {f: n for n, f in ACTIVATIONS.items()}


def resolve_activation(activation):
    if isinstance(activation, str):
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation name '{activation}'. "
                f"Known: {sorted(ACTIVATIONS)}"
            )
        return ACTIVATIONS[activation], activation
    return activation, _ACT_TO_NAME.get(activation, "custom")

# Loss weighting schemes. "local_ntk" is Algorithm 1 of Wang, Wang & Perdikaris
# (2022), "Improved architectures and training algorithms for deep operator
# networks": ntk_alpha = 1 gives their "NTK weights", 0.5 "moderate NTK weights".
# "global_ntk" is the per-term variant from their advection notebook.
WEIGHTINGS = ("none", "local_ntk", "global_ntk")

def MLP(layers, activation=relu):
    """ Vanilla MLP"""
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b

        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs

    return init, apply

@partial(jit, static_argnums=(4,))
def _sample_batch(key, inputs, output, branch_table, batch_size):
    # Memory helper
    idx = random.choice(key, output.shape[0], (batch_size,), replace=False)
    if branch_table is None:
        in_batch = tuple(arr[idx] for arr in inputs)
    else:
        branch_idx = inputs[0][idx]                    # (batch,)
        branch     = branch_table[branch_idx]          # (batch, J)
        rest       = tuple(arr[idx] for arr in inputs[1:])
        in_batch   = (branch,) + rest
    out_batch = output[idx]
    return in_batch, out_batch


class DataGenerator(data.Dataset):
    def __init__(self, inputs, output, batch_size=1024,
                 rng_key=random.PRNGKey(1234), branch_table=None):
        self.inputs     = tuple(np.asarray(arr) for arr in inputs)
        self.output     = np.asarray(output)
        self.N          = self.output.shape[0]
        self.batch_size = batch_size
        self.key        = rng_key
        self.branch_table = None if branch_table is None else np.asarray(branch_table)

    def __getitem__(self, index):
        # Generate one batch of data
        self.key, subkey = random.split(self.key)
        return _sample_batch(subkey, self.inputs, self.output,
                             self.branch_table, self.batch_size)


def build_val_batch(ds):
    """
    Build a single validation (inputs, outputs) tuple.
    """
    Q     = np.asarray(ds['Q'])            # (N, J)
    phi_0 = np.asarray(ds['phi_0'])        # (N, J)
    x     = np.asarray(ds['x'])            # (J,)
    N, J  = Q.shape
    Q_flat   = np.repeat(Q, J, axis=0)
    x_flat   = np.tile(x, N)
    phi_flat = phi_0.reshape(-1)
    return (Q_flat, x_flat), phi_flat


def build_data_arrays(ds):
    """
    Flat arrays for the supervised phi_0 loss.
    """
    Q     = np.asarray(ds['Q'])            # (N, J)
    phi_0 = np.asarray(ds['phi_0'])        # (N, J)
    x     = np.asarray(ds['x'])            # (J,)
    N, J  = Q.shape
    Q_flat   = np.repeat(Q, J, axis=0)     # (N*J, J)
    x_flat   = np.tile(x, N)               # (N*J,)
    phi_flat = phi_0.reshape(-1)           # (N*J,)

    return (Q_flat, x_flat), phi_flat


def build_bcs_arrays(ds, X, n_per_sample=50,
                     rng_key=random.PRNGKey(2025), N_angles=16):
    """
    Vacuum-BC evaluation points, with mu drawn from the Gauss-Legendre
    quadrature nodes.

    Every sample gets both boundaries: half of its points at x=0 with mu > 0
    (incoming from left is zero), the other half at x=X with mu < 0 (incoming
    from right is zero). Target is zero for every point.
    """
    Q    = np.asarray(ds['Q'])
    N, J = Q.shape
    total = N * n_per_sample

    mu_nodes, _ = leggauss(N_angles)
    mu_nodes = np.asarray(mu_nodes)
    pos_nodes = mu_nodes[mu_nodes > 0.0]      # left-boundary angles
    neg_nodes = mu_nodes[mu_nodes < 0.0]      # right-boundary angles

    sample_idx = np.repeat(np.arange(N), n_per_sample)
    is_left    = np.tile(np.arange(n_per_sample) % 2 == 0, N)

    k1, k2 = random.split(rng_key)
    mu_left  = random.choice(k1, pos_nodes, (total,))
    mu_right = random.choice(k2, neg_nodes, (total,))

    x_bc  = np.where(is_left, 0.0, X)
    mu_bc = np.where(is_left, mu_left, mu_right)

    y = np.stack([x_bc, mu_bc], axis=-1)   # (total, 2)
    s = np.zeros((total,))
    return (sample_idx, y), s, Q


def build_res_arrays(ds, X, n_per_sample=100,
                     rng_key=random.PRNGKey(2026), N_angles=16):
    """
    Interior collocation points for the PDE residual loss: x continuous in
    (0, X), mu drawn from the Gauss-Legendre nodes. Target is zero because
    residual_net already absorbs Q/2 via jnp.interp.
    """
    Q    = np.asarray(ds['Q'])
    N, J = Q.shape
    total = N * n_per_sample

    mu_nodes, _ = leggauss(N_angles)
    mu_nodes = np.asarray(mu_nodes)

    k1, k2 = random.split(rng_key)
    x_r  = random.uniform(k1, (total,), minval=0.0, maxval=X)
    mu_r = random.choice(k2, mu_nodes, (total,))

    sample_idx = np.repeat(np.arange(N), n_per_sample)

    y = np.stack([x_r, mu_r], axis=-1)
    s = np.zeros((total,))
    return (sample_idx, y), s, Q


class PI_DeepONet:
    def __init__(self, branch_layers, trunk_layers, N_angles,
                 Sigma_t, Sigma_s0, Sigma_s1,
                 x_sensors, X, Q_shift, Q_scale,
                 lambda_data=1.0, lambda_res=1.0, lambda_bcs=1.0,
                 weighting="none",
                 ntk_alpha=1.0,
                 ntk_chunk_size=None,
                 branch_activation=relu,
                 trunk_activation=relu,
                 lr_init=1e-3,
                 lr_decay_rate=0.9,
                 lr_transition_steps=2000,
                 lr_schedule=None,
                 seed=None):
        branch_activation, self.branch_activation_name = resolve_activation(branch_activation)
        trunk_activation,  self.trunk_activation_name  = resolve_activation(trunk_activation)
        self.branch_activation = branch_activation
        self.trunk_activation  = trunk_activation

        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=branch_activation)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=trunk_activation)
        self.N_angles = N_angles

        # Initialize Parameters (use seed for reproducible init per trial)
        if seed is None:
            seed = 1234
        key = random.PRNGKey(seed)
        k1, k2 = random.split(key)
        branch_params = self.branch_init(rng_key=k1)
        trunk_params = self.trunk_init(rng_key=k2)
        self.params = (branch_params, trunk_params)

        # Cross sections
        self.Sigma_t  = Sigma_t
        self.Sigma_s0 = Sigma_s0
        self.Sigma_s1 = Sigma_s1

        # Gauss-Legendre quadrature — fixed constants, not trainable
        _mu, _w = leggauss(N_angles)
        self.mu_GL = np.array(_mu)   # shape (N_angles,)
        self.w_GL  = np.array(_w)    # shape (N_angles,)

        # Spatial sensor grid — needed to interpolate Q(x) at arbitrary
        # collocation points via jnp.interp inside residual_net.
        self.x_sensors = np.asarray(x_sensors)   # shape (J,)
        self.X         = float(X)                # slab length

        # Branch-input transform: (Q - Q_shift) / Q_scale, constants taken from
        # the training set. Applied only where Q enters the branch; the source
        # term in residual_net keeps raw Q.
        self.Q_shift = float(Q_shift)
        self.Q_scale = float(Q_scale)

        # Loss-term weights
        self.lambda_data = float(lambda_data)
        self.lambda_res  = float(lambda_res)
        self.lambda_bcs  = float(lambda_bcs)

        # Adaptive per-point weights (see ntk_weights), applied on top of the
        # fixed lambdas above. With NTK weighting the lambdas should normally
        # be 1.0; a lambda of 0 still switches its term off entirely.
        if weighting not in WEIGHTINGS:
            raise ValueError(f"Unknown weighting '{weighting}'. Known: {WEIGHTINGS}")
        self.weighting = weighting
        self.ntk_alpha = float(ntk_alpha)
        self.ntk_chunk_size = ntk_chunk_size

        # Learning rate.
        if lr_schedule is None:
            lr_schedule = lr_init
        elif lr_schedule == "exp_decay":
            lr_schedule = optax.exponential_decay(
                init_value=lr_init,
                transition_steps=lr_transition_steps,
                decay_rate=lr_decay_rate,
            )
        self.lr_schedule = lr_schedule
        self.optimizer = optax.adam(learning_rate=lr_schedule)
        self.opt_state = self.optimizer.init(self.params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(self.params)

        self.itercount = itertools.count()

        # Loggers
        self.loss_log       = []
        self.loss_data_log  = []
        self.loss_bcs_log   = []
        self.loss_res_log   = []
        self.lam_data_log   = []    # batch-mean of the adaptive weights
        self.lam_bcs_log    = []
        self.lam_res_log    = []

    # DeepONet architecture
    def operator_net(self, params, Q, x, mu):
        branch_params, trunk_params = params
        y = np.stack([x, mu])
        B = self.branch_apply(branch_params, (Q - self.Q_shift) / self.Q_scale)
        T = self.trunk_apply(trunk_params, y)
        outputs = np.sum(B * T)
        return outputs

    def residual_net(self, params, Q, x, mu):
        """
        1D transport residual at a single evaluation point.
        """
        # Angular flux at (x, mu_k) for every GL quadrature node.
        psi_vec = vmap(
            lambda mu_k: self.operator_net(params, Q, x, mu_k)
        )(self.mu_GL)                                # shape (N_angles,)

        # Moments via GL quadrature.
        phi_0 = np.dot(self.w_GL, psi_vec)
        phi_1 = np.dot(self.w_GL * self.mu_GL, psi_vec)

        # psi and its x-derivative at the point (x, mu).
        psi_at_mu = self.operator_net(params, Q, x, mu)
        psi_x     = grad(self.operator_net, argnums=2)(params, Q, x, mu)

        # Q(x) via linear interpolation on the sensor grid.
        Q_x = np.interp(x, self.x_sensors, Q)

        res = (
            mu * psi_x
            + self.Sigma_t * psi_at_mu
            - 0.5 * (self.Sigma_s0 * phi_0 + 3.0 * mu * self.Sigma_s1 * phi_1)
            - 0.5 * Q_x
        )
        return res

    def phi0_net(self, params, Q, x):
        """Scalar flux phi_0(x) via GL quadrature of the angular flux."""
        psi_vec = vmap(
            lambda mu_k: self.operator_net(params, Q, x, mu_k)
        )(self.mu_GL)
        return np.dot(self.w_GL, psi_vec)

    def data_net(self, params, Q, x):
        """The prediction the data loss compares against its labels."""
        return self.phi0_net(params, Q, x)

    # Neural Tangent Kernel diagonal
    def ntk_diag(self, f, params, *args):
        """
        Diagonal NTK entries H_kk = ||d f_k / d theta||^2 (Eq. 3.7 of the paper)
        for every point in the batch (args batched along axis 0) and every
        output component of f. Returns shape (batch,) + f's output shape.
        """
        param_leaves = tree_leaves(params)

        def one(a):
            J = tree_leaves(jacrev(f)(params, *a))   # each: out_shape + param_shape
            return sum(np.sum(j ** 2, axis=tuple(range(j.ndim - p.ndim, j.ndim)))
                       for j, p in zip(J, param_leaves))

        # The per-example Jacobians are materialized: batch * out_size * n_params
        # floats. ntk_chunk_size bounds that by looping over chunks of the batch.
        if self.ntk_chunk_size is None:
            return vmap(one)(args)
        return lax.map(one, args, batch_size=self.ntk_chunk_size)

    @partial(jit, static_argnums=(0,))
    def ntk_weights(self, params, data_batch, bcs_batch, res_batch):
        """
        NTK-guided weights for every point of the three batches.

        local_ntk  (Algorithm 1):  lambda_k = (max_j H_jj / H_kk)^alpha,
                                   the max running over all active terms.
        global_ntk (one weight per loss term t):
                                   lambda_t = (sum_s mean(H_s) / mean(H_t))^alpha.
        Means rather than the notebook's traces keep the global weights
        independent of the batch sizes; with equal batch sizes they coincide.

        Terms whose fixed lambda is 0 are left out of the kernel and get weight 1.
        """
        (Q_d, x_d), _ = data_batch
        (Q_b, y_b), _ = bcs_batch
        (Q_r, y_r), _ = res_batch

        K = {}
        if self.lambda_data > 0:
            K["data"] = self.ntk_diag(self.data_net, params, Q_d, x_d)
        if self.lambda_bcs > 0:
            K["bcs"] = self.ntk_diag(self.operator_net, params, Q_b, y_b[:, 0], y_b[:, 1])
        if self.lambda_res > 0:
            K["res"] = self.ntk_diag(self.residual_net, params, Q_r, y_r[:, 0], y_r[:, 1])

        if self.weighting == "local_ntk":
            K_max = np.max(np.stack([k.max() for k in K.values()]))
            lam = {t: (K_max / k) ** self.ntk_alpha for t, k in K.items()}
        else:
            K_mean = {t: k.mean() for t, k in K.items()}
            K_sum  = sum(K_mean.values())
            lam = {t: (K_sum / m) ** self.ntk_alpha * np.ones_like(K[t])
                   for t, m in K_mean.items()}

        lam = lax.stop_gradient(lam)
        return tuple(lam.get(t, 1.0) for t in ("data", "bcs", "res"))

    def loss_weights(self, params, data_batch, bcs_batch, res_batch):
        """(lam_data, lam_bcs, lam_res): per-point weights, or 1.0 when unweighted."""
        if self.weighting == "none":
            return 1.0, 1.0, 1.0
        return self.ntk_weights(params, data_batch, bcs_batch, res_batch)

    # Boundary loss
    def loss_bcs(self, params, batch, lam=1.0):
        inputs, outputs = batch
        Q, y = inputs
        phi_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, Q, y[:, 0], y[:, 1])
        loss = np.mean(lam * (outputs.flatten() - phi_pred) ** 2)
        return loss

    # Residual loss
    def loss_res(self, params, batch, lam=1.0):
        inputs, outputs = batch
        Q, y = inputs
        pred = vmap(self.residual_net, (None, 0, 0, 0))(params, Q, y[:, 0], y[:, 1])
        loss = np.mean(lam * (outputs.flatten() - pred) ** 2)
        return loss

    # Supervised data loss on the output of data_net.
    def loss_data(self, params, batch, lam=1.0):
        inputs, outputs = batch
        Q, x = inputs
        pred = vmap(self.data_net, (None, 0, 0))(params, Q, x)
        return np.mean(lam * (outputs.reshape(pred.shape) - pred) ** 2)

    # Total loss. lams are the adaptive weights; computed here when not given.
    def loss(self, params, data_batch, bcs_batch, res_batch, lams=None):
        if lams is None:
            lams = self.loss_weights(params, data_batch, bcs_batch, res_batch)
        lam_data, lam_bcs, lam_res = lams
        l_data = self.loss_data(params, data_batch, lam_data)
        l_bcs  = self.loss_bcs(params, bcs_batch, lam_bcs)
        l_res  = self.loss_res(params, res_batch, lam_res)
        return (
            self.lambda_data * l_data
            + self.lambda_bcs * l_bcs
            + self.lambda_res * l_res
        )

    # Update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, params, opt_state, data_batch, bcs_batch, res_batch):
        # Algorithm 1: weights from the current parameters, then a gradient
        # step on the weighted loss with the weights held fixed.
        lams  = self.loss_weights(params, data_batch, bcs_batch, res_batch)
        grads = grad(self.loss)(params, data_batch, bcs_batch, res_batch, lams)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # Optimize parameters in a loop
    def train(self, data_dataset, bcs_dataset, res_dataset,
              nIter=10000, log_every=100, callback=None,
              val_batch=None, val_every=None):
        data_iter = iter(data_dataset)
        bcs_iter  = iter(bcs_dataset)
        res_iter  = iter(res_dataset)

        if val_every is None:
            val_every = log_every

        # Validation bookkeeping
        self.val_ARE_log    = []
        self.val_iter_log   = []
        self.best_params    = self.params
        self.best_val_ARE   = float("inf")
        self.best_val_iter  = 0

        for it in range(nIter):
            data_batch = next(data_iter)
            bcs_batch  = next(bcs_iter)
            res_batch  = next(res_iter)

            self.params, self.opt_state = self.step(
                next(self.itercount), self.params, self.opt_state,
                data_batch, bcs_batch, res_batch,
            )

            if it % log_every == 0:
                # L is the (weighted) training objective; the individual
                # terms are logged unweighted so runs stay comparable.
                lams   = self.loss_weights(self.params, data_batch, bcs_batch, res_batch)
                l      = self.loss(self.params, data_batch, bcs_batch, res_batch, lams)
                l_data = self.loss_data(self.params, data_batch)
                l_bcs  = self.loss_bcs(self.params, bcs_batch)
                l_res  = self.loss_res(self.params, res_batch)

                self.loss_log.append(float(l))
                self.loss_data_log.append(float(l_data))
                self.loss_bcs_log.append(float(l_bcs))
                self.loss_res_log.append(float(l_res))

                lam_means = [float(np.mean(lam)) for lam in lams]
                self.lam_data_log.append(lam_means[0])
                self.lam_bcs_log.append(lam_means[1])
                self.lam_res_log.append(lam_means[2])

                line = (f"Iter {it:6d}: L={float(l):.3e}  "
                        f"L_data={float(l_data):.3e}  "
                        f"L_bcs={float(l_bcs):.3e}  "
                        f"L_res={float(l_res):.3e}")
                if self.weighting != "none":
                    line += ("  mean lam data/bcs/res="
                             + "/".join(f"{m:.2e}" for m in lam_means))

                v = None
                if val_batch is not None and it % val_every == 0:
                    v = float(self.val_ARE(self.params, val_batch))
                    self.val_ARE_log.append(v)
                    self.val_iter_log.append(it)

                    if v < self.best_val_ARE:
                        self.best_val_ARE  = v
                        self.best_val_iter = it
                        self.best_params   = self.params
                        flag = " *"
                    else:
                        flag = ""
                    line += f"  val_ARE={v:.3f}%{flag}"

                print(line)

                if callback is not None:
                    callback(it, float(l), float(l_data),
                             float(l_bcs), float(l_res), v)

        # Restore the parameters that achieved the lowest validation ARE
        if val_batch is not None:
            print(f"\nBest validation ARE = {self.best_val_ARE:.3f}% "
                  f"at iter {self.best_val_iter}; restoring those params.")
            self.params = self.best_params

    @partial(jit, static_argnums=(0,))
    def val_ARE(self, params, val_batch):
        """
        Validation average relative error (%). Used by optimization.py as the Optuna objective.
        """
        (Q, x), phi_true = val_batch

        def phi0_at(Q_i, x_j):
            psi_vec = vmap(
                lambda mu_k: self.operator_net(params, Q_i, x_j, mu_k)
            )(self.mu_GL)
            return np.dot(self.w_GL, psi_vec)

        phi_pred = vmap(phi0_at)(Q, x)
        return np.mean(np.abs((phi_true.flatten() - phi_pred) / phi_true.flatten())) * 100.0

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, Q_star, Y_star):
        psi_fn = vmap(self.operator_net, (None, 0, 0, 0))
        return psi_fn(params, Q_star, Y_star[:, 0], Y_star[:, 1])

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, Q_star, Y_star):
        return vmap(self.residual_net, (None, 0, 0, 0))(params, Q_star, Y_star[:, 0], Y_star[:, 1])

    @partial(jit, static_argnums=(0,))
    def predict_phi0(self, params, Q_batch, x_points):
        def phi0_at(Q_i, x_j):
            psi_vec = vmap(
                lambda mu_k: self.operator_net(params, Q_i, x_j, mu_k)
            )(self.mu_GL)
            return np.dot(self.w_GL, psi_vec)
        phi0_for_one_Q = vmap(phi0_at, in_axes=(None, 0))
        phi0_all = vmap(phi0_for_one_Q, in_axes=(0, None))
        return phi0_all(Q_batch, x_points)


def build_psi_data_arrays(ds):
    """
    Flatten an angular-flux dataset into per-sample supervision with a full angular flux target
    """
    Q     = np.asarray(ds['Q'])        # (N, J)
    psi   = np.asarray(ds['psi'])      # (N, A, J)
    x     = np.asarray(ds['x'])        # (J,)

    N, A, J = psi.shape

    # psi is (N, A, J); we want one A-vector per (sample, x), so reorder to (N, J, A) and flatten the (N, J) axes -> (N*J, A).
    psi_xa   = np.transpose(psi, (0, 2, 1))   # (N, J, A)
    psi_flat = psi_xa.reshape(N * J, A)       # (N*J, A)

    Q_flat = np.repeat(Q, J, axis=0)   # (N*J, J)
    x_flat = np.tile(x, N)             # (N*J,)

    return (Q_flat, x_flat), psi_flat


def build_psi_val_batch(ds):
    """
    Validation batch for the angular regime's best-params tracking.
    """
    return build_val_batch(ds)


class PI_DeepONet_Angular(PI_DeepONet):
    def angular_net(self, params, Q, x):
        """
        Full angular flux vector at (Q, x): psi_tilde(x, mu_1..mu_A).

        Returns shape (A,).
        """
        branch_params, trunk_params = params
        y = np.atleast_1d(x)                       # trunk input is x only, shape (1,)
        B = self.branch_apply(branch_params, (Q - self.Q_shift) / self.Q_scale)   # (p,)
        T = self.trunk_apply(trunk_params, y)      # (A*p,)
        p = B.shape[0]
        T = T.reshape(self.N_angles, p)            # (A, p)
        return T @ B                               # (A,)

    def operator_net(self, params, Q, x, mu):
        """
        Scalar psi_tilde(x, mu) for a single node angle mu, by selecting the matching channel from angular_net.
        """
        psi_vec = self.angular_net(params, Q, x)               # (A,)
        onehot  = (self.mu_GL == mu).astype(psi_vec.dtype)     # (A,)
        # Fallback to nearest node if mu is not exactly a node (robustness);
        # exact-match is the normal path and yields a true one-hot.
        onehot = lax.cond(
            np.sum(onehot) > 0,
            lambda _: onehot,
            lambda _: (np.argmin(np.abs(self.mu_GL - mu))
                       == np.arange(self.N_angles)).astype(psi_vec.dtype),
            operand=None,
        )
        return np.dot(onehot, psi_vec)

    def phi0_net(self, params, Q, x):
        """Scalar flux from one angular_net call instead of A operator_net calls."""
        return np.dot(self.w_GL, self.angular_net(params, Q, x))

    def data_net(self, params, Q, x):
        """
        Vector data loss: MSE over the full angular vector at each (Q, x).
        Each of the A components is its own term in the NTK weighting.
        """
        return self.angular_net(params, Q, x)                  # (A,)