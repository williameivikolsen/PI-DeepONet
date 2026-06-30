import itertools
from functools import partial
import jax.numpy as np
from jax import random, grad, vmap, jit, lax
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
    "swish": swish,
    "silu": swish,   # alias
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

class DataGenerator(data.Dataset):
    def __init__(self, inputs, output, batch_size=1024,
                 rng_key=random.PRNGKey(1234), branch_table=None):
        # Initialization
        self.inputs     = inputs
        self.output     = output
        self.N          = output.shape[0]
        self.batch_size = batch_size
        self.key        = rng_key
        self.branch_table = branch_table

    def __getitem__(self, index):
        # Generate one batch of data
        self.key, subkey = random.split(self.key)
        return self._batch(subkey)

    @partial(jit, static_argnums=(0,))
    def _batch(self, key):
        # Generates data containing batch_size samples
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        if self.branch_table is None:
            in_batch = tuple(arr[idx] for arr in self.inputs)
        else:
            # inputs[0] is an index array; gather the actual branch rows from
            # the unique table. Remaining inputs are sliced normally.
            branch_idx = self.inputs[0][idx]               # (batch,)
            branch     = self.branch_table[branch_idx]     # (batch, J)
            rest       = tuple(arr[idx] for arr in self.inputs[1:])
            in_batch   = (branch,) + rest
        out_batch = self.output[idx]
        return in_batch, out_batch


def build_val_batch(ds, output_scale: float):
    """
    Build a single validation (inputs, outputs) tuple.
    """
    Q     = np.asarray(ds['Q'])            # (N, J)
    phi_0 = np.asarray(ds['phi_0'])        # (N, J)
    x     = np.asarray(ds['x'])            # (J,)
    N, J  = Q.shape
    Q_flat   = np.repeat(Q, J, axis=0)
    x_flat   = np.tile(x, N)
    phi_flat = (phi_0 / output_scale).reshape(-1)
    return (Q_flat, x_flat), phi_flat


def build_data_arrays(ds, normalize=True):
    """
    Flat arrays for the supervised phi_0 loss.

    inputs = (Q_flat, x_flat)
        Q_flat : (N*J, J)    branch inputs, one row per (sample, x-point)
        x_flat : (N*J,)      scalar spatial coord
    outputs = phi_flat       : (N*J,)  scalar flux targets (normalized by
                                       training-set mean if normalize=True)

    When normalize=True, ALL fluxes and the source Q are rescaled by
    phi_scale = mean(phi_0). This is exact: the steady-state transport
    equation is linear, so substituting psi = phi_scale * psi_tilde
    (and same for phi_0, phi_1) leaves the equation identical except Q
    is replaced by Q/phi_scale. The network therefore solves the same
    physical problem in dimensionless units, with all fields O(1).

    Returns
    -------
    inputs    : tuple (Q_flat, x_flat) — Q in RAW units; the model
                divides by phi_scale internally in residual_net.
    outputs   : phi_flat (normalized).
    phi_scale : float — mean of raw phi_0. Pass as output_scale to the
                model so loss_data, residual_net, and predict_s all
                stay consistent.
    """
    Q     = np.asarray(ds['Q'])            # (N, J)
    phi_0 = np.asarray(ds['phi_0'])        # (N, J)
    x     = np.asarray(ds['x'])            # (J,)
    N, J  = Q.shape
    Q_flat   = np.repeat(Q, J, axis=0)     # (N*J, J)
    x_flat   = np.tile(x, N)               # (N*J,)
    phi_flat = phi_0.reshape(-1)           # (N*J,)

    if normalize:
        phi_scale = float(np.mean(phi_flat))
        phi_flat  = phi_flat / phi_scale
        # Q stays in raw units — residual_net divides by phi_scale internally.
    else:
        phi_scale = 1.0

    return (Q_flat, x_flat), phi_flat, phi_scale


def build_bcs_arrays(ds, X, n_per_sample=50,
                     rng_key=random.PRNGKey(2025), N_angles=16):
    """
    Vacuum-BC evaluation points, with mu drawn from the Gauss-Legendre
    quadrature nodes.

    Half the points are at x=0 with mu > 0 (incoming from left is zero),
    the other half at x=X with mu < 0 (incoming from right is zero).
    Target is zero for every point.
    """
    Q    = np.asarray(ds['Q'])
    N, J = Q.shape
    total = N * n_per_sample
    half  = total // 2

    mu_nodes, _ = leggauss(N_angles)
    mu_nodes = np.asarray(mu_nodes)
    pos_nodes = mu_nodes[mu_nodes > 0.0]      # left-boundary angles
    neg_nodes = mu_nodes[mu_nodes < 0.0]      # right-boundary angles

    k1, k2 = random.split(rng_key)
    mu_left  = random.choice(k1, pos_nodes, (half,))
    mu_right = random.choice(k2, neg_nodes, (total - half,))

    x_left   = np.zeros((half,))
    x_right  = np.full((total - half,), X)

    x_bc  = np.concatenate([x_left,  x_right])
    mu_bc = np.concatenate([mu_left, mu_right])
    sample_idx = np.repeat(np.arange(N), n_per_sample)

    # Shuffle so that left/right halves are interleaved with sample indices.
    perm = random.permutation(random.PRNGKey(7), total)
    sample_idx = sample_idx[perm]
    x_bc       = x_bc[perm]
    mu_bc      = mu_bc[perm]

    # Memory note: we do NOT materialize Q[sample_idx] (which would be
    # (total, J) and explodes as N*n_per_sample*J — 100 MB+ at the large
    # dataset). Instead we return the integer sample index and the unique
    # source table Q (N, J); DataGenerator gathers the per-batch branch
    # rows on the fly. The returned inputs[0] is the INDEX, and the table
    # is the third return value, to be passed to DataGenerator(branch_table=...).
    y = np.stack([x_bc, mu_bc], axis=-1)   # (total, 2)
    s = np.zeros((total,))
    return (sample_idx, y), s, Q


def build_res_arrays(ds, X, n_per_sample=100,
                     rng_key=random.PRNGKey(2026), N_angles=16):
    """
    Interior collocation points for the PDE residual loss: x continuous in
    (0, X), mu drawn from the Gauss-Legendre nodes. Target is zero because
    residual_net already absorbs Q/2 via jnp.interp.

    mu is restricted to the GL nodes (not continuous on (-1, 1)) for the
    same reason as build_bcs_arrays: the vector-output angular model only
    defines psi on the nodes. This costs no physics fidelity — the transport
    residual's only derivative is in x, and its scattering source is a node
    quadrature sum, so the residual is fully determined by the node angles.
    x stays continuous because the residual DOES differentiate in x.
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
    # See build_bcs_arrays: return the index + unique Q table rather than
    # materializing Q[sample_idx], so memory stays O(N*J) not O(total*J).
    y = np.stack([x_r, mu_r], axis=-1)
    s = np.zeros((total,))
    return (sample_idx, y), s, Q


class PI_DeepONet:
    def __init__(self, branch_layers, trunk_layers, N_angles,
                 Sigma_t, Sigma_s0, Sigma_s1,
                 x_sensors, X,
                 lambda_data=1.0, lambda_res=1.0, lambda_bcs=1.0,
                 activation=relu,
                 lr_init=1e-3,
                 lr_decay_rate=0.9,
                 lr_transition_steps=2000,
                 output_scale=1.0,
                 seed=None):
        activation, self.activation_name = resolve_activation(activation)
        self.activation = activation

        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=activation)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=activation)
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

        # Flux normalization constant: phi_scale = mean(phi_0) on training set.
        # The network learns the normalized fields psi_tilde = psi/phi_scale,
        # phi_0_tilde = phi_0/phi_scale, phi_1_tilde = phi_1/phi_scale.
        # The transport equation in these variables is identical to the
        # original except Q is replaced by Q/phi_scale (linearity).
        # - loss_data: targets are pre-normalized in build_data_arrays.
        # - loss_bcs:  zero targets, zero predictions; unaffected.
        # - residual_net: divides interpolated Q by output_scale.
        # - predict_s:  multiplies network output by output_scale to
        #               return raw psi.
        self.output_scale = float(output_scale)

        # Loss-term weights
        self.lambda_data = float(lambda_data)
        self.lambda_res  = float(lambda_res)
        self.lambda_bcs  = float(lambda_bcs)

        # 1. Define schedule
        self.lr_schedule = optax.exponential_decay(
            init_value=lr_init,
            transition_steps=lr_transition_steps,
            decay_rate=lr_decay_rate,
        )
        # 2. Define optimizer
        self.optimizer = optax.adam(learning_rate=self.lr_schedule)
        
        # 3. Initialize optimizer state
        self.opt_state = self.optimizer.init(self.params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(self.params)

        self.itercount = itertools.count()

        # Loggers
        self.loss_log       = []
        self.loss_data_log  = []
        self.loss_bcs_log   = []
        self.loss_res_log   = []

    # Define DeepONet architecture
    def operator_net(self, params, Q, x, mu):
        branch_params, trunk_params = params
        y = np.stack([x, mu])
        B = self.branch_apply(branch_params, Q)
        T = self.trunk_apply(trunk_params, y)
        outputs = np.sum(B * T)
        return outputs

    # Define PDE residual at a single evaluation point (x, mu)
    def residual_net(self, params, Q, x, mu):
        """
        1D transport residual at a single evaluation point.

            R(x, mu) = mu * dpsi/dx + Sigma_t * psi
                       - 0.5 * (Sigma_s0 * phi_0 + 3 * mu * Sigma_s1 * phi_1)
                       - Q(x) / 2

        Q is the branch vector of length J evaluated on self.x_sensors;
        Q(x) at x is obtained via piecewise-linear interpolation.
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
        # Divide by output_scale: in the normalized PDE that the network
        # learns, the source term is Q/phi_scale (linearity of transport).
        Q_x = np.interp(x, self.x_sensors, Q) / self.output_scale

        res = (
            mu * psi_x
            + self.Sigma_t * psi_at_mu
            - 0.5 * (self.Sigma_s0 * phi_0 + 3.0 * mu * self.Sigma_s1 * phi_1)
            - 0.5 * Q_x
        )
        return res

    # Boundary loss
    def loss_bcs(self, params, batch):
        inputs, outputs = batch
        Q, y = inputs
        phi_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, Q, y[:, 0], y[:, 1])
        loss = np.mean((outputs.flatten() - phi_pred) ** 2)
        return loss

    # Residual loss
    def loss_res(self, params, batch):
        inputs, outputs = batch
        Q, y = inputs
        pred = vmap(self.residual_net, (None, 0, 0, 0))(params, Q, y[:, 0], y[:, 1])
        loss = np.mean((outputs.flatten() - pred) ** 2)
        return loss

    # Supervised data loss on the scalar flux phi_0.
    def loss_data(self, params, batch):
        """
        phi_0_pred at (Q_i, x_j) is computed by Gauss-Legendre quadrature
        over the scalar operator_net.
        """
        inputs, outputs = batch
        Q, x = inputs

        def phi0_at(Q_i, x_j):
            psi_vec = vmap(
                lambda mu_k: self.operator_net(params, Q_i, x_j, mu_k)
            )(self.mu_GL)
            return np.dot(self.w_GL, psi_vec)

        phi_0_pred = vmap(phi0_at)(Q, x)
        return np.mean((outputs.flatten() - phi_0_pred) ** 2)

    # Define total loss
    def loss(self, params, data_batch, bcs_batch, res_batch):
        l_data = self.loss_data(params, data_batch)
        l_bcs  = self.loss_bcs(params, bcs_batch)
        l_res  = self.loss_res(params, res_batch)
        return (
            self.lambda_data * l_data
            + self.lambda_bcs * l_bcs
            + self.lambda_res * l_res
        )

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, params, opt_state, data_batch, bcs_batch, res_batch):
        grads = grad(self.loss)(params, data_batch, bcs_batch, res_batch)
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
                l      = self.loss(self.params, data_batch, bcs_batch, res_batch)
                l_data = self.loss_data(self.params, data_batch)
                l_bcs  = self.loss_bcs(self.params, bcs_batch)
                l_res  = self.loss_res(self.params, res_batch)

                self.loss_log.append(float(l))
                self.loss_data_log.append(float(l_data))
                self.loss_bcs_log.append(float(l_bcs))
                self.loss_res_log.append(float(l_res))

                line = (f"Iter {it:6d}: L={float(l):.3e}  "
                        f"L_data={float(l_data):.3e}  "
                        f"L_bcs={float(l_bcs):.3e}  "
                        f"L_res={float(l_res):.3e}")

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
                             float(l_bcs), float(l_res))

        # Restore the parameters that achieved the lowest validation ARE
        if val_batch is not None:
            print(f"\nBest validation ARE = {self.best_val_ARE:.3f}% "
                  f"at iter {self.best_val_iter}; restoring those params.")
            self.params = self.best_params

    @partial(jit, static_argnums=(0,))
    def val_ARE(self, params, val_batch):
        """
        Validation average relative error (%). Used by optimization.py as
        the Optuna objective.
        """
        (Q, x), phi_norm = val_batch

        def phi0_at(Q_i, x_j):
            psi_vec = vmap(
                lambda mu_k: self.operator_net(params, Q_i, x_j, mu_k)
            )(self.mu_GL)
            return np.dot(self.w_GL, psi_vec)

        phi_pred = vmap(phi0_at)(Q, x)
        return np.mean(np.abs((phi_norm.flatten() - phi_pred) / phi_norm.flatten())) * 100.0

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, Q_star, Y_star):
        # Network outputs psi_tilde = psi/output_scale; multiply back to
        # return raw psi in original units.
        psi_norm = vmap(self.operator_net, (None, 0, 0, 0))(params, Q_star, Y_star[:, 0], Y_star[:, 1])
        return self.output_scale * psi_norm

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, Q_star, Y_star):
        # residual_net returns the residual of the normalized PDE.
        # The raw-PDE residual is output_scale times that (linearity).
        r_pred = vmap(self.residual_net, (None, 0, 0, 0))(params, Q_star, Y_star[:, 0], Y_star[:, 1])
        return self.output_scale * r_pred

    @partial(jit, static_argnums=(0,))
    def predict_phi0(self, params, Q_batch, x_points):
        def phi0_at(Q_i, x_j):
            psi_vec = vmap(
                lambda mu_k: self.operator_net(params, Q_i, x_j, mu_k)
            )(self.mu_GL)
            return np.dot(self.w_GL, psi_vec)
        phi0_for_one_Q = vmap(phi0_at, in_axes=(None, 0))
        return self.output_scale * vmap(phi0_for_one_Q, in_axes=(0, None))(Q_batch, x_points)


def build_psi_data_arrays(ds, normalize=True):
    """
    Flatten an angular-flux dataset into per-(sample, x) supervision with a
    full angular vector target.

    inputs = (Q_flat, x_flat)
        Q_flat : (N*J, J)   branch input, one row per (sample, x-point)
        x_flat : (N*J,)     scalar spatial coordinate
    outputs = psi_flat : (N*J, A)   angular-flux targets, psi / phi_scale
    where A = N_angles, J = #cells, N = #sources.

    Returns
    -------
    inputs, outputs, phi_scale
    """
    Q     = np.asarray(ds['Q'])        # (N, J)
    psi   = np.asarray(ds['psi'])      # (N, A, J)
    phi_0 = np.asarray(ds['phi_0'])    # (N, J)
    x     = np.asarray(ds['x'])        # (J,)

    N, A, J = psi.shape

    if normalize:
        phi_scale = float(np.mean(phi_0))
    else:
        phi_scale = 1.0

    # psi is (N, A, J); we want one A-vector per (sample, x), so reorder to
    # (N, J, A) and flatten the (N, J) axes -> (N*J, A).
    psi_xa   = np.transpose(psi, (0, 2, 1))            # (N, J, A)
    psi_flat = (psi_xa.reshape(N * J, A) / phi_scale)  # (N*J, A)

    Q_flat = np.repeat(Q, J, axis=0)   # (N*J, J)
    x_flat = np.tile(x, N)             # (N*J,)

    return (Q_flat, x_flat), psi_flat, phi_scale


def build_psi_val_batch(ds, output_scale: float):
    """
    Validation batch for the angular regime's best-params tracking.

    val_ARE in PI_DeepONet measures phi_0 ARE (angle-integrated), which is
    the quantity we ultimately care about and keeps the early-stopping
    criterion identical across all regimes. So the val batch is built in
    the phi_0 form expected by PI_DeepONet.val_ARE: ((Q_flat, x_flat),
    phi_norm). This is identical to build_val_batch; kept as a separate
    name so the angular training script reads self-documentingly.
    """
    return build_val_batch(ds, output_scale=output_scale)


class PI_DeepONet_Angular(PI_DeepONet):
    def angular_net(self, params, Q, x):
        """
        Full angular flux vector at (Q, x): psi_tilde(x, mu_1..mu_A).

        Returns shape (A,).
        """
        branch_params, trunk_params = params
        y = np.atleast_1d(x)                       # trunk input is x only, shape (1,)
        B = self.branch_apply(branch_params, Q)    # (p,)
        T = self.trunk_apply(trunk_params, y)      # (A*p,)
        p = B.shape[0]
        T = T.reshape(self.N_angles, p)            # (A, p)
        return T @ B                               # (A,)

    def operator_net(self, params, Q, x, mu):
        """
        Scalar psi_tilde(x, mu) for a single node angle mu, by selecting the
        matching channel from angular_net.

        mu is expected to be one of the GL nodes (the data, BC and residual
        builders all sample on the nodes). Selection uses a one-hot dot so
        the result stays differentiable in x (grad argnums=2 in residual_net
        flows through angular_net; the channel mask does not depend on x).
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

    def loss_data(self, params, batch):
        """
        Vector data loss: MSE over the full angular vector at each (Q, x).

            batch = ((Q, x), psi_target)
            Q : (B, J)    x : (B,)    psi_target : (B, A)
            prediction = angular_net(params, Q, x)   -> (B, A)
        """
        inputs, outputs = batch
        Q, x = inputs
        psi_pred = vmap(self.angular_net, (None, 0, 0))(params, Q, x)  # (B, A)
        return np.mean((outputs - psi_pred) ** 2)