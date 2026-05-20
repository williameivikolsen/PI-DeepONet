import itertools
from functools import partial
import jax.numpy as np
from jax import random, grad, vmap, jit, lax
from jax import config
from jax.flatten_util import ravel_pytree
from jax.nn import relu
from numpy.polynomial.legendre import leggauss
import optax 
from torch.utils import data

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
                 rng_key=random.PRNGKey(1234)):
        # Initialization
        self.inputs     = inputs
        self.output     = output
        self.N          = output.shape[0]
        self.batch_size = batch_size
        self.key        = rng_key

    def __getitem__(self, index):
        # Generate one batch of data
        self.key, subkey = random.split(self.key)
        return self._batch(subkey)

    @partial(jit, static_argnums=(0,))
    def _batch(self, key):
        # Generates data containing batch_size samples
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        in_batch  = tuple(arr[idx] for arr in self.inputs)
        out_batch = self.output[idx]
        return in_batch, out_batch


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


def build_bcs_arrays(ds, X, n_per_sample=500,
                     rng_key=random.PRNGKey(2025)):
    """
    Random vacuum-BC evaluation points.

    Half the points are at x=0 with mu > 0 (incoming from left is zero),
    the other half at x=X with mu < 0 (incoming from right is zero).
    Target is zero for every point.

    mu is sampled with a small epsilon away from 0 to avoid the grazing
    direction.
    """
    Q    = np.asarray(ds['Q'])
    N, J = Q.shape
    total = N * n_per_sample
    half  = total // 2

    k1, k2 = random.split(rng_key)
    mu_left  = random.uniform(k1, (half,),          minval=1e-4, maxval=1.0)
    mu_right = random.uniform(k2, (total - half,),  minval=-1.0, maxval=-1e-4)

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

    Q_flat = Q[sample_idx]                  # (total, J)
    y      = np.stack([x_bc, mu_bc], axis=-1)   # (total, 2)
    s      = np.zeros((total,))
    return (Q_flat, y), s


def build_res_arrays(ds, X, n_per_sample=1000,
                     rng_key=random.PRNGKey(2026)):
    """
    Random interior collocation points in (0, X) x (-1, 1) for the PDE
    residual loss. Target is zero because residual_net already absorbs
    Q/2 via jnp.interp.
    """
    Q    = np.asarray(ds['Q'])
    N, J = Q.shape
    total = N * n_per_sample

    k1, k2 = random.split(rng_key)
    x_r  = random.uniform(k1, (total,), minval=0.0, maxval=X)
    mu_r = random.uniform(k2, (total,), minval=-1.0, maxval=1.0)

    sample_idx = np.repeat(np.arange(N), n_per_sample)
    Q_flat     = Q[sample_idx]              # (total, J)
    y          = np.stack([x_r, mu_r], axis=-1)
    s          = np.zeros((total,))
    return (Q_flat, y), s


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
        # l_bcs  = self.loss_bcs(params, bcs_batch)
        # l_res  = self.loss_res(params, res_batch)
        return (
            self.lambda_data * l_data
            # + self.lambda_bcs * l_bcs
            # + self.lambda_res * l_res
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
              nIter=10000, log_every=100, callback=None):
        data_iter = iter(data_dataset)
        bcs_iter  = iter(bcs_dataset)
        res_iter  = iter(res_dataset)

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

                print(
                    f"Iter {it:6d}: L={float(l):.3e}  "
                    f"L_data={float(l_data):.3e}  "
                    f"L_bcs={float(l_bcs):.3e}  "
                    f"L_res={float(l_res):.3e}"
                )

                if callback is not None:
                    callback(it, float(l), float(l_data),
                             float(l_bcs), float(l_res))

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