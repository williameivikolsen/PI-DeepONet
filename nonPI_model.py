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

    Returns
    -------
    inputs    : tuple (Q_flat, x_flat)
    outputs   : phi_flat (possibly normalized)
    phi_scale : float — the mean of raw phi_0 used for normalization
                (==1.0 if normalize=False). Pass this to DeepONet so
                predict_s returns un-normalized predictions.
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
    else:
        phi_scale = 1.0

    return (Q_flat, x_flat), phi_flat, phi_scale

class DeepONet:
    def __init__(self, branch_layers, trunk_layers,
                 Sigma_t, Sigma_s0, Sigma_s1,
                 x_sensors, X,
                 lambda_data=1.0, lambda_res=1.0, lambda_bcs=1.0,
                 activation=None,
                 lr_init=1e-3,
                 lr_decay_rate=0.9,
                 lr_transition_steps=10000,
                 output_scale=1.0,
                 seed=None):
        # Network initialization and evaluation functions
        if activation is None:
            activation = np.tanh
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=activation)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=activation)

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

        # Spatial sensor grid — needed to interpolate Q(x) at arbitrary
        # collocation points via jnp.interp inside residual_net.
        self.x_sensors = np.asarray(x_sensors)   # shape (J,)
        self.X         = float(X)                # slab length

        # Output normalization constant.
        # Network learns phi_0 / output_scale; predict_s multiplies back.
        # Loss is computed on normalized values (targets already divided).
        self.output_scale = float(output_scale)

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

    # Define DeepONet architecture
    def operator_net(self, params, Q, x):
        branch_params, trunk_params = params
        y = np.atleast_1d(x)          # scalar → shape (1,)
        B = self.branch_apply(branch_params, Q)
        T = self.trunk_apply(trunk_params, y)
        return np.sum(B * T)

    # Supervised data loss on the scalar flux phi_0.
    def loss(self, params, batch):
        inputs, outputs = batch
        Q, x = inputs
        phi_0_pred = vmap(self.operator_net, (None, 0, 0))(params, Q, x)
        return np.mean((outputs.flatten() - phi_0_pred) ** 2)

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, params, opt_state, data_batch):
        grads = grad(self.loss)(params, data_batch)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # Optimize parameters in a loop
    def train(self, data_dataset,
              nIter=10000, log_every=100, callback=None):
        data_iter = iter(data_dataset)

        for it in range(nIter):
            data_batch = next(data_iter)

            self.params, self.opt_state = self.step(
                next(self.itercount), self.params, self.opt_state,
                data_batch
            )

            if it % log_every == 0:
                l      = self.loss(self.params, data_batch)

                self.loss_log.append(float(l))

                print(
                    f"Iter {it:6d}: L={float(l):.3e}  "
                )

                if callback is not None:
                    callback(it, float(l))

    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, Q_star, x_star):
        # Network is trained on phi_0 / output_scale, so multiply back
        # to return predictions in the original (un-normalized) units.
        phi_norm = vmap(self.operator_net, (None, 0, 0))(params, Q_star, x_star)
        return self.output_scale * phi_norm

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, Q_star, Y_star):
        r_pred = vmap(self.residual_net, (None, 0, 0, 0))(params, Q_star, Y_star[:, 0], Y_star[:, 1])
        return r_pred