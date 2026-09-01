import itertools
from functools import partial
import jax.numpy as np
from jax import random, grad, vmap, jit, lax
from jax import config
from jax.flatten_util import ravel_pytree
from jax.nn import relu, tanh, gelu, softplus, sigmoid, elu, swish

ACTIVATIONS = {"relu": relu, "tanh": tanh, "gelu": gelu, "softplus": softplus,
               "sigmoid": sigmoid, "elu": elu, "swish": swish, "silu": swish}
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


class DeepONet:
    def __init__(self, branch_layers, trunk_layers,
                 Sigma_t, Sigma_s0, Sigma_s1,
                 x_sensors, X, Q_shift, Q_scale,
                 lambda_data=1.0, lambda_res=1.0, lambda_bcs=1.0,
                 branch_activation=relu,
                 trunk_activation=relu,
                 lr_init=1e-3,
                 lr_decay_rate=0.9,
                 lr_transition_steps=5000,
                 lr_schedule=None,
                 seed=None):
        # Network initialization and evaluation functions
        branch_activation, self.branch_activation_name = resolve_activation(branch_activation)
        trunk_activation,  self.trunk_activation_name  = resolve_activation(trunk_activation)
        self.branch_activation = branch_activation
        self.trunk_activation  = trunk_activation
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=branch_activation)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=trunk_activation)

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

        # Branch-input transform: (Q - Q_shift) / Q_scale, constants taken from
        # the training set. Applied only where Q enters the branch; callers pass
        # Q in raw physical units.
        self.Q_shift = float(Q_shift)
        self.Q_scale = float(Q_scale)

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

    # DeepONet architecture
    def operator_net(self, params, Q, x):
        branch_params, trunk_params = params
        y = np.atleast_1d(x)          # scalar → shape (1,)
        B = self.branch_apply(branch_params, (Q - self.Q_shift) / self.Q_scale)
        T = self.trunk_apply(trunk_params, y)
        return np.sum(B * T)

    # Supervised data loss on the scalar flux phi_0.
    def loss(self, params, batch):
        inputs, outputs = batch
        Q, x = inputs
        phi_0_pred = vmap(self.operator_net, (None, 0, 0))(params, Q, x)
        return np.mean((outputs.flatten() - phi_0_pred) ** 2)

    # Update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, params, opt_state, data_batch):
        grads = grad(self.loss)(params, data_batch)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    # Optimize parameters in a loop
    def train(self, data_dataset,
              nIter=10000, log_every=100, callback=None,
              val_batch=None, val_every=None):
        data_iter = iter(data_dataset)

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

            self.params, self.opt_state = self.step(
                next(self.itercount), self.params, self.opt_state,
                data_batch
            )

            if it % log_every == 0:
                l      = self.loss(self.params, data_batch)

                self.loss_log.append(float(l))

                inputs_b, outputs_b = data_batch
                Q_b, x_b = inputs_b
                pred_b = vmap(self.operator_net, (None, 0, 0))(self.params, Q_b, x_b)
                are = float(np.mean(np.abs((outputs_b.flatten() - pred_b) / outputs_b.flatten())) * 100)

                line = f"Iter {it:6d}: L={float(l):.3e}  ARE={are:.3f}%"

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
                    callback(it, float(l), v)

        # Restore the parameters that achieved the lowest validation ARE
        if val_batch is not None:
            print(f"\nBest validation ARE = {self.best_val_ARE:.3f}% "
                  f"at iter {self.best_val_iter}; restoring those params.")
            self.params = self.best_params

    @partial(jit, static_argnums=(0,))
    def val_ARE(self, params, val_batch):
        """
        Validation average relative error (%). Used by optimization.py as the
        Optuna objective.
        """
        (Q, x), phi_true = val_batch
        phi_pred = vmap(self.operator_net, (None, 0, 0))(params, Q, x)
        return np.mean(np.abs((phi_true.flatten() - phi_pred) / phi_true.flatten())) * 100.0

    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, Q_star, x_star):
        phi_fn = vmap(self.operator_net, (None, 0, 0))
        return phi_fn(params, Q_star, x_star)

    @partial(jit, static_argnums=(0,))
    def predict_phi0(self, params, Q_batch, x_points):
        f_for_one_Q = vmap(
            lambda Q_i, x_j: self.operator_net(params, Q_i, x_j),
            in_axes=(None, 0),
        )
        phi0_all = vmap(f_for_one_Q, in_axes=(0, None))
        return phi0_all(Q_batch, x_points)