import optuna
import jax.numpy as jnp
import numpy as onp
from jax import random
from model import PI_DeepONet, DataGenerator, build_data_arrays, build_bcs_arrays, build_res_arrays

ds_np = onp.load("datasets/M_Iso_train.npz")
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files}
X_slab = 10.0
J = int(ds['x'].shape[0])

def objective(trial):
    # lambda_res    = trial.suggest_float("lambda_res", 1e-3, 10.0, log=True)
    # lambda_bcs    = trial.suggest_float("lambda_bcs", 1e-3, 10.0, log=True)
    branch_width  = trial.suggest_categorical("branch_width", [50, 100, 250])
    trunk_width   = trial.suggest_categorical("trunk_width", [100, 250, 500])
    # trunk_width   = 100
    n_per_sample  = trial.suggest_categorical("n_per_sample", [500, 750, 1000])
    n_layers      = trial.suggest_int("n_layers", 3, 6)
    n_iter_trial = 50000
    lr_transition_steps = trial.suggest_categorical("lr_transition_steps", [n_iter_trial//20, n_iter_trial//10, n_iter_trial//5])

    branch_layers = [J] + [branch_width] * n_layers + [100]
    trunk_layers  = [2] + [trunk_width]  * n_layers + [100]

    data_in, data_out, phi_scale = build_data_arrays(ds, normalize=True)
    bcs_in, bcs_out = build_bcs_arrays(ds, X=X_slab, n_per_sample=n_per_sample)
    res_in, res_out = build_res_arrays(ds, X=X_slab, n_per_sample=n_per_sample)

    data_dataset = DataGenerator(data_in, data_out, batch_size=1000, rng_key=random.PRNGKey(101))
    bcs_dataset  = DataGenerator(bcs_in,  bcs_out,  batch_size=1000, rng_key=random.PRNGKey(202))
    res_dataset  = DataGenerator(res_in,  res_out,  batch_size=1000, rng_key=random.PRNGKey(303))

    model = PI_DeepONet(
        branch_layers, trunk_layers,
        N_angles=16,
        Sigma_t=1.0, Sigma_s0=0.5, Sigma_s1=0.0,
        x_sensors=ds['x'], X=X_slab,
        lambda_data=0.1, lambda_res=0.7, lambda_bcs=0.2,
        output_scale=phi_scale,
        lr_transition_steps=lr_transition_steps
    )
    model.train(data_dataset, bcs_dataset, res_dataset,
                nIter=n_iter_trial, log_every=500)

    return float(model.loss_log[-1]) 

study = optuna.create_study(
    storage="sqlite:///pi_deeponet.db",
    study_name="pi_deeponet",
    direction="minimize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=30)