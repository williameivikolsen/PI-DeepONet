import os
import pickle
import time

import numpy as onp
import jax.numpy as jnp
from jax import vmap


CHECKPOINT_PATH = "trained_models/weights.pkl"   # change to PI ckpt as needed
TEST_DIR        = "datasets/test"

SCENARIOS = [
    # GRF-based (200 samples each)
    ("M_Iso_test_NS.npz",     "NS  (no shift)",       "grf"),
    ("M_Iso_test_SS1.npz",    "SS_1 (small shift)",   "grf"),
    ("M_Iso_test_SS2.npz",    "SS_2 (small shift)",   "grf"),
    ("M_Iso_test_SS3.npz",    "SS_3 (small shift)",   "grf"),
    ("M_Iso_test_LS1.npz",    "LS_1 (large shift)",   "grf"),
    ("M_Iso_test_LS2.npz",    "LS_2 (large shift)",   "grf"),
    ("M_Iso_test_LS3.npz",    "LS_3 (large shift)",   "grf"),
    ("M_Iso_test_LS4.npz",    "LS_4 (large shift)",   "grf"),
    ("M_Iso_test_LS5.npz",    "LS_5 (large shift)",   "grf"),
    ("M_Iso_test_LS6.npz",    "LS_6 (large shift)",   "grf"),
    # Special sources (1 sample each)
    ("M_Iso_test_LC.npz",     "LC  (linear comb)",    "single"),
    ("M_Iso_test_NLC.npz",    "NLC (nonlinear comb)", "single"),
    ("M_Iso_test_SIN_LF.npz", "SIN_LF (f=0.2)",       "single"),
    ("M_Iso_test_SIN_HF.npz", "SIN_HF (f=0.8)",       "single"),
]

def load_model(path: str):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)

    cfg = dict(ckpt["config"])
    cfg["x_sensors"] = jnp.asarray(cfg["x_sensors"])

    is_PI = "N_angles" in cfg

    if is_PI:
        from model import PI_DeepONet
        model = PI_DeepONet(**cfg)
        kind = "PI"
    else:
        from nonPI_model import DeepONet
        model = DeepONet(**cfg)
        kind = "nonPI"

    model.params   = ckpt["params"]
    model.loss_log = ckpt["loss_log"]

    return model, kind

def r2_score(true: onp.ndarray, pred: onp.ndarray) -> float:
    ss_res = onp.sum((true - pred) ** 2)
    ss_tot = onp.sum((true - true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def are(true: onp.ndarray, pred: onp.ndarray) -> float:
    return float(onp.mean(onp.abs((true - pred) / true)) * 100.0)

def predict_phi0_batch_nonPI(model, Q_batch: onp.ndarray,
                             x: onp.ndarray) -> onp.ndarray:
    """Non-PI: predict_s returns phi_0 directly, un-normalized."""
    n, J = Q_batch.shape
    Q_jax = jnp.asarray(Q_batch)
    x_jax = jnp.asarray(x)

    out = onp.empty((n, J))
    for i in range(n):
        Q_rep = jnp.broadcast_to(Q_jax[i], (J, J))
        out[i] = onp.asarray(model.predict_s(model.params, Q_rep, x_jax))
    return out


def predict_phi0_batch_PI(model, Q_batch: onp.ndarray,
                          x: onp.ndarray) -> onp.ndarray:
    """
    PI: operator_net predicts NORMALIZED psi(x, mu); reconstruct phi_0 by
    Gauss-Legendre quadrature over mu, then multiply by output_scale to
    return raw phi_0 in physical units.
    """
    n, J = Q_batch.shape
    Q_jax = jnp.asarray(Q_batch)
    x_jax = jnp.asarray(x)

    def phi0_at(Q_i, x_j):
        psi_vec = vmap(
            lambda mu_k: model.operator_net(model.params, Q_i, x_j, mu_k)
        )(model.mu_GL)
        return jnp.dot(model.w_GL, psi_vec)

    out = onp.empty((n, J))
    for i in range(n):
        phi0 = vmap(lambda x_j: phi0_at(Q_jax[i], x_j))(x_jax)
        out[i] = onp.asarray(model.output_scale * phi0)
    return out


PREDICTORS = {
    "nonPI": predict_phi0_batch_nonPI,
    "PI":    predict_phi0_batch_PI,
}


# ---------------------------------------------------------------------------
# Per-scenario evaluation
# ---------------------------------------------------------------------------
def evaluate_scenario(model, kind: str, filepath: str, label: str,
                      scenario_kind: str) -> dict:
    ds = onp.load(filepath, allow_pickle=True)
    Q     = ds["Q"]
    phi_0 = ds["phi_0"]
    x     = ds["x"]

    predictor = PREDICTORS[kind]

    t0 = time.time()
    phi_0_pred = predictor(model, Q, x)
    t_pred = time.time() - t0

    r2_vals  = onp.array([r2_score(phi_0[i], phi_0_pred[i]) for i in range(len(Q))])
    are_vals = onp.array([are(phi_0[i],      phi_0_pred[i]) for i in range(len(Q))])

    return {
        "label":     label,
        "kind":      scenario_kind,
        "n":         len(Q),
        "R2_mean":   float(onp.mean(r2_vals)),
        "R2_std":    float(onp.std(r2_vals)),
        "ARE_mean":  float(onp.mean(are_vals)),
        "ARE_std":   float(onp.std(are_vals)),
        "t_predict": t_pred,
    }

model, kind = load_model(CHECKPOINT_PATH)

results = []
for filename, label, scenario_kind in SCENARIOS:
    path = os.path.join(TEST_DIR, filename)
    if not os.path.exists(path):
        continue
    res = evaluate_scenario(model, kind, path, label, scenario_kind)
    results.append(res)

print("\n" + "=" * 78)
print(f"Model type: {kind}")
print(f"{'Scenario':<24s} {'n':>4s}  {'R^2 (mean +/- std)':>25s}  {'ARE % (mean +/- std)':>22s}")
print("-" * 78)
for r in results:
    if r["kind"] == "grf":
        r2_s  = f"{r['R2_mean']:.5f} +/- {r['R2_std']:.5f}"
        are_s = f"{r['ARE_mean']:.3f} +/- {r['ARE_std']:.3f}"
    else:
        r2_s  = f"{r['R2_mean']:.5f}"
        are_s = f"{r['ARE_mean']:.3f}"
    print(f"{r['label']:<24s} {r['n']:>4d}  {r2_s:>25s}  {are_s:>22s}")
print("=" * 78)