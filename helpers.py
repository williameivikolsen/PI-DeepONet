import pickle

import jax.numpy as jnp
import numpy as onp


def load_model(path: str):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    cfg = dict(ckpt["config"])
    cfg["x_sensors"] = jnp.asarray(cfg["x_sensors"])

    is_PI = "N_angles" in cfg
    if is_PI:
        # Distinguish the vector-output angular model from the scalar PI model.
        model_type = cfg.pop("model_type", None)
        trunk_in = cfg["trunk_layers"][0] if "trunk_layers" in cfg else 2
        is_angular_vec = (model_type == "angular_vec") or \
                         (model_type is None and trunk_in == 1)

        if model_type == "angular_vec_scalar_data":
            # Same vector-output trunk as PI_DeepONet_Angular, but trained on scalar phi_0 labels
            from model_angular_scalar import PI_DeepONet_AngularScalar
            model = PI_DeepONet_AngularScalar(**cfg)
            kind = "PI_angular_scalar_data"
        elif is_angular_vec:
            from model import PI_DeepONet_Angular
            model = PI_DeepONet_Angular(**cfg)
            kind = "PI_angular"
        else:
            from model import PI_DeepONet
            model = PI_DeepONet(**cfg)
            kind = "PI"
    else:
        from nonPI_model import DeepONet
        cfg.pop("model_type", None)
        model = DeepONet(**cfg)
        kind = "nonPI"

    model.params   = ckpt["params"]
    model.loss_log = ckpt["loss_log"]
    if is_PI:
        for key in ("loss_data_log", "loss_bcs_log", "loss_res_log"):
            if key in ckpt:
                setattr(model, key, ckpt[key])
    return model, kind, ckpt


def r2_score(true: onp.ndarray, pred: onp.ndarray) -> float:
    ss_res = onp.sum((true - pred) ** 2)
    ss_tot = onp.sum((true - true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def are(true: onp.ndarray, pred: onp.ndarray) -> float:
    return float(onp.mean(onp.abs((true - pred) / true)) * 100.0)