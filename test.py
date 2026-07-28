import pickle
from pathlib import Path

TRAINED_MODELS_DIR = Path("trained_models")


def classify(cfg: dict) -> str:
    is_PI = "N_angles" in cfg
    if not is_PI:
        return "DeepONet"
    model_type = cfg.get("model_type")
    trunk_layers = cfg.get("trunk_layers")
    is_angular_vec = (model_type == "angular_vec") or (
        model_type is None and trunk_layers is not None and trunk_layers[0] == 1
    )
    return "PI-DeepONet (angular)" if is_angular_vec else "PI-DeepONet"


def main():
    paths = sorted(TRAINED_MODELS_DIR.rglob("*.pkl"))
    if not paths:
        print(f"No .pkl files found under {TRAINED_MODELS_DIR}/")
        return

    for path in paths:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        cfg = ckpt.get("config", {})

        kind = classify(cfg)
        activation = cfg.get("activation", "not recorded")
        branch_layers = cfg.get("branch_layers", "?")
        trunk_layers = cfg.get("trunk_layers", "?")
        n_per_sample = cfg.get("n_per_sample", "?")

        print(f"{path}")
        print(f"  kind          : {kind}")
        print(f"  activation    : {activation}")
        print(f"  branch_layers : {branch_layers}")
        print(f"  trunk_layers  : {trunk_layers}")
        print(f"  n_per_sample  : {n_per_sample}")
        if "N_angles" in cfg:
            print(f"  N_angles      : {cfg['N_angles']}")
        print()


if __name__ == "__main__":
    main()
