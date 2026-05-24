import pickle
with open("trained_models/optimal_model.pkl", "rb") as f:
    ckpt = pickle.load(f)
print("Keys in config:", list(ckpt["config"].keys()))
print("output_scale in config:", ckpt["config"].get("output_scale", "MISSING"))