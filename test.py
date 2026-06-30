import pickle

path = "trained_models/large/pideeponet_angular.pkl"

with open(path, "rb") as f:
    ckpt = pickle.load(f)

ckpt["config"]["activation"] = "softplus"

with open(path, "wb") as f:
    pickle.dump(ckpt, f)

print(f"Added activation='softplus' to {path}")