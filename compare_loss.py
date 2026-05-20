import pickle
import matplotlib.pyplot as plt
import os

with open("./trained_models/n_samples.pkl", "rb") as f:
    ckpt = pickle.load(f)
plt.plot(ckpt["loss_log"], label="n_samples")
with open("./trained_models/PI_test.pkl", "rb") as f:
    ckpt = pickle.load(f)
plt.plot(ckpt["loss_log"], label="PI")
with open("./trained_models/architecture.pkl", "rb") as f:
    ckpt = pickle.load(f)
plt.plot(ckpt["loss_log"], label="First architecture")
with open("./trained_models/another_architecture.pkl", "rb") as f:
    ckpt = pickle.load(f)
plt.plot(ckpt["loss_log"], label="Another architecture")

plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()