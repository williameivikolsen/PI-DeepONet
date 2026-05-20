import pickle
import matplotlib.pyplot as plt
import os

with open("./trained_models/n_samples.pkl", "rb") as f:
    ckpt = pickle.load(f)
plt.plot(ckpt["loss_log"], label="n_samples")
with open("./trained_models/PI_test.pkl", "rb") as f:
    ckpt = pickle.load(f)
plt.plot(ckpt["loss_log"], label="PI")

plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()