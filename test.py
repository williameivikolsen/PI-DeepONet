import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

with open("./trained_models/sahadath.pkl", "rb") as f:
    ckpt = pickle.load(f)
plt.plot(np.linspace(1, 2000, len(ckpt["loss_log"])), ckpt["loss_log"], label="Sahadath")
# with open("./trained_models/data_only_16_angles.pkl", "rb") as f:
#     ckpt = pickle.load(f)
# plt.plot(ckpt["loss_log"], label="Data 16 angles")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()