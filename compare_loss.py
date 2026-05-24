import pickle
import matplotlib.pyplot as plt
import os

for file in os.listdir("trained_models/"):
    with open("trained_models/" + file, "rb") as f:
        ckpt = pickle.load(f)
    if len(ckpt["loss_log"]) > 600:
        plt.plot(ckpt["loss_log"], label=file)

plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()