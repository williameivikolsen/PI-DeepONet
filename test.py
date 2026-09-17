import pickle
import numpy as onp
import matplotlib.pyplot as plt

CHECKPOINT = "trained_models/lr_search/large/pideeponet_angular_relu_tanh_nodata.pkl"

ckpt = pickle.load(open(CHECKPOINT, "rb"))
cfg = ckpt["config"]
iters = onp.arange(len(ckpt["loss_log"])) * ckpt["log_every"]

hp = ckpt.get("hyperparameters")
if hp is None:
    raise SystemExit(f"{CHECKPOINT} records no hyperparameters (it predates the architecture "
                     f"search): set lr_schedule and the three loss weights by hand to continue it.")
lr_schedule = hp["lr"]
lr_config   = ckpt["lr_config"]
lambda_data = 1.0
lambda_res  = float(hp["res_over_data"])
lambda_bcs  = float(hp["bcs_over_data"])

plt.figure(figsize=(8, 5))
plt.plot(iters, ckpt["loss_log"], lw=2.0, color="black", label="total")
plt.plot(iters, ckpt["loss_data_log"], lw=1.4, label="data")
plt.plot(iters, ckpt["loss_bcs_log"], lw=1.4, label="BC")
plt.plot(iters, ckpt["loss_res_log"], lw=1.4, label="residual")

plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title(f"branch {cfg['branch_activation']} / trunk {cfg['trunk_activation']}"
          f"   ({ckpt['lr_config']}, best val ARE {ckpt['best_val_ARE']:.3f}%)")
plt.legend()
plt.tight_layout()
plt.grid("--")
# plt.savefig("results/relu_tanh_loss.pdf", bbox_inches="tight")
plt.show()
