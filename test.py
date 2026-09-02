import pickle
import numpy as onp
import matplotlib.pyplot as plt

CHECKPOINT = "trained_models/lr_search/relu_gelu/pideeponet_angular.pkl"

ckpt = pickle.load(open(CHECKPOINT, "rb"))
cfg = ckpt["config"]
iters = onp.arange(len(ckpt["loss_log"])) * ckpt["log_every"]

plt.figure(figsize=(8, 5))
plt.plot(iters, ckpt["loss_log"], lw=2.0, color="black", label="total")
# plt.plot(iters, ckpt["loss_data_log"], lw=1.4, label="data")
# plt.plot(iters, ckpt["loss_bcs_log"], lw=1.4, label="BC")
# plt.plot(iters, ckpt["loss_res_log"], lw=1.4, label="residual")

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
