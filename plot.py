import numpy as onp
import jax.numpy as jnp
import matplotlib.pyplot as plt

from helpers import load_model, plot_loss_curves, plot_sample_predictions


CHECKPOINT = "trained_models/large/pideeponet.pkl"
TEST_FILE  = "datasets/test/M_Iso_test_NS.npz"
SAMPLE_IDX = 2


model, kind, ckpt = load_model(CHECKPOINT)

fig1 = plot_loss_curves(model, kind=kind, log_every=ckpt["log_every"])

ds_np = onp.load(TEST_FILE)
ds    = {k: jnp.asarray(ds_np[k]) for k in ds_np.files
         if ds_np[k].dtype.kind in "fiu"}

fig2 = plot_sample_predictions(model, ds, sample_index=SAMPLE_IDX)

plt.show()
