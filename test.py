import numpy as np

ds_np = np.load("datasets/M_Iso_train.npz")
ds    = {k: np.asarray(ds_np[k]) for k in ds_np.files}
print(ds_np)