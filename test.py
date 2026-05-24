import numpy as np

size = "medium"

ds_np = np.load("datasets/" + size + "/M_Iso_train.npz")
ds    = {k: np.asarray(ds_np[k]) for k in ds_np.files}
n = len(ds["Q"])*len(ds["x"])
print(n)