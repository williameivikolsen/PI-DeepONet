import pickle
import numpy as onp
import matplotlib.pyplot as plt
from helpers import load_model

relu_tanh_path = "trained_models/lr_search/large/pideeponet_angular_relu_tanh_nodata.pkl"
benchmark = "trained_models/training_testing/large/benchmark.pkl"


relu_tanh, _, _ = load_model(relu_tanh_path)
benchmark, _, _ = load_model(benchmark)
N = 100
x = onp.linspace(0, 10, N)
Q = onp.zeros(N)
mask = (x > 2.5) & (x < 7.5)
Q[mask] = 50
# Q = -(x-5)**2 + 25
relu_tanh_pred = onp.asarray(relu_tanh.predict_phi0(relu_tanh.params, Q[None, :], x)[0])   # (1, J) batch -> (N,)
benchmark_pred = onp.asarray(benchmark.predict_phi0(benchmark.params, Q[None, :], x)[0])   # (1, J) batch -> (N,)
plt.plot(x, Q, lw=1.2, alpha=0.6, label="Source $Q(x)$")
plt.fill_between(x, 0, Q, alpha=0.10)
plt.plot(x, relu_tanh_pred, label=r"Relu-Tanh $\phi(x)$")
plt.plot(x, benchmark_pred, label=r"Benchmark $\phi(x)$")
plt.legend()
plt.show()