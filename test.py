import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 20, 1000)
softplus = np.log(1 + np.exp(x))
dfdx = np.exp(x)*1/(1 + np.exp(x))
plt.plot(x, softplus, label="Softplus")
plt.plot(x, dfdx, label="Derivative")
plt.legend()
plt.show()