import matplotlib.pyplot as plt
import numpy as np
def normal(x,sigma, mu):
 return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))
def fermi(x, c, mu):
 return 1/(np.exp(c*(x- mu) + 1))
x = np.linspace(0, 1, 100)

a = [normal(i, 0.1, 0.5) for i in x]
b = [fermi(i, 3, 1) for i in x]
plt.plot(x, a)
plt.plot(x, b)
plt.grid(visible=True)
plt.show()
# plt.savefig("distro.svg", format="svg", dpi=300)