from scipy import integrate
from cmath import exp
import numpy as np
import matplotlib.pyplot as plt

# 2 <= n is the number of independent uniformly distributed variables
n = 10

j = complex(0.0, 1.0)  # imaginary unit
mu = 0.5 * n  # effective mu of the sum of independent uniformly distributed variables
sigma = ((1.0 / 12.0) * n)**0.5  # effective standard deviation of independent uniformly distributed variables

# Kernel for the inverse Fourier transform
def imag(p,x):
    return np.real((j**n / (2.0 * np.pi * p**n)) * (exp(-j * p) - 1.0)**n * exp(j * p * x))

x = np.linspace(0, n, num=100)  # x-points for the graphs
y1 = []
y2 = []
for i in range(0, 100):
    val1, error = integrate.quad(imag, -np.inf, np.inf, args=(x[i],), limit=5000, limlst=5000,
                              maxp1=5000, epsabs=1.5e-5, epsrel=1.5e-5)  # real part of the inverse Fourier transform
    # +/-np.inf are the infinite integration limits
    val2 = (1.0 / (sigma * (2.0 * np.pi)**0.5)) * np.real(exp(-0.5 * (x[i] - mu)**2 / sigma**2))
    y1.append(val1)
    y2.append(val2)

plt.plot(x, y1)  # distribution of the sum of independent uniformly distributed variables
plt.plot(x, y2)  # normal distribution
plt.grid()
plt.show()