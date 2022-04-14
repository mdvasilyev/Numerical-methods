import math
import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt

# define parameters of the function
A1, A2 = 1, 0.3
w1, w2 = 1, 4
L = 1
N = 100
n_max = 30

def function(x):
    return A1 * np.cos(w1 * x) + A2 * np.sin(w2 * x)

x_array = np.linspace(-L, L, N)
function_values = function(x_array)

def function_for_an(x, n):
    return function(x) * np.cos(np.pi * n * x)

def function_for_bn(x, n):
    return function(x) * np.sin(np.pi * n * x)

n_array = np.arange(n_max)
integrals = [[n, quad(function_for_an, -L, L, args=(n))[0], quad(function_for_bn, -L, L, args=(n))[0]] for n in n_array]
An_array = np.array(integrals).T[1]
Bn_array = np.array(integrals).T[2]

def S_N(x):
    for i in range(n_max):
        if i == 0:
            sum = 0
            sum = sum + An_array[i] / 2
        else:
            sum = sum + An_array[i] * np.cos(np.pi * i * x) + Bn_array[i] * np.sin(np.pi * i * x)
    return sum

class Chebyshev:
    """
    Chebyshev(a, b, n, func)
    Given a function func, lower and upper limits of the interval [a,b],
    and maximum degree n, this class computes a Chebyshev approximation
    of the function.
    Method eval(x) yields the approximated function value.
    """
    def __init__(self, a, b, n, func):
        self.a = a
        self.b = b
        self.func = func

        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)
        f = [func(math.cos(math.pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
        fac = 2.0 / n
        self.c = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / n)
                  for k in range(n)]) for j in range(n)]

    def eval(self, x):
        a,b = self.a, self.b
        assert(a <= x <= b)
        y = (2.0 * x - a - b) * (1.0 / (b - a))
        y2 = 2.0 * y
        (d, dd) = (self.c[-1], 0)             # Special case first step for efficiency
        for cj in self.c[-2:0:-1]:            # Clenshaw's recurrence
            (d, dd) = (y2 * d - dd + cj, d)
        return y * d - dd + 0.5 * self.c[0]   # Last step is different

n_cheb_max = 30 # define n_cheb_max
ch = Chebyshev(-L, L, n_cheb_max, function)
Chebyshev_array = np.zeros_like(x_array)

for i in range(len(x_array)):
    Chebyshev_array[i] = (ch.eval(x_array[i]))

plt.plot(x_array, function_values)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Original curve')

fig, ax = plt.subplots()
ax.plot(x_array, S_N(x_array), label='Trigonometric fitting curve')
ax.plot(x_array, Chebyshev_array, label='Chebyshev fitting curve')
legend = ax.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Approximation curves\nN_trig = ' + str(n_max) + '\nN_Cheb = ' + str(n_cheb_max))
plt.show()

trig_difference = np.abs(np.subtract(function_values, S_N(x_array)))
cheb_difference = np.abs(np.subtract(function_values, Chebyshev_array))

fig, ax = plt.subplots()
ax.plot(x_array, trig_difference)
legend = ax.legend()
plt.xlabel('x')
plt.ylabel('|f(x) - S_N|')
plt.title('Deviation of trig. pol., n_trig = ' + str(n_max))
plt.show()

plt.plot(x_array, cheb_difference)
plt.xlabel('x')
plt.ylabel('|f(x) - S_N|')
plt.title('Deviation of Cheb. pol., n_Cheb = ' + str(n_cheb_max))
plt.show()