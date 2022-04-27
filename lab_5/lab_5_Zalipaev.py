import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

filename = 'lab_5\epsdata.txt'
data = np.loadtxt(filename, delimiter=' ', dtype=float)
time_array = np.transpose(data)[0]
E_tr = np.transpose(data)[1]
t_min = time_array[0]
t_max = time_array[-1]
c = 3 # e8
z = 1 # e-3
f_0 = 1 # e11

def inverse_F(freq, tp, freq_0):
    return (2 * np.sqrt(np.pi) * tp * np.exp(- 2 * np.pi * (freq - freq_0) ** 2 * tp ** 2)) ** -1

def exponent(time, freq):
    return np.exp(1j * 2 * np.pi * freq * (time - z / c))

def T(freq, tp, freq_0, time):
    return inverse_F(freq, tp, freq_0) * integrate.quad(exponent(time, freq), t_min, t_max, args=(time))

#plt.plot(np.linspace(0, 2, 1000), inverse_F(np.linspace(0, 2, 1000), 1, 1))

plt.plot(time_array, E_tr)
plt.show()