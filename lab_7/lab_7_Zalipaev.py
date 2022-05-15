import numpy as np
from numpy import linspace
from random import randint
from scipy.integrate import dblquad, nquad
from math import sqrt

A = 4
B = 2
C = 3
R = 1
N = 100000
dotsNumber = 1000

xDots = linspace(-R, R, dotsNumber)
yDots = linspace(-R, R, dotsNumber)
z = lambda x, y: A*x**2 + B*y**2 + C
circ = lambda x, y: x**2 + y**2

# True integral
def bx():
  return [-1,1]
def by(x):
  return [-sqrt(1-x**2), sqrt(1-x**2)]
trueIntegral = nquad(z, [by, bx])[0]
print('True integral: ', trueIntegral)

# First method
count = 0
sum = 0
xTemp, yTemp, zTemp = [], [], []
for i in range(N):
  xTemp.append(xDots[randint(0,dotsNumber-1)])
  yTemp.append(yDots[randint(0,dotsNumber-1)])
  zTemp.append(z(xTemp[i], yTemp[i]))
  if circ(xTemp[i], yTemp[i]) <= 1:
    sum += z(xTemp[i], yTemp[i])
    count += 1
firstIntegral = 4*R/N*sum
print('First method: ', firstIntegral)

# Second method
zMax = max(zTemp)
zSpace = linspace(0, zMax, dotsNumber)
count = 0
for i in range(N):
  zTemp = zSpace[randint(0,dotsNumber-1)]
  if circ(xTemp[i], yTemp[i]) <= 1 and zTemp < z(xTemp[i], yTemp[i]):
    count += 1
secondIntegral = 4*R*zMax*count/N
print('Second method: ', secondIntegral)