import numpy as np
import sympy as smp
from sympy import *

A = Matrix([[8,3,2],[3,6,1],[2,1,7]])
n = np.shape(A)[0]
l = smp.symbols('l')
l_array = np.array(np.abs(solve(det(A - l * np.identity(n)), l)), dtype=float)
number_of_digits = 4
l_array = l_array.round(decimals=number_of_digits)
A = np.array(A, dtype=float)
I = np.identity(n)
y = np.ones((n, 1))
number_of_iterations = 10

def y_of_l(number_of_l):
    return np.vstack((y, l_array[number_of_l]))

def newton_eigenvalues(y_iterative):
    for k in range(number_of_iterations):
        x = y_iterative[0:n]
        l = y_iterative[n]
        J = np.vstack((np.concatenate((A - l * I, -x), axis=1),np.concatenate((2 * np.transpose(x)[0], [0]))))
        V = np.vstack((np.dot(A, x) - l * x, [np.sum(np.transpose(x)[0] ** 2) - 1]))
        y_iterative = y_iterative - np.dot(np.linalg.inv(J), V)
    return y_iterative

eival_array = np.zeros_like(l_array)
eivec_array = np.zeros((n,n))

for i in range(len(l_array)):
    eival_array[i] = newton_eigenvalues(y_of_l(i))[n][0]

for i in range(n):
    for j in range(n):
        eivec_array[i][j] = newton_eigenvalues(y_of_l(i))[j][0]
eivec_array = np.transpose(eivec_array)

print("Newton's method applied:")
print("Number of iterations in Newton's method =", str(number_of_iterations))
print('Eigen values:\n', eival_array)
print('Eigen vectors:\n', eivec_array, '\n')

# check the results
A = np.array(A, dtype=float)
w, v = np.linalg.eig(A)

print('Built-in function used:')
print('Eigen values:\n', str(w))
print('Eigen vectors:\n', str(v))

'''
    Below is something that does not work yet, but may work in the future 
'''


'''

A = Matrix([[8,3,2],[3,6,1],[2,1,7]])
n = np.shape(A)[0]
l = smp.symbols('l')
l_array = np.array(np.abs(solve(det(A - l * np.identity(n)), l)), dtype=float)
number_of_digits = 4
l_array = l_array.round(decimals=number_of_digits)
e = 0.0001
countSimpleIter = 0
countJacobi = 0
countZeidel = 0
countSOR = 0

# decompose A into A1, A2 and D
D = np.array(np.diag(np.diag(A)),dtype=float)
A1 = np.array(np.triu(A), dtype=float)
for i in range(n):
    for j in range(n):
        A1[i,i] = 0
A2 = np.array(np.tril(A), dtype=float)
for i in range(n):
    for j in range(n):
        A2[i,i] = 0

# simple-iteration method
A = np.array([[8,3,2],[3,6,1],[2,1,7]], dtype=float)
xiterative = (np.array([[1,1,1]])).reshape(n,1) # solution initial approximation
previous_step_size = 1
max_iters = 1000

def simple_iter_method(prev_step_size, xiter, counter, number_of_eigenvalue):
    while prev_step_size > e and countSimpleIter < max_iters:
        previous_x = xiter
        xiter = np.dot(np.linalg.inv(A), l_array[number_of_eigenvalue] * xiter)
        difference = np.array(previous_x - xiter, dtype=float)
        prev_step_size = np.linalg.norm(difference)
        norm = np.linalg.norm(xiter)
        normalized_vector = xiter / norm
        counter += 1
    return normalized_vector

#print(l_array[0], simple_iter_method(previous_step_size, xiterative, countSimpleIter, 0))
#print(l_array[0], simple_iter_method(previous_step_size, xiterative, countSimpleIter, 2))
#print(simple_iter_method(previous_step_size, xiterative, countSimpleIter, 0),simple_iter_method(previous_step_size, xiterative, countSimpleIter, 1),simple_iter_method(previous_step_size, xiterative, countSimpleIter, 2))

# Jacobi method
xiterative = (np.array([[1,1,1]])).reshape(n,1) # solution initial approximation
invD = np.linalg.inv(D)
previous_step_size = 1
max_iters = 1000

while previous_step_size > e and countJacobi < max_iters:
    previous_x = xiterative
    xiterative = - np.dot(invD,np.dot(A1,xiterative)) - np.dot(invD,np.dot(A2,xiterative)) + np.dot(invD,l_array[0] * xiterative)
    difference = np.array(previous_x - xiterative, dtype=float)
    previous_step_size = np.linalg.norm(difference)
    countJacobi += 1
xJacobi = xiterative

print('Absolute error in the Jacobi method:',previous_step_size)
print('Number of iterations in the Jacobi method:',countJacobi)
print('Solution obtained by the Jacobi method: '+'\n', xJacobi, '\n')

# Seidel method
xiterative = (np.array([[1,1,1]])).reshape(n,1) # solution initial approximation
invMatr = np.linalg.inv(D + A1)
previous_step_size = 1
max_iters = 1000

while previous_step_size > e and countZeidel < max_iters:
    previous_x = xiterative
    xiterative = - np.dot(invMatr,np.dot(A2,xiterative)) + np.dot(invMatr,l_array[0] * xiterative)
    difference = np.array(previous_x - xiterative, dtype=float)
    previous_step_size = np.linalg.norm(difference)
    countZeidel += 1
xZeidel = xiterative

print('Absolute error in the Seidel method:',previous_step_size)
print('Number of iterations in the Seidel method:',countZeidel)
print('Solution obtained by the Seidel method: '+'\n', xZeidel, '\n')

# SOR method
xiterative = (np.array([[1,1,1]])).reshape(n,1) # solution initial approximation
w = 1.01
invFactor = np.linalg.inv(np.identity(n) + np.dot(w*invD,A1))
previous_step_size = 1
max_iters = 1000

while previous_step_size > e and countSOR < max_iters:
    previous_x = xiterative
    xiterative = np.dot(invFactor,np.dot((1-w)*np.identity(n) - np.dot(w*invD,A2),xiterative)) + np.dot(invFactor,np.dot(w*invD,l_array[0] * xiterative))
    difference = np.array(previous_x - xiterative, dtype=float)
    previous_step_size = np.linalg.norm(difference)
    countSOR += 1
xSOR = xiterative

print('Parameter \u03C9 =',w)
print('Absolute error in the SOR method:',previous_step_size)
print('Number of iterations in the SOR method:',countSOR)
print('Solution obtained by the SOR method: '+'\n', xSOR)

'''
