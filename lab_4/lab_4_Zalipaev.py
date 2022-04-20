import numpy as np
import sympy as smp
from sympy import *

# here we find eigen values
A = Matrix([[16, 3, 2], [3, 5, 1], [2, 1, 10]])
n = np.shape(A)[0]
l = smp.symbols('l')
l_array = np.array(np.abs(solve(det(A - l * np.identity(n)), l)), dtype=float)
number_of_digits = 4
l_array = l_array.round(decimals=number_of_digits)
A = np.array(A, dtype=float)
I = np.identity(n)
y = np.ones((n, 1))
number_of_iterations = 10

# concatenate initial guess and eigen value into one vector 
def y_of_l(number_of_l):
    return np.vstack((y, l_array[number_of_l]))

# find eigen vectors with Newton's method
## P.S. this also works if eigen values are not known but replaced with initial guess as well as eigen vectors
def newton_eigenval_and_vec_problem(y_iterative):
    for k in range(number_of_iterations):
        x = y_iterative[0:n]
        l = y_iterative[n]
        J = np.vstack((np.concatenate((A - l * I, -x), axis=1), np.concatenate((2 * np.transpose(x)[0], [0]))))
        V = np.vstack((np.dot(A, x) - l * x, [np.sum(np.transpose(x)[0] ** 2) - 1]))
        y_iterative = y_iterative - np.dot(np.linalg.inv(J), V)
    return y_iterative

eival_array = np.zeros_like(l_array)
eivec_array = np.zeros((n, n))

for i in range(len(l_array)):
    eival_array[i] = newton_eigenval_and_vec_problem(y_of_l(i))[n][0]

for i in range(n):
    for j in range(n):
        eivec_array[i][j] = newton_eigenval_and_vec_problem(y_of_l(i))[j][0]
eivec_array = np.transpose(eivec_array)

print("Newton's method applied:")
print("Number of iterations in Newton's method =", str(number_of_iterations))
print('Eigen values:\n', eival_array)
print('Eigen vectors:\n', eivec_array, '\n')

'''
    Below one can find Jacobi, Zeidel and SOR methods for finding eigen vectors
'''

A = Matrix([[16, 3, 2], [3, 5, 1], [2, 1, 10]])
n = np.shape(A)[0]
l = smp.symbols('l')
l_array = np.array(np.abs(solve(det(A - l * np.identity(n)), l)), dtype=float)
number_of_digits = 4
l_array = l_array.round(decimals=number_of_digits)
e = 0.001
count_simple_iter = 0

# decompose A into A1, A2 and D
D = np.array(np.diag(np.diag(A)), dtype=float)
mutated_D = np.zeros((n, n, n)) # contains D - l[i] * I
for i in range(n):
    mutated_D[i] = D
    mutated_D[i] = mutated_D[i] - l_array[i] * np.identity(n)
A1 = np.array(np.triu(A), dtype=float)
for i in range(n):
    for j in range(n):
        A1[i, i] = 0
A2 = np.array(np.tril(A), dtype=float)
for i in range(n):
    for j in range(n):
        A2[i, i] = 0

# Jacobi method
count_Jacobi = 0
x_iterative = (np.array([[1, 1, 1]])).reshape(n, 1) # solution initial approximation
inverse_D = np.zeros_like(mutated_D)
for i in range(n):
    inverse_D[i] = np.linalg.inv(mutated_D[i])
previous_step_size = 1
max_iters = 50

# Jacobi method messes up one component of the first eigen value -> need to check formulas
def Jacobi_method(prev_step_size, x_iter, counter, number_of_eigenvalue):
    while prev_step_size > e and counter < max_iters:
        previous_x = x_iter
        x_iter = - np.dot(inverse_D[number_of_eigenvalue], np.dot(A1, x_iter)) - np.dot(inverse_D[number_of_eigenvalue], np.dot(A2, x_iter))
        difference = np.array(previous_x - x_iter, dtype=float)
        prev_step_size = np.linalg.norm(difference)
        norm = np.linalg.norm(x_iter)
        normalized_vector = x_iter / norm
        counter += 1
    return normalized_vector, prev_step_size, counter

Jacobi_solution_array = np.zeros((n, n))
Jacobi_pr_st_s_array = np.zeros(n)
Jacobi_counter_array = np.zeros(n)

print('Applying Jacobi method:')
for i in range(n):
    Jacobi_solution_array[:, [i]], Jacobi_pr_st_s_array[i], Jacobi_counter_array[i] = Jacobi_method(previous_step_size, x_iterative, count_Jacobi, i)
    print('For eigen value', str(l_array[i]), 'with absolute error', str(Jacobi_pr_st_s_array[i]), 'Jacobi method converges in', str(Jacobi_counter_array[i]), 'iterations to:\n', Jacobi_solution_array[:, [i]], '\n')

# Seidel method
count_Zeidel = 0
x_iterative = (np.array([[1, 1, 1]])).reshape(n, 1) # solution initial approximation
inverse_matr = np.zeros_like(mutated_D)
for i in range(n):
    inverse_matr[i] = np.linalg.inv(mutated_D[i] + A1)
previous_step_size = 1
max_iters = 1000

def Zeidel_method(prev_step_size, x_iter, counter, number_of_eigenvalue):
    while prev_step_size > e and counter < max_iters:
        previous_x = x_iter
        x_iter = - np.dot(inverse_matr[number_of_eigenvalue], np.dot(A2, x_iter))
        difference = np.array(previous_x - x_iter, dtype=float)
        prev_step_size = np.linalg.norm(difference)
        norm = np.linalg.norm(x_iter)
        normalized_vector = x_iter / norm
        counter += 1
    return normalized_vector, prev_step_size, counter

Zeidel_solution_array = np.zeros((n, n))
Zeidel_pr_st_s_array = np.zeros(n)
Zeidel_counter_array = np.zeros(n)

print('Applying Zeidel method:')
for i in range(n):
    Zeidel_solution_array[:, [i]], Zeidel_pr_st_s_array[i], Zeidel_counter_array[i] = Zeidel_method(previous_step_size, x_iterative, count_Zeidel, i)
    print('For eigen value', str(l_array[i]), 'with absolute error', str(Zeidel_pr_st_s_array[i]), 'Zeidel method converges in', str(Zeidel_counter_array[i]), 'iterations to:\n', Zeidel_solution_array[:, [i]], '\n')


# SOR method
count_SOR = 0
x_iterative = (np.array([[1, 1, 1]])).reshape(n, 1) # solution initial approximation
w = 1.01
inverse_factor = np.zeros_like(mutated_D)
for i in range(n):
    inverse_factor[i] = np.linalg.inv(np.identity(n) + np.dot(w*inverse_D[i], A1))
previous_step_size = 1
max_iters = 1000

def SOR_method(prev_step_size, x_iter, counter, number_of_eigenvalue):
    while prev_step_size > e and counter < max_iters:
        previous_x = x_iter
        x_iter = np.dot(inverse_factor[number_of_eigenvalue], np.dot((1-w)*np.identity(n) - np.dot(w*inverse_D[number_of_eigenvalue], A2), x_iter))
        difference = np.array(previous_x - x_iter, dtype=float)
        prev_step_size = np.linalg.norm(difference)
        norm = np.linalg.norm(x_iter)
        normalized_vector = x_iter / norm
        counter += 1
    return normalized_vector, prev_step_size, counter

SOR_solution_array = np.zeros((n, n))
SOR_pr_st_s_array = np.zeros(n)
SOR_counter_array = np.zeros(n)

print('Applying SOR method:')
for i in range(n):
    SOR_solution_array[:, [i]], SOR_pr_st_s_array[i], SOR_counter_array[i] = SOR_method(previous_step_size, x_iterative, count_SOR, i)
    print('For eigen value', str(l_array[i]), 'with absolute error', str(SOR_pr_st_s_array[i]), 'and \u03C9 =', str(w), 'SOR method converges in', str(SOR_counter_array[i]), 'iterations to:\n', SOR_solution_array[:, [i]], '\n')

# check the results
A = np.array(A, dtype=float)
w, v = np.linalg.eig(A)

print('Built-in function used:')
print('Eigen values:\n', str(w))
print('Eigen vectors:\n', str(v))
