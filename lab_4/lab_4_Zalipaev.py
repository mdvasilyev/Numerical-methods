import numpy as np
import sympy as smp
from sympy import *
'''
A = np.array([[8,3,2],[3,6,1],[2,1,7]]) # original matrix
n = np.shape(A)[0]
I = np.identity(n)
y = np.ones((n+1, 1))
number_of_iterations = 10


def newton_eigenvalues(y_iterative):
    for k in range(number_of_iterations):
        x = y_iterative[0:n]
        l = y_iterative[n]
        J = np.vstack((np.concatenate((A - l * I, -x), axis=1),np.concatenate((2 * np.transpose(x)[0], [0]))))
        V = np.vstack((np.dot(A, x) - l * x, [np.sum(np.transpose(x)[0] ** 2) - 1]))
        y_iterative = y_iterative - np.dot(np.linalg.inv(J), V)
    return y_iterative

y_1 = newton_eigenvalues(y)
x_1 = y_1[0:n]
l_1 = y_1[n]
y_2 = newton_eigenvalues(10 * y)
x_2 = y_2[0:n]
l_2 = y_2[n]
y_3 = newton_eigenvalues(0.1 * y)
x_3 = y_3[0:n]
l_3 = y_3[n]

print("Number of iterations in Newton's method =", str(number_of_iterations), '\n')

print('Approximated x_1: ', str(np.transpose(x_1)[0]))
print('Approximated l_1: ', str(l_1))
print('Approximated x_2: ', str(np.transpose(x_2)[0]))
print('Approximated l_2: ', str(l_2))
print('Approximated x_3: ', str(np.transpose(x_3)[0]))
print('Approximated l_3: ', str(l_3), '\n')

w, v = np.linalg.eig(A)
print('Eigen values:\n', str(w))
print('Eigen vectors:\n', str(v))


x1, x2, x3 = smp.symbols('x_1 x_2 x_3')
x = np.transpose([[x1,x2,x3]])
'''


A = Matrix([[8,3,2],[3,6,1],[2,1,7]])
n = np.shape(A)[0]
l = smp.symbols('l')
l_array = np.array(solve(det(A - l * np.identity(n)), l))
#print(l_array)

f = np.abs(l_array[0] * (np.array([[2,12,5]])).reshape(n,1)) # constant vector
e = 0.0001
countJacobi = 0
countZeidel = 0
countSOR = 0

# decompose A into A1, A2 and D
D = np.array(np.diag(np.diag(A)), dtype=float)
A1 = np.triu(A)
for i in range(n):
    for j in range(n):
        A1[i,i] = 0
A2 = np.tril(A)
for i in range(n):
    for j in range(n):
        A2[i,i] = 0

U = np.zeros((n,n))
x = np.zeros((n,1))
y = np.zeros((n,1))
U2 = np.hstack([U,y])
Af = np.hstack([A,f])
L1 = np.zeros((n,n))

# Gauss method
for j in range(n+1):
    U2[0,j] = Af[0,j]/Af[0,0]
for k in range(1,n):
    for i in range(k,n):
        for j in range(k,n+1):
            Af[i,j] = Af[i,j] - Af[i,k-1]*U2[k-1,j]
            U2[k,j] = Af[k,j]/Af[k,k]
U = U2[:,:n]
y = np.reshape((U2[:,n]),(n,1))

# backward substitution
for i in range(n-1,-1,-1):
    if i == n-1:
        x[i] = y[i]
    else:
        sum = 0
        for j in range(n-1,i,-1):
            sum = sum + U[i,j]*x[j]
        x[i] = y[i] - sum

print('Exact solution of SLE with Gauss method: '+'\n', x)
print("Let's solve the system using iterative methods with relative error \u03B5 =", e, '\n')

# Jacobi method
xiterative = (np.array([[1,1,1]], dtype=float)).reshape(n,1) # solution initial approximation
invD = np.linalg.inv(D)

while np.linalg.norm(x - xiterative)/np.linalg.norm(x) >= e:
    countJacobi += 1
    xiterative = - np.dot(invD,np.dot(A1,xiterative)) - np.dot(invD,np.dot(A2,xiterative)) + np.dot(invD,f)
xJacobi = xiterative

print('Relative error in the Jacobi method:',np.linalg.norm(x - xiterative)/np.linalg.norm(x))
print('The number of iterations in the Jacobi method:',countJacobi)
print('The solution obtained by the Jacobi method: '+'\n', xJacobi, '\n')

# Seidel method
xiterative = (np.array([[1,1,1]])).reshape(n,1) # solution initial approximation
invMatr = np.linalg.inv(D + A1)

while np.linalg.norm(x - xiterative)/np.linalg.norm(x) >= e:
    countZeidel += 1
    xiterative = - np.dot(invMatr,np.dot(A2,xiterative)) + np.dot(invMatr,f)
xZeidel = xiterative

print('Relative error in the Seidel method:',np.linalg.norm(x - xiterative)/np.linalg.norm(x))
print('The number of iterations in the Seidel method:',countZeidel)
print('The solution obtained by the Seidel method: '+'\n', xZeidel, '\n')

# SOR method
xiterative = (np.array([[1,1,1]])).reshape(n,1) # solution initial approximation
w = 1.01
invFactor = np.linalg.inv(np.identity(n) + np.dot(w*invD,A1))

while np.linalg.norm(x - xiterative)/np.linalg.norm(x) >= e:
    countSOR += 1
    xiterative = np.dot(invFactor,np.dot((1-w)*np.identity(n) - np.dot(w*invD,A2),xiterative)) + np.dot(invFactor,np.dot(w*invD,f))
xSOR = xiterative

print('Parameter \u03C9 =',w)
print('Relative error in the SOR method:',np.linalg.norm(x - xiterative)/np.linalg.norm(x))
print('The number of iterations in the SOR method:',countSOR)
print('The solution obtained by the SOR method: '+'\n', xSOR)