import numpy as np

A = np.array([[10,3,0],[3,15,1],[0,1,7]],dtype=float) # original matrix
size = A.shape
n = int(size[0])
f =(np.array([[2,12,5]])).reshape(n,1) # constant vector
e = 0.0001
countJacobi = 0
countZeidel = 0
countSOR = 0

# decompose A into A1, A2 and D
D = np.diag(np.diag(A))
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
xiterative = (np.array([[1,1,1]])).reshape(n,1) # solution initial approximation
invD = np.linalg.inv(D)

while np.linalg.norm(x - xiterative)/np.linalg.norm(x) >= e:
    countJacobi += 1
    xiterative = - np.dot(invD,np.dot(A1,xiterative)) - np.dot(invD,np.dot(A2,xiterative)) + np.dot(invD,f)
xJacobi = xiterative

print('Relative error in the Jacobi method:',np.linalg.norm(x - xiterative)/np.linalg.norm(x))
print('Number of iterations in the Jacobi method:',countJacobi)
print('Solution obtained by the Jacobi method: '+'\n', xJacobi, '\n')

# Seidel method
xiterative = (np.array([[1,1,1]])).reshape(n,1) # solution initial approximation
invMatr = np.linalg.inv(D + A1)

while np.linalg.norm(x - xiterative)/np.linalg.norm(x) >= e:
    countZeidel += 1
    xiterative = - np.dot(invMatr,np.dot(A2,xiterative)) + np.dot(invMatr,f)
xZeidel = xiterative

print('Relative error in the Seidel method:',np.linalg.norm(x - xiterative)/np.linalg.norm(x))
print('Number of iterations in the Seidel method:',countZeidel)
print('Solution obtained by the Seidel method: '+'\n', xZeidel, '\n')

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
print('Number of iterations in the SOR method:',countSOR)
print('Solution obtained by the SOR method: '+'\n', xSOR)