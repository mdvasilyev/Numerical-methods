import numpy as np

''' 
        Crout LU-decomposition
'''

A = np.array([[3,-1,0,2],[-2,1,1,2],[0,-1,7,2],[-1,2,3,5]],dtype=float) # original matrix 

size = A.shape
n = int(size[0])
U = np.zeros((n,n))
x = np.zeros((n,1))
y = np.zeros((n,1))
f =(np.array([[2,3,-3,2]])).reshape(n,1)
U2 = np.hstack([U,y])
A1 = np.hstack([A,f])
L1 = np.zeros((n,n))

for j in range(n+1):
    U2[0,j] = A1[0,j]/A1[0,0]
for k in range(1,n):
    for i in range(k,n):
        for j in range(k,n+1):
            A1[i,j] = A1[i,j] - A1[i,k-1]*U2[k-1,j]
            U2[k,j] = A1[k,j]/A1[k,k]

U = U2[:,:n]
y = np.reshape((U2[:,n]),(n,1))

for i in range(n-1,-1,-1):
    if i == n-1:
        x[i] = y[i]
    else:
        sum = 0
        for j in range(n-1,i,-1):
            sum = sum + U[i,j]*x[j]
        x[i] = y[i] - sum

print('SLE solution by the Gauss method: '+'\n', x)

L = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        if j == 0:
            L[i,j] = A[i,j]
        if i >= j and j!=0:
            L[i,j] = A1[i,j]

LU = np.dot(L,U)
print('Original mathix A: \n', A,'\n Desired matrices:'+'\n L matrix: \n', L,'\n U matrix: \n', U,'\n LU matrix: \n', LU)

det_lu = np.linalg.det(LU)
det_a = np.linalg.det(A)

print('Check determinants')
print('Determinant of A:',det_a,'\nDeterminant of LU: ',det_lu)

e = 1e-6

if np.abs(det_a - det_lu) <= e:
    print('Determinant deviation is less than', str(e))
else:
    print('Everything is bad :(')