import numpy as np
import sympy as sp

a, b, c, d, e, f = sp.symbols('a b c d e f')
x, y = sp.symbols('x y')

def target_function(x, y):
    return 0.5 * a * x ** 2 + b * x * y + 0.5 * c * y ** 2 - d * x - e * y - f
def diff_target_function(variable):
    return sp.diff(target_function(x, y), variable)

solution = sp.solve((diff_target_function(x), diff_target_function(y)), (x, y))

print('Analytical solution:')
print('Minimal x is equal to', str(solution[x]))
print('Minimal y is equal to', str(solution[y]))


number_of_digits = 4
xMin_value = solution[x].subs([(a, 1),(b, 0.1),(c, 3),(d, 2),(e, 4),(f, 0.5)]).evalf(number_of_digits)
yMin_value = solution[y].subs([(a, 1),(b, 0.1),(c, 3),(d, 2),(e, 4),(f, 0.5)]).evalf(number_of_digits)

print('Substituting numbers: a, b, c, d, e, f = 1, 0.1, 3, 2, 4, 0.5, we get the minimal point A[' + str(xMin_value) + ',' + str(yMin_value) + '].', '\n')

# gradient descent
a, b, c, d, e, f = 1, 0.1, 3, 2, 4, 0.5
cur_p = np.array([1, 1], dtype=float) # the algorithm starts at point = [1, 1]
rate = 0.25 # learning rate
precision = 0.0001 # this tells us when to stop the algorithm
previous_step_size = 1
max_iters = 10000 # maximum number of iterations
iters = 0 # iteration counter

def gradf(x_value, y_value):
    return np.array([diff_target_function(x).subs([(x, x_value),(y, y_value)]).evalf(number_of_digits), diff_target_function(y).subs([(x, x_value),(y, y_value)]).evalf(number_of_digits)]) #Gradient of our function

while previous_step_size > precision and iters < max_iters:
    prev_p = cur_p # store current p value in prev_p
    cur_p = cur_p - rate * gradf(prev_p[0], prev_p[1]) # grad descent
    difference = np.array(cur_p - prev_p, dtype=float)
    previous_step_size = np.linalg.norm(difference) # change in point
    iters += 1 # iteration count

cur_p = np.array(cur_p, float)

print('Gradient descent with precision =', str(precision) + ', learning rate =', str(rate) + ', and initial guess =', str([1, 1]), 'converges to A' + str(cur_p.round(decimals=3)), 'in', str(iters), 'iterations.\n')

# Newton method
cur_p = np.array([1, 1], dtype=float) # the algorithm starts at point = [1, 1]
precision = 0.0001
number_of_iterations = 1000

def newton_local_minimum(gradient_of_f, initial_p, epsilon, max_iter):
    pn = initial_p
    for n in range(0, max_iter):
        fpn = np.array([gradient_of_f[0].subs([(x, pn[0]),(y, pn[1])]), gradient_of_f[1].subs([(x, pn[0]),(y, pn[1])])], dtype=float)
        if np.linalg.norm(fpn) < epsilon:
            print("Newton's method converges in", n, 'iterations to')
            return pn.round(decimals=3)
        Dfpn = np.array([gradf(fpn[0], fpn[1])[0].subs([(x, pn[0]),(y, pn[1])]), gradf(fpn[0], fpn[1])[1].subs([(x, pn[0]),(y, pn[1])])], dtype=float)
        if any(Dfpn) == 0:
            print('Zero derivative. No solution found.')
            return None
        pn = pn - fpn / abs(Dfpn)
    print('Exceeded maximum iterations. No solution found.')
    return None

print(newton_local_minimum(gradf(x, y), cur_p, precision, number_of_iterations))