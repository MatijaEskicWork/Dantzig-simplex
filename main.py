import numpy as np
import scipy.optimize as opt
import math

server1 = 1000
server2 = 600
server3 = 500
time = 2880

def checkConstraints(A, x, b):
    for i in range(b.size):
        tmp = 0
        for j in range(x.size):
            tmp += (x[j] * A[i][j])
        if (tmp > b[i]):
            return False
    return True

def evalFunction(X, Z):
    f = 0
    for i in range(X.size):
        f += X[i] * Z[i]
    return f

if __name__ == '__main__':
    Z = np.array([10, 8, 6, 9, 18, 20, 15, 17, 15, 16, 13, 17])
    b_ub = np.array([server1, server2, server3, time, time, time, time])
    A_ub = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                     [5, 0, 0, 0, 6, 0, 0, 0, 13, 0, 0, 0],
                     [0, 7, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0],
                     [0, 0, 4, 0, 0, 0, 8, 0, 0, 0, 9, 0],
                     [0, 0, 0, 10, 0, 0, 0, 15, 0, 0, 0, 17]])

    dantzig = opt.linprog(c=-Z, A_ub=A_ub, b_ub=b_ub, method='simplex', options={'disp': True})
    f = -1 * dantzig.fun
    values = dantzig.x
    print("Value of function before rounding: {}".format(f))
    print("Values of variables before rounding:")
    print(values)

    for i in range(values.size):
        values[i] = round(values[i])

    if (checkConstraints(A_ub, values, b_ub) == False):
        for i in range(values.size):
            values[i] = math.floor(values[i])

    f = evalFunction(values, Z)

    print("Value of function after rounding: {}".format(f))
    print("Values of variables after rounding:")
    print(values)