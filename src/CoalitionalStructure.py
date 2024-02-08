import numpy as np
from itertools import combinations

def CoalitionalStructure(number_Players):
    """
    It computes the matrix of all the possible subcoalitions obtainable with n players
    :param number_Players:
    :return: matrix number of combinations x n
    """
    print("Hello")
    # create a Matrix with rows as possible number of combinations
    n=0
    for k in range(1, number_Players+1):
        n += np.math.factorial(number_Players)/ (np.math.factorial(number_Players - k) * np.math.factorial(k))

    Matrix = np.zeros((int(n), number_Players))

    # compute the location for ones on the coefficient matrix
    v = np.arange(1, number_Players+1)

    for i in range(1, number_Players+1):
        H = np.array(list(combinations(v,i))) # find combinations of agents
        M = np.zeros((len(H), number_Players))
        for k in range(len(H)): # row
            for h in range(len(H[k])): # column
                M[k][h] = H[k][h]
        if i == 1:
            J = M
        else:
            J = np.vstack((J,M))

    # populate the coefficient matrix
    for i in range(int(n)): # row
        for j in range(number_Players):
            if J[i, j] > 0:
                Matrix[i][int(J[i,j])-1] = 1

    return Matrix
