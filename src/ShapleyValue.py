import numpy as np
from math import factorial
from scipy.special import comb
from src.CoalitionalStructure import CoalitionalStructure

def shapley(n, v_list):
    """
    This program determines the shapley value where

    :param n: is the number of players
    :param v: list of payoffs for the different coalitions, the order for the coalitions
    is the same as the combinations from the routine CoalitionStructure
    :return:
    """
    m = 2**n - 1 # number of possible subcoalitions. It is also the nrows in A

    v = np.array(v_list)
    A = CoalitionalStructure(n)
    ss = np.ones(n)
    A = np.vstack((A, ss))

    S = np.zeros(n)
    for k in range(n): # agent k
        k1 = np.zeros(m)
        k2 = np.zeros(m)
        A2 = np.zeros_like(A)
        for i in range(m):
            if A[i,k] > 0:
                k1[i] = v[i] # values from v that applies to agent k
                A2[i,:] = A[i,:] # subcoalitions that agent k participates in

        A3 = A2.copy()
        A3[:, k] = 0 # subcoalitions without agent k (S\{k})
        for i in range(m):
            for j in range(m):
                if np.array_equal(A3[i, :n], A[j, :n]):
                    k2[i] = v[j] # values from v that applies to S\{k}

        k3 = k1-k2 # calculates the term c(S with k) - c(S without k)
        k4 = np.sum(A2, axis=1) # the number of elements in S (without k)
        r = np.zeros(m)
        mm = factorial(n - 1)
        for i in range(m):
            if k4[i] > 0:
                r[i] = mm / comb(n-1, int(k4[i]-1))

        S[k] = np.sum(k3 * r) / np.sum(r)

    return S


