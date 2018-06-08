from __future__ import division
import numpy as np
import math

def kleinberg(offsets, s=2, gamma=1, n=None, T=None, k=None):
    """Kleinberg's algorithm (described in 'Bursty and Hierarchical Structure 
    in Streams'). The algorithm models activity bursts in a time series as an 
    infinite hidden Markov model.
    
    Input:
        offsets: a list of time offsets (numeric)
        s: the base of the exponential distribution that is used for modeling 
        the event frequencies
        gamma: coefficient for the transition costs between states
        n, T: to have a fixed cost function (not dependent of the given offsets). 
        Which is needed if you want to compare bursts for different inputs.
        k: maximum burst level"""

    if s <= 1:
        raise ValueError("s must be greater than 1!")
    if gamma <= 0:
        raise ValueError("gamma must be positive!")
    if not n is None and n <= 0:
        raise ValueError("n must be positive!")
    if not T is None and T <= 0:
        raise ValueError("T must be positive!")
    if len(offsets) < 1:
        raise ValueError("offsets must be non-empty!")

    offsets = np.array(offsets, dtype=object)
    
    if offsets.size == 1:
        bursts = np.array([0, offsets[0], offsets[0]], ndmin=2, dtype=object)
        return bursts

    offsets = np.sort(offsets)
    gaps = np.diff(offsets)

    if not np.all(gaps):
        raise ValueError("Input cannot contain events with zero time between!")

    if T is None:
        T = np.sum(gaps)

    if n is None:
        n = np.size(gaps)

    g_hat = T / n
    gamma_log_n = gamma * math.log(n)

    if k is None:
        k = int(math.ceil(float(1 + math.log(T, s) + math.log(1 / np.amin(gaps), s))))

    def tau(i, j):
        if i >= j:
            return 0
        else:
            return (j - i) * gamma_log_n
    
    alpha_function = np.vectorize(lambda x: s ** x / g_hat)
    alpha = alpha_function(np.arange(k))

    def f(j, x):
        return alpha[j] * math.exp(-alpha[j] * x)

    C = np.repeat(float("inf"), k)
    C[0] = 0

    q = np.empty((k, 0))
    for t in range(np.size(gaps)):
        C_prime = np.repeat(float("inf"), k)
        q_prime = np.empty((k, t+1))
        q_prime.fill(np.nan)

        for j in range(k):
            cost_function = np.vectorize(lambda x: C[x] + tau(x, j))
            cost = cost_function(np.arange(0, k))

            el = np.argmin(cost)

            if f(j, gaps[t]) > 0:
                C_prime[j] = cost[el] - math.log(f(j, gaps[t]))
            
            if t > 0:
                q_prime[j,:t] = q[el,:]

            q_prime[j, t] = j + 1

        C = C_prime
        q = q_prime

    j = np.argmin(C)
    q = q[j,:]

    prev_q = 0
    
    N = 0
    for t in range(np.size(gaps)):
        if q[t] > prev_q:
            N = N + q[t] - prev_q
        prev_q = q[t]

    bursts = np.array([np.repeat(np.nan, N), np.repeat(offsets[0],N),np.repeat(offsets[0], N)], ndmin=2, dtype=object).transpose()

    burst_counter = -1
    prev_q = 0
    stack = np.zeros(int(N), dtype=int)
    stack_counter = -1
    for t in range(np.size(gaps)):
        if q[t] > prev_q:
            num_levels_opened = q[t] - prev_q
            for i in range(int(num_levels_opened)):
                burst_counter += 1
                bursts[burst_counter, 0] = prev_q + i
                bursts[burst_counter, 1] = offsets[t]
                stack_counter += 1
                stack[stack_counter] = int(burst_counter)
        elif q[t] < prev_q:
            num_levels_closed = prev_q - q[t]
            for i in range(int(num_levels_closed)):
                bursts[int(stack[stack_counter]), 2] = offsets[t]
                stack_counter -= 1
        prev_q = q[t] 

    while stack_counter >= 0:
        bursts[int(stack[stack_counter]), 2] = offsets[np.size(gaps)]
        stack_counter -= 1

    return bursts
    
