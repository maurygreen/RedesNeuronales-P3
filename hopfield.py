import numpy as np
import argparse
import math

from tqdm import tqdm
from time import time

def initialize(p, N, seed):
    """
    Initialize the matrices XI and W

    Parameters
    ----------
    p    : int, amount of memories to store
    N    : int, amount of neurons to use in the network
    seed : int, random seed used to initialize 
    """
    print("Initializing XI and W")
    np.random.seed(seed)
    # This initializes the XI. By the numpy documentation, these values are
    # sampled from a "discrete uniform" distribution, in concordance with the
    # pseudo code given.
    xi = np.random.randint(0, high=2, size=(N, p)) * 2 - 1

    w = np.zeros((N,N))

    for i in range(1,N):
        for j in range(i-1):
            # Compute XI[i,m] * XI[j,m] all at once 
            row_prod = xi[i,:] * xi[j,:]
            w[i,j] = row_prod.sum()
            w[j,i] = w[i,j]
    print("Done!")

    return xi, w

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=500,
                        help='Amount of neurons to use')
    parser.add_argument("-seed", type=int, default=0,
                        help='Random seed to use')
    parser.add_argument("-pinit", type=int, default=10,
                        help='Initial amount of stored memories')
    parser.add_argument("-pmax", type=int, default=100,
                        help='Maximum amount of stored memories')
    parser.add_argument("-step", type=int, default=10,
                        help='Step with which the amount of stored memories changes')

    args  = parser.parse_args()

    # 
    N     = args.n
    SEED  = args.seed
    STEP  = args.step
    PINIT = args.pinit
    PMAX  = args.pmax

    # Needed to store the results
    results = []

    for p in [PINIT + j*STEP for j in range(math.floor((PMAX-PINIT)/STEP)+1)]:
    
        # Time the amount of time that took each iteration wrt p
        start = time()
        xi, w = initialize(p, N, SEED)

        m_mean = 0

        for mu in tqdm(range(p), desc="p={}".format(p)):
            # As numpy arrays are managed by reference, make a copy to save
            # the original values of the memories
            s     = np.copy(xi[:,mu])
            cond = True

            while cond:
                # Make a copy of s, as s is going to be changed and numpy
                # array are managed by reference
                s_aux = np.copy(s)
                for i in range(N):
                    # This is equivalent to do:
                    # for j in range(N):
                    #     h += w[i:j] * s[j]
                    h = np.dot(w[i,:], s)
                    # h > 0 is either 1 or 0. Make a linear transform and get 1
                    # or -1
                    s[i] = 2 * int(h > 0) - 1 

                # This is equivalent to do:
                # d = 0
                # for n in range(N):
                #    if s[j] != s_aux[j]:
                #        d += 1
                # if d == 0:
                #    cond = True
                cond = not np.all(s == s_aux)

            # Compute the superposition 
            m = np.dot(xi[:,mu], s.T)

            m_mean += np.abs(m)/N

        m_mean = m_mean / float(p)

        results.append((p/N,m_mean))
        print("Elapsed: {}".format(time() - start))

    results = np.array(results)
    # Save the results 
    np.savetxt('hopfield_sim-{}.txt'.format(N), results) 

