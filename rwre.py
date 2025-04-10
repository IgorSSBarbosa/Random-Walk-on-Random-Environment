import random
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.stats import bernoulli


'''Simple Symmetric Exclusion Process (SSEP)'''

# The random environment is a SSEP

def space_env(L , mu=0.5):
    total_space = np.arange(L)
    space = np.random.choice(total_space, int(L*mu), replace=False)  # choose L/2 of the bins to contain a particle
    return(np.sort(space))                                           # return the index of the particles sorted


def move_particle(space,s,L,mu=0.5):
    #Moves s particles to a neighboring site if that new position is empty
    particles_index = np.arange(len(space))
    particles_to_move = np.random.choice(particles_index,s)

    # Pick a random direction  for each step (left or right)
    direction = np.random.choice([-1, 1], s)
    
    # this is a torus, if the particle at L-1 jumps to right it goes to 0, and vice-versa
    for idx_atual, dir_particle in zip(particles_to_move, direction):
        # Check if the new position is empty (no particle present)
        # Since we are in Z, the order of partiles is never changed, 
        # therefore it's enough to check if new_postition is equal to position of next particle.
        if ((space[idx_atual] + dir_particle)% L  ) != ((space[(idx_atual + dir_particle)%(int(L/2))])%L):
            # Move the particle
            space[idx_atual] = (space[idx_atual] + dir_particle)%L


def search_in_rotated_array(space, a)-> bool: # This is an adaptation of binary search that decides wheter a belongs to space or not
    left, right = 0, len(space) - 1 

    while left <= right:
        mid = (left + right) // 2

        if space[mid] == a:
            return True  # Found the element

        # Determine which side is properly sorted
        if space[left] <= space[mid]:  # Left half is sorted
            if space[left] <= a < space[mid]:  # Target in left half
                right = mid - 1
            else:  # Target in right half
                left = mid + 1
        else:  # Right half is sorted
            if space[mid] < a <= space[right]:  # Target in right half
                left = mid + 1
            else:  # Target in left half
                right = mid - 1

    return False  # Element not found


'''Random Walk'''


def random_walk_step(space,rw,right_drift=1/3,left_drift=2/3):  
    if search_in_rotated_array(space,rw): # in blue particles there is a right shift
        direction = bernoulli.rvs(right_drift)  # sample 1 with prob 1/3 and 0 with probability 2/3 
        direction = 2*direction -1              # change bernoulli(0,1) to bernoulli(-1,+1)
        rw = rw + direction

        return(rw)
    
    else :# in red particles there is a left shift
        direction = bernoulli.rvs(left_drift)   # sample 1 with prob 2/3 and 0 with probability 1/3 
        direction = 2*direction -1              # change bernoulli(0,1) to bernoulli(-1,+1)
        rw = rw + direction

        return(rw)


def random_walk(L,t,right_drift=1/3,left_drift=2/3,mu=1/2,v=1): 
    # L means size of lattice,
    # v is the velocity of the environment particles, 
    # t is the number of steps of RW, 
    # mu is the proportion of particlesin the environment.

    rw = 0 # random walk starts at zero
    
    space = space_env(L,mu)
    numb_particle = len(space)

    for t0 in range(t+1):
        s = np.random.poisson(numb_particle*v)

        move_particle(space,s,L,mu)
        rw= random_walk_step(space,rw,right_drift,left_drift)
    return(rw)


def generate_data(Numbsimul,L,t,right_drift,left_drift,mu=1/2,v=1):
    # L,v,t,mu  are exactly thte same parameters used in random_walk
    # N is the number of simulations made

    data = np.array([random_walk(L,t,right_drift,left_drift,mu,v) for i in range(Numbsimul)])
    return(data)


def full_data_random_walk(k1,k2,Numbsimul,right_drift=1/3,left_drift=2/3,mu=1/2,v=1):
# k_1 is the minum size simulation and k2 is biggest 
    if k2<=k1:
        print('------------k2 must be bigger than k1-----------')
        return('error')

    start_time = time.time()

    # (L,v,t,mu) are exactly the same parameters used in random walk

    data = np.zeros((k2+1-k1, Numbsimul))

    for i, k in tqdm(enumerate(range(k1, k2+1))):
        n = pow(2,k)
        data[i,:] = np.array(generate_data(Numbsimul,n,n,right_drift,left_drift,mu,v))
    print('\n')
    print("--- %s seconds ---" % (time.time() - start_time))
    print('\n')

    return(data)
