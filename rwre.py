import random
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.stats import bernoulli

def space_env(L , mu=0.5):
    total_space = np.arange(L)
    space = np.random.choice(total_space, int(L*mu), replace=False)  # choose L/2 of the bins to contain a particle
    space = np.sort(space)
    space = np.append(space,[0])
    return(space)                                           # return the index of the particles sorted

def search(space,L,old_idx=0): # This function checks if the position of rw(space[-1]) is at some particle
    a = (space[-1])%L 
    numbpart = len(space)-1
    left,right = (old_idx%numbpart),((old_idx+1)%numbpart)

    if space[left]== a:
        return True,left
    if space[right]==a:
        return True,left
    
    if (space[left]> space[right]):  # in that case there is a jump between left and right
        if a> space[left] or space[right]>a:
            return False, left

    # Determine a properly sorted interval [left, right]
    while (space[left]> a)and(space[(left-1)%numbpart]<space[left]):
        # first part checks if      a < left
        # second part checks for a jump
        left = (left - 1)%(numbpart)
    while (space[right]<a)and(space[right]<space[(right+1)%numbpart]):
        # first part checks if right < a
        # second part checks for a jump
        right = (right + 1)%(numbpart)    
    # Binary search on [left,right]
    while left <= right:
        mid = (left+right)//2
        if space[mid] == a:
            return True,left
        
        if space[mid]< a<=space[right]:
            left = mid+1
        elif space[left]<= a < space[mid]:
            right = mid-1
        else:
            return False,left
    return False,left

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
        rw= random_walk_step(space,rw,L,right_drift,left_drift)
    return(rw)

def pipeline_to_move(space,s,l_drift=1/3, r_drift=2/3,v=1):
    # space is a vector with index of particles and the random walk in the last
    # s is the number of steps taken by the random walk
    numb_particles = len(space) -1                   # subtract one to not count rw as a particle
    p_index = np.arange(numb_particles)      # enumerating the particles
    p_steps = np.random.poisson(v*numb_particles,s)+1
    total = np.cumsum(p_steps)              # determining the times whether random walks jumps
    to_move = np.random.choice(p_index,total[-1])# determining which particle/rw jumps each time
                
    to_dir_l = np.random.choice([-1,1],total[-1])      # determining the vector of the directions to jump
    to_dir_r = np.zeros(total[-1])                     # to avoid bugs
    for n in total:                                    # random walk steps
        to_dir_l[n-1] = 2*np.random.binomial(1,l_drift) -1 # sampling rw steps if there is a particle
        to_dir_r[n-1] = 2*np.random.binomial(1,r_drift) -1 # sampling rw steps if there is not a particle
        to_move[n-1]= numb_particles
        
    return to_move,to_dir_l,to_dir_r


def can_particle_move(space, L, i, dir_l): # Boolean saying if the particle i can move dir_l
    numb_particles = len(space)-1
    actual_pos= space[i] 
    new_pos = (actual_pos + dir_l)%L
    next_part_idx = (i + dir_l)% numb_particles
    next_part_pos = space[next_part_idx]

    return new_pos != next_part_pos

def move_all(space,L,to_move,to_dir_l,to_dir_r,old_part=0):
    numb_particles = len(space)-1
    for i,dir_l,dir_r in zip(to_move,to_dir_l,to_dir_r):
        if i == numb_particles:                     # i is the rw to_move
            rw_isinpart, old_part = search(space,L,old_part)
            if rw_isinpart:        # rw is at a particle
                space[-1] = space[-1] + dir_l
            else:                                       # rw is not at a particle       
                space[-1] = space[-1] + dir_r
        else:                                       # i is a particle to move
            if can_particle_move(space,L,i,dir_l):  # if particle i can move 
                space[i] = (space[i] + dir_l)%L     # Move the particle

def rw(L,s,mu=1/2,l_drift=1/3,r_drift=2/3,v=1):
    space = space_env(L,mu)
    to_move,to_dir_l,to_dir_r = pipeline_to_move(space,s,l_drift,r_drift,v)
    move_all(space,L,to_move,to_dir_l,to_dir_r)
    return space[-1] # returns the position of the random walk


def generate_data(Numbsimul,L,t,mu=1/2,l_drift=1/3,r_drift=2/3,v=1):
    # L,v,t,mu  are exactly thte same parameters used in random_walk
    # N is the number of simulations made

    data = np.array([rw(L,t,mu,l_drift,r_drift,v) for i in range(Numbsimul)])
    return(data)


def full_data_random_walk(k1,k2,Numbsimul,mu=1/2,l_drift=1/3,r_drift=2/3,v=1):
# k_1 is the minum size simulation and k2 is biggest 
    if k2<=k1:
        print('------------k2 must be bigger than k1-----------')
        return('error')

    start_time = time.time()

    # (L,v,t,mu) are exactly the same parameters used in random walk

    data = np.zeros((k2+1-k1, Numbsimul))

    for i, k in tqdm(enumerate(range(k1, k2+1))):
        n = pow(2,k)
        data[i,:] = np.array(generate_data(Numbsimul,n,n+1,mu,l_drift,r_drift,v))
    print('\n')
    print("--- %s seconds ---" % (time.time() - start_time))
    print('\n')

    return(data)
