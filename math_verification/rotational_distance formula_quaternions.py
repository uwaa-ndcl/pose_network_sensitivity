'''
brute force check the rotational distance formula for quaternions
'''
import numpy as np
import utils

# generate a bunch of random rotation matrices
#np.random.seed(2)
n = 10**7
R1 = utils.random_rotation_matrix(n)
R2 = utils.random_rotation_matrix(n)
dist_mat = utils.geodesic_distance(R1,R2)

# convert to quaternions and calculate distance
q1 = utils.mat_to_quat(R1)
q2 = utils.mat_to_quat(R2)
q1_dot_q2 = np.einsum('ij,ij-> j', q1, q2)
dist_quat = 2*np.arccos(np.abs(q1_dot_q2))
errs = np.abs(dist_mat - dist_quat)
print('largest error:', np.max(errs))
