'''
brute force check of the identity:

dist(q1,q2) = 2*phi,          phi in [0,pi/2]
            = 2*pi - 2*phi,   phi in [pi/2,pi]
'''
import numpy as np
import utils

n = 10**6
R1 = utils.random_rotation_matrix(n)
R2 = utils.random_rotation_matrix(n)
q1 = utils.mat_to_quat(R1)
q2 = utils.mat_to_quat(R2)
q1_dot_q2 = np.einsum('ij,ij-> j', q1, q2)
lhs = 2*np.arccos(np.abs(q1_dot_q2))

# check the equation
phi = np.arccos(q1_dot_q2)
mask_1 = (phi<=np.pi/2)
mask_2 = (phi>=np.pi/2)
rhs_1 = 2*phi 
rhs_2 = 2*np.pi - 2*phi 
err1 = np.abs(lhs[mask_1] - rhs_1[mask_1])
err2 = np.abs(lhs[mask_2] - rhs_2[mask_2])
print('max err, phi in [0,pi/2]', np.max(err1))
print('max err, phi in [pi/2,pi]', np.max(err2))
