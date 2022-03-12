'''
check the planar equations for rotational and euclidean distances of quaternions

c := ||q2 - q1||

dist(q1,q2) = 4*arcsin(c/2),             c in [0,sqrt(2)] 
            = 2*pi - 4*arcsin(c/2),      c in [sqrt(2),2] 
'''
import numpy as np
import utils

n = 10**6
R1 = utils.random_rotation_matrix(n)
R2 = utils.random_rotation_matrix(n)
q1 = utils.mat_to_quat(R1)
q2 = utils.mat_to_quat(R2)
q1_dot_q2 = np.einsum('ij,ij-> j', q1, q2)
dist_rot = 2*np.arccos(np.abs(q1_dot_q2))
dist_euc = np.linalg.norm(q2 - q1, axis=0)

# planar
c = np.linalg.norm(q2 - q1, axis=0)
print('c, min:', np.min(c))
print('c, max:', np.max(c))
dist_rot_1 = 4*np.arcsin(c/2)
dist_rot_2 = 2*np.pi - 4*np.arcsin(c/2)
mask_1 = (c<=np.sqrt(2))
mask_2 = (c>=np.sqrt(2))
err1 = np.abs(dist_rot[mask_1] - dist_rot_1[mask_1])
err2 = np.abs(dist_rot[mask_2] - dist_rot_2[mask_2])
print('rotational distance max error, c in [0,sqrt(2)]', np.max(err1))
print('rotational distance max error, c in [sqrt(2),2]', np.max(err2))
