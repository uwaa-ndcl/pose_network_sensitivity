'''
monte carlo calculation of the distance ratio constant for rotation matrices
(this isn't used in the paper, but I'm just curious)
'''
import numpy as np
import utils as ut

n_rots = 10**7
R1 = ut.random_rotation_matrix(n_rots)
R2 = ut.random_rotation_matrix(n_rots)
dist_rot = ut.geodesic_distance(R1,R2)
dist_euc = np.linalg.norm(R2-R1, axis=(0,1))
dist_ratio = dist_rot/dist_euc
print('dist(R1,R2)/||R2-R1||, min:', np.min(dist_ratio))
print('dist(R1,R2)/||R2-R1||, max:', np.max(dist_ratio))

print('\nnote: sqrt(2)/2 = .707106781')
print('note: pi/sqrt(8) = 1.1107207')
