'''
test the triangle property of the rotational distance function (which helps determine if the function is a pseudometric)
'''
import numpy as np
import transforms3d as t3d
import utils as ut

# brute force
n_rots = 10**7
R1 = ut.random_rotation_matrix(n_rots)
R2 = ut.random_rotation_matrix(n_rots)
R3 = ut.random_rotation_matrix(n_rots)
dist12 = ut.geodesic_distance(R1, R2)
dist23 = ut.geodesic_distance(R2, R3)
dist13 = ut.geodesic_distance(R1, R3)
check = (dist12 < (dist23 + dist13))
count = np.sum(check)
print('triangle inequality holds for', count, '/', n_rots, 'rotations')
