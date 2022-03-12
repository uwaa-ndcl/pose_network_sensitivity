'''
test the distance ratio constant for quaternions
'''
import numpy as np
import transforms3d as t3d
import utils as ut

print('NOTE: pi/sqrt(2) = ', np.pi/np.sqrt(2))

# manual
###############################################################################
print('\nSPECIFIC PAIRS OF QUATERNIONS')

# two quaternions which we know have a distance ratio of nearly 0
eps = 10**-8
theta_1 = 0
theta_2 = np.pi + eps
q1 = np.array([np.cos(theta_1), np.sin(theta_1), 0, 0])
q2 = np.array([np.cos(theta_2), np.sin(theta_2), 0, 0])
R1 = t3d.quaternions.quat2mat(q1)
R2 = t3d.quaternions.quat2mat(q2)
dist = ut.geodesic_distance(R1, R2)
frac = dist/np.linalg.norm(q2-q1)
print('1st pair')
print('q1 = ', q1)
print('q2 = ', q2)
print('distance ratio = ', frac)

# two quaternions which we know have a distance ratio of pi/sqrt(2)
q1 = np.array([1, 0, 0, 0])
q2 = np.array([0, 1, 0, 0])
R1 = t3d.quaternions.quat2mat(q1)
R2 = t3d.quaternions.quat2mat(q2)
dist = ut.geodesic_distance(R1, R2)
frac = dist/np.linalg.norm(q2-q1)
print('\n2nd pair')
print('q1 = ', q1)
print('q2 = ', q2)
print('distance ratio = ', frac)


# brute force
###############################################################################

n_rots = 10**7
R1 = ut.random_rotation_matrix(n_rots)
R2 = ut.random_rotation_matrix(n_rots)
q1 = ut.mat_to_quat(R1)
q2 = ut.mat_to_quat(R2)
q_diff = np.linalg.norm(q2 - q1, axis=0)
dist = ut.geodesic_distance(R1, R2)
frac = dist/q_diff
print('\nBRUTE FORCE (' + '{:.0e}'.format(n_rots) + ' TRIALS)')
print('dist(q1,q2)/||q2 - q1||, min: ', np.min(frac))
print('dist(q1,q2)/||q2 - q1||, max: ', np.max(frac))
