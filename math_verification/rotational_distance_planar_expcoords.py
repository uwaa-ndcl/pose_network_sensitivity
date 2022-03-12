'''
check the planar equations for rotational and euclidean distances of exponential coordinates

||s2 - s1|| = sqrt((theta_1 - theta_2)**2 + 4*theta_1*theta_2*t)

dist(s1,s2) = 2*arccos(abs(t*cos(theta_1/2 + theta_2/2) + (1-t)*cos(theta_1/2 - theta_2/2)))
'''
import numpy as np

n = 10**6
el_max = 10**1
s1 = np.random.uniform(-el_max,el_max,(3,n))
s2 = np.random.uniform(-el_max,el_max,(3,n))
theta_1 = np.linalg.norm(s1, axis=0)
theta_2 = np.linalg.norm(s2, axis=0)
e1 = s1/theta_1
e2 = s2/theta_2
e1_dot_e2 = np.einsum('ik,ik -> k', e1, e2) # dot product between each pair of axes

# distances
dist_euc = np.linalg.norm(s2-s1, axis=0)
dist_rot = 2*np.arccos(np.abs(np.cos(theta_1/2)*np.cos(theta_2/2) + e1_dot_e2*np.sin(theta_1/2)*np.sin(theta_2/2)))

# planar equations
t = .5*(1 - e1_dot_e2)
dist_euc_2d = ((theta_1 - theta_2)**2 + 4*theta_1*theta_2*t)**.5
dist_rot_2d = 2*np.arccos(np.abs(t*np.cos(theta_1/2 + theta_2/2) + (1-t)*np.cos(theta_1/2 - theta_2/2)))

# errors
err_euc = np.linalg.norm(dist_euc - dist_euc_2d)
err_rot = np.linalg.norm(dist_rot - dist_rot_2d)
print('cumulative error in euclidean distances', err_euc)
print('cumulative error in rotational distances', err_rot)
