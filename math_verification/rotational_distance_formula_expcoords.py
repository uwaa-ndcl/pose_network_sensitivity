'''
test the rotational distance formula for exponential coordinates
'''
import numpy as np
import utils as ut

# generate a bunch of random exponential coordinates and convert them to rotation matrices
n = 10**6
el_max = 10**1
s1 = np.random.uniform(-el_max,el_max,(3,n))
s2 = np.random.uniform(-el_max,el_max,(3,n))
ang1 = np.linalg.norm(s1, axis=0)
ang2 = np.linalg.norm(s2, axis=0)
ax1 = s1/ang1
ax2 = s2/ang2
ax_dot = np.einsum('ik,ik -> k', ax1, ax2) # dot product between each pair of axes
R1 = ut.axang_to_mat(s1)
R2 = ut.axang_to_mat(s2)

# find the rotational distance between each pair of matrices, and using the formula, then compare
dist_mat = ut.geodesic_distance(R1,R2)
dist_formula = 2*np.arccos(np.abs(np.cos(ang1/2)*np.cos(ang2/2) + ax_dot*np.sin(ang1/2)*np.sin(ang2/2)))
dist_errors = np.abs(dist_mat - dist_formula)
dist_cml_error = np.linalg.norm(dist_mat - dist_formula)
print('largest error:', np.max(dist_errors))
print('cumulative dist error:', dist_cml_error)
