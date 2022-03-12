'''
monte carlo calcuation of the distance ratio constant for unconstrained exponential coordinates
'''
import numpy as np
import utils as ut

n = 10**7      # number of pairs of coordinates to compare
el_max = 10**2 # elements of all coordinages will be in range (-el_max,el_max)
s1 = np.random.uniform(-el_max,el_max,(3,n))
s2 = np.random.uniform(-el_max,el_max,(3,n))
R1 = ut.axang_to_mat(s1)
R2 = ut.axang_to_mat(s2)
dist_rot = ut.geodesic_distance(R1,R2) 
dist_euc = np.linalg.norm(s2-s1, axis=0)
frac = dist_rot/dist_euc
print('dist(s_1,s_2)/||s2 - s1|| min:', np.min(frac))
print('dist(s_1,s_2)/||s2 - s1|| max:', np.max(frac))
