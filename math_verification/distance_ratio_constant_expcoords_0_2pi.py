'''
monte carlo verification of the proof of distance ratio constant for [0,2*pi]
'''
import numpy as np
import matplotlib.pyplot as pp

n = 10**6
theta_1 = np.random.uniform(0,2*np.pi,n)
theta_2 = np.random.uniform(0,2*np.pi,n)
sum = theta_1 + theta_2
mask = sum<2*np.pi
theta_1 = np.delete(theta_1, mask)
theta_2 = np.delete(theta_2, mask)
theta_1_p = 2*np.pi -  theta_1
theta_2_p = 2*np.pi -  theta_2
sum = theta_1_p + theta_2_p

# make sure theta_1 and theta_2 are in [0,2*pi] and theta_1+theta_2 is in [0,2*pi]
print('theta_1 min & max')
print(np.min(theta_1_p))
print(np.max(theta_1_p))
print('theta_2 min & max')
print(np.min(theta_2_p))
print(np.max(theta_2_p))
print('theta_1 + theta_2 min & max')
print(np.min(sum))
print(np.max(sum))

# plot to make sure (theta_1,theta_2) and (theta_1',theta_2') cover all of [0,2*pi]x[0,2*pi]
pp.scatter(theta_1,theta_2,color='r')
pp.scatter(theta_1_p,theta_2_p,color='b')
pp.show()

# make sure (theta_1,theta_2,t) and (theta_1',theta_2',t) have same rotational distance
n = len(theta_1)
t = np.random.uniform(0,1,n)
dist_rot = 2*np.arccos(np.abs(t*np.cos(theta_1/2 + theta_2/2) + (1-t)*np.cos(theta_1/2 - theta_2/2)))
dist_rot_p = 2*np.arccos(np.abs(t*np.cos(theta_1_p/2 + theta_2_p/2) + (1-t)*np.cos(theta_1_p/2 - theta_2_p/2)))
err = np.linalg.norm(dist_rot - dist_rot_p)
print('\nrot dist diff')
print(err)

# make sure euclidean distance of (theta_1,theta_2,t) is larger than that of (theta_1',theta_2',t)
dist_euc = ((theta_1 - theta_2)**2 + 4*theta_1*theta_2*t)**.5
dist_euc_p = ((theta_1_p - theta_2_p)**2 + 4*theta_1_p*theta_2_p*t)**.5
euc_diff = np.sum(dist_euc >= dist_euc)
print('\neuc dist diff')
print(euc_diff, '/', n, 'orignal euc dists are larger than or equal to prime euc dist')
