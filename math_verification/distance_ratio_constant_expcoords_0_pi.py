'''
monte carlo verification of the proof of distance ratio constant for [0,pi]
'''
import numpy as np
import matplotlib.pyplot as pp

n = 10**8
theta_1 = np.random.uniform(0,np.pi,n)
theta_2 = np.random.uniform(0,np.pi,n)
t = np.random.uniform(0,1,n)
dist_rot = 2*np.arccos(np.abs(t*np.cos(theta_1/2 + theta_2/2) + (1-t)*np.cos(theta_1/2 - theta_2/2)))
dist_euc = ((theta_1 - theta_2)**2 + 4*theta_1*theta_2*t)**.5
frac = dist_rot/dist_euc
print('dist(p1,p2)/||p2 - p1||, min: ', np.min(frac))
print('dist(p1,p2)/||p2 - p1||, max: ', np.max(frac))
i_max = np.argmax(frac)
