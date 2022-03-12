'''
monte carlo check for Theorem 12
'''
import numpy as np

def dist_rot_fun(theta_1, theta_2, t):
    dist = 2*np.arccos(np.abs(t*np.cos(theta_1/2 + theta_2/2) + (1-t)*np.cos(theta_1/2 - theta_2/2)))
    return dist

def dist_euc_fun(theta_1, theta_2, t):
    dist = ((theta_1 - theta_2)**2 + 4*theta_1*theta_2*t)**.5
    return dist

n = 10**7
THETA_1 = np.random.uniform(0,100,n)
THETA_2 = np.random.uniform(0,100,n)
T = np.random.uniform(0,1,n)
DIST_ROT = dist_rot_fun(THETA_1, THETA_2, T)
DIST_EUC = dist_euc_fun(THETA_1, THETA_2, T)
RATIO = DIST_ROT/DIST_EUC

mask = np.abs(THETA_1-THETA_2)>np.pi

###############################################################################
# case: |theta_1-theta_2|>pi
###############################################################################
print('\nCASE A: |theta_1-theta_2|>pi\n')
theta_1 = THETA_1[mask]
theta_2 = THETA_2[mask]
t = T[mask]
n_case = len(theta_1)

term = (theta_1-theta_2)**2
print('pi**2:', np.pi**2)
print('min term (should be greater than pi**2):', np.min(term))

term = 4*theta_1*theta_2*t 
print('min term (should be greater than 0):', np.min(term))

dist_rot = dist_rot_fun(theta_1, theta_2, t)
dist_euc = dist_euc_fun(theta_1, theta_2, t)
check = np.sum(dist_euc>np.pi)
print('euc dist larger than pi:', np.sum(check), '/', n_case)
ratio = dist_rot/dist_euc
print('ratio max:', np.max(ratio))


###############################################################################
# case: |theta_1+theta_2|<=pi
###############################################################################
print('\nCASE B: |theta_1+theta_2|<=pi\n')
theta_1 = THETA_1[~mask]
theta_2 = THETA_2[~mask]
t = T[~mask]
n_case = len(theta_1)

# determine theta_1 prime and theta_2 prime
n_p_1 = theta_1 // (2*np.pi)
n_p_2 = theta_2 // (2*np.pi)
n_p = np.maximum(n_p_1, n_p_2)
n_p = n_p.astype(int)

# shift angles to [-pi,2pi]
theta_1_p = theta_1 - 2*np.pi*n_p
theta_2_p = theta_2 - 2*np.pi*n_p

# sanity check
print('min n (should be an integer 0 or larger):', np.min(n_p))
print('max n (should be an integer 0 or larger):', np.max(n_p))
print('min theta_1_p (should be larger than -pi):', np.min(theta_1_p))
print('max theta_1_p (should be smaller than 2*pi):', np.max(theta_1_p))
print('min theta_2_p (should be larger than -pi):', np.min(theta_2_p))
print('max theta_2_p (should be smaller than 2*pi):', np.max(theta_2_p))
max_theta_1_p_theta_2_p = np.maximum(theta_1_p, theta_2_p)
print('min of maximum(theta_1_p,theta_2_p) (should be equal to 0 or larger):', np.min(max_theta_1_p_theta_2_p))

# mask WLOG
mask_wlog = theta_1_p>theta_2_p
mask_case1 = mask_wlog & (theta_1_p>=0) & (theta_2_p>=0)
mask_case2 = mask_wlog & (theta_1_p>=0) & (theta_2_p<0)

###############################################################################
# case 1: theta_1'>=0 and theta_2'>=0
print('\nCASE B1: theta_1\'>=0 and theta2\' >= 0\n')
t_case1 = t[mask_case1]
theta_1_case1 =  theta_1[mask_case1]
theta_2_case1 =  theta_2[mask_case1]
theta_1_p_case1 =  theta_1_p[mask_case1]
theta_2_p_case1 =  theta_2_p[mask_case1]

rot_dist_case1 = dist_rot_fun(theta_1_case1, theta_2_case1, t_case1)
rot_dist_p_case1 = dist_rot_fun(theta_1_p_case1, theta_2_p_case1, t_case1)
err = np.abs(rot_dist_case1 - rot_dist_p_case1)
print('max rotational distance error', np.max(err))

euc_dist_case1 = dist_euc_fun(theta_1_case1, theta_2_case1, t_case1)
euc_dist_p_case1 = dist_euc_fun(theta_1_p_case1, theta_2_p_case1, t_case1)
diff = euc_dist_case1 - euc_dist_p_case1
print('min euclidean distance diff (should be >= 0):', np.min(diff))

ratio_case1 = rot_dist_case1/euc_dist_case1
ratio_p_case1 = rot_dist_p_case1/euc_dist_p_case1
diff = ratio_p_case1 - ratio_case1
print('min distance ratio diff (should be >= 0):', np.min(diff))

###############################################################################
# case 2: theta_1'>=0 and theta_2'<0
print('\nCASE B2: theta_1\'>=0 and theta2\' < 0\n')
t_case2 = t[mask_case2]
theta_1_case2 = theta_1[mask_case2]
theta_2_case2 = theta_2[mask_case2]
theta_1_p_case2 = theta_1_p[mask_case2]
theta_2_p_case2 = theta_2_p[mask_case2]

rot_dist_case2 = dist_rot_fun(theta_1_case2, theta_2_case2, t_case2)
rot_dist_p_case2 = dist_rot_fun(theta_1_p_case2, theta_2_p_case2, t_case2)
err = np.abs(rot_dist_case2 - rot_dist_p_case2)
print('max rotational distance error', np.max(err))

euc_dist_case2 = dist_euc_fun(theta_1_case2, theta_2_case2, t_case2)
euc_dist_p_case2 = dist_euc_fun(theta_1_p_case2, theta_2_p_case2, t_case2)
diff = euc_dist_case2 - euc_dist_p_case2
print('min euclidean distance diff (should be >=0):', np.min(diff))

# pair (-theta_1',theta_2',1-t)
print('min -theta_2\' (should be >= 0):', np.min(-theta_2_p_case2))
print('max -theta_2\' (should be <= pi):', np.max(-theta_2_p_case2))
rot_dist_pair = dist_rot_fun(theta_1_p_case2, -theta_2_p_case2, 1-t_case2)
euc_dist_pair = dist_euc_fun(theta_1_p_case2, -theta_2_p_case2, 1-t_case2)
err = np.abs(rot_dist_p_case2 - rot_dist_pair)
print('max rotational distance error, pair', np.max(err))
err = np.abs(euc_dist_p_case2 - euc_dist_pair)
print('max euclidean distance error, pair', np.max(err))

ratio_case2 = rot_dist_case2/euc_dist_case2
ratio_p_case2 = rot_dist_p_case2/euc_dist_p_case2
diff = ratio_p_case2 - ratio_case2
print('min distance ratio diff (should be >= 0):', np.min(diff))
