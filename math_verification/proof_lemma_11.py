'''
monte carlo check for the proof of Lemma 11
'''
import numpy as np

def dist_rot_fun(theta_1, theta_2, t):
    dist = 2*np.arccos(np.abs(t*np.cos(theta_1/2 + theta_2/2) + (1-t)*np.cos(theta_1/2 - theta_2/2)))
    return dist

def dist_euc_fun(theta_1, theta_2, t):
    dist = ((theta_1 - theta_2)**2 + 4*theta_1*theta_2*t)**.5
    return dist

n = 10**6
THETA_1 = np.random.uniform(0,2*np.pi,n)
THETA_2 = np.random.uniform(0,2*np.pi,n)
T = np.random.uniform(0,1,n)
DIST_ROT = dist_rot_fun(THETA_1, THETA_2, T)
DIST_EUC = dist_euc_fun(THETA_1, THETA_2, T)
RATIO = DIST_ROT/DIST_EUC

# equation 32
print('equation 32')
x1 = np.random.uniform(0,np.pi**2,n)
x2 = np.random.uniform(0,np.pi**2,n)
t = np.random.uniform(0,1,n)
lhs = t*np.cos(np.sqrt(x1)) + (1-t)*np.cos(np.sqrt(x2))
rhs = np.cos(np.sqrt(t*x1 + (1-t)*x2))
check = np.sum(lhs>=rhs)
print('equation true', np.sum(check), '/', n)

# equation 33
print('\nequation 33')
mask = THETA_1+THETA_2 <= 2*np.pi
theta_1 = THETA_1[mask]
theta_2 = THETA_2[mask]
t = T[mask]
n_cases = len(theta_1)
x1 = (theta_1/2 + theta_2/2)**2
x2 = (theta_1/2 - theta_2/2)**2
print('pi**2', np.pi**2)
print('smallest x1', np.min(x1))
print('largest x1', np.max(x1))
print('smallest x2', np.min(x2))
print('largest x2', np.max(x2))

lhs = t*np.cos(theta_1/2 + theta_2/2) + (1-t)*np.cos(theta_1/2 - theta_2/2)
rhs = np.cos(.5*np.sqrt((theta_1-theta_2)**2 + 4*theta_1*theta_2*t))
check = np.sum(lhs>=rhs)
print('equation true', np.sum(check), '/', n_cases)
print('min lhs', np.min(lhs))
print('max lhs', np.max(lhs))
rhs_arg = .5*np.sqrt((theta_1-theta_2)**2 + 4*theta_1*theta_2*t)
print('max rhs arg', np.max(rhs_arg))

# equation 34
print('\nequation 34')
lhs = np.arccos(t*np.cos(theta_1/2 + theta_2/2) + (1-t)*np.cos(theta_1/2 - theta_2/2))
rhs = .5*np.sqrt((theta_1-theta_2)**2 + 4*theta_1*theta_2*t)
check = np.sum(lhs<=rhs)
print('equation true', check, '/', n_cases)

# equation 35
print('\nequation 35')
lhs = 2*np.arccos(np.abs(t*np.cos(theta_1/2 + theta_2/2) + (1-t)*np.cos(theta_1/2 - theta_2/2)))
rhs = np.sqrt((theta_1-theta_2)**2 + 4*theta_1*theta_2*t)
check = np.sum(lhs<=rhs)
print('equation true', check, '/', n_cases)

# equation 36
print('\nequation 36')
lhs = lhs/rhs
rhs = 1
check = np.sum(lhs<=rhs)
print('equation true', check, '/', n_cases)

# equation 37
print('\nequation 37')
mask = THETA_1+THETA_2 >= 2*np.pi
theta_1 = THETA_1[mask]
theta_2 = THETA_2[mask]
t = T[mask]
n_cases = len(theta_1)
theta_1_p = 2*np.pi - theta_1
theta_2_p = 2*np.pi - theta_2

# text after equation 37
print('2*pi:', 2*np.pi)
print('theta_1_p min', np.min(theta_1_p))
print('theta_1_p max', np.max(theta_1_p))
print('theta_2_p min', np.min(theta_2_p))
print('theta_2_p max', np.max(theta_2_p))

rot_dist = dist_rot_fun(theta_1, theta_2, t)
rot_dist_p = dist_rot_fun(theta_1_p, theta_2_p, t)
err = np.abs(rot_dist - rot_dist_p)
print('max rotational distance error:', np.max(err))

euc_dist = dist_euc_fun(theta_1, theta_2, t)
euc_dist_p = dist_euc_fun(theta_1_p, theta_2_p, t)
diff = np.abs(euc_dist - euc_dist_p)
print('min euclidean distance difference (should be >=0):', np.min(diff))
print('max euclidean distance difference:', np.max(diff))

# equation 38
print('\nequation 38')
lhs = ((theta_1 - theta_2)**2 + 4*theta_1*theta_2*t) - ((theta_1_p - theta_2_p)**2 + 4*theta_1_p*theta_2_p*t)
rhs = 8*np.pi*t*(2*np.pi - (theta_1_p + theta_2_p))
err = np.abs(lhs - rhs)
print('max lhs v. rhs error:', np.max(err))

# upper bound
theta_1 = np.random.uniform(1e-6,np.pi)
theta_2 = 0
t = np.random.uniform(0,1)
rot_dist = dist_rot_fun(theta_1,theta_2,t)
euc_dist = dist_euc_fun(theta_1,theta_2,t)
print('theta_1', theta_1)
print('rotational distance', rot_dist)
print('euclidean distance', euc_dist)
