'''
brute force check of the identity:

arccos(abs(x)) = pi - arccos(x),   x in [-1,0]
               = arccos(x),        x in [0,1]
'''
import numpy as np

n = 10**8

# x in [-1,0]
x = np.random.uniform(-1,0,n)
lhs = np.arccos(np.abs(x))
rhs = np.pi - np.arccos(x)
print('x in [-1,0]')
print('cumulative error:', np.linalg.norm(rhs-lhs))

# x in [0,1]
x = np.random.uniform(0,1,n)
lhs = np.arccos(np.abs(x))
rhs = np.arccos(x)
print('\nx in [0,1]')
print('cumulative error:', np.linalg.norm(rhs-lhs))
