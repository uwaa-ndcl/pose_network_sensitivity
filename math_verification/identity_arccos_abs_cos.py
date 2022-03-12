'''
brute force check of the identity:

arccos(abs(cos)) = phi,        phi in [0,pi/2]
                 = pi - phi,   phi in [pi/2,pi]
'''

phi = np.random.uniform(0,np.pi,n)
lhs = np.arccos(np.abs(np.cos(phi)))
mask_1 = (phi<=np.pi/2)
mask_2 = (phi>=np.pi/2)
rhs_1 = phi
rhs_2 = np.pi - phi
err_1 = np.abs(lhs[mask_1] - rhs_1[mask_1])
err_2 = np.abs(lhs[mask_2] - rhs_2[mask_2])
print('max err, phi in [0,pi/2]', np.max(err1))
print('max err, phi in [pi/2,pi]', np.max(err2))
