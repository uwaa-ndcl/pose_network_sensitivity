import os
import math
import torch
import transforms3d as t3d

import config

def skew_mat(s):
    '''
    cross product matrix, for vectors s and w, s x w = skew_mat(s) @ w

    s is an array of vectors of size (3,n_vectors)
    '''

    n = s.shape[1]
    zer1 = torch.zeros(n, device=config.device)
    zer2 = torch.zeros(n, device=config.device)
    zer3 = torch.zeros(n, device=config.device)
    row1 = torch.stack([   zer1, -s[2,:],  s[1,:]])
    row2 = torch.stack([ s[2,:],    zer2, -s[0,:]])
    row3 = torch.stack([-s[1,:],  s[0,:],   zer3])
    S = torch.stack([row1, row2, row3])

    return S


def exp(S):
    '''
    use the Rodrigues formula to calculate the matrix exponential of a
    skew-symmetric matrix

    S is an array of skew-symmetric matrices of size (3,3,n_matrices)
    '''
    # setup
    n = S.shape[2]
    s = torch.stack([S[2,1,:], S[0,2,:], S[1,0,:]])
    theta = torch.linalg.norm(s, axis=0)
    #eye = np.dstack([np.eye(3)]*n)
    eye = torch.zeros((3,3,n), device=config.device)
    eye[0,0,:] = 1
    eye[1,1,:] = 1
    eye[2,2,:] = 1
    SS = torch.einsum('ijn,jkn -> ikn', S, S)

    # indices for which theta==0 and theta!=0
    ind_0 = torch.where(theta==0)[0]
    ind_all = torch.arange(n)
    mask = torch.ones(n, dtype=bool, device=config.device)
    mask[ind_0] = False
    ind_not0 = ind_all[mask]

    # get solution
    a = torch.full((n,), float('nan'), device=config.device)
    b = torch.full((n,), float('nan'), device=config.device)
    a[ind_0] = 1
    b[ind_0] = .5
    a[ind_not0] = torch.sin(theta[ind_not0])/theta[ind_not0]
    b[ind_not0] = .5*(torch.sin(theta[ind_not0]/2)/(theta[ind_not0]/2))**2
    exp = eye + a*S + b*SS

    return exp


def geodesic_distance(R1, R2):
    '''
    geodesic distance between two rotation matrices
    R1 and R2 are arrays of matrices, each of size (3,3,n_matrices)
    '''

    R2T = torch.transpose(R2, 0, 1)
    R1_R2T = torch.einsum('ijn,jkn -> ikn', R1, R2T)
    #trace_R1_R2T = np.trace(R1_R2T)
    trace_R1_R2T = torch.einsum('iin -> n', R1_R2T) 

    # sometimes, due to rounding errors, the input to arccos may be a tiny
    # bit above 1, which will create a nan
    eq_inds = torch.all(torch.flatten(R1, end_dim=1)==torch.flatten(R2, end_dim=1), dim=0).nonzero().flatten() # indices where R1==R2
    arccos_arg = .5*(trace_R1_R2T - 1)
    arccos_arg[eq_inds] = 1
    dist = torch.arccos(arccos_arg)

    return dist


def geodesic_distance_expcoords(s1, s2):
    '''
    geodesic_distance between two expcoords
    expcoords are arrays, eah of size (3,n_expcoords)
    '''

    theta1 = torch.norm(s1, dim=0)
    theta2 = torch.norm(s2, dim=0)
    e1 = s1/theta1[None,:]
    e2 = s2/theta2[None,:]
    e_dot = torch.einsum('ij,ij -> j', e1, e2)
    arg = torch.abs(torch.cos(theta1/2)*torch.cos(theta2/2) + e_dot*torch.sin(theta1/2)*torch.sin(theta2/2))
    dist = 2*torch.acos(arg)

    return dist


def expcoords_to_mat(expcoords):
    '''
    convert an axis-angle to a rotation matrix

    expcoords is an array of angles*axes of size (3,n_expcoords)
    '''

    S = skew_mat(expcoords)
    R = exp(S)

    return R


def quat_to_expcoords(q):
    '''
    convert a quaternion to axis-angle representation (output as angle*axis)

    q is an array of quaternions of size (4,n_quaternions)
    '''

    # setup
    n = q.shape[1]
    s = torch.full((3,n), float('nan'), device=config.device)
    ang = 2*torch.arccos(q[0,:])

    # indices for which ang==0 and ang!=0
    ind_all = torch.arange(n)
    ind_0 = torch.where(ang==0)[0]
    mask = torch.ones(n, dtype=bool, device=config.device)
    mask[ind_0] = False
    ind_not0 = ind_all[mask]
    
    # when theta==0
    s[:,ind_0] = 0
    
    # when theta!=0
    sin_ang_over_2 = torch.sin(ang[ind_not0]/2)
    ax = q[1:,ind_not0]/sin_ang_over_2
    s[:,ind_not0] = ang[ind_not0]*ax
        
    return s


def quat_to_mat(q):
    '''
    convert a quaternion to a rotation matrix

    q is an array of quaternions of size (4,n_quaternions)
    '''

    s = quat_to_expcoords(q)
    R = expcoords_to_mat(s)

    return R
