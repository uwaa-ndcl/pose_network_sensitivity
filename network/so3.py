import numpy as np
import transforms3d as t3d
import cv2

import config


################################################################################
#                                  EXPONENTIAL                                 #
################################################################################


def skew_mat(s):
    '''
    cross product matrix, for vectors s and w, s x w = skew_mat(s) @ w

    s can either be a single vector of size 3
    or an array of vectors of size (3,n_vectors)
    '''

    s_dim = len(s.shape)

    # s is a single vector
    if s_dim==1:
        S = np.array([[    0, -s[2],  s[1]],
                      [ s[2],     0, -s[0]],
                      [-s[1],  s[0],    0]])

    # s is an array of vectors
    if s_dim==2:
        n = s.shape[1]
        S = np.array([[np.zeros(n),     -s[2,:],        s[1,:]],
                      [     s[2,:], np.zeros(n),       -s[0,:]],
                      [    -s[1,:],      s[0,:],  np.zeros(n)]])

    return S


def exp(S):
    '''
    use the Rodrigues formula to calculate the matrix exponential of a
    skew-symmetric matrix

    S can either be a single skew-symmetric matrix of size (3,3)
    or an array of skew-symmetric matrices of size (3,3,n_matrices)
    '''

    # setup
    S_dim = len(S.shape)

    # S is a single matrix
    if S_dim==2:

        s = np.array([S[2,1], S[0,2], S[1,0]])
        theta = np.linalg.norm(s)

        if theta == 0.0:
            a = 1
            b = .5
        else:
            a = np.sin(theta)/theta
            b = .5*(np.sin(theta/2)/(theta/2))**2

        exp = np.eye(3) + a*S + b*S@S

    # S is an array of matrices
    if S_dim==3:
        
        # setup
        n = S.shape[2]
        s = np.array([S[2,1,:], S[0,2,:], S[1,0,:]])
        theta = np.linalg.norm(s, axis=0)
        eye = np.dstack([np.eye(3)]*n)
        SS = np.einsum('ijn,jkn -> ikn', S, S)

        # indices for which theta==0 and theta!=0
        ind_0 = np.where(theta==0)[0]
        ind_all = np.arange(n)
        ind_not0 = np.delete(ind_all, ind_0)

        # get solution
        a = np.full(n, np.nan)
        b = np.full(n, np.nan)
        a[ind_0] = 1
        b[ind_0] = .5
        a[ind_not0] = np.sin(theta[ind_not0])/theta[ind_not0]
        b[ind_not0] = .5*(np.sin(theta[ind_not0]/2)/(theta[ind_not0]/2))**2
        exp = eye + a*S + b*SS

    return exp


def log(R):
    '''
    matrix logarithm of a rotation matrix

    R can either be a single matrix of size (3,3)
    or an array of matrices of size (3,3,n_matrices)
    '''
    
    R_dim = len(R.shape)

    # R is a single matrix
    if R_dim==2:
        theta = np.arccos((np.trace(R) - 1)/2)
        if theta == 0.0:
            log = 0.5*(R - R.T)
        else:
            log = (theta/(2*np.sin(theta)))*(R - R.T)

    # R is an array of matrices
    elif R_dim==3:
        # set up
        n = R.shape[2]
        log = np.full((3,3,n), np.nan)
        RT = np.transpose(R, axes=(1,0,2))

        # sometimes, due to rounding errors, the input to arccos may be a tiny
        # bit above 1, which will create a nan
        eps = 1e-15
        arccos_arg = (np.trace(R) - 1)/2
        ind_0 = np.where(np.logical_and(arccos_arg>=1, arccos_arg<1+eps))[0]

        # when theta does not equal zero
        ind_all = np.arange(n)
        ind_not0 = np.delete(ind_all, ind_0)
        R_not0 = R[:,:,ind_not0]
        RT_not0 = RT[:,:,ind_not0]
        theta_not0 = np.arccos((np.trace(R_not0) - 1)/2)
        log[:,:,ind_not0] = (theta_not0/(2*np.sin(theta_not0)))*(R_not0 - RT_not0)

        # when theta equals 0
        R_0 = R[:,:,ind_0]
        RT_0 = RT[:,:,ind_0]
        log[:,:,ind_0] = 0.5*(R_0 - RT_0)

    return log


def geodesic_distance_log(R1, R2):
    '''
    geodesic distance between two rotation matrices using the log function
    R1 and R2 can either be a single matrices of size (3,3)
    or can be arrays of matrices, each of size (3,3,n_matrices)
    '''

    R1_dim = len(R1.shape)
    R2_dim = len(R2.shape)

    # R1 & R2 are single matrices
    if R1_dim==2 and R2_dim==2:
        if np.all(R1==R2):
            dist = 0
        else:
            dist = (1/np.sqrt(2)) * np.linalg.norm(log(R1.T @ R2), ord='fro')

    # R1 and R2 are arrays of matrices
    if R1_dim==3 and R2_dim==3:
        #equal = np.all(R1==R2, axis=(0,1))
        R1T = np.transpose(R1, axes=(1,0,2))
        R1T_R2 = np.einsum('ijn,jkn -> ikn', R1T, R2)
        log_R1T_R2 = log(R1T_R2)
        dist = (1/np.sqrt(2)) * np.linalg.norm(log_R1T_R2, ord=None, axis=(0,1))

    return dist


def geodesic_distance(R1, R2):
    '''
    geodesic distance between two rotation matrices
    R1 and R2 can either be a single matrices of size (3,3)
    or can be arrays of matrices, each of size (3,3,n_matrices)
    '''

    R1_dim = len(R1.shape)
    R2_dim = len(R2.shape)

    # R1 & R2 are single matrices
    if R1_dim==2 and R2_dim==2:
        if np.all(R1==R2):
            dist = 0
        else:
            eps = 1e-15
            arccos_arg = .5*(np.trace(R1 @ R2.T) - 1)
            if arccos_arg>1 and arccos_arg<1+eps:
                arccos_arg = 1
            elif arccos_arg<-1 and arccos_arg>-1-eps:
                arccos_arg = -1
            dist = np.arccos(arccos_arg)

    # R1 and R2 are arrays of matrices
    if R1_dim==3 and R2_dim==3:
        R2T = np.transpose(R2, axes=(1,0,2))
        R1_R2T = np.einsum('ijn,jkn -> ikn', R1, R2T)
        trace_R1_R2T = np.trace(R1_R2T)

        # sometimes, due to rounding errors, the input to arccos may be a tiny
        # bit above 1, which will create a nan
        eq_inds = np.all(R1==R2, axis=(0,1)).nonzero()[0] # indices where R1==R2
        arccos_arg = .5*(trace_R1_R2T - 1)
        arccos_arg[eq_inds] = 1
        dist = np.arccos(arccos_arg)

    return dist


################################################################################
#                                    RANDOM                                    #
################################################################################


def random_rotation_matrix(n=None):
    '''
    generate a random rotation matrix
    from Graphics Gems III
    input n is the number of matrices to generate
    '''

    # generate a single random rotation matrix
    if n is None:
        (x1, x2, x3) = np.random.uniform(0, 1, size=3)
        theta = 2*np.pi*x1
        phi = 2*np.pi*x2
        z = x3
        V = np.array([np.cos(phi)*np.sqrt(z), np.sin(phi)*np.sqrt(z), np.sqrt(1-z)])
        R_z = np.array([[ np.cos(theta), np.sin(theta), 0],
                        [-np.sin(theta), np.cos(theta), 0],
                        [             0,             0, 1]])
        M = np.matmul(2*np.outer(V,V) - np.eye(3), R_z)

    # generate an array of random rotation matrices
    else:
        (x1, x2, x3) = np.random.uniform(0, 1, size=(3,n))
        theta = 2*np.pi*x1
        phi = 2*np.pi*x2
        z = x3
        V = np.array([np.cos(phi)*np.sqrt(z), np.sin(phi)*np.sqrt(z), np.sqrt(1-z)])
        R_z = np.array([[ np.cos(theta), np.sin(theta), np.zeros(n)],
                        [-np.sin(theta), np.cos(theta), np.zeros(n)],
                        [   np.zeros(n),   np.zeros(n),  np.ones(n)]])
        V_outer_V = np.einsum('ij,kj -> ikj', V, V)
        eye3 = np.eye(3)[:,:,None]
        mat = 2*V_outer_V - eye3
        M = np.einsum('ijn,jkn -> ikn', mat, R_z)

    return M


################################################################################
#                                  AXIS-ANGLES                                 #
################################################################################


def axang_to_mat(axang):
    '''
    convert an axis-angle to a rotation matrix

    axang can either be a single angle*axis of size 3
    or an array of angles*axes of size (3,n_axangs)
    '''

    S = skew_mat(axang)
    R = exp(S)

    return R


def mat_to_axang(R):
    '''
    convert a rotation matrix to a axis-angle (angle*axis)

    R can either be a single matrix of size (3,3)
    or an array of matrices of size (3,3,n_matrices)
    '''

    R_dim = len(R.shape)

    # R is a single matrix
    if R_dim==2:
        S = log(R)
        s = np.array([S[2,1], S[0,2], S[1,0]])

    # R is an array of matrices
    elif R_dim==3:
        S = log(R)
        s = np.array([S[2,1,:], S[0,2,:], S[1,0,:]])

    return s


################################################################################
#                                  QUATERNIONS                                 #
################################################################################


def axang_to_quat(s):
    '''
    convert an axis-angle roation (input as angle*axis) to a quaternion

    s can either be a single axis-angle vector of size 3
    or an array of vectors of size (3,n_vectors)
    '''

    # setup
    s_dim = len(s.shape)

    # R is a single matrix
    if s_dim==1:
        ang = np.linalg.norm(s)
        if ang==0:
            q = np.array([1,0,0,0])
        else:
            ax = s/ang
            q = np.array([np.cos(ang/2), ax[0]*np.sin(ang/2), ax[1]*np.sin(ang/2), ax[2]*np.sin(ang/2)])

    # R is an array of matrices
    elif s_dim==2:
        # setup
        n = s.shape[1]
        ax = np.full((3,n), np.nan)
        ang = np.linalg.norm(s, axis=0)

        # determine the indices in which the angle equals 0
        ind_all = np.arange(n)
        ind_0 = np.where(ang==0)[0]
        ind_not0 = np.delete(ind_all, ind_0)

        # angle==0
        ax[:,ind_0] = 0

        # angle!=0
        ax[:,ind_not0] = s[:,ind_not0]/ang[ind_not0]
    
        # calculate the quaternion
        q = np.array([np.cos(ang/2), ax[0]*np.sin(ang/2), ax[1]*np.sin(ang/2), ax[2]*np.sin(ang/2)])
        q_norm = np.linalg.norm(q, axis=0)
        q = q/q_norm

    return q


def quat_to_axang(q):
    '''
    convert a quaternion to axis-angle representation (output as angle*axis)

    q can either be a single quaternion of size 4
    or an array of quaternions of size (4,n_quaternions)
    '''

    q_dim = len(q.shape)

    # q is a single quaternion
    if q_dim==1:
        ang = 2*np.arccos(q[0])
        if ang==0:
            s = np.zeros(3)
        else:
            sin_ang_over_2 = np.sin(ang/2)
            ax = q[1:]/sin_ang_over_2
            s = ang*ax

    # q is an array of quaternions
    elif q_dim==2:
        # setup
        n = q.shape[1]
        s = np.full((3,n), np.nan)
        ang = 2*np.arccos(q[0,:])

        # indices for which ang==0 and ang!=0
        ind_all = np.arange(n)
        ind_0 = np.where(ang==0)[0]
        ind_not0 = np.delete(ind_all, ind_0)
        
        # when theta==0
        s[:,ind_0] = 0
        
        # when theta!=0
        sin_ang_over_2 = np.sin(ang[ind_not0]/2)
        ax = q[1:,ind_not0]/sin_ang_over_2
        s[:,ind_not0] = ang[ind_not0]*ax
        
    return s


def mat_to_quat(R):
    '''
    convert a rotation matrix to a quaternion

    R can either be a single matrix of size (3,3)
    or an array of matrices of size (3,3,n_matrices)
    '''

    s = mat_to_axang(R)
    q = axang_to_quat(s)

    return q


def quat_to_mat(q):
    '''
    convert a quaternion to a rotation matrix

    q can either be a single quaternion of size 4
    or an array of quaternions of size (4,n_quaternions)
    '''

    s = quat_to_axang(q) 
    R = axang_to_mat(s)

    return R


################################################################################
#                                    PNP                                       #
################################################################################


def xyz_to_uv(xyz):
    '''
    convert xyz to uv (in pixels w.r.t. the center of the image)

    input: xyz (n_batch,8,3)
    output: uv (n_batch,8,3)
    '''

    f = config.f
    sensor_width = config.sensor_width
    sensor_height = config.sensor_height
    pix_width = config.pix_width
    pix_height = config.pix_height
    scale_u = f*(1/(sensor_width/2))*(pix_width/2)
    scale_v = f*(1/(sensor_height/2))*(pix_height/2)
    u = scale_u*xyz[0,:,:]/xyz[1,:,:]
    v = scale_v*xyz[2,:,:]/xyz[1,:,:]
    uv = np.stack([u, v], axis=0)

    return uv


def q_pos_to_uv(q, pos, points):
    '''
    shift points by quaternion q and position pos,
    then project them to uv

    q is an array of quaterions of shape (n_batch,4)
    pos is an array of shape (n_batch,3)
    points is an array of shape (3,8)
    '''

    R = quat_to_mat(q)

    # multiply p (3,8) by each R (3,3,n_batch) to get a (3,8,n_batch)
    xyz = np.einsum('ijk, jl -> ilk', R, points)

    xyz = xyz + pos[:,None,:]
    uv = xyz_to_uv(xyz)

    return uv


def apply_pnp(uv, ref_points):
    '''
    apply PnP using opencv's algorithm

    in cv2, z is away from the camera,
    in my work, y is away from the camera,
    so we have to swap the y and z axes

    uv is of shape (2,8,n_batch) 
    ref_points is of shape (3,8)
    '''

    # camera properties
    f = config.f
    sensor_width = config.sensor_width
    sensor_height = config.sensor_height
    pix_width = config.pix_width
    pix_height = config.pix_height
    scale_u = f*(1/(sensor_width/2))*(pix_width/2)
    scale_v = f*(1/(sensor_height/2))*(pix_height/2)

    # reference points
    ref_points = ref_points[[0,2,1]] # swap y and z

    # apply PnP
    center = (0,0)
    camera_matrix = np.array([[scale_u, 0, center[0]],
                              [0, scale_v, center[1]],
                              [0, 0, 1]], dtype='double')
    dist_coeffs = np.zeros((4,1)) # assuming no lens distortion
    uv = np.swapaxes(uv,0,1) # cv2 wants uv to be of size (8,2,n_batch)
    (success, s_est, p_est) = cv2.solvePnP(
        ref_points.T, uv, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # permute
    if p_est.shape==(3,1):
        p_est = p_est.flatten()
    else:
        print('UH OH')
    p_est = p_est[[0,2,1]] # swap y and z
    R_est, jac_est = cv2.Rodrigues(s_est)
    perm_mat = np.array([[1,0,0],[0,0,1],[0,1,0]])
    R_est = perm_mat.T @ R_est @ perm_mat

    return p_est, R_est
