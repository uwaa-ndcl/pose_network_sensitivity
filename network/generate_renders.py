import os.path
import sys
import subprocess
import math
import pickle
import numpy as np
import transforms3d as t3d

# my files
import dirs
import image as ti
import so3
import blender_render as br
import my_net


def render_random(n_renders, save_dir):
    '''
    render a bunch of random poses to create the training data

    x,y,z (in model, x is left-right, y is forward-backward, z is up-down)
    '''
    
    x_min = -1
    x_max = 1
    z_min = -1
    z_max = 1
    y_min = .4
    y_max = 1.5

    x = np.random.uniform(x_min, x_max, size=n_renders)
    y = np.random.uniform(y_min, y_max, size=n_renders)
    z = np.random.uniform(z_min, z_max, size=n_renders)
    
    # "camera cone" xyz
    y = np.random.uniform(y_min, y_max, size=n_renders)
    #y = [y_min]*n_renders # debug
    #y = [y_max]*n_renders # debug
    scl = .3
    for i in range(n_renders):
        x[i] = scl*y[i]*np.random.uniform(-1, 1)
        #x[i] = scl*y[i] # debug
        z[i] = scl*y[i]*np.random.uniform(-1, 1)
        #z[i] = scl*y[i] # debug

    p = np.stack((x, y, z), axis=0)

    # rotations
    R = so3.random_rotation_matrix(n_renders)
    quat = so3.mat_to_quat(R)

    # render
    br.soup_gen(p, R, save_dir, transparent=[True]*n_renders)

# render
render_random(n_renders=10**3, save_dir=dirs.training_renders_dir)
render_random(n_renders=10**4, save_dir=dirs.test_renders_dir)
