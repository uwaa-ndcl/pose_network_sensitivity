'''
test determination of (u,v) coordinates from PnP using my formula, vs. using PnP algorithm
'''
import os
import cv2
import numpy as np
import torch

import dirs
import config
import so3

device = config.device

# position and orientation
#pos = np.array([-1,3,5.0])
pos = np.random.uniform(-5,5,3)
pos = np.expand_dims(pos,1)
q = np.random.uniform(0,1,4)
q = q/np.linalg.norm(q)
q = np.expand_dims(q,1)

# reference points
bounding_box_npz = os.path.join(dirs.data_dir, 'bounding_box.npz')
dat = np.load(bounding_box_npz, allow_pickle=True)
ref_points = dat['p']

# get uv
uv = so3.q_pos_to_uv(q, pos, ref_points)

# convert to numpy
q = np.squeeze(q)
R = so3.quat_to_mat(q)

# PnP
pos_est, R_est = so3.apply_pnp(uv, ref_points)
pos_est = np.expand_dims(pos_est,1)
print('position error:', np.linalg.norm(pos_est - pos))
print('rotation error: ', so3.geodesic_distance(R_est, R))
