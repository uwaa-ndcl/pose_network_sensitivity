import os
import itertools
import numpy as np
import torch
import cv2
import matplotlib.pyplot as pp
import matplotlib.lines as lines

import my_net
import dirs
import config
import so3

device = config.device 

# reference points
bounding_box_npz = os.path.join(dirs.data_dir, 'bounding_box.npz')
dat = np.load(bounding_box_npz, allow_pickle=True)
ref_points = dat['p']
#ref_points = torch.from_numpy(ref_points_np).to(config.device).float()

# datasets
batch_size = 1
png_dir = dirs.test_data_dir
data_npz = os.path.join(dirs.test_data_dir, 'test_data.npz')
evalset = my_net.MyDataset(png_dir, data_npz)
evalloader = torch.utils.data.DataLoader(
        evalset, batch_size=batch_size, shuffle=False,
        num_workers=my_net.n_workers)
n_eval = len(evalset)

# network & cuda
net = my_net.MyNet(1.0)
device = torch.device('cuda')
net.to(device)

# loss
ckpt_restore_ind = 235
ckpt_file = os.path.join(my_net.ckpt_dir, 'ckpt_' + str(ckpt_restore_ind) + '.pt')
ckpt = torch.load(ckpt_file)
net.load_state_dict(ckpt['net_state_dict'])
net.eval()

# starting index of image
i = 0

def get_data(i):
    #ind = np.random.randint(0,8)
    eval_data_i = next(itertools.islice(evalloader, i, None))
    ims = eval_data_i['image']
    posq_true = eval_data_i['posq']
    ims, posq_true = ims.to(device), posq_true.to(device)
    pos_true = posq_true[:,:3].cpu().detach().numpy().T
    quat_true = posq_true[:,3:].cpu().detach().numpy().T
    uv_true = so3.q_pos_to_uv(quat_true, pos_true, ref_points)

    # squeeze
    pos_true = np.squeeze(pos_true)
    uv_true = np.squeeze(uv_true)
    quat_true = np.squeeze(quat_true)

    # true rotation matrix
    R_true = so3.quat_to_mat(quat_true)

    # evaluate the network
    pose_pred = net(ims)
    pose_pred = pose_pred.cpu().detach().numpy().T
    pos_pred = pose_pred[:3,:]
    quat_pred = so3.axang_to_quat(pose_pred[3:,:])
    uv_pred = so3.q_pos_to_uv(quat_pred, pos_pred, ref_points)
    uv_pred = np.squeeze(uv_pred)

    # PnP
    p_est, R_est = so3.apply_pnp(uv_pred, ref_points)
    pos_err = np.linalg.norm(p_est*100 - pos_true*100) # in cm
    rot_err = so3.geodesic_distance(R_est, R_true)
    rot_err_deg = rot_err*180/np.pi

    return ims, uv_true, uv_pred, pos_err, rot_err_deg


def plot(ims, uv_true, uv_pred, pos_err, rot_err_deg):
    # plot
    im = ims.cpu().numpy()
    im = im/255.0
    im = np.squeeze(im)
    im = np.transpose(im, (1,2,0))
    #im = pp.imread(im)

    pp.imshow(im, extent=[-256//2, 256//2, -256//2, 256//2])
    pp.plot(uv_true[0,:], uv_true[1,:], 'r.', label='true')
    pp.plot(uv_pred[0,:], uv_pred[1,:], 'b.', label='predicted')
    red_dot = lines.Line2D([], [], color='red', marker='.', linestyle='None', label='true')
    blue_dot = lines.Line2D([], [], color='blue', marker='.', linestyle='None', label='predicted')
    pp.legend(handles=[blue_dot, red_dot], loc='center left', bbox_to_anchor=(1.0,.9), frameon=True)
    pp.text(140,0,'image index:   '+ str(i) + ' / ' + str(n_eval) + '\nposition error: ' + '{:.2f}'.format(pos_err) + ' cm\n' + 'rotation error: ' + '{:.2f}'.format(rot_err_deg) + ' deg')
    pp.text(140,-50,'CLICK FOR NEW IMAGE', fontsize=16)


def onclick(event):
    global i
    i += 1
    pp.clf()
    ims, uv_true, uv_pred, pos_err, rot_err_deg = get_data(i)
    plot(ims, uv_true, uv_pred, pos_err, rot_err_deg)
    pp.draw()

# first image
fig,ax = pp.subplots()
fig.set_size_inches(12,6)
#pp.axes(ax[0])
#pp.figure(figsize=(3,4))
ims, uv_true, uv_pred, pos_err, rot_err_deg = get_data(i)
plot(ims, uv_true, uv_pred, pos_err, rot_err_deg)

fig.canvas.mpl_connect('button_press_event', onclick)
pp.show()
pp.draw()
