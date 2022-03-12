import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision

import dirs
import config
import so3_pytorch as so3

device = config.device

# general parameters
n_epochs_per_display = 1
n_epochs_per_checkpoint_save = 1
n_ckpt_to_keep = 5
pos_scale_factor = 1
rot_scale_factor = 1

# training parameters
batch_size = 64*4
dropout_keep_prob = 0.5
learning_rate = 1e-1
eps = 1e-6 # "term added to the denominator to improve numerical stability (default 1e-6)"
#n_workers = 40 # I think this caused a problem
n_workers = 8

# files
model_name = 'soup_can' # name of the .blend file in the models directory
renders_train_pkl = os.path.join(dirs.training_renders_dir, 'to_render.pkl')
renders_test_pkl = os.path.join(dirs.test_renders_dir, 'to_render.pkl')
training_data_npz = os.path.join(dirs.training_data_dir, 'training_data.npz')
test_data_npz = os.path.join(dirs.test_data_dir, 'test_data.npz')

ckpt_dir = dirs.ckpt_dir
ckpt_file = os.path.join(ckpt_dir, 'ckpt_%d.pt')
loss_npz = os.path.join(ckpt_dir, 'loss.npz')

# functions
def get_data(png_dir, data_npz):
    '''
    png_dir: a directory containing the png files (which will be read in
    alphabetical order)
    data_npz: an npz file with entries 'pos' and 'quat'
    '''

    # images
    png_files = sorted(glob.glob(os.path.join(png_dir, '*.png')))
    n_total_datapts = len(png_files)

    # posq values
    data = np.load(data_npz)
    pos = data['pos'] # each column is one training datapoint
    quat = data['quat']
    posq_np = np.concatenate((pos, quat), axis=0)
    posq_np = posq_np.astype(np.float32)

    return posq_np, png_files, n_total_datapts


def get_all_bkgd_pngs():
    '''
    get all of the background images which renders are superimposed upon
    '''

    all_bkgd_pngs = sorted(glob.glob(os.path.join(dirs.bkgd_dir, '*.png')))

    return all_bkgd_pngs


class MyDataset():
    # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

    def __init__(self, png_dir, data_npz):
        self.png_dir = png_dir
        posq_np, png_files, n_total_datapts = get_data(png_dir, data_npz)
        self.png_files = png_files
        self.posq = torch.from_numpy(posq_np)
        #self.transform = torchvision.transforms.Normalize(0,255)

    def __len__(self):
        return len(self.png_files)

    def __getitem__(self, i):
        png_name = self.png_files[i]
        im_i = torchvision.io.read_image(png_name).float()
        #im_i = self.transform(im_i)
        posq_i = self.posq[:,i]
        sample = {'image': im_i, 'posq': posq_i}
        return  sample


class MyNet(nn.Module):

    def __init__(self, dropout_keep_prob):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=3, padding=2)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.conv2 = nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(128*9*9, 4096)
        self.drop1 = nn.Dropout(p=1.0 - dropout_keep_prob)
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(p=1.0 - dropout_keep_prob)
        self.fc3 = nn.Linear(4096, 6)

        self.layers = [
                self.conv1, nn.ReLU(inplace=False),
                self.pool1,
                self.conv2, nn.ReLU(inplace=False),
                self.pool2,
                self.conv3, nn.ReLU(inplace=False),
                self.conv4, nn.ReLU(inplace=False),
                self.conv5, nn.ReLU(inplace=False),
                self.pool3,
                nn.Flatten(),
                self.fc1, nn.ReLU(inplace=False),
                self.drop1,
                self.fc2, nn.ReLU(inplace=False),
                self.drop2,
                self.fc3]


    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = self.pool3(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.drop1(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x

'''
https://discuss.pytorch.org/t/how-to-fix-define-the-initialization-weights-seed/20156/2
'''
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight.data, -1, 1)
        nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight.data, -1, 1)
        nn.init.zeros_(m.bias.data)


def loss_fun(pose_actual, pose_pred):
    '''
    loss function based on pos and quaternions

    pose_actual comes in as postion and quaternion 
    '''

    # get position and rotation portions of estimate 
    pos_pred, s_pred = pose_pred[:,:3], pose_pred[:,3:]
    pos_actual, q_actual = pose_actual[:,:3], pose_actual[:,3:]
    s_actual = so3.quat_to_expcoords(q_actual.T).T

    # positional loss
    loss_pos = torch.sum((pos_actual - pos_pred)**2)

    # angular loss
    dist_s = so3.geodesic_distance_expcoords(s_actual.T, s_pred.T)
    loss_s = torch.sum(dist_s)

    # sum
    loss = pos_scale_factor*loss_pos + rot_scale_factor*loss_s

    return loss
