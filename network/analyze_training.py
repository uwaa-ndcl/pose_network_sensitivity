import numpy as np
import torch
import torch.nn as nn

# my modules
import utils
import dirs
import config
import my_net
import so3_pytorch as so3

# restore
restore_epoch = 235

# pytorch train and test datasets
trainset = my_net.MyDataset(
    dirs.training_data_dir, my_net.training_data_npz)
testset = my_net.MyDataset(
    dirs.test_data_dir, my_net.test_data_npz)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=my_net.batch_size, shuffle=True,
    num_workers=my_net.n_workers)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=my_net.batch_size, shuffle=False,
    num_workers=my_net.n_workers)
n_train_datapts = len(trainset)
n_test_datapts = len(testset)

# network & cuda
net = my_net.MyNet(my_net.dropout_keep_prob)
device = config.device
net.to(device)
net.eval()
loss_fun = my_net.loss_fun

# load checkpoint
ckpt = torch.load(my_net.ckpt_file % restore_epoch)
ckpt_state_dict = ckpt['net_state_dict']
net.load_state_dict(ckpt_state_dict)

# iterate over each layer and get the output of each function
x0 = torch.rand((1,3,256,256)).to(device)
y0 = net(x0)
layers = net.layers
n_layers = len(layers)
X = [x0]
for i in range(n_layers):
    f = layers[i]
    X.append(f(X[-1]))

# sanity check
output_error = torch.norm(y0 - X[-1]).item()
print('one-shot v. layer-by-layer error', output_error)

# calculate Lipschitz bound
lipschitz_bound = 1
n_layers = len(layers)
for i in range(n_layers):
    layer = layers[i]

    # convolution
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        spec_norm, V = utils.affine_norm(layer, X[i].shape)
        L = spec_norm.item()

    # max pooling
    elif isinstance(layer, nn.MaxPool2d):
        k = layer.kernel_size 
        s = layer.stride 
        L = np.ceil(k/s)

    # relu
    elif isinstance(layer, nn.ReLU):
        L = 1

    # flatten
    elif isinstance(layer, nn.Flatten):
        L = 1

    # dropout
    elif isinstance(layer, nn.Dropout):
        L = 1

    # some other type of layer
    else:
        print('ERROR: this loop is not implemented for layers of type', layer)

    lipschitz_bound *= L

# split last layer into position and rotation components,
# and compute lipschitz constant for each 
last_layer = layers[-1]
W = last_layer.weight.detach()
W_pos = W[:3,:]
W_rot = W[3:,:]
W_pos_norm = torch.linalg.norm(W_pos, ord=2).item()
W_rot_norm = torch.linalg.norm(W_rot, ord=2).item()
lipschitz_bound_rot = lipschitz_bound*W_pos_norm 
lipschitz_bound_pos = lipschitz_bound*W_rot_norm 

print('position Lipschitz bound:', lipschitz_bound_rot)
print('rotation Lipschitz bound:', lipschitz_bound_pos)

# loop over both train and test loaders
loaders = [trainloader, testloader]
n_datapts = [n_train_datapts, n_test_datapts]
loader_names = ['TRAINING DATA', 'TEST DATA']
with torch.no_grad():
    for i, loader in enumerate(loaders):
        loss = 0
        loss_p = 0
        err_p = 0
        err_s = 0
        n_ob = 0
        for data in loader:
            # get estimates and true values
            #print('a')
            ims = data['image']
            posq_actual = data['posq']
            ims = ims.to(device)
            posq_actual = posq_actual.to(device)
            posq_pred = net(ims)
            loss_batch = loss_fun(posq_actual, posq_pred)
            loss += loss_batch.item()
            #print('b')

            # rotation error
            s_actual = so3.quat_to_expcoords(posq_actual[:,3:].T).T
            s_pred = posq_pred[:,3:]
            dist_s = so3.geodesic_distance_expcoords(s_actual.T, s_pred.T)
            err_s += torch.sum(dist_s).item()

            # position error
            err_p += torch.sum(torch.norm(posq_actual[:,:3] - posq_pred[:,:3], dim=1)).item()
            loss_p += torch.sum((posq_actual[:,:3] - posq_pred[:,:3])**2).item()

            # "out of bounds" parameters which have norm greater than pi
            s_pred_norm = torch.norm(s_pred, dim=0)
            #print('d')
            n_ob += torch.sum(s_pred_norm > np.pi).item()

        print('\n' + loader_names[i])
        print('average cost function error:', loss/n_datapts[i])
        print('average angular distance error:', err_s/n_datapts[i], '(rad)')
        print('average cost function error, position:', loss_p/n_datapts[i])
        print('average position error:', err_p/n_datapts[i], '(m)')
        print('estimates out of bounds:', 100*n_ob/n_datapts[i], '%')
