import os
import time
import glob
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torchsummary

# my modules
import dirs
import config
import my_net
import so3_pytorch as so3

fancy_print = 0 # print additional information at each epoch

def train_net(n_epochs, continue_from_ckpt, restore_epoch, reset_optimizer_params=False):

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

    # loss
    loss_fun = my_net.loss_fun
    if my_net.eps is not None:
        optimizer = torch.optim.Adadelta(net.parameters(), lr=my_net.learning_rate, eps=my_net.eps)
    else:
        optimizer = torch.optim.Adadelta(net.parameters(), lr=my_net.learning_rate)

    # colors for printing
    ansi_blue, ansi_green, ansi_red, ansi_end = \
        '\033[94m', '\033[92m', '\033[91m', '\033[0m'

    # start training from a previous checkpoint
    if continue_from_ckpt:
        start_epoch = restore_epoch + 1
        ckpt = torch.load(my_net.ckpt_file % restore_epoch)
        ckpt_state_dict = ckpt['net_state_dict']
        optimizer_state_dict = ckpt['optimizer_state_dict']

        net.load_state_dict(ckpt_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        if reset_optimizer_params:
            for group in optimizer.param_groups:
                group['lr'] = my_net.learning_rate
                if group['eps'] is not None:
                    group['eps'] = my_net.eps

        # load data from previous training
        data = np.load(my_net.loss_npz)
        epoch_array = data['epoch_array']
        loss_train = data['loss_train']
        loss_test = data['loss_test']

    # start training from scratch
    else:
        start_epoch = 0

        # for loss npz
        epoch_array = np.array([])
        loss_train = np.array([])
        loss_test = np.array([])

    print(torchsummary.summary(net, (3, config.pix_width, config.pix_height)))
    if fancy_print:
        print(ansi_blue + '{:<9}'.format('epoch') +
              ansi_green + '{:<9}'.format('train loss') +
              ansi_green + '{:<9}'.format('rot') +
              ansi_green + '{:<9}'.format('pos') +
              ansi_green + '{:<9}'.format('OB') +
              ansi_red + '{:<9}'.format('test loss') +
              ansi_red + '{:<9}'.format('rot') +
              ansi_red + '{:<9}'.format('pos') +
              ansi_red + '{:<9}'.format('OB') +
              ansi_end)
    else:
        print(ansi_blue + '{:<9}'.format('epoch') +
              ansi_green + '{:<14}'.format('train loss') +
              ansi_red + '{:<14}'.format('test loss') +
              ansi_end)
         
    # training: loop over epochs 
    for epoch in range(start_epoch, n_epochs):
        loss_train_sum, loss_test_sum = 0.0, 0.0
        if fancy_print:
            loss_s_train, loss_s_test = 0.0, 0.0
            loss_p_train, loss_p_test = 0.0, 0.0
            n_ob_train, n_ob_test = 0.0, 0.0

        # training data
        net.train()
        for i, data in enumerate(trainloader):
            ims = data['image']
            posq_actual = data['posq']
            ims = ims.to(device)
            posq_actual = posq_actual.to(device)
            ll = [net.conv1, net.conv2, net.conv3, net.fc1, net.fc2, net.fc3]
            for l in ll:
                if torch.any(torch.isnan(l.weight)):
                    import pdb; pdb.set_trace()
                if torch.any(torch.isnan(l.bias)):
                    import pdb; pdb.set_trace()

            # train
            optimizer.zero_grad()
            posq_pred = net(ims)
            loss = loss_fun(posq_actual, posq_pred)
            if np.isnan(loss.item()): import pdb; pdb.set_trace()
            loss.backward()
            optimizer.step()
            loss_train_sum += loss.item()

            if fancy_print:
                # position and rotation distances
                loss_p_train += torch.sum((posq_actual[:,:3] - posq_pred.detach()[:,:3])**2)
                s_actual = so3.quat_to_expcoords(posq_actual[:,3:].T).T
                s_pred = posq_pred.detach()[:,3:]
                dist_s = so3.geodesic_distance_expcoords(s_actual.T, s_pred.T)
                loss_s_train += torch.sum(dist_s)
                s_pred_norm = torch.norm(s_pred, dim=0)
                n_ob_train += torch.sum(s_pred_norm > np.pi)

        # test data
        with torch.no_grad():
            net.eval()
            for i, data in enumerate(testloader):
                ims = data['image']
                posq_actual = data['posq']
                ims = ims.to(device)
                posq_actual = posq_actual.to(device)
                posq_pred = net(ims)
                loss = loss_fun(posq_actual, posq_pred)
                loss_test_sum += loss.item()

                if fancy_print:
                    # position and rotation distances
                    loss_p_test += torch.sum((posq_actual[:,:3] - posq_pred[:,:3])**2)
                    s_actual = so3.quat_to_expcoords(posq_actual[:,3:].T).T
                    s_pred = posq_pred[:,3:]
                    dist_s = so3.geodesic_distance_expcoords(s_actual.T, s_pred.T)
                    loss_s_test += torch.sum(dist_s)
                    s_pred_norm = torch.norm(s_pred, dim=0)
                    n_ob_test += torch.sum(s_pred_norm > np.pi)

        # save data to file
        loss_train_avg = loss_train_sum/n_train_datapts
        loss_test_avg = loss_test_sum/n_test_datapts
        loss_train = np.append(loss_train, loss_train_avg) 
        loss_test = np.append(loss_test, loss_test_avg) 
        epoch_array = np.append(epoch_array, epoch)
        np.savez(my_net.loss_npz, epoch_array=epoch_array, loss_train=loss_train,
                 loss_test=loss_test)

        # make checkpoint files every so often
        if epoch % my_net.n_epochs_per_checkpoint_save == 0:
            torch.save({'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        my_net.ckpt_file % epoch)

            # delete old checkpoints
            try:
                os.remove(my_net.ckpt_file % (epoch -
                    my_net.n_epochs_per_checkpoint_save * my_net.n_ckpt_to_keep))
            except:
                pass

        # print epoch data
        if fancy_print:
            print(ansi_blue + '{:<9d}'.format(epoch) +
                  ansi_green + '{:<9.3f}'.format(loss_train_avg) +
                  ansi_green + '{:<9.3f}'.format(loss_s_train/n_train_datapts) +
                  ansi_green + '{:<9.3f}'.format(loss_p_train/n_train_datapts) +
                  ansi_green + '{:<9.3f}'.format(n_ob_train/n_train_datapts) +
                  ansi_red + '{:<9.3f}'.format(loss_test_avg) +
                  ansi_red + '{:<9.3f}'.format(loss_s_test/n_test_datapts) +
                  ansi_red + '{:<9.3f}'.format(loss_p_test/n_test_datapts) +
                  ansi_red + '{:<9.3f}'.format(n_ob_test/n_test_datapts) +
                  ansi_end)
        else:
            print(ansi_blue + '{:<9d}'.format(epoch) +
                  ansi_green + '{:<14.3f}'.format(loss_train_avg) +
                  ansi_red + '{:<14.3f}'.format(loss_test_avg) +
                  ansi_end)

        # exit loop if losses have become nan
        if np.isnan(loss_train_avg) and np.isnan(loss_test_avg):
            break
