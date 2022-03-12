'''
utilities for computing the matrix norm of convolution layers
'''
import copy
import torch
import torch.nn as nn

import config
device = config.device

def conv_trans_from_conv(conv):
    '''
    create a torch.nn.ConvTranspose2d() layer based on a torch.nn.Conv2d()
    layer (conv)
    '''

    conv_trans = nn.ConvTranspose2d(
            conv.out_channels,
            conv.in_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False)
    weight = conv.weight
    conv_trans.weight = torch.nn.Parameter(weight)

    return conv_trans


def affine_norm(func, input_shape, n_iter=100):
    '''
    The largest singular value of matrix M can be found by taking the square
    root of largest eigenvlaue of the matrix P = M.T @ M. The largest
    eigenvalue of matrix M (which is the square of the largest singular value)
    can be found with a power iteration. The matrix P can also be found by
    applying a convolution operator to the image, and then applying a
    transposed convolution on that result.

    Note that since we're using a power iteration, we are applying the
    operation:

    func: function, either nn.Conv2d or nn.Linear
    input_shape for conv (shape of input array): =  batch, chan, H, W
    n_iter: number of iterations
    '''

    ########## conv2d ##########
    if isinstance(func, nn.Conv2d):

        # create conv trans layer
        conv = func
        conv_trans = conv_trans_from_conv(conv)

        # create new conv layer (which will have no bias)
        conv_no_bias = copy.deepcopy(conv)
        conv_no_bias.bias = None

        # determine batch size from zero_output_inds variable
        b, ch, n_row, n_col = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        # power iteration
        #torch.manual_seed(0)
        v = torch.rand(b*ch*n_row*n_col)
        v = v.to(device)
        for i in range(n_iter):
            with torch.no_grad(): # this prevents out of memory errors
                # apply A
                V = torch.reshape(v, (b,ch,n_row,n_col)) # reshape to 4D array
                C1 = conv_no_bias(V) # output shape: (batch, out chan, H, W)

                # apply A.T
                C2 = conv_trans(C1, output_size=(b,ch,n_row,n_col))
                c2 = C2.view(b,-1) # reshape to 1D array

                # normalize over each batch
                v = nn.functional.normalize(c2, dim=1)

        norm = torch.norm(c2, dim=1) # largest eigenvalue of M.T @ M
        spec_norm = torch.sqrt(norm) # largest singular value of M

    ########## fully-connnected ##########
    elif isinstance(func, nn.Linear):

        fc = func
        m,n = fc.weight.shape

        # create conv trans layer
        #conv_trans = conv_trans_from_conv(conv)
        fc_trans = copy.deepcopy(fc)
        fc_trans.weight = torch.nn.Parameter(fc_trans.weight.T)
        fc_trans.bias = None

        # create new conv layer (which will have no bias)
        fc_new = copy.deepcopy(fc)
        fc_new.bias = None

        # spectral norm of function
        b = 1

        # power iteration
        V = torch.rand(b,n)
        V = V.to(device)
        for i in range(n_iter):
            with torch.no_grad(): # this prevents out of memory errors
                # apply A
                C1 = fc_new(V)

                # apply A.T
                C2 = fc_trans(C1)

                # normalize over each batch
                V = nn.functional.normalize(C2, dim=1)

        norm = torch.norm(C2, dim=1) # largest eigenvalue of M.T @ M
        spec_norm = torch.sqrt(norm) # largest singular value of M

    return spec_norm, V
