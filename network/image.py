import os
import cv2
import glob
import warnings
import imageio
import numpy as np


def load_im_np(filename):
    '''
    load 1 image to a numpy array
    resulting array will have entries on the interval [0, 1] of type float32
    '''
        
    a = imageio.imread(filename) # will load integers on interval [0,255]
    b = np.asarray(a)
    c = np.float32(b)
    d = c/255.0 # put elements on interval [0,1]
            
    return d


def write_im_np(filename, im):
    '''
    take an array with entries on the interval [0, 1] and save it as an image
    '''
    warnings.simplefilter('always')
    if np.any(im < 0) or np.any(im > 1):
        warnings.warn('image has values outside of the interval [0,1], ')
        #warnings.warn('image has values outside of the interval [0,1], ' + \
                      #'saturating...')
        im = np.clip(im, 0, 1)

    im = np.uint8(im * 255.0)
    imageio.imwrite(filename, im)


def overlay(im_overlay, im_background, mode='0to1'):
    '''
    overlay an RGBA image onto an RGB background image

    INPUTS
    im_overlay: RGBA image with values on interval [0, 255]
    im_background: RGB image with values on interval [0, 255]
    mode: '0to1' (image elements are from 0 to 1),
          '0to255' (image elements are from 0 to 255)
    '''

    alph = im_overlay[:,:,3]

    if mode == '0to255':
        alph = alph/255.0 # now alpha values are in interval [0,1]

    im_overlay_scaled = np.stack((alph*im_overlay[:,:,0],
                                  alph*im_overlay[:,:,1],
                                  alph*im_overlay[:,:,2]), 2)
    im_background_scaled = np.stack(((1 - alph)*im_background[:,:,0],
                                     (1 - alph)*im_background[:,:,1],
                                     (1 - alph)*im_background[:,:,2]), 2)
    im_out = im_overlay_scaled + im_background_scaled

    return im_out


def scale_and_crop(w_des, h_des, input_dir, output_dir):
    '''
    take an image and reduce it to the desired size (by cropping the centermost bit)
    inputs:   w_des and h_des: desired number of width and height pixels,
                  both should be multiples of 2!
              input dir: directory of images to be converted
              output_dir: directory of images for saving
    '''

    if w_des%2 != 0 or h_des%2 != 0:
        raise ValueError('Desired width and height must be multiples of 2!')

    aspect_ratio_des = w_des/h_des

    # get list of filenames in input directory and subdirectories
    filename_list = []
    basename_list = []
    all_filenames = glob.iglob(os.path.join(input_dir, '**/*.jpg'), recursive=True)
    for filename in all_filenames:
        filename_list.append(filename)
        basename = os.path.basename(filename)
        basename_list.append(basename)

    # check if any of the files have the same name
    n_files = len(filename_list)
    n_unique_files = len(set(basename_list))
    print('there are', n_files, 'total files, and', n_unique_files, 'unique file names')
    input('press Enter to continue...')

    # iterate over files
    for i in range(n_files):

        filename = filename_list[i]
        basename = os.path.basename(filename)
        
        # read image
        im = cv2.imread(filename) # loads image as BGR!

        # if image is corrupted, im will be None type
        if im is None:
            print(filename, 'IS CORRUPTED!!!')
            continue
        else:
            print(filename)
            h, w, c = im.shape # (row, column, channel) note the order!!!

        # if image is too small, ignore it
        if w < w_des or h < h_des:
                continue

        aspect_ratio = w/h

        # image fatter than desired
        if aspect_ratio > aspect_ratio_des:
            w_scl = int(round(h_des*aspect_ratio)) # scaled image width
            im_scl = cv2.resize(im, dsize=(w_scl, h_des), interpolation=cv2.INTER_CUBIC) # note the dsize order!
            w_scl_mid = int(round(w_scl/2)) # middle of scaled width
            w_des_hlf = int(w_des/2) # half of desired width
            im_scl_crp = im_scl[:, w_scl_mid-w_des_hlf:w_scl_mid+w_des_hlf, :] # crop center

        # image skinnier than desired
        else:
            h_scl = int(round(w_des/aspect_ratio))
            im_scl = cv2.resize(im, dsize=(w_des, h_scl), interpolation=cv2.INTER_CUBIC)
            h_scl_mid = int(round(h_scl/2))
            h_des_hlf = int(h_des/2)
            im_scl_crp = im_scl[h_scl_mid-h_des_hlf:h_scl_mid+h_des_hlf, :, :]

        # save new file
        filename_new = os.path.join(output_dir, basename[:-4] + '.png')
        cv2.imwrite(filename_new, im_scl_crp)
