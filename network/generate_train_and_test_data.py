'''
overlay RGBA images onto backgrounds
'''
import os
import glob
import pickle
import numpy as np

import dirs
import config
import image as ti
import so3
import my_net

def superimpose(render_pkl, render_pngs, bkgd_type, n_bkgds_per_render, save_dir, save_npz):
    '''
    add noise to a render and superimpose it onto a background
    '''

    # get renders info
    with open(render_pkl, 'rb') as input:
        render_props = pickle.load(input)
    n_renders = render_props.n_renders
    pos_renders = render_props.pos
    rot_mat_renders = render_props.rot_mat
    quat_renders = s03.mat_to_quat(rot_mat_renders)
    world_RGB_renders = render_props.world_RGB
    pix_width = config.pix_width
    pix_height = config.pix_height
    
    # get background info
    all_bkgd_pngs = my_net.get_all_bkgd_pngs() 
    n_bkgd_pngs = len(all_bkgd_pngs)

    # number of images
    n_ims = n_renders*n_bkgds_per_render

    # generate random indices to choose background files
    if bkgd_type == 'FROM_FILE':
        #import pdb; pdb.set_trace()
        bkgd_inds = np.random.randint(0, n_bkgd_pngs, size=n_ims)

    # data for all images
    pos = np.full((3, n_ims), np.nan)
    quat = np.full((4, n_ims), np.nan)
    world_RGB = np.full((3, n_ims), np.nan)

    # loop over all renders
    for i in range(n_renders):

        # set filename for render i as filename of render plus a number
        png_i = render_pngs[i]
        path_i, filename_i = os.path.split(png_i)
        save_name_i = filename_i[:-4] + '_%03d' + '.png' # [:-4] removes .png

        # render image
        im = ti.load_im_np(png_i)

        # add noise
        noise_rgb = .05*np.random.uniform(
            low=-1.0, high=1.0, size=(pix_width, pix_height, 3))
        noise_alpha = np.full((pix_width, pix_height, 1), 0.0) # don't add noise to alpha
        noise = np.concatenate((noise_rgb, noise_alpha), axis=2)
        im = np.clip(im + noise, 0.0, 1.0) # add noise and keep values on interval [0, 1]

        # properties of i'th render
        pos_i = pos_renders[:,i]
        quat_i = quat_renders[:,i]
        world_RGB_i = world_RGB_renders[:,i]

        # loop over all backgrounds to overlay the image on
        for j in range(n_bkgds_per_render):

            # fill arrays for all training data
            ij = n_bkgds_per_render*i + j
            pos[:, ij] = pos_i
            quat[:, ij] = quat_i
            world_RGB[:, ij] = world_RGB_i

            # random background image from file
            if bkgd_type=='FROM_FILE':
                bkgd_ind_j = bkgd_inds[ij]
                bkgd_png_j = all_bkgd_pngs[bkgd_ind_j]
                im_bkgd = ti.load_im_np(bkgd_png_j)

            # random noise background
            elif bkgd_type=='RANDOM_NOISE':
                im_bkgd = np.random.uniform(0.0, 1.0, (pix_width, pix_height, 3))

            # white background
            elif bkgd_type=='WHITE':
                im_bkgd = 1.0*np.ones((pix_width, pix_height, 3))

            # random solid colors background
            elif bkgd_type=='SOLID':
                im_bkgd = np.random.uniform(0,1,3)*np.ones((pix_width, pix_height, 3))
            
            # superimpose transparent render on top of background
            im_f = ti.overlay(im, im_bkgd)

            # save combined render/background as a file
            save_file_j = os.path.join(save_dir, save_name_i % j)
            ti.write_im_np(save_file_j, im_f)


        # display progress
        if i==0:
            print('in directory', save_dir)
        if i % 100 == 0:
            print('processed', i, '/', n_renders, 'renders', end="\r")
        if i==n_renders-1:
            print('processed', i+1, '/', n_renders, 'renders')

    if bkgd_type == 'FROM_FILE':
        np.savez(save_npz, pos=pos, quat=quat, bkgd_inds=bkgd_inds, world_RGB=world_RGB)
    else:
        np.savez(save_npz, pos=pos, quat=quat, world_RGB=world_RGB)


def generate_training_data():
    # get renders info
    render_dir = dirs.training_renders_dir
    render_pkl = my_net.renders_train_pkl
    render_pngs = sorted(glob.glob(os.path.join(render_dir, '*.png')))
    save_npz = my_net.training_data_npz

    # background properties
    n_bkgds_per_render = 1
    bkgd_type = 'FROM_FILE'
    #bkgd_type = 'RANDOM_NOISE'
    #bkgd_type = 'WHITE'
    #bkgd_type = 'SOLID'

    # run and save
    save_dir = dirs.training_data_dir
    superimpose(render_pkl, render_pngs, bkgd_type, n_bkgds_per_render, save_dir, save_npz)


def generate_test_data():
    # get renders info
    render_dir = dirs.test_renders_dir
    render_pkl = my_net.renders_test_pkl
    render_pngs = sorted(glob.glob(os.path.join(render_dir, '*.png')))
    save_npz = my_net.test_data_npz

    # background properties
    n_bkgds_per_render = 1
    bkgd_type = 'FROM_FILE'
    #bkgd_type = 'RANDOM_NOISE'
    #bkgd_type = 'WHITE'
    #bkgd_type = 'SOLID'

    # run and save
    save_dir = dirs.test_data_dir
    superimpose(render_pkl, render_pngs, bkgd_type, n_bkgds_per_render, save_dir, save_npz)


generate_training_data()
generate_test_data()
