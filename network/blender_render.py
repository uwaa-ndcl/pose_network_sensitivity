import os
import sys
import pickle
import pkgutil
import pathlib
import subprocess
import numpy as np
import transforms3d as t3d

import config
import image as ti

class RenderProperties:

    def __init__(self):
        # name of .blend file
        self.model_name = 'soup_can'

        # list of names of images, if none then images will be named
        # 000000.png, 000001.png, ...
        self.image_names = None

        # directory to save output images, etc.
        self.save_dir = None # this will be filled later

        # object
        #self.ob = None # this will be set inside of Blender
        self.n_renders = 1
        self.pos = np.array([0,0,0])[:,np.newaxis] # size (3, n_renders)
        self.rot_mat = np.eye(3)[:,:,np.newaxis] # size (3, 3, n_renders)

        # world lighting, size (3, n_renders)
        self.world_RGB = None  

        # lighting energy (sometimes called power in Blender) of all lights
        self.lighting_energy = None

        # transparent background?
        self.transparent = None # list of booleans, size (n_renders)

        # camera
        #self.cam_ob = None # this will be set inside of Blender
        self.cam_pos = [0, 0, 0]
        self.cam_rot_mat = t3d.euler.euler2mat(np.pi/2, 0, 0, axes='sxyz')
        #self.pix_width = 640
        self.pix_width = config.pix_width
        #self.pix_height = 480
        self.pix_height = config.pix_height
        self.sensor_fit = 'AUTO'
        self.angle_w = 2*np.arctan(18/50) # Blender default
        self.angle_h = 2*np.arctan(18/50)


def blender_render(render_dir):
    '''
    call a blender command which will generate renders in render_dir
    '''

    # get path to render script, and add
    main_dir = pathlib.Path(__file__).parent.absolute()
    render_script = os.path.join(main_dir, 'blender_process_renders.py')

    # run blender command
    blender_cmd = 'blender --background --python-use-system-env --python ' \
                  + render_script + ' -- ' + render_dir
    subprocess.run([blender_cmd], shell=True)


def soup_gen(p, R, save_dir, transparent=None,
                    lighting_energy=20.0, world_RGB=np.array([.0, .0, .0])):

    # render properties
    n_ims = p.shape[1]
    to_render_pkl = os.path.join(save_dir, 'to_render.pkl')
    render_props = RenderProperties()
    render_props.n_renders = n_ims
    render_props.pos = p
    render_props.rot_mat = R
    render_props.world_RGB = np.repeat(world_RGB[:,np.newaxis], n_ims, axis=1)
    render_props.lighting_energy = lighting_energy
    render_props.transparent = transparent
    with open(to_render_pkl, 'wb') as output:
        pickle.dump(render_props, output, pickle.HIGHEST_PROTOCOL)
    blender_render(save_dir)
