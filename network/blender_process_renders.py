'''
this script is to be run as a blender command:

blender --background --python script_name.py -- data_dir

data_dir: directory containing to_render.pkl, which should contain a
          RenderProperties object
'''
import bpy
import os.path
import sys
import math
import pickle
import numpy as np
import transforms3d as t3d

import dirs
import config

# get arguments, see:
# https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after "--"
data_dir = argv[0]
print('ddd', data_dir)

# load data from pkl file
to_render_pkl = os.path.join(data_dir, 'to_render.pkl')
with open(to_render_pkl, 'rb') as input:
    render_props = pickle.load(input)
render_props.save_dir = data_dir

# model info
blender_models_dir = dirs.blender_models_dir
blend_file = os.path.join(blender_models_dir, render_props.model_name+'.blend')
bpy.ops.wm.open_mainfile(filepath=blend_file)

# object
ob_name = 'all_parts'
ob = bpy.data.objects[ob_name]
ob.rotation_mode = 'QUATERNION'

# camera
camera_name = 'cam0'
cam = bpy.data.cameras.new(camera_name)  # create a new camera
cam_ob = bpy.data.objects.new(camera_name, cam)  # create a new camera object
bpy.context.scene.camera = cam_ob  # set the active camera to be the new camera
cam_ob.location = render_props.cam_pos
cam_ob.rotation_mode = 'QUATERNION'
cam_ob.rotation_quaternion = t3d.quaternions.mat2quat(render_props.cam_rot_mat)

# lens and sensor
f = config.f
bpy.data.cameras[camera_name].lens = f
# set the sensor size to produce the correct angle
# (Blender doesn't work too well when you try to set the angle itself)
bpy.data.cameras[camera_name].sensor_fit = render_props.sensor_fit
bpy.data.cameras[camera_name].sensor_width = 2*f*math.tan(render_props.angle_w/2)
bpy.data.cameras[camera_name].sensor_height = 2*f*math.tan(render_props.angle_h/2)

# scene properties
bpy.data.scenes['Scene'].render.resolution_x = render_props.pix_width
bpy.data.scenes['Scene'].render.resolution_y = render_props.pix_height
bpy.data.scenes['Scene'].render.resolution_percentage = 100
bpy.data.scenes['Scene'].render.image_settings.file_format = 'PNG'
bpy.data.scenes['Scene'].render.image_settings.color_mode = 'RGB'
bpy.data.scenes['Scene'].cycles.film_transparent = False
#bpy.data.scenes['Scene'].render.engine = 'CYCLES'
bpy.data.scenes['Scene'].render.engine = 'BLENDER_EEVEE' # faster than CYCLES
bpy.data.scenes['Scene'].cycles.device = 'GPU'

# lighting energy
if render_props.lighting_energy is not None:
    for light in bpy.data.lights:
        light.energy = render_props.lighting_energy

# loop through poses to generate images
for i in range(render_props.n_renders):

    # different world color?
    if render_props.world_RGB is not None:
        world_RGB_i = render_props.world_RGB[:, i]
        A = np.array([1.0]) # alpha for world RGBA lighting
        RGBA = np.concatenate((world_RGB_i, A))
        bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[0].default_value = RGBA
   
    # transparent background?
    if (render_props.transparent is not None) and (render_props.transparent[i]):
        bpy.data.scenes['Scene'].render.image_settings.color_mode = 'RGBA'
        bpy.data.scenes['Scene'].render.film_transparent = True
    else:
        bpy.data.scenes['Scene'].render.image_settings.color_mode = 'RGB'
        bpy.data.scenes['Scene'].cycles.film_transparent = False

    # object location and rotation
    ob.location = render_props.pos[:,i]
    ob_quat = t3d.quaternions.mat2quat(render_props.rot_mat[:,:,i])
    ob.rotation_quaternion = ob_quat 

    # set the file to save to
    if render_props.image_names is None:
        image_file_name_i = '%06d.png' % i # generic numerical name
    else:
        image_file_name_i = render_props.image_names[i]
    image_file_i = os.path.join(render_props.save_dir, image_file_name_i)
    bpy.data.scenes['Scene'].render.filepath = image_file_i

    # render the image
    bpy.ops.render.render(write_still=True)
