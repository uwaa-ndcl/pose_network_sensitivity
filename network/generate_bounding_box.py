'''
this script is to be run as a blender command:

blender --background --python script_name.py
'''
import bpy
import numpy as np
import dirs

blend_file = '/home/trevor/simple_pose_net/blender_models/soup_can.blend'
#blend_file = '/home/trevor/Downloads/cone.blend'
bpy.ops.wm.open_mainfile(filepath=blend_file)

# object
ob_name = 'all_parts'
ob = bpy.data.objects[ob_name]
ob.rotation_mode = 'AXIS_ANGLE'
ob.rotation_axis_angle = [0,1,0,0]
ob.location = [2,3,4]

verts = np.array([v.co for v in bpy.data.objects['all_parts'].data.vertices])

x_min = np.min(verts[:,0])
x_max = np.max(verts[:,0])
y_min = np.min(verts[:,1])
y_max = np.max(verts[:,1])
z_min = np.min(verts[:,2])
z_max = np.max(verts[:,2])
xyz_min_max = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max, 'z_min': z_min, 'z_max': z_max}

# points
'''
                    5------6
                   /|     /|         z
                  / |    / |         |
                 8------7  |         |
                 |  |   |  |         |
                 |  1---|--2         *------y 
                 | /    | /         /
                 |/     |/         /
                 4------3         x
'''
p1 = np.array([x_min, y_min, z_min])
p2 = np.array([x_min, y_max, z_min])
p3 = np.array([x_max, y_max, z_min])
p4 = np.array([x_max, y_min, z_min])
p5 = np.array([x_min, y_min, z_max])
p6 = np.array([x_min, y_max, z_max])
p7 = np.array([x_max, y_max, z_max])
p8 = np.array([x_max, y_min, z_max])
p = np.array([p1,p2,p3,p4,p5,p6,p7,p8]).T

save_npz = '/home/trevor/Downloads/bounding_box.npz'
np.savez(save_npz, xyz_min_max=xyz_min_max, p=p)
