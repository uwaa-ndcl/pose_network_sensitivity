# scale and crop images

import image as ti
from network import my_net

#input_dir = '/home/trevor/Downloads/SUN2012/'
input_dir = '/home/trevor/Downloads/SUN397/'
#output_dir = '/home/trevor/Downloads/SUN397_300_by_300/'
output_dir = '/home/trevor/Downloads/SUN397_256_by_256/'
w_des = my_net.pix_width
h_des = my_net.pix_height
ti.scale_and_crop(w_des, h_des, input_dir, output_dir)
