import os

# background images for superimposing
#bkgd_dir = '/home/trevor/Downloads/val2017_256_by_256/'
bkgd_dir = '/home/trevor/Downloads/SUN397_256_by_256/'

# highest level dirs
blender_models_dir = 'blender_models/'
data_dir = 'data/'

# data subdirs
ckpt_dir = os.path.join(data_dir, 'checkpoints')
training_renders_dir = os.path.join(data_dir, 'training_renders')
test_renders_dir = os.path.join(data_dir, 'test_renders')
training_data_dir = os.path.join(data_dir, 'training_data')
test_data_dir = os.path.join(data_dir, 'test_data')
