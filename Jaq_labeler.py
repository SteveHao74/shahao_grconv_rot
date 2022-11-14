import os
import matplotlib.pyplot as plt
from imageio import imread


f_path = '/media/abb/Data/Data/Jacquard_Dataset/Jacquard_Dataset_6/54667ba816ea03dd738e43095496b061/'
depth_fname = os.path.join(f_path,'3_54667ba816ea03dd738e43095496b061_perfect_depth.tiff')
# depth_fname = os.path.join(f_path,'3_54667ba816ea03dd738e43095496b061_stereo_depth.tiff')
depth_img = imread(depth_fname)
plt.imshow(depth_img)
plt.show()
