import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate


dir_root = '/home/abb/Pictures/Dataset_SIH/GGCNN_rot2/'
file = os.listdir(dir_root)
#
# print()
# for f in file:
#     npz = np.load(dir_root+f)
#     d_img = npz['d_img']
#     q_img = npz['q_img']
#     for i in range(9):
#         d_i = d_img[i]
#         q_i = q_img[i]
#         plt.imshow(d_i,alpha=1)
#         plt.show()    
#         plt.imshow(q_i,alpha=0.8)
#         plt.show()
#
#     print()

npz = np.load(dir_root+'10.npz')
d_img = npz['d_img']
q_img = npz['q_img']
gt = npz['q_gt']
gs = npz['gs']


plt.imshow(d_img[6])
plt.show()
plt.imshow(gt)
plt.show()
plt.imshow(gs)
plt.show()

# d_img_rot = rotate(d_i, (np.pi/3) / np.pi * 180, center=None, mode='edge', preserve_range=True).astype(
#     d_i.dtype)
for i in range(12):

    d_i = d_img[i]
    q_i = q_img[i]
    # print(q_i.max)
    plt.figure(1)
    ax1 = plt.subplot(4, 3, i+1)
    plt.imshow(q_i,alpha=1)
    plt.title(str(round(np.max(q_i),3)))

    plt.figure(2)
    ax2 = plt.subplot(4, 3, i+1)
    plt.imshow(d_i,alpha=1)

    # plt.show()
plt.show()