#!/usr/bin/env python
# python file:find_grasp2.py
# date:2020.1.16
# function:1.Given an image use the width-network to calculate its opening-width
#          2.Use Pyro4 to transmite a grasp planning result (p0,p1,d0,d1,q) for a given img_array

import torch
import Pyro4
import torch.utils.data
import cv2
import matplotlib.pyplot as plt
import numpy as np
from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset
from skimage.transform import rotate
import time
from torchvision import transforms
from skimage.feature import peak_local_max

args_network = '/home/abb/ggcnn-DQN/ggcnn-master_3/output/models/200525_2219_anglesless5_withoutTnn_rot_12angle_mindistance5_70width_randomFalseData/epoch_11_iou_0.53'
net = torch.load(args_network)
device = torch.device("cuda:0")

step = 18

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(300),
    transforms.ToTensor()
])

def find_grasp(img_array):

    crop_img = img_array.copy()
    crop_img = np.clip((crop_img - crop_img.mean()), -1, 1)
    crop_img[np.where(crop_img < -0.01)] *= 1.2
    crop_img = np.clip(crop_img, -0.04, 0.1)
    plt.imshow(crop_img)
    plt.show()
    with torch.no_grad():

        input_img = []
        for k in range(step):
            depth_img_rot = rotate(crop_img, (np.pi / 2 - k * np.pi / step) / np.pi * 180, center=None,
                                   mode='edge', preserve_range=True).astype(crop_img.dtype)
            depth_img_rot = data_transforms(depth_img_rot)
            # print(depth_img_rot.shape)
            input_img.append(torch.tensor(depth_img_rot).unsqueeze(0).float())
        input_img = torch.cat(input_img)
        # print(input_img.shape)

        xc = input_img.to(device)
        ggcnn_start = time.time()

        pos_output, width_output = net.forward(xc)
        ggcnn_end = time.time()
        process_time = ggcnn_end - ggcnn_start
        print('ggcnn process time = {}'.format(process_time))
        q_img, width_img = post_process_output(pos_output, width_output)

        gs = evaluation.get_best_grasp(q_img,
                                       no_grasps=1,
                                       grasp_width=width_img,
                                       zoom_factor=torch.tensor([1])
                                       )

    g = gs[0]

    plt.figure(5)
    plt.imshow(crop_img, alpha=0.8)
    cv2.circle(crop_img, (g.center[1], g.center[0]), 2, (0, 0, 255))

    gr = g.as_gr
    cv2.line(crop_img, (int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)),
             (int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)),
             (255, 0, 0), 2)
    cv2.line(crop_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
             (255, 0, 0), 2)
    cv2.line(crop_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
             (255, 0, 0), 2)

    plt.imshow(crop_img, alpha=0.8)
    plt.show()
    #
    # d_img = xc.cpu().squeeze(1).numpy()
    # q_img = q_img
    #
    # for i in range(step):
    #
    #     d_i = d_img[i]
    #     q_i = q_img[i]
    #
    #     plt.figure(1)
    #     ax1 = plt.subplot(int(step/3), 3, i + 1)
    #     plt.imshow(q_i, alpha=1)
    #     plt.title(str(round(np.max(q_i), 3)))
    #
    #     plt.figure(2)
    #     ax2 = plt.subplot(int(step/3), 3, i + 1)
    #     plt.imshow(d_i, alpha=1)
    # plt.show()

    p0 = np.array([int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)])
    p1 = np.array([int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)])
    d0 = img_array[p0[1],p0[0]]
    d1 = img_array[p1[1],p1[0]]
    dmin = min(d0,d1)-0.005
    dmin = min(d0, d1) - 0.005 - 0.005
    cv2.line(crop_img, (p0[0],p0[1]),
             (p1[0],p1[1]),
             (255, 0, 0), 2)
    cv2.line(crop_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
             (255, 0, 0), 2)
    cv2.line(crop_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
             (255, 0, 0), 2)
    plt.imshow(crop_img)
    plt.show()

    return np.array([p0,p1,dmin,dmin,process_time])





@Pyro4.expose
class GraspServer(object):
    def plan(self, name):
        return find_grasp(name)

if __name__ == "__main__":

    # for i in range(5,12,1):
    #     img_path = '/home/abb/Pictures/npy/' + str(i) + '.npy'
    #     img_array = np.load(img_path)
    #     print(find_grasp(img_array))
    # img_path = '/home/abb/Download/gmnet_robot/npy/007.npy'
    # img_array = np.load(img_path)
    # print(find_grasp(img_array))
    img_path = '/home/abb/Download/gmnet_robot/npy/000.npy'
    img_array = np.load(img_path)
    print(find_grasp(img_array))

    #
    #
    # Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
    # Pyro4.Daemon.serveSimple({GraspServer: 'grasp'}, ns=False, host='', port=6665)








