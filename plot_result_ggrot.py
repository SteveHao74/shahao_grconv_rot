import datetime
import os
import sys
import argparse
import logging
import cv2
import math
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt
from torchsummary import summary
import tensorboardX
from utils.visualisation.gridshow import gridshow
from utils.dataset_processing import evaluation
from utils.data import get_dataset
from models import get_network
from models.common import post_process_output
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
from skimage.filters import gaussian
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

class Section():
    def __init__(self,pixel,i):
        self.pixel = np.array(pixel)
        self.sum = np.sum(self.pixel)
        self.end = i
        self.start = i - self.pixel.size
        self.mid = int(i- self.pixel.size/2)

# Calculate depth between two finger and decide whether need to grasp
def myline(startx, starty, endx, endy):
    line = []
    if abs(endy - starty) > abs(endx - startx):
        if endy > starty:
            for y in range(starty, endy):
                x = int((y - starty) * (endx - startx) / (endy - starty)) + startx
                line.append([y, x])
        else:
            for y in range(endy, starty):
                x = int((y - starty) * (endx - startx) / (endy - starty)) + startx
                line.append([y, x])
        return line
    if abs(endy - starty) <= abs(endx - startx):
        if endx > startx:
            for x in range(startx, endx):
                y = int((x - startx) * (endy - starty) / (endx - startx)) + starty
                line.append([y, x])
        else:
            for x in range(endx, startx):
                y = int((x - startx) * (endy - starty) / (endx - startx)) + starty
                line.append([y, x])
        return line

def section_plt(section):
    plt.subplot(1, 1, 1)
    num = len(section)
    width = 0.2
    index = np.arange(num)
    p2 = plt.bar(index, section, width, label='num', color='#87CEFA')
    plt.xlabel('clusters')
    xtick_step = 5
    plt.xticks(range(0, num, xtick_step), range(0, num, xtick_step))
    plt.ylabel('pixel height')
    plt.title('Grasp section distribution')
    plt.show()
    # plt.ion()
    # plt.pause(1)
    return

avail_length = []
def find_grasp(img_array,w=0,plot=False):

    if plot:
        plt.imshow(img_array)
        plt.show()
    depth_img = img_array.copy()

    # depth_img = np.clip((depth_img - 0.891), -1, 1)
    depth_img = gaussian(depth_img, 1.0, preserve_range=True).astype(np.float32)
    step = 12
    with torch.no_grad():
        input_img = []
        for k in range(step):
            depth_img_rot = rotate(depth_img, (np.pi / 2 - k * np.pi / step) / np.pi * 180, center=None,
                                   mode='edge', preserve_range=True).astype(depth_img.dtype)
            depth_img_rot = data_transforms(depth_img_rot)
            # print(depth_img_rot.shape)
            input_img.append(torch.tensor(depth_img_rot).unsqueeze(0).float())
        input_img = torch.cat(input_img)
        # print(input_img.shape)

        xc = input_img.to(device)
        ggcnn_start = time.time()

        pos_output, width_output = net.forward(xc)
        ggcnn_end = time.time()

        print('ggcnn process time = {}'.format(ggcnn_end - ggcnn_start))
        q_img, width_img = post_process_output(pos_output, width_output)

        gs = evaluation.get_best_grasp(q_img,
                                       no_grasps=1,
                                       grasp_width=width_img,
                                       zoom_factor=torch.tensor([1])
                                       )

    g = gs[0]
    gr = g.as_gr

    if plot:
        plt.figure(5)
        plt_img = depth_img.copy()
        plt.imshow(plt_img, alpha=0.8)
        cv2.circle(plt_img, (g.center[1], g.center[0]), 2, (0, 0, 255))

        cv2.line(plt_img, (int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)),
                 (int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)),
                 (255, 0, 0), 2)
        cv2.line(plt_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
                 (255, 0, 0), 2)
        cv2.line(plt_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
                 (255, 0, 0), 2)
        plt.imshow(plt_img, alpha=0.8)
        plt.show()

    plt_img = depth_img.copy()
    # plt.imshow(plt_img)
    # plt.show()
    # p0[u,v]
    p0 = np.array([int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)])
    p1 = np.array([int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)])
    line = myline(p0[0], p0[1], p1[0], p1[1])

    total_section = []

    for i in range(len(line)):
        try:
            pixel_value = -plt_img[line[i][0], line[i][1]]
        except IndexError:
            break
        total_section.append(pixel_value)

    total_section = np.array(total_section)
    # np.save('test_section.npy', total_section)
    if plot:
        section_plt(total_section)

    Sections = []
    sub_section = []
    cur_flag = 0
    count_flag = 0
    bais = 12
    for i in range(total_section.size):
        if total_section[i] > total_section.max()*0.65:
            if cur_flag == 0:
                cur_flag = 1
                sub_section.append(total_section[i])
                continue
            else:
                sub_section.append(total_section[i])
                continue
        if total_section[i] < total_section.max()*0.65:
            if cur_flag == 0:
                if len(sub_section):
                    if count_flag>=bais or i == total_section.size-1:
                        Sections.append(Section(sub_section, i-count_flag-1))
                        sub_section = []
                        count_flag = 0
                    else:
                        count_flag += 1
                    continue
            else:
                cur_flag = 0
                continue

    max_sum = 0
    main_Section = Section([], 0)
    for i in range(len(Sections)):
        if Sections[i].sum > max_sum:
            max_sum = Sections[i].sum
            main_Section = Sections[i]
    if plot:
        section_plt(main_Section.pixel)
    p2 = line[max(0, main_Section.start - 10)]
    p3 = line[min(len(line) - 1, main_Section.end + 10)]

    g.center = line[main_Section.mid-1]
    g.length = int(((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2) ** 0.5)
    # avail_length.append(g.length)
    g.width = int(g.length / 4)

    return g


args_train_dataset = '/home/abb/Downloads/Jacquard/'
# args_train_dataset = '/media/abb/Data/Data/fang'
args_dataset = 'jacquard'
# args_dataset = 'gmnet'
args_ds_rotate = 0
args_batch_size = 1
args_num_workers = 1
args_network = 'ggcnn'

Dataset = get_dataset(args_dataset)
train_dataset = Dataset(args_train_dataset, start=0.9, end=1.0, ds_rotate=args_ds_rotate,
                        random_rotate=False, random_zoom=False,
                        include_depth=True, include_rgb=True)
train_data = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args_batch_size,
    shuffle=True,
    num_workers=args_num_workers
)

fig = plt.figure(figsize=(40, 40))
# plt.title('Training Set')
for idx, (x, y, didx, rot, zoom) in enumerate(train_data):

    rgb_img = train_data.dataset.get_rgb(didx, rot, 0.5, normalise=False)
    depth_img = train_data.dataset.get_depth(didx, rot, zoom)

    # g = find_grasp(depth_img,w=0,plot=False)

    ax = fig.add_subplot(2, 8, idx%16+1)
    ax.imshow(rgb_img)
    # if idx == 0:
    #     ax.set_title('RGB')
    ax.axis('off')

    if idx ==15:
        plt.show()
        break

    # ax = fig.add_subplot(5, 2, 2*idx+2)
    # ax.imshow(rgb_img)
    # g.plot(ax)
    # if idx == 0:
    #     ax.set_title('Optimal Grasp')
    # ax.axis('off')

    # plt.colorbar(plot)

