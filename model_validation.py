#!/usr/bin/env python
# python file:sim_test_SIH.py
# date:2020.2.25
# function:Given a series of object image from simulation, use GGCNN Network to  calculate the grasping candidates for each one

import sys
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import os
# import tkinter as tk
import h5py
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from models.common import post_process_output
from utils.dataset_processing import evaluation
from utils.dataset_processing import grasp, image
from imageio import imread
from skimage.transform import rotate

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


if __name__ == "__main__":

    # args_model = '/home/abb/ggcnn-DQN/ggcnn-master_3/output/models/200525_2219_anglesless5_withoutTnn_rot_12angle_mindistance5_70width_randomFalseData/epoch_11_iou_0.53'
    args_model = '/media/abb/Data/Project/ggcnn_rot/output/models/200525_2219_anglesless5_withoutTnn_rot_12angle_mindistance5_70width_randomFalseData/epoch_24_iou_0.51'
    # args_model = '/media/abb/Data/Project/ggcnn_rot/output/models/200717_1746_ggrot_xgb_all_100/epoch_53_iou_0.59'
    # args_model = '/media/abb/Data/Project/ggcnn_rot/output/models/200717_2339_ggrot_Jacquard/epoch_29_iou_0.83'
    # args_model = '/media/abb/Data/Project/ggcnn_rot/output/models/200717_2150_ggrot_xgb_inforce_all_100/epoch_24_iou_0.31'
    # args_model = '/media/abb/Data/Project/ggcnn_rot/output/models/200717_2316_ggrot_sim_100/epoch_81_iou_0.57'
    net = torch.load(args_model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    print('+++++++++++++++++++++++++++++++++')
    print(args_model)
    print('+++++++++++++++++++++++++++++++++')
    optimizer = optim.Adam(net.parameters())
    net.eval()

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(300),
        transforms.ToTensor()
    ])

    img_save_path = '/media/abb/Data/Data/shapenet_4/'

    img_files = os.listdir(img_save_path)

    plot = 1

    count = 0
    succ_count = 0.0
    succ_rate = 0.0
    for img_array in img_files:
        n1_img = np.load(img_save_path+img_array)
        n1_img = np.clip((n1_img - n1_img.mean()), -1, 1)
        # plt.imshow(n1_img)
        # plt.show()
        obj = np.where(n1_img < -0.01)
        center = [np.mean(obj[0]).astype(int), np.mean(obj[1]).astype(int)]
        output_size = 300
        left = max(0, min(center[1] - output_size // 2, 640 - output_size))
        top = max(0, min(center[0] - output_size // 2, 480 - output_size))
        # crop_img = depth_img[top:top+300,left:left+300]
        depth_img = n1_img[top:top + output_size, left:left + output_size]

        gs = grasp.GraspRectangles()
        # if plot:
        #     plt.imshow(depth_img)
        #     plt.show()

        step = 18
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

        # if plot:
        #     plt.figure(5)
        #     plt_img = depth_img.copy()
        #     plt.imshow(plt_img, alpha=0.8)
        #     cv2.circle(plt_img, (g.center[1], g.center[0]), 2, (0, 0, 255))
        #
        #     cv2.line(plt_img, (int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)),
        #              (int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)),
        #              (255, 0, 0), 2)
        #     cv2.line(plt_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
        #              (255, 0, 0), 2)
        #     cv2.line(plt_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
        #              (255, 0, 0), 2)
        #     plt.imshow(plt_img, alpha=0.8)
        #     plt.show()

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
        # if plot:
        #     section_plt(total_section)

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
        # if plot:
        #     section_plt(main_Section.pixel)
        p2 = line[max(0, main_Section.start - 10)]
        p3 = line[min(len(line) - 1, main_Section.end + 10)]

        g.center = line[main_Section.mid-1]
        g.length = int(((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2) ** 0.5)
        # avail_length.append(g.length)
        g.width = int(g.length / 4)

        if plot:
            plt_img = depth_img.copy()
            plt.imshow(plt_img, alpha=0.8)
            cv2.circle(plt_img, (g.center[1], g.center[0]), 2, (0, 0, 255))
            gr = g.as_gr
            cv2.line(plt_img, (int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)),
                     (int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)),
                     (255, 0, 0), 2)
            cv2.line(plt_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
                     (255, 0, 0), 2)
            cv2.line(plt_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
                     (255, 0, 0), 2)
            plt.imshow(plt_img, alpha=0.8)
            plt.savefig('./output/plt/ggrot-GMD-sim_'+str(count)+'.png')
            plt.ion()
            plt.pause(0.2)

        try:
            succ = input()
        except:
            print('Error,retry')
            succ = input()
        if succ == '1':
            succ_count  = succ_count + 1.0
            print(succ_count)
            succ_rate = succ_count / (count + 1.0)
            print(succ_rate)
        else:
            print(succ_count)
            print(succ_rate)
            succ_rate = succ_count / (count + 1.0)
        count = count + 1
        with open('200717_2316_ggrot_sim_100.txt', 'a') as file:
            file.write('g_num:{} succ:{} succ_count:{} succ_rate:{:.04f} \n'.format(count, succ, succ_count, succ_rate))
        print('g_num:{} succ:{} succ_count:{} succ_rate:{:.04f} '.format(count, succ, succ_count, succ_rate))
