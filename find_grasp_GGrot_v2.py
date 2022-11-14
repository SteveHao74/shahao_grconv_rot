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
from skimage.filters import gaussian
from skimage.feature import peak_local_max

args_network = '/media/abb/Data/Project/ggcnn_rot/output/models/200525_2219_anglesless5_withoutTnn_rot_12angle_mindistance5_70width_randomFalseData/epoch_24_iou_0.51'
# args_network = '/home/abb/ggcnn-DQN/ggcnn-master_3/output/models/200525_2219_anglesless5_withoutTnn_rot_12angle_mindistance5_70width_randomFalseData/epoch_11_iou_0.53'
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

    # depth_img = np.clip((depth_img - 0.5), -1, 1)
    # # depth_img = np.clip((depth_img - 0.891), -1, 1)
    depth_img[depth_img<0.7]=0.825
    depth_img = np.clip((depth_img - 0.825), -1, 1)
    depth_img = gaussian(depth_img, 1.0, preserve_range=True).astype(np.float32)
    if plot:
        plt.imshow(depth_img)
        plt.show()
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
        if total_section[i] > total_section.max()*0.35:
            if cur_flag == 0:
                cur_flag = 1
                sub_section.append(total_section[i])
                continue
            else:
                sub_section.append(total_section[i])
                continue
        if total_section[i] < total_section.max()*0.35:
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
    p2 = line[max(0, main_Section.start - 16)]
    p3 = line[min(len(line) - 1, main_Section.end + 16)]

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
        plt.show()


    # for i in range(main_Section.start, main_Section.end, 1):
    #     if img_array[line[i][0], line[i][1]] < grasp_depth:
    #         grasp_depth = img_array[line[i][0], line[i][1]]
    average_depth = 0
    for i in range(main_Section.start, main_Section.end, 1):
        average_depth +=  img_array[line[i][0], line[i][1]]
    average_depth = average_depth / (main_Section.end - main_Section.start)
    d0 = average_depth
    d1 = d0
    region = 3
    d2 = np.mean(img_array[p2[0]-region:p2[0]+region,p2[1]-region:p2[1]+region]) - 0.006
    d3 = np.mean(img_array[p3[0]-region:p3[0]+region,p3[1]-region:p3[1]+region]) - 0.006

    dmin = d0 + 0.03

    print(dmin)
    print(min(dmin,d2))
    print(min(dmin,d3))
    grasp_depth = min(min(dmin,d2),min(dmin,d3))
    print(avail_length)
    # print(max(avail_length))
    # return np.array([p2[::-1], p3[::-1], grasp_depth, grasp_depth, [g.length, max(avail_length)]])

    # return np.array([p2[::-1],p3[::-1],grasp_depth,grasp_depth,g.length])
    return np.array([p2[::-1],p3[::-1],grasp_depth,grasp_depth,g.angle])





@Pyro4.expose
class GraspServer(object):
    def plan(self, name,width):
        np.save('saved.npy',name)
        return find_grasp(name,width)

if __name__ == "__main__":

    # for i in range(5,12,1):
    #     img_path = '/home/abb/Pictures/npy/' + str(i) + '.npy'
    #     img_array = np.load(img_path)
    #     print(find_grasp(img_array))
    img_path = 'npy/001.npy'
    img_array = np.load(img_path)
    print(find_grasp(img_array,plot=False))
    img_path = 'npy/000.npy'
    img_array = np.load(img_path)
    print(find_grasp(img_array,plot=False))
    img_path = 'npy/002.npy'
    img_array = np.load(img_path)
    print(find_grasp(img_array,plot=False))
    img_path = 'npy/003.npy'
    img_array = np.load(img_path)
    print(find_grasp(img_array,plot=False))


    # Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
    # Pyro4.Daemon.serveSimple({GraspServer: 'grasp'}, ns=False, host='', port=6666)








