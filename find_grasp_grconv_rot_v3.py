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
import math
from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset
from skimage.transform import rotate
import time
from torchvision import transforms
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from pathlib import Path

# args_network = '/home/abb/ggcnn-DQN/ggcnn-master_3/output/models/200525_2219_anglesless5_withoutTnn_rot_12angle_mindistance5_70width_randomFalseData/epoch_11_iou_0.53'
# args_model = '/media/abb/Data/Project/ggcnn_rot/output/models/200525_2219_anglesless5_withoutTnn_rot_12angle_mindistance5_70width_randomFalseData/epoch_24_iou_0.51'
# args_model = '/media/abb/Data/Project/ggcnn_rot/output/models/200717_1746_ggrot_xgb_all_100/epoch_53_iou_0.59'
# args_model = '/media/abb/Data/Project/ggcnn_rot/output/models/200717_2339_ggrot_Jacquard/epoch_29_iou_0.83'
# args_model = '/media/abb/Data/Project/ggcnn_rot/output/models/200717_2150_ggrot_xgb_inforce_all_100/epoch_24_iou_0.31'
# args_model = '/media/abb/Data/Project/ggcnn_rot/output/models/200717_2316_ggrot_sim_100/epoch_81_iou_0.57'
# args_model = '/media/abb/Data/Project/ggcnn_rot/output/models/200801_1137_ggrot_jacquard_1_10/epoch_10_iou_0.80'
# args_model = '/media/abb/Data/Project/ggcnn_rot/output/models/200819_1136_ggrot_JAQ_100/epoch_94_iou_0.75'

def get_model(model_path):
    model_path = Path(model_path).resolve()
    max_fn = 0
    max_f = None
    for f in model_path.iterdir():
        fs = f.name.split('_')
        if len(fs) == 4:
            fn = int(fs[1])
            if fn > max_fn:
                max_fn = fn
                max_f = f#这里是想找到最后一次epoch训练的参数结果，也就是使用最新参数
    return max_f

ggrot_path = Path.home().joinpath('Project/grconv_rot')
args_model = get_model(ggrot_path.joinpath('output/models/220222_1159_no_normal_gauss_gmd'))#220211_1507_correct_gauss_gmd#220119_1027_grrot_gmd#200801_1137_ggrot_jacquard_1_10
model_name = "gmd"
net = torch.load(args_model)
device = torch.device("cuda:0")



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
# cal_time = []
def find_grasp(img_array,width,plot=False):
    # ggcnn_start = time.time()
    if plot:
        plt.imshow(img_array)
        plt.show()
    image_r = img_array.copy()

    if np.std(image_r)  == 0:
        print(np.mean(image_r),image_r)
        return  [None,None,None,None,None,None] 
    # import pdb; pdb.set_trace()
    if model_name == 'cor':
        normalize_depth = (image_r-np.mean(image_r)*np.ones(image_r.shape))/np.std(image_r)  * 0.32395524*0.1 #+1.1981683*np.ones(image_r.shape)
    elif model_name == 'gmd' or model_name == 'gmd_tense'or model_name == "single_gmd" :# or 'single_gmd' :
        normalize_depth = (image_r-np.mean(image_r)*np.ones(image_r.shape))/np.std(image_r) * 0.005129744*0.1#*0.1 #+0.69948566*np.ones(image_r.shape)
        # normalize_depth=image_r
        # normalize_depth = (image_r-np.mean(image_r)*np.ones(image_r.shape))/np.std(image_r)  * 0.015 #+1.5*np.ones(image_r.shape) 
    elif model_name == 'jaq':
        normalize_depth = (image_r-np.mean(image_r)*np.ones(image_r.shape))/np.std(image_r)* 0.04099764*0.1 #+1.5008891*np.ones(image_r.shape)    
        # normalize_depth = (image_r-np.mean(image_r)*np.ones(image_r.shape))/np.std(image_r)  * 0.015# +1.5*np.ones(image_r.shape)   
    else:
        normalize_depth = image_r
    print(normalize_depth.std())
    # depth of bullet sim:
    depth_img = np.clip((normalize_depth - normalize_depth.mean()), -1, 1).astype(np.float32)#0.6
    # depth of real owrld

    # depth_img[depth_img<0.7]=0.83
    # depth_img = np.clip((depth_img - 0.83), -1, 1)

    # depth_img = gaussian(depth_img, 1.0, preserve_range=True).astype(np.float32)
    if plot:
        plt.imshow(depth_img)
        plt.show()
    # depth of vrep
    # depth_img = np.clip((depth_img - 0.5), -1, 1)

    step = 18
    print(1)
    with torch.no_grad():
        input_img = []
        for k in range(step):
            depth_img_rot = rotate(depth_img, (np.pi / 2 - k * np.pi / step) / np.pi * 180, center=None,
                                   mode='edge', preserve_range=True).astype(depth_img.dtype)
            depth_img_rot = data_transforms(depth_img_rot)
            # print(depth_img_rot.shape)
            input_img.append(depth_img_rot.clone().detach().unsqueeze(0).float())
        input_img = torch.cat(input_img)
        # print(input_img.shape)

        xc = input_img.to(device)

        print(2)
        pos_output, width_output = net.forward(xc)
        print(3)
        q_img, width_img = post_process_output(pos_output, width_output)
        print(4)
        gs = evaluation.get_best_grasp(q_img,
                                       no_grasps=20,
                                       grasp_width=width_img,
                                       zoom_factor=torch.tensor([1])
             
                               )
        print(gs)
    if gs == []:
        return  [[None,None,None,None,None,None]] 
    # import pdb;pdb.set_trace()
    grasp_array =  []
    for i in range(len(gs)):
        g = gs[i]
        print(g.center,g.angle,g.length,g.width)
        # g.length = width
        # index = 12-int((90-g.angle*180/np.pi)/15)
        index = 18-math.ceil((90-g.angle*180/np.pi)/10)
        print("index:",index)
        gr = g.as_gr

        p0 = np.array([int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)])
        p1 = np.array([int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)])
        print(p0,p1)
        region = 3
        d2 = np.mean(img_array[p0[1]-region:p0[1]+region,p0[0]-region:p0[0]+region]) - 0.006
        d3 = np.mean(img_array[p1[1]-region:p1[1]+region,p1[0]-region:p1[0]+region]) - 0.006
        print(d2,d3)
        grasp_depth = min(d2,d3)

        print([p0,p1,grasp_depth,grasp_depth,g.angle])
        grasp_array.append([p0,p1,grasp_depth,grasp_depth,g.angle,q_img[index]])

    return grasp_array


@Pyro4.expose
class GraspServer(object):
    def plan(self, name,width):
        # np.save(name,'img.npy')
        return find_grasp(name,width)

if __name__ == "__main__":

    # for i in range(5,12,1):
    #     img_path = '/home/abb/Pictures/npy/' + str(i) + '.npy'
    #     img_array = np.load(img_path)
    #     print(find_grasp(img_array))
    # img_path = '/home/abb/Download/gmnet_robot/npy/002.npy'
    # img_array = np.load(img_path)
    # print(find_grasp(img_array,True))
    # a = np.load('/media/abb/Data/Project/ggcnn_rot/npy/000.npy')
    # plt.imshow(a)
    # plt.show()
    # print(find_grasp(a,60,True))
    # model_name =""

    Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
    Pyro4.Daemon.serveSimple({GraspServer: 'grasp'}, ns=False, host='', port=6665)








