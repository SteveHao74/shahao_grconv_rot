import numpy as np
import random
import time
from time import sleep
import PIL
import cv2
import matplotlib.pyplot as plt
import math
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.dataset_processing import grasp, image
# from torchvision import transforms
from torchvision import datasets, models, transforms
from models.common import post_process_output
from utils.dataset_processing import evaluation
from skimage.feature import peak_local_max


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load the first model
model_1 = models.resnet18(pretrained=True)
# model_ft = models.ResNet()
num_ftrs = model_1.fc.in_features
model_1.fc = nn.Linear(num_ftrs, 1)
model_1.load_state_dict(torch.load('/home/abb/gg_cnn/gripper_classification/models/Gripper8Class_48_0.7460567823343849.pt', map_location='cuda:0'))
model_1 = model_1.to(device)

# load the second model
args_model = '/home/abb/ggcnn-DQN/ggcnn-master_3/output/models/200525_2219_anglesless5_withoutTnn_rot_12angle_mindistance5_70width_randomFalseData/epoch_11_iou_0.53'
model_2 = torch.load(args_model)
model_2 = model_2.to(device)

# load the third model
model_3 = models.resnet18(pretrained=True)
# model_2 = models.resnet34(pretrained=True)
num_ftrs = model_3.fc.in_features
model_3.fc = nn.Linear(num_ftrs, 2)
model_3.load_state_dict(torch.load('/home/abb/gg_cnn/model/grasp_88_standard.pt', map_location='cuda:0'))
model_3 = model_3.to(device)

composed1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop((300, 300)),
    transforms.ToTensor(),
])
composed2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


plt_flag = 1

def find_grasp(img_array,bais=10):
    #deal with the input array
    print('-----------------------')
    print("Image array recevied.")
    img_array_ = img_array.copy()
    if plt_flag:
        plt.imshow(img_array_)
        plt.show()

    depth_img = img_array.copy()
    depth_img[np.where(depth_img < -0.01)] *= 1.2
    depth_img = np.clip(depth_img, -0.03, 0.1)
    plt.figure(3)
    plt.imshow(depth_img)
    plt.show()

    obj = np.where(depth_img < -0.01)
    center = [np.mean(obj[0]).astype(int), np.mean(obj[1]).astype(int)]
    #
    output_size = 300
    left = max(0, min(center[1] - output_size // 2, 640 - output_size))
    top = max(0, min(center[0] - output_size // 2, 480 - output_size))
    # crop_img = depth_img[top:top+300,left:left+300]
    crop_img = depth_img[top:top + output_size, left:left + output_size].copy()
    plt.figure(4)
    plt.imshow(crop_img)
    plt.show()
    obj = np.where(crop_img < -0.01)
    print('obj.size={}'.format(obj[0].size))

    # with torch.no_grad():
    #     xc = torch.tensor(crop_img).unsqueeze(0).unsqueeze(0).float().to(device)
    #     pos_output, width_output = model_2.forward(xc)
    #     q_img, width_img = post_process_output(pos_output, width_output)
    #     plt.imshow(q_img)
    #     plt.show()
    #     vu = peak_local_max(depth_img, min_distance=5, threshold_abs=0.0, num_peaks=1)
    #     print(vu)


    n2_img = (crop_img * 255).astype(np.uint8)
    n2_img = np.expand_dims(n2_img, axis=2)
    n2_img = np.concatenate((n2_img, n2_img, n2_img), axis=2)
    plt.imshow(n2_img)
    plt.show()
    input = composed2(n2_img).unsqueeze(0).to(device)
    with torch.no_grad():
        width = int(model_1(input).item()*100)
        print(width)


    pos_sample = 15
    ang_sample = 4
    grip_len = width-bais


    img_list_flag = 1
    sampleNo = 2
    mu = 0
    sigma = grip_len / 3
    vu_list = []
    img_list = []
    quality = 0
    index_max = 0
    while (1):
        # grip_len += 1
        with torch.no_grad():

            vu_0 = center[0] - top
            vu_1 = center[1] - left
            vu_list.append((vu_0, vu_1))

            for i in range(-18, 17, ang_sample):
                u = vu_1
                v = vu_0
                a = -5 * i
                M = cv2.getRotationMatrix2D((u, v), a, 1)

                ro_n2 = cv2.warpAffine(n2_img, M, (300, 300))
                crop_n2 = ro_n2[int(v - grip_len / 2):int(v + grip_len / 2), int(u - grip_len / 2):int(u + grip_len / 2)]
                crop_n2 = composed2(crop_n2)
                crop_n2 = crop_n2.unsqueeze(0)
                img_list.append(crop_n2)

            for k in range(pos_sample-1):
                np.random.seed(random.randint(0, 1000))
                s = np.random.normal(mu, sigma, sampleNo)
                vu_0 = center[0] - top + int(s[0])
                vu_1 = center[1] - left + int(s[1])
                vu_list.append((vu_0, vu_1))
                for i in range(-18, 17, ang_sample):
                    u = vu_1
                    v = vu_0
                    a = -5 * i
                    M = cv2.getRotationMatrix2D((u, v), a, 1)

                    ro_n2 = cv2.warpAffine(n2_img, M, (300, 300))
                    crop_n2 = ro_n2[int(v - grip_len / 2):int(v + grip_len / 2), int(u - grip_len / 2):int(u + grip_len / 2)]
                    crop_n2 = composed2(crop_n2)
                    crop_n2 = crop_n2.unsqueeze(0)
                    img_list.append(crop_n2)

            inputs = torch.cat(img_list).to(device)
            outputs = model_3(inputs)
            outputs = F.softmax(outputs, dim=1).cpu().detach().numpy()[:, 1]

            # ang_num = 36 // ang_sample
            # for i in range(pos_sample):
            #     print('Sample point:'+str(i+1))
            #     plt.figure(i+1)
            #     for j in range(ang_num):
            #         ax = plt.subplot(3,int(ang_num//3),j+1)
            #         k = i*ang_num+j
            #         print(outputs[k])
            #         plt.imshow(img_list[k].cpu().detach().squeeze().numpy()[0])
            #     plt.show()

            index_max = outputs.argmax()
            quality = outputs[index_max]
            print('max index = {},quality = {}'.format(index_max,quality ))

        if quality>0.93:
            inp=img_list[index_max].squeeze(0)
            inp = inp.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            if plt_flag:
                plt.imshow(inp)
                plt.show()

            g = grasp.Grasp(vu_list[(index_max // int((36/ang_sample)))],-math.pi/2 + (index_max % (36/ang_sample)) * math.pi/(36/ang_sample),grip_len,grip_len/3)
            # crop_img = n2_img.copy()
            # cv2.circle(crop_img, (g.center[1], g.center[0]), 2, (0, 0, 255))
            # gr = g.as_gr
            # p0 = np.array([int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)])
            # p1 = np.array([int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)])
            # d0 = img_array_[p0[1],p0[0]]
            # d1 = img_array_[p1[1],p1[0]]
            # dmin = min(d0,d1)-0.005
            # # print(dmin)
            # if plt_flag:
            #     cv2.line(crop_img, (p0[0],p0[1]),
            #              (p1[0],p1[1]),
            #              (255, 0, 0), 2)
            #     cv2.line(crop_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
            #              (255, 0, 0), 2)
            #     cv2.line(crop_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
            #              (255, 0, 0), 2)
            #
            # if plt_flag:
            #     plt.imshow(crop_img)
            #     plt.show()
            # if quality>0.85:
            #     print(np.array([p0,p1,dmin,dmin,quality]))
            #     return np.array([p0,p1,dmin,dmin,quality])
            # else:
            #     grip_len = int(grip_len*1.1)
            #     print('gripper length = {}'.format(grip_len))
            plt.imshow(crop_img, alpha=0.8)
            cv2.circle(crop_img, (g.center[1], g.center[0]), 2, (0, 0, 255))

            gr = g.as_gr
            cv2.line(crop_img,
                     (int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)),
                     (int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)),
                     (255, 0, 0), 2)
            cv2.line(crop_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
                     (255, 0, 0), 2)
            cv2.line(crop_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
                     (255, 0, 0), 2)

            plt.imshow(crop_img, alpha=0.8)
            plt.show()

            return
        else:
            vu_list = []
            img_list = []
            quality = 0
            index_max = 0

# dir_root = '/home/abb/gg_cnn/gripper_classification/gripper_8_modified/val'
# # depth_img = 0.00123 - np.load(dir_root+'1.npz')['d_img']
# png_file = glob.glob(os.path.join(dir_root,'*','*.png'))
# png = cv2.imread(png_file[0])
# plt.imshow(png)
# plt.show()

dir_root = '/home/abb/Pictures/Dataset_SIH/GGCNN_Gazebo/'
# depth_img = 0.00123 - np.load(dir_root+'1.npz')['d_img']
depth_img = np.load(dir_root+'12.npz')['d_img']

find_grasp(depth_img,0)
