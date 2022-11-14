import argparse
import logging

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

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(300),
    transforms.ToTensor()
])

dir_root = '/home/abb/Pictures/Dataset_SIH/GGCNN_Gazebo/'
# depth_img = 0.00123 - np.load(dir_root+'1.npz')['d_img']
depth_img = np.load(dir_root+'0.npz')['d_img']

# png_file = '/home/abb/gg_cnn/gripper_classification/gripper_8_modified/val/34.0/2061__20_donut_1.5_1_1.png'
# png_f = cv2.imread(png_file)
# plt.imshow(png_f)
# plt.show()
#
# depth_img = png_f.transpose(2,0,1)[0]/256

depth_img[np.where(depth_img<-0.01)] *= 1.2
depth_img = np.clip(depth_img,-0.03,0.1)
plt.figure(3)
plt.imshow(depth_img)
plt.show()

obj = np.where(depth_img<-0.01)
center = [np.mean(obj[0]).astype(int),np.mean(obj[1]).astype(int)]
#
output_size = 300
left = max(0, min(center[1] - output_size // 2, 640 - output_size))
top = max(0, min(center[0] - output_size // 2, 480 - output_size))
# crop_img = depth_img[top:top+300,left:left+300]
crop_img = depth_img[top:top+output_size,left:left+output_size].copy()
plt.figure(4)
plt.imshow(crop_img)
plt.show()
obj = np.where(crop_img<-0.01)
print('obj.size={}'.format(obj[0].size))
#
flag = 0
if obj[0].size<700:
    flag = 1
    output_size = 150
    left = max(0, min(center[1] - output_size // 2, 640 - output_size))
    top = max(0, min(center[0] - output_size // 2, 480 - output_size))
    # crop_img = depth_img[top:top+300,left:left+300]
    crop_img = data_transforms(depth_img[top:top+output_size,left:left+output_size].copy())[0].numpy()
    plt.figure(4)
    plt.imshow(crop_img)
    plt.show()
    obj = np.where(crop_img<-0.01)
    print('obj.size={}'.format(obj[0].size))
elif obj[0].size>3000:
    flag = 2
    output_size = 400
    left = max(0, min(center[1] - output_size // 2, 640 - output_size))
    top = max(0, min(center[0] - output_size // 2, 480 - output_size))
    # crop_img = depth_img[top:top+300,left:left+300]
    crop_img = data_transforms(depth_img[top:top+output_size,left:left+output_size].copy())[0].numpy()
    plt.figure(4)
    plt.imshow(crop_img)
    plt.show()
    obj = np.where(crop_img<-0.01)
    print('obj.size={}'.format(obj[0].size))

# args_network = '/home/abb/ggcnn-DQN/ggcnn-master_3/output/models/200518_1650_anglesless10_withoutTnn_rot_12angle_mindistance5_70width_randomFalseData/epoch_27_iou_0.55'
args_network = '/home/abb/ggcnn-DQN/ggcnn-master_3/output/models/200525_2219_anglesless5_withoutTnn_rot_12angle_mindistance5_70width_randomFalseData/epoch_11_iou_0.53'
net = torch.load(args_network)
device = torch.device("cuda:0")

step = 18
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

    print('ggcnn process time = {}'.format(ggcnn_end - ggcnn_start))
    q_img, width_img = post_process_output(pos_output, width_output)

    gs = evaluation.get_best_grasp(q_img,
                                   no_grasps=1,
                                   grasp_width=width_img,
                                   zoom_factor=torch.tensor([1])
                                   )

g = gs[0]

plt.figure(5)
plt_img = crop_img.copy()
plt.imshow(plt_img,alpha=0.8)
cv2.circle(plt_img, (g.center[1], g.center[0]), 2, (0, 0, 255))
gr = g.as_gr
cv2.line(plt_img, (int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)),
         (int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)),
         (255, 0, 0), 2)
cv2.line(plt_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
         (255, 0, 0), 2)
cv2.line(plt_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
         (255, 0, 0), 2)
plt.imshow(plt_img,alpha=0.8)
plt.show()

plt_img = crop_img.copy()
plt.imshow(plt_img)
plt.show()
#p0[u,v]
p0 = np.array([int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)])
p1 = np.array([int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)])
line = myline(p0[0], p0[1], p1[0], p1[1])

total_section = []
Sections = []
sub_section = []

for i in range(len(line)):
    pixel_value = -plt_img[line[i][0], line[i][1]]
    total_section.append(pixel_value)

total_section = np.array(total_section)
np.save('test_section.npy',total_section)
section_plt(total_section)

Sections = []
sub_section = []
cur_flag = 0
for i in range(total_section.size):
    if total_section[i] > 0.01:
        if cur_flag == 0:
            cur_flag = 1
            sub_section.append(total_section[i])
            continue
        else:
            sub_section.append(total_section[i])
            continue
    if total_section[i] < 0.01:
        if cur_flag == 0:
            continue
        else:
            cur_flag = 0
            Sections.append(Section(sub_section,i))
max_sum = 0
main_Section = Section([],0)
for i in range(len(Sections)):
    if Sections[i].sum>max_sum:
        max_sum = Sections[i].sum
        main_Section = Sections[i]
section_plt(main_Section.pixel)
p2 = line[max(0,main_Section.start-8)][::-1]
p3 = line[min(len(line),main_Section.end+8)][::-1]

g.center = line[main_Section.mid]
g.length = int(((p2[0] - p3[0])**2 + (p2[1]-p3[1])**2)**0.5)
g.width = int(g.length/4)

plt_img = crop_img.copy()
plt.imshow(plt_img,alpha=0.8)
cv2.circle(plt_img, (g.center[1], g.center[0]), 2, (0, 0, 255))
gr = g.as_gr
cv2.line(plt_img, (int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)),
         (int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)),
         (255, 0, 0), 2)
cv2.line(plt_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
         (255, 0, 0), 2)
cv2.line(plt_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
         (255, 0, 0), 2)
plt.imshow(plt_img,alpha=0.8)
plt.show()



d0 = plt_img[p0[1], p0[0]]
d1 = plt_img[p1[1], p1[0]]
dmin = min(d0, d1) - 0.005

# ## visualization module
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
#
# if flag==1:
#     g.center = (int(g.center[0] * (1.5) / 3) + top, int(g.center[1] * (1.5) / 3) + left)
# elif flag==2:
#     g.center = (int(g.center[0] * 4 / 3) + top, int(g.center[1] * 4 / 3) + left)
# else:
#     g.center = (g.center[0] + top, g.center[1] + left)
#
# plt.figure(6)
# plt.imshow(depth_img,alpha=0.8)
# cv2.circle(depth_img, (g.center[1], g.center[0]), 2, (0, 0, 255))
#
# gr = g.as_gr
# cv2.line(depth_img, (int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)),
#          (int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)),
#          (255, 0, 0), 2)
# cv2.line(depth_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
#          (255, 0, 0), 2)
# cv2.line(depth_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
#          (255, 0, 0), 2)
# plt.imshow(depth_img,alpha=0.8)
# plt.show()

