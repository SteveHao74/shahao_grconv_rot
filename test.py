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

# args_train_dataset = '/home/abb/Downloads/Cornell_Dataset'
# args_dataset = 'cornell'
# args_ds_rotate = 0
# args_batch_size = 1
# args_num_workers = 1
# args_network = 'ggcnn'
#
# Dataset = get_dataset(args_dataset)
# train_dataset = Dataset(args_train_dataset, start=0.0, end=1.0, ds_rotate=args_ds_rotate,
#                         random_rotate=False, random_zoom=False,
#                         include_depth=True, include_rgb=False)
# train_data = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=args_batch_size,
#     shuffle=True,
#     num_workers=args_num_workers
# )
#
# val_dataset = Dataset(args_train_dataset, start=0.0, end=1.0,ds_rotate=args_ds_rotate,
#                       random_rotate=True, random_zoom=True,
#                       include_depth=True, include_rgb=False, is_training=False)
# val_data = torch.utils.data.DataLoader(
#     val_dataset,
#     batch_size=1,
#     shuffle=False,
#     num_workers=args_num_workers
# )
#
# for x, y, _, _, _ in train_data:
#     print(x.size)

args_train_dataset = '/home/abb/Downloads/Jacquard/'
# args_train_dataset = '/media/abb/Data/Data/fang'
args_dataset = 'jacquard'
# args_dataset = 'gmnet'
args_ds_rotate = 0
args_batch_size = 1
args_num_workers = 1
args_network = 'ggcnn'

Dataset = get_dataset(args_dataset)
train_dataset = Dataset(args_train_dataset, start=0.0, end=0.9, ds_rotate=args_ds_rotate,
                        random_rotate=False, random_zoom=False,
                        include_depth=True, include_rgb=True)
train_data = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args_batch_size,
    shuffle=True,
    num_workers=args_num_workers
)

for idx, (x, y, didx, rot, zoom) in enumerate(train_data):
    rgb_img = train_data.dataset.get_rgb(didx, rot, zoom, normalise=False)
    depth_img = train_data.dataset.get_depth(didx, rot, zoom)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(depth_img, cmap='gray')
    ax.set_title('Depth')
    ax.axis('off')


    # plt.colorbar(plot)
    plt.show()

# val_dataset = Dataset(args_train_dataset, start=0.0, end=1.0,ds_rotate=args_ds_rotate,
#                       random_rotate=True, random_zoom=True,
#                       include_depth=True, include_rgb=True, is_training=False)
# val_data = torch.utils.data.DataLoader(
#     val_dataset,
#     batch_size=1,
#     shuffle=False,
#     num_workers=args_num_workers
# )
#
# for x, y, _, _, _ in train_data:
#     print(x.size)


# input_channels = 1
# ggcnn = get_network(args_network)
#
# policy_net = ggcnn(input_channels=input_channels)
# target_net = ggcnn(input_channels=input_channels)
# device = torch.device("cuda:0")
# policy_net = policy_net.to(device)
#
# # Anti-Clockwise rotate
# def rotate_state(state,a,device):
#     angle = (math.pi/2 - a)
#     theta = torch.tensor([
#         [math.cos(angle), math.sin(-angle), 0],
#         [math.sin(angle), math.cos(angle), 0]
#     ], dtype=torch.float,device=device)
#     grid = F.affine_grid(theta.unsqueeze(0), state.size())
#     output = F.grid_sample(state, grid)
#     return output
#
# with torch.no_grad():
#     for x, y, didx, rot, zoom_factor in val_data:
#
#         xc = x.transpose(1, 0).to(device)
#         xc = []
#         for i in range(9):
#             xc.append(rotate_state(x.to(device), math.pi / 2 - i * math.pi / 9, device))
#         xc = torch.cat(xc)
#
#         yc = []
#         for j in range(len(y)):
#             yjc = []
#             for i in range(9):
#                 yjc.append(rotate_state(y[j].to(device), math.pi / 2 - i * math.pi / 9, device))
#             yc.append(torch.cat(yjc))
#         lossd = policy_net.compute_loss(xc, yc)
#
#         q_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['width'])
#         print()
#
#         s = evaluation.calculate_iou_match_rot(q_out,
#                                                val_data.dataset.get_gtbb_val(didx, rot, zoom_factor),
#                                                no_grasps=1,
#                                                grasp_width=w_out,
#                                                )
