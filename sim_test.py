#!/usr/bin/env python
# python file:sim_test_SIH.py
# date:2020.2.25
# function:Given a series of object image from simulation, use GGCNN Network to  calculate the grasping candidates for each one
import rospy
import numpy as np
import random
import time
from sensor_msgs.msg import Image
import cv2, cv_bridge
import matplotlib.pyplot as plt
import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from get_stable_pose_SIH import sim_init,model_spawn_qua,\
    move_gripper,cal_shaking,control_gripper,model_get,position_to_pixel,\
    model_set_qua,pixel_to_position,orientation_to_rpy,rpy_to_orientation
from models.common import post_process_output
from utils.dataset_processing import evaluation
from utils.dataset_processing import grasp, image
from skimage.transform import rotate
from imageio import imread
bridge = cv_bridge.CvBridge()


def image_callback(msg):

    if msg.encoding == '32FC1':
        global img_array
        # img_array = bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        img_array = np.array(img, dtype=np.float32)

    if msg.encoding == 'rgb8':
        rgb = bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        cv2.namedWindow("window", 1)
        cv2.imshow("window", rgb)
        cv2.waitKey(1)


if __name__ == "__main__":

    rospy.init_node('grasping_demo', disable_signals=True)
    # initialize the env
    sim_init()

    img = rospy.Subscriber('/camera/depth/image_raw', Image, image_callback, queue_size=2)
    move_gripper()
    control_gripper('open', 2.5)

    # args_model = '/home/abb/ggcnn-DQN/ggcnn-master_3/output/models/200518_1650_anglesless10_withoutTnn_rot_12angle_mindistance5_70width_randomFalseData/epoch_27_iou_0.55'
    args_model = '/home/abb/ggcnn-DQN/ggcnn-master_3/output/models/200525_2219_anglesless5_withoutTnn_rot_12angle_mindistance5_70width_randomFalseData/epoch_11_iou_0.53'

    net = torch.load(args_model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    print('+++++++++++++++++++++++++++++++++')
    print(args_model)
    print('+++++++++++++++++++++++++++++++++')

    net.eval()

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(300),
        transforms.ToTensor()
    ])


    # read the object name and pose from the H5 file
    pose_info_storage_root = '/home/abb/Pictures/Dataset_SIH/stable pose'
    h5_name = 'Stable_1.2'
    f_p = h5py.File(pose_info_storage_root + '/' + h5_name + '.h5', 'r')

    succ_count = 0.0
    count = 0
    succ_rate = 0

    for i, name in enumerate(f_p):
        # if i < 18:
        #     continue
        if i == 45:
            break
        for j,pose in enumerate(f_p[name]):
            control_gripper('open', 2.5)
            # read obj name from H5 file
            print('Num:{} Name:{}'.format(i, name))
            print('**************************************')
            print('Pose {}:'.format(j))
            # read pose from H5df file
            Orien = f_p[name][pose]['orientation']
            Posi = f_p[name][pose]['position']
            print('Orientation:{}'.format(Orien))
            print('Position:{}'.format(Posi))
            sampleNo = 3
            mu = 0.0
            sigma = 0.03
            np.random.seed(random.randint(0, 1000))
            delta = np.random.normal(mu, sigma, sampleNo)

            spawn_rpy = orientation_to_rpy(Orien)
            spawn_rpy[2] += delta[2]*20
            Orien = rpy_to_orientation(spawn_rpy)
            ox = Orien[0]
            oy = Orien[1]
            oz = Orien[2]
            ow = Orien[3]
            px = Posi[0] + delta[0]
            py = Posi[1] + delta[1]
            pz = Posi[2]
            # spawn model in the center of camera
            model_spawn_qua(name,[ox,oy,oz,ow,px,py,pz])

            rospy.sleep(1)
            n1_img = img_array.copy()
            lift_height = model_get(name)
            print('-------------------------')
            print('lift_height = {}'.format(lift_height))

            n1_img = np.clip((n1_img - n1_img.mean()), -1, 1)
            depth_img = n1_img.astype(np.float32)
            # plt.imshow(depth_img)
            # plt.show()
            depth_img[np.where(depth_img < -0.01)] *= 1.2
            depth_img = np.clip(depth_img, -0.03, 0.1)

            np.savez(
                '/home/abb/Pictures/Dataset_SIH/GGCNN_Gazebo/' + str(count) + '.npz', \
                d_img=depth_img
            )
            # model_set_qua(name,reset=True)
            count += 1
            # continue


            #######################################
            obj = np.where(depth_img < -0.01)
            center = [np.mean(obj[0]).astype(int), np.mean(obj[1]).astype(int)]
            output_size = 300
            left = max(0, min(center[1] - output_size // 2, 640 - output_size))
            top = max(0, min(center[0] - output_size // 2, 480 - output_size))
            # crop_img = depth_img[top:top+300,left:left+300]
            crop_img = depth_img[top:top + output_size, left:left + output_size]
            plt.figure(4)
            plt.imshow(crop_img,alpha=1)
            # plt.show()
            plt.ion()
            plt.pause(0.2)


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
            plt.imshow(crop_img, alpha=0.8)
            cv2.circle(crop_img, (g.center[1], g.center[0]), 2, (0, 0, 255))

            gr = g.as_gr
            cv2.line(crop_img,
                     (int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)),
                     (int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)),
                     (255, 0, 0), 2)
            cv2.line(crop_img, (int(gr.points[1][1]), int(gr.points[1][0])),
                     (int(gr.points[2][1]), int(gr.points[2][0])),
                     (255, 0, 0), 2)
            cv2.line(crop_img, (int(gr.points[0][1]), int(gr.points[0][0])),
                     (int(gr.points[3][1]), int(gr.points[3][0])),
                     (255, 0, 0), 2)

            plt.imshow(crop_img, alpha=0.8)
            # plt.show()
            plt.ion()
            plt.pause(0.2)

            d_img = xc.cpu().squeeze(1).numpy()
            q_img = q_img

            # for a in range(step):
            #
            #     d_i = d_img[a]
            #     q_i = q_img[a]
            #
            #     plt.figure(1)
            #     ax1 = plt.subplot(int(step/3), 3, a + 1)
            #     plt.imshow(q_i, alpha=1)
            #     plt.title(str(round(np.max(q_i), 3)))
            #
            #     plt.figure(2)
            #     ax2 = plt.subplot(int(step/3), 3, a + 1)
            #     plt.imshow(d_i, alpha=1)
            # plt.show()
            ###########################################

            uv = pixel_to_position([g.center[0]+top,g.center[1]+left])

            gx = uv[0]
            gy = uv[1]
            gz = -0.315
            ga = g.angle

            move_gripper([gx, gy, 0.1, ga])
            rospy.sleep(1)
            move_gripper([gx, gy, gz, ga])
            control_gripper('close',5 )
            rospy.sleep(0.5)
            move_gripper([gx, gy, 0.1, ga])
            rospy.sleep(0.5)
            lift_pose = model_get(name)
            lift_height = lift_pose.position.z
            print('lift_height = {}'.format(lift_height))

            if lift_height>0.2:
                succ = 1
                succ_count += 1.0
                print(succ_count)
                succ_rate = succ_count/(count+1)
                print(succ_rate)
            else:
                succ = 0
                print(succ_count)
                print(succ_rate)
                succ_rate = succ_count/(count+1)
            with open('ggcnn_sim_result_200225_0016_training_gmd_SIH.txt', 'a') as file:
                file.write('g_num:{} succ:{} succ_count:{} succ_rate:{:.04f} height:{:.04f}\n'.format(count,succ,succ_count,succ_rate,lift_height))
            print('g_num:{} succ:{} succ_count:{} succ_rate:{:.04f} height:{:.04f} '.format(count,succ,succ_count,succ_rate,lift_height))
            rospy.sleep(0.5)
            move_gripper()
            control_gripper('open',3)
            model_set_qua(name,reset=True)
            count += 1