#!/usr/bin/env python
# python file: get_stable_pose_SIH.py
# date:2020.2.18
# function:an app to record the stable pose of 129 models
import rospy
import actionlib
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from my_simple_grasping_platform.msg import Console
from std_msgs.msg import Float64MultiArray, Float64, MultiArrayDimension
import numpy as np

import random
import time
from sensor_msgs.msg import Image
import cv2, cv_bridge
import matplotlib.pyplot as plt
import math
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import h5py
from imageio import imread
import tkinter as tk

bridge = cv_bridge.CvBridge()
dc_x = 0.5
dc_y = 0.5
dc_z = 1


def image_callback(msg):
    if msg.encoding == '32FC1':
        global img_array
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        img_array = np.array(img, dtype=np.float32)

    if msg.encoding == 'rgb8':
        rgb = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.namedWindow("window", 1)
        cv2.imshow("window", rgb)
        cv2.waitKey(1)


def sim_init():
    # spawn object
    # ------------------------------------------------------------------
    spawn_model_req = SpawnModelRequest()

    spawn_model_req.model_name = 'depth_camera'
    sdf_path = rospy.get_param('/camera_path')

    spawn_model_req.model_xml = ''
    with open(sdf_path + '/model.sdf', "r") as xml_file:
        spawn_model_req.model_xml = xml_file.read()

    spawn_model_req.robot_namespace = 'depth_camera'
    spawn_model_req.initial_pose.position.x = 0.5
    spawn_model_req.initial_pose.position.y = 0.5
    spawn_model_req.initial_pose.position.z = 1.2
    spawn_model_req.initial_pose.orientation.x = -0.5
    spawn_model_req.initial_pose.orientation.y = 0.5
    spawn_model_req.initial_pose.orientation.z = 0.5
    spawn_model_req.initial_pose.orientation.w = 0.5
    spawn_model_req.reference_frame = 'world'

    # print(spawn_model_req)

    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    spawn_model_resp = spawn_model(spawn_model_req)

    # #------------------------------------------------------------------
    # #set_model_state

    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    set_model_state_req = SetModelStateRequest()
    # print(set_model_state_req)
    set_model_state_req.model_state.model_name = 'depth_camera'
    set_model_state_req.model_state.pose = spawn_model_req.initial_pose
    set_model_state_req.model_state.pose.position.x = dc_x
    set_model_state_req.model_state.pose.position.y = dc_y
    set_model_state_req.model_state.pose.position.z = dc_z
    set_model_state_req.model_state.pose.orientation.x = -0.5
    set_model_state_req.model_state.pose.orientation.y = 0.5
    set_model_state_req.model_state.pose.orientation.z = 0.5
    set_model_state_req.model_state.pose.orientation.w = 0.5
    # print(set_model_state_req)

    set_model_state_resp = set_state(set_model_state_req)

    # rospy.sleep(1)
    # img = rospy.Subscriber('/camera/depth/image_raw', Image, image_callback, queue_size=2)
    # rospy.sleep(1)


def move_gripper(input_pose=[0.0, 0.0, 0.0, 0.0], duration=0.5):
    gripper_joints = ['virtual_1_2_joint', 'virtual_2_3_joint', 'virtual_3_4_joint',
                      'virtual_4_ee_joint']  # ['base_l_finger_joint', 'base_r_finger_joint'] #
    # Connect to the right arm trajectory action server
    # print "Connecting to the right arm trajectory action server..."
    #    gripper_client = actionlib.SimpleActionClient('/my_simple_2f_gripper/gripper_finger_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    gripper_client = actionlib.SimpleActionClient('/end_position_and_pose_controller/follow_joint_trajectory',
                                                  FollowJointTrajectoryAction)
    gripper_client.wait_for_server()
    # print "Connected"

    # Create an arm trajectory with the arm_goal as the end-point
    # print "Setting up a goal trajectory..."
    gripper_trajectory = JointTrajectory()
    gripper_trajectory.joint_names = gripper_joints
    gripper_trajectory.points.append(JointTrajectoryPoint())
    # gripper_trajectory.points[0].positions =  [-0.5, -0.35, 0.8, 0.0]   # [0.0, 0.0] ### [0.04, 0.04] #gripper_goal # Not cartesian positions, joint angles !!!

    gripper_trajectory.points[0].positions = input_pose
    gripper_trajectory.points[0].velocities = [0.0 for i in gripper_joints]
    gripper_trajectory.points[0].accelerations = [0.0 for i in gripper_joints]
    gripper_trajectory.points[0].time_from_start = rospy.Duration(
        duration)  # want the trajectory to finish on time, i.e. 3 seconds # set finish time point???
    # probably, when duration is set, the speed will be automatically modified to reach time requirement, as long as the velocity set in URDF satisfies.

    # Create an empty trajectory goal
    gripper_goal = FollowJointTrajectoryGoal()
    # set the trajectory component to the goal trajectory
    gripper_goal.trajectory = gripper_trajectory
    # specify zero tolerance for the execution time
    gripper_goal.goal_time_tolerance = rospy.Duration(0.0)

    # Send the goal to the action server
    # print "Sending the goal to the action server..."
    gripper_client.send_goal(gripper_goal)

    # Wait for up to 5 seconds for the motion to complete
    #    print "Wait for the goal transitions to complete..."
    #    gripper_client.wait_for_result(rospy.Duration(10.0))
    #    if gripper_client.get_state() == actionlib.SimpleClientGoalState.SUCCEEDED:
    #        print "The action is done."
    #    else:
    #        print "action is not yet done."

    gripper_client.wait_for_result(rospy.Duration(0.0))
    # print "Done."

    result = gripper_client.get_result()
    # print result.SUCCESSFUL


def control_gripper(comd, force=80):
    # open the finger
    topic_name = "/gripper_finger_controller/command"
    pub = rospy.Publisher(topic_name, Float64MultiArray, queue_size=100)
    # Create the message
    msg = Float64MultiArray()
    msg.layout.dim.append(MultiArrayDimension())
    msg.layout.dim[0].size = 2
    msg.layout.dim[0].stride = 1
    msg.layout.dim[0].label = 'open_fingers'
    if comd == 'open':
        msg.data = [force, force]
        pub.publish(msg)
        rospy.sleep(0.5)
        return
    else:
        msg.data = [-force, -force]
        pub.publish(msg)
        rospy.sleep(0.5)
        return


def position_to_pixel(pose=[0.0, 0.0, 0.0]):
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Intrinsic = np.array([[554.254691191187, 0.0, 320.5], [0.0, 554.254691191187, 240.5], [0.0, 0.0, 1.0]])
    pose = np.array([[pose[0], pose[1], pose[2]]]).T
    Co_camera = np.dot(R, pose) + np.array([[-0.5], [0.5], [1]])
    Co_camera = Co_camera * (1 / Co_camera[2])
    return np.dot(Intrinsic, Co_camera)


def pixel_to_position(center=[240, 320]):
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Intrinsic = np.array([[554.254691191187, 0.0, 320.5], [0.0, 554.254691191187, 240.5], [0.0, 0.0, 1.0]])
    uv = np.array([[center[1], center[0], 1]]).T
    uv = np.dot(np.linalg.inv(Intrinsic), uv)
    uv = uv * 1
    uv = uv - np.array([[-0.5], [0.5], [1]])
    uv = np.dot(np.linalg.inv(R), uv)
    return uv


def orientation_to_rpy(orientation=[0.0, 0.0, 0.0, 0.0]):
    x = orientation[0]
    y = orientation[1]
    z = orientation[2]
    w = orientation[3]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [roll, pitch, yaw]


def rpy_to_orientation(rpy=[0.0, 0.0, 0.0]):
    a = rpy[0]
    b = rpy[1]
    r = rpy[2]
    w = math.cos(a / 2) * math.cos(b / 2) * math.cos(r / 2) + math.sin(a / 2) * math.sin(b / 2) * math.sin(r / 2)
    x = math.sin(a / 2) * math.cos(b / 2) * math.cos(r / 2) - math.cos(a / 2) * math.sin(b / 2) * math.sin(r / 2)
    y = math.cos(a / 2) * math.sin(b / 2) * math.cos(r / 2) + math.sin(a / 2) * math.cos(b / 2) * math.sin(r / 2)
    z = math.cos(a / 2) * math.cos(b / 2) * math.sin(r / 2) - math.sin(a / 2) * math.sin(b / 2) * math.cos(r / 2)
    return [x, y, z, w]


def cal_gripper_len(dc_x, dc_y, dc_z, grip_len):
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Intrinsic = np.array([[554.254691191187, 0.0, 320.5], [0.0, 554.254691191187, 240.5], [0.0, 0.0, 1.0]])

    x = 0.50
    y = 0.50
    a = 0 * 0.5235987
    # a = 0.7

    x_l = -grip_len / 2
    y_l = 0.0
    x_r = grip_len / 2
    y_r = 0.0
    # object anticlockwise = the coordinate clockwise
    yaw = -a
    Rz = np.array([[math.cos(yaw), math.sin(yaw), 0], [-math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    Tz = np.array([[x, y, 0]]).T

    Co_l = np.array([[x_l, y_l, 0]]).T
    Co_l = np.dot(Rz, Co_l) + Tz
    Co_camera_l = np.dot(R, Co_l) + np.array([[-dc_x], [dc_y], [dc_z]])
    Co_camera_l = Co_camera_l * (1 / Co_camera_l[2])
    result = np.dot(Intrinsic, Co_camera_l)
    # print(result)
    u_l = int(result[0])
    v_l = int(result[1])

    Co_r = np.array([[x_r, y_r, 0]]).T
    Co_r = np.dot(Rz, Co_r) + Tz
    Co_camera_r = np.dot(R, Co_r) + np.array([[-dc_x], [dc_y], [dc_z]])
    Co_camera_r = Co_camera_r * (1 / Co_camera_r[2])
    result = np.dot(Intrinsic, Co_camera_r)
    # print(result)
    u_r = int(result[0])
    v_r = int(result[1])

    return ((u_l - u_r) ** 2 + (v_l - v_r) ** 2) ** 0.5


# spawn modal by roll pitch yaw
def model_spawn(name, spawn_pose=[0, 0, 0, 0, 0, 0], read_from_path=False):
    # for name in ['1_Amicelli_800_tex_1',
    #              # '2_CeylonTea_800_tex_1','3_CokePlasticSmallGrasp_800_tex_1',
    #              # '4_CondensedMilk_800_tex_1','5_DanishHam_800_tex_1','6_Glassbowl_800_tex_1',
    #              # '7_GreenCup_800_tex_1','8_HamburgerSauce_800_tex_1','9_InstantSoup_800_tex_1'
    #              ]:
    sampleNo = 2
    mu = 0
    sigma = 0.001
    np.random.seed(random.randint(0, 1000))
    delta = np.random.normal(mu, sigma, sampleNo)
    # name_ = name + '_1.6_1'
    name_ = name
    # ********************************************************************************
    spawn_model_req_obj = SpawnModelRequest()
    spawn_model_req_obj.model_name = name_
    # sdf_path = rospy.get_param('/sdf_path')
    sdf_path = '/home/abb/.gazebo/models/'

    if read_from_path:
        spawn_model_req_obj.model_xml = ''
        # ********************************************************************************
        with open(sdf_path + name[:-4] + '/model.sdf', "r") as xml_file:
            spawn_model_req_obj.model_xml = xml_file.read()
        # ********************************************************************************
        spawn_model_req_obj.robot_namespace = name_
        # read_name = name_.replace('_1.6_', '_1_')
        # print(read_name)
        read_path = '/home/abb/Pictures/Data_1_1/' + name_ + '/'
        with open(read_path + name_ + '_pose.txt', 'r') as f:
            list_result = f.readlines()
        a = float(list_result[1].rstrip('\n').split()[2])
        b = float(list_result[2].rstrip('\n').split()[2])
        r = float(list_result[3].rstrip('\n').split()[2])
        spawn_model_req_obj.initial_pose.position.x = float(list_result[4].rstrip('\n').split()[2]) + delta[0]
        spawn_model_req_obj.initial_pose.position.y = float(list_result[5].rstrip('\n').split()[2]) + delta[1]
        spawn_model_req_obj.initial_pose.position.z = float(list_result[6].rstrip('\n').split()[2]) + 0.1
        r = r + delta[0] * math.pi * 10.0
        spawn_model_req_obj.initial_pose.orientation.w = math.cos(a / 2) * math.cos(b / 2) * math.cos(r / 2) + math.sin(
            a / 2) * math.sin(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.x = math.sin(a / 2) * math.cos(b / 2) * math.cos(r / 2) - math.cos(
            a / 2) * math.sin(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.y = math.cos(a / 2) * math.sin(b / 2) * math.cos(r / 2) + math.sin(
            a / 2) * math.cos(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.z = math.cos(a / 2) * math.cos(b / 2) * math.sin(r / 2) - math.sin(
            a / 2) * math.sin(b / 2) * math.cos(r / 2)
    else:
        spawn_model_req_obj.model_xml = ''
        # ********************************************************************************
        with open(sdf_path + name + '/model.sdf', "r") as xml_file:
            spawn_model_req_obj.model_xml = xml_file.read()
        # ********************************************************************************
        spawn_model_req_obj.robot_namespace = name_
        # read_name = name_.replace('_1.6_', '_1_')
        # print(read_name)
        a = spawn_pose[0]
        b = spawn_pose[1]
        r = spawn_pose[2]
        spawn_model_req_obj.initial_pose.position.x = spawn_pose[3]
        spawn_model_req_obj.initial_pose.position.y = spawn_pose[4]
        spawn_model_req_obj.initial_pose.position.z = spawn_pose[5]
        spawn_model_req_obj.initial_pose.orientation.w = math.cos(a / 2) * math.cos(b / 2) * math.cos(r / 2) + math.sin(
            a / 2) * math.sin(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.x = math.sin(a / 2) * math.cos(b / 2) * math.cos(r / 2) - math.cos(
            a / 2) * math.sin(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.y = math.cos(a / 2) * math.sin(b / 2) * math.cos(r / 2) + math.sin(
            a / 2) * math.cos(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.z = math.cos(a / 2) * math.cos(b / 2) * math.sin(r / 2) - math.sin(
            a / 2) * math.sin(b / 2) * math.cos(r / 2)

    spawn_model_req_obj.reference_frame = 'world'
    # print(spawn_model_req)
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    spawn_model_resp = spawn_model(spawn_model_req_obj)
    # ------------------------------------------------------------------
    # set_model_state

    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    set_model_state_req_obj = SetModelStateRequest()
    # print(set_model_state_req)
    # ********************************************************************************
    set_model_state_req_obj.model_state.model_name = name_
    set_model_state_req_obj.model_state.pose = spawn_model_req_obj.initial_pose
    set_model_state_resp = set_state(set_model_state_req_obj)
    rospy.sleep(0.5)


# spawn modal by quaternion
def model_spawn_qua(name, spawn_pose=[0, 0, 0, 0, 0, 0, 0], read_from_path=False):
    # for name in ['1_Amicelli_800_tex_1',
    #              # '2_CeylonTea_800_tex_1','3_CokePlasticSmallGrasp_800_tex_1',
    #              # '4_CondensedMilk_800_tex_1','5_DanishHam_800_tex_1','6_Glassbowl_800_tex_1',
    #              # '7_GreenCup_800_tex_1','8_HamburgerSauce_800_tex_1','9_InstantSoup_800_tex_1'
    #              ]:
    sampleNo = 2
    mu = 0
    sigma = 0.001
    np.random.seed(random.randint(0, 1000))
    delta = np.random.normal(mu, sigma, sampleNo)
    # name_ = name + '_1.6_1'
    name_ = name
    # ********************************************************************************
    spawn_model_req_obj = SpawnModelRequest()
    spawn_model_req_obj.model_name = name_
    # sdf_path = rospy.get_param('/sdf_path')
    sdf_path = '/home/abb/.gazebo/models/'

    if read_from_path:
        spawn_model_req_obj.model_xml = ''
        # ********************************************************************************
        with open(sdf_path + name[:-4] + '/model.sdf', "r") as xml_file:
            spawn_model_req_obj.model_xml = xml_file.read()
        # ********************************************************************************
        spawn_model_req_obj.robot_namespace = name_
        # read_name = name_.replace('_1.6_', '_1_')
        # print(read_name)
        read_path = '/home/abb/Pictures/Data_1_1/' + name_ + '/'
        with open(read_path + name_ + '_pose.txt', 'r') as f:
            list_result = f.readlines()
        a = float(list_result[1].rstrip('\n').split()[2])
        b = float(list_result[2].rstrip('\n').split()[2])
        r = float(list_result[3].rstrip('\n').split()[2])
        spawn_model_req_obj.initial_pose.position.x = float(list_result[4].rstrip('\n').split()[2]) + delta[0]
        spawn_model_req_obj.initial_pose.position.y = float(list_result[5].rstrip('\n').split()[2]) + delta[1]
        spawn_model_req_obj.initial_pose.position.z = float(list_result[6].rstrip('\n').split()[2]) + 0.1
        r = r + delta[0] * math.pi * 10.0
        spawn_model_req_obj.initial_pose.orientation.w = math.cos(a / 2) * math.cos(b / 2) * math.cos(r / 2) + math.sin(
            a / 2) * math.sin(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.x = math.sin(a / 2) * math.cos(b / 2) * math.cos(r / 2) - math.cos(
            a / 2) * math.sin(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.y = math.cos(a / 2) * math.sin(b / 2) * math.cos(r / 2) + math.sin(
            a / 2) * math.cos(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.z = math.cos(a / 2) * math.cos(b / 2) * math.sin(r / 2) - math.sin(
            a / 2) * math.sin(b / 2) * math.cos(r / 2)
    else:
        spawn_model_req_obj.model_xml = ''
        # ********************************************************************************
        with open(sdf_path + name + '/model.sdf', "r") as xml_file:
            spawn_model_req_obj.model_xml = xml_file.read()
        # ********************************************************************************
        spawn_model_req_obj.robot_namespace = name_
        # read_name = name_.replace('_1.6_', '_1_')
        # print(read_name)

        spawn_model_req_obj.initial_pose.orientation.x = spawn_pose[0]
        spawn_model_req_obj.initial_pose.orientation.y = spawn_pose[1]
        spawn_model_req_obj.initial_pose.orientation.z = spawn_pose[2]
        spawn_model_req_obj.initial_pose.orientation.w = spawn_pose[3]

        spawn_model_req_obj.initial_pose.position.x = spawn_pose[4]
        spawn_model_req_obj.initial_pose.position.y = spawn_pose[5]
        spawn_model_req_obj.initial_pose.position.z = spawn_pose[6]

    spawn_model_req_obj.reference_frame = 'world'
    # print(spawn_model_req)
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    spawn_model_resp = spawn_model(spawn_model_req_obj)
    # ------------------------------------------------------------------
    # set_model_state

    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    set_model_state_req_obj = SetModelStateRequest()
    # print(set_model_state_req)
    # ********************************************************************************
    set_model_state_req_obj.model_state.model_name = name_
    set_model_state_req_obj.model_state.pose = spawn_model_req_obj.initial_pose
    set_model_state_resp = set_state(set_model_state_req_obj)
    rospy.sleep(0.5)


# set modal by roll pitch yaw
def model_set(name, spawn_pose=[0, 0, 0, 0, 0, 0], reset=False):
    sampleNo = 2
    mu = 0
    sigma = 0.006
    np.random.seed(random.randint(0, 1000))
    delta = np.random.normal(mu, sigma, sampleNo)
    # name_ = name + '_1.6_1'
    name_ = name
    # ********************************************************************************
    spawn_model_req_obj = SpawnModelRequest()
    spawn_model_req_obj.model_name = name_
    # sdf_path = rospy.get_param('/sdf_path')
    sdf_path = '/home/abb/.gazebo/models/'

    spawn_model_req_obj.model_xml = ''
    # ********************************************************************************
    with open(sdf_path + name + '/model.sdf', "r") as xml_file:
        spawn_model_req_obj.model_xml = xml_file.read()
    # ********************************************************************************
    spawn_model_req_obj.robot_namespace = name_
    # read_name = name_.replace('_1.6_', '_1_')
    # print(read_name)
    a = spawn_pose[0]
    b = spawn_pose[1]
    r = spawn_pose[2]

    if reset:
        seed = np.random.seed(int(time.time() % 100))
        spawn_model_req_obj.initial_pose.position.x = -0.5 - np.random.randint(1, 20, 1) * 0.1
        spawn_model_req_obj.initial_pose.position.y = -0.5 - np.random.randint(1, 20, 1) * 0.1
        spawn_model_req_obj.initial_pose.position.z = spawn_pose[5]
        spawn_model_req_obj.initial_pose.orientation.w = math.cos(a / 2) * math.cos(b / 2) * math.cos(r / 2) + math.sin(
            a / 2) * math.sin(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.x = math.sin(a / 2) * math.cos(b / 2) * math.cos(r / 2) - math.cos(
            a / 2) * math.sin(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.y = math.cos(a / 2) * math.sin(b / 2) * math.cos(r / 2) + math.sin(
            a / 2) * math.cos(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.z = math.cos(a / 2) * math.cos(b / 2) * math.sin(r / 2) - math.sin(
            a / 2) * math.sin(b / 2) * math.cos(r / 2)
    else:
        spawn_model_req_obj.initial_pose.position.x = spawn_pose[3]
        spawn_model_req_obj.initial_pose.position.y = spawn_pose[4]
        spawn_model_req_obj.initial_pose.position.z = spawn_pose[5]
        spawn_model_req_obj.initial_pose.orientation.w = math.cos(a / 2) * math.cos(b / 2) * math.cos(r / 2) + math.sin(
            a / 2) * math.sin(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.x = math.sin(a / 2) * math.cos(b / 2) * math.cos(r / 2) - math.cos(
            a / 2) * math.sin(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.y = math.cos(a / 2) * math.sin(b / 2) * math.cos(r / 2) + math.sin(
            a / 2) * math.cos(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.z = math.cos(a / 2) * math.cos(b / 2) * math.sin(r / 2) - math.sin(
            a / 2) * math.sin(b / 2) * math.cos(r / 2)

    spawn_model_req_obj.reference_frame = 'world'

    # ------------------------------------------------------------------
    # set_model_state

    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    set_model_state_req_obj = SetModelStateRequest()
    # print(set_model_state_req)
    # ********************************************************************************
    set_model_state_req_obj.model_state.model_name = name_
    set_model_state_req_obj.model_state.pose = spawn_model_req_obj.initial_pose
    set_model_state_resp = set_state(set_model_state_req_obj)
    rospy.sleep(0.5)
    return spawn_model_req_obj.initial_pose


# set modal quaternion
def model_set_qua(name, spawn_pose=[0, 0, 0, 0, 0, 0, 0], reset=False):
    sampleNo = 2
    mu = 0
    sigma = 0.006
    np.random.seed(random.randint(0, 1000))
    delta = np.random.normal(mu, sigma, sampleNo)
    # name_ = name + '_1.6_1'
    name_ = name
    # ********************************************************************************
    spawn_model_req_obj = SpawnModelRequest()
    spawn_model_req_obj.model_name = name_
    # sdf_path = rospy.get_param('/sdf_path')
    sdf_path = '/home/abb/.gazebo/models/'

    spawn_model_req_obj.model_xml = ''
    # ********************************************************************************
    with open(sdf_path + name + '/model.sdf', "r") as xml_file:
        spawn_model_req_obj.model_xml = xml_file.read()
    # ********************************************************************************
    spawn_model_req_obj.robot_namespace = name_
    # read_name = name_.replace('_1.6_', '_1_')
    # print(read_name)
    a = spawn_pose[0]
    b = spawn_pose[1]
    r = spawn_pose[2]

    if reset:
        seed = np.random.seed(int(time.time() % 100))
        spawn_model_req_obj.initial_pose.position.x = -0.5 - np.random.randint(1, 20, 1) * 0.1
        spawn_model_req_obj.initial_pose.position.y = -0.5 - np.random.randint(1, 20, 1) * 0.1
        spawn_model_req_obj.initial_pose.position.z = spawn_pose[5]
        spawn_model_req_obj.initial_pose.orientation.w = math.cos(a / 2) * math.cos(b / 2) * math.cos(r / 2) + math.sin(
            a / 2) * math.sin(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.x = math.sin(a / 2) * math.cos(b / 2) * math.cos(r / 2) - math.cos(
            a / 2) * math.sin(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.y = math.cos(a / 2) * math.sin(b / 2) * math.cos(r / 2) + math.sin(
            a / 2) * math.cos(b / 2) * math.sin(r / 2)
        spawn_model_req_obj.initial_pose.orientation.z = math.cos(a / 2) * math.cos(b / 2) * math.sin(r / 2) - math.sin(
            a / 2) * math.sin(b / 2) * math.cos(r / 2)
    else:
        spawn_model_req_obj.initial_pose.orientation.x = spawn_pose[0]
        spawn_model_req_obj.initial_pose.orientation.y = spawn_pose[1]
        spawn_model_req_obj.initial_pose.orientation.z = spawn_pose[2]
        spawn_model_req_obj.initial_pose.orientation.w = spawn_pose[3]

        spawn_model_req_obj.initial_pose.position.x = spawn_pose[4]
        spawn_model_req_obj.initial_pose.position.y = spawn_pose[5]
        spawn_model_req_obj.initial_pose.position.z = spawn_pose[6]

    spawn_model_req_obj.reference_frame = 'world'

    # ------------------------------------------------------------------
    # set_model_state

    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    set_model_state_req_obj = SetModelStateRequest()
    # print(set_model_state_req)
    # ********************************************************************************
    set_model_state_req_obj.model_state.model_name = name_
    set_model_state_req_obj.model_state.pose = spawn_model_req_obj.initial_pose
    set_model_state_resp = set_state(set_model_state_req_obj)
    rospy.sleep(0.5)
    return spawn_model_req_obj.initial_pose


# get modal's state
def model_get(name):
    # name_ = name + '_1.6_1'
    name_ = name
    # #------------------------------------------------------------------
    # #get_model_state
    rospy.wait_for_service('/gazebo/get_model_state')
    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    get_model_state_req = GetModelStateRequest()
    # record initial state----------------------------------------------------------
    get_model_state_req.model_name = name_
    get_model_state_resp = get_state(get_model_state_req)

    return get_model_state_resp.pose


# delete modal
# DeleteModel,DeleteModelRequest
def model_delete(name):
    # name_ = name + '_1.6_1'
    name_ = name
    # #------------------------------------------------------------------
    # #get_model_state
    rospy.wait_for_service('/gazebo/delete_model')
    delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
    delete_model_state_req = DeleteModelRequest()
    delete_model_state_req.model_name = name_
    delete_model_state_resp = delete_model(delete_model_state_req)


def cal_shaking(input_pos, shake_dist, shake_angle):
    x_l = -0.049
    y_l = 0.0
    x_r = 0.049
    y_r = 0.0
    yaw = -shake_angle
    Rz = np.array([[math.cos(yaw), math.sin(yaw), 0], [-math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    Tz = np.array([[input_pos[0], input_pos[1], 0]]).T
    Co_l = np.array([[-shake_dist, 0.0, 0]]).T
    Co_l = np.dot(Rz, Co_l) + Tz
    Co_r = np.array([[shake_dist, 0.0, 0]]).T
    Co_r = np.dot(Rz, Co_r) + Tz
    return [Co_l[0], Co_l[1], input_pos[2], input_pos[3]], [Co_r[0], Co_r[1], input_pos[2], input_pos[3]]


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


def can_grasp(img, grasp_pose, gripper_len):
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Intrinsic = np.array([[554.254691191187, 0.0, 320.5], [0.0, 554.254691191187, 240.5], [0.0, 0.0, 1.0]])

    x = grasp_pose[0]
    y = grasp_pose[1]
    a = grasp_pose[3]

    x_l = -gripper_len / 2
    y_l = 0.0
    x_r = gripper_len / 2
    y_r = 0.0
    yaw = -a
    Rz = np.array([[math.cos(yaw), math.sin(yaw), 0], [-math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    Tz = np.array([[x, y, 0]]).T
    dc_x = 0.5
    dc_y = 0.5
    dc_z = 1

    Co_l = np.array([[x_l, y_l, 0]]).T
    Co_l = np.dot(Rz, Co_l) + Tz
    Co_camera_l = np.dot(R, Co_l) + np.array([[-dc_x], [dc_y], [dc_z]])
    Co_camera_l = Co_camera_l * (1 / Co_camera_l[2])
    result = np.dot(Intrinsic, Co_camera_l)
    # print(result)
    u_l = int(result[0])
    v_l = int(result[1])

    Co_r = np.array([[x_r, y_r, 0]]).T
    Co_r = np.dot(Rz, Co_r) + Tz
    Co_camera_r = np.dot(R, Co_r) + np.array([[-dc_x], [dc_y], [dc_z]])
    Co_camera_r = Co_camera_r * (1 / Co_camera_r[2])
    result = np.dot(Intrinsic, Co_camera_r)
    # print(result)
    u_r = int(result[0])
    v_r = int(result[1])

    # print(u_l, v_l, u_r, v_r)
    line = myline(u_l, v_l, u_r, v_r)
    # print(line)
    section = []
    for i in range(len(line)):
        section.append(dc_z - img[line[i][0], line[i][1]])
    section = np.array(section)
    print(section)
    if sum(section[:3]) > 0 or sum(section[-3:]) > 0:
        return False
    else:
        return True
    # img_copy = img.copy()
    # for i in range(len(line)):
    #     img_copy[line[i][0],line[i][1]] = 0
    # plt.figure(2)
    # plt.imshow(img_copy)
    # plt.show()
    # section_plt(section)
    # npz = np.load('/home/abb/Pictures/Dataset_SIH/npz/0_100_Sauerkraut_800_tex_1.2_pose0.npz')
    # npz = np.load('/home/abb/Pictures/Dataset_SIH/npz/3_101_Seal_800_tex_1.2_pose1.npz')
    # # TEST can_grasp
    # npz = np.load('/home/abb/Pictures/Dataset_SIH/npz/96_1_Amicelli_800_tex_1.2_pose1.npz')
    # img = npz['image.npy']
    # plt.figure(1)
    # plt.imshow(img)
    # plt.show()
    # d_img = img.copy()
    # d_img = (1-d_img)*500
    # # plt.imshow(d_img)
    # # plt.show()
    # # get the pixel'position with value
    # all_pixel = np.argwhere(d_img>5)
    # # print(all_pixel)
    # # get the center in pixel
    # center_pixel = np.mean(all_pixel, axis=0)  # center is a list of (y,1,2) axis = 0 cal the mean of center
    # # print(center_pixel)
    # # get the center in world
    # center = pixel_to_position(center_pixel)
    # print(center)
    # # can_grasp(img, [center[0], center[1], 0, math.pi / 2 -math.pi/18], 0.1)
    # for i in range(18):
    #     can_grasp(img,[center[0],center[1],0,math.pi/2-math.pi/18*i],0.12)
    # return


global count

file_list = '/home/abb/.gazebo/models/'
name_list = os.listdir(file_list)
name_list.sort()
pose_info_storage_root = '/home/abb/Pictures/Dataset_SIH/stable pose'
h5_name = 'Stable_1.2'
f = h5py.File(pose_info_storage_root + '/' + h5_name + '.h5', 'a')
count = len(f)


# count = 29
class APP:
    def __init__(self, root):
        root.title("Hello Test")
        frame = tk.Frame(root)
        frame.pack()
        self.cur_obj = tk.Button(frame, text='Show current obj', bg='white', fg='black', command=self.app_cur_obj)
        self.cur_obj.pack(side=tk.LEFT)
        self.spawn = tk.Button(frame, text='Spawn_New_Model', fg='blue', command=self.app_spawn)
        self.spawn.pack(side=tk.LEFT)
        self.last_i = tk.Button(frame, text='Last_index', fg='red', command=self.last_index)
        self.last_i.pack(side=tk.LEFT)
        self.next_i = tk.Button(frame, text='Next_index', fg='green', command=self.next_index)
        self.next_i.pack(side=tk.LEFT)
        self.record = tk.Button(frame, text='Record_Model_Pose', fg='black', command=self.app_pose_record)
        self.record.pack(side=tk.LEFT)
        self.spawn = tk.Button(frame, text='Set_Model_Away', fg='blue', command=self.app_set)
        self.spawn.pack(side=tk.LEFT)
        # self.app = tk.Button(frame,text='Show current obj',bg='white',fg ='black',command=self.app_cur_obj)
        # self.app = tk.Button(frame,text='Spawn_New_Model',fg='blue',command=self.app_spawn)
        # self.app = tk.Button(frame,text='Record_Model_Pose',fg='black',command=self.app_pose_record)
        # self.app = tk.Button(frame,text='Last_index',fg='red',command=self.last_index)

        # self.app = tk.Button(frame,text='Next_index',fg='green',command=self.next_index)
        # self.app.pack(side=tk.RIGHT)

    def app_cur_obj(self):
        global count
        print('~~Current obj is {}:{}'.format(count, name_list[count]))

    def app_spawn(self):
        global count
        # count += 1
        model_spawn(name_list[count], [0.0, 0.0, 0.0, 0.5, 0.5, 0.1], False)
        print('**Spawn Model:{}\n'.format(name_list[count]))

    def app_set(self):
        global count
        model_set(name_list[count], [0.0, 0.0, 0.0, 0.5, 0.5, 0.1], reset=True)
        print('**Reset Last Model:{}'.format(name_list[count]))

    def last_index(self):
        global count
        if count == 0:
            print('!Error: the very FIRST index')
        else:
            count -= 1
        print('----------------------------------------')
        print('--current index:{}'.format(count))

    def next_index(self):
        global count
        if count == len(name_list):
            print('!Error: the very LAST index')
        else:
            count += 1
        print('--current index:{}'.format(count))

    def app_pose_record(self):
        global count
        current_pose = model_get(name_list[count])
        cpx = current_pose.position.x
        cpy = current_pose.position.y
        cpz = current_pose.position.z
        cox = current_pose.orientation.x
        coy = current_pose.orientation.y
        coz = current_pose.orientation.z
        cow = current_pose.orientation.w

        if not (name_list[count] in f):

            obj_filename_group = f.create_group(name_list[count] + '/pose0')
            obj_filename_group.create_dataset('position', data=[cpx, cpy, cpz])
            obj_filename_group.create_dataset('orientation', data=[cox, coy, coz, cow])
            # obj_filename_group.create_dataset('position_y', data=cpy)
            # obj_filename_group.create_dataset('position_z', data=cpz)
            # obj_filename_group.create_dataset('orientation_x', data=cox)
            # obj_filename_group.create_dataset('orientation_y', data=coy)
            # obj_filename_group.create_dataset('orientation_z', data=coz)
            # obj_filename_group.create_dataset('orientation_w', data=cow)
        else:
            count_pos = 'pose' + str(len(f[name_list[count]]))
            obj_filename_group = f.create_group(name_list[count] + '/' + count_pos)
            obj_filename_group.create_dataset('position', data=[cpx, cpy, cpz])
            obj_filename_group.create_dataset('orientation', data=[cox, coy, coz, cow])
        print('**Record Model Pose:{}\n'.format(name_list[count]))
        print(current_pose)


if __name__ == "__main__":
    rospy.init_node('grasping_demo', disable_signals=True)
    sim_init()
    # file_list = glob.glob(os.path.join('/home/abb/Pictures/Data_1_1/','*','*pose.txt'))
    # file_list.sort()
    # name_list = [file_list[i][28:28+int((len(file_list[i])-38)/2)] for i in range(len(file_list))]

    root = tk.Tk()
    app = APP(root)
    root.mainloop()
    # for i in range(3):
    #     name = name_list[i]
    #     a_ = 0.0
    #     b_ = 0.0
    #     r_ = 0.0
    #     x_ = 0.5
    #     y_ = 0.5
    #     z_ = 0.1
    #     model_spawn(name,[a_,b_,r_,x_,y_,z_],False)
    #     print(model_set(name,[a_,b_,r_,x_,y_,z_],False))
    #     rospy.sleep(2)
    #     print(model_get(name))
    #     model_set(name,[a_,b_,r_,x_,y_,z_],True)

    rospy.spin()

    # pixel_to_position([240,320])
    # position_to_pixel([0.5,0.5,0])




