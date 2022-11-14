import numpy as np

import torch
import torch.utils.data
import matplotlib.pyplot as plt
import random
import cv2
from skimage.transform import rotate

class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training GG-CNNs in a common format.
    """
    def __init__(self, output_size=300, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False, is_training = True):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.grasp_files = []
        self.is_training =  is_training

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)#返回的是随机选了一个角度以及和其相邻的抓取簇

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        ######################################################
        # plt.imshow(depth_img)
        # plt.show()
        # crop_img = depth_img.copy()
        # for gr in bbs:
        #     g = gr.as_grasp
        #     cv2.circle(crop_img, (g.center[1],g.center[0]),2,(0, 0, 255))
        #     cv2.line(crop_img, (int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)),
        #              (int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)),
        #              (255, 0, 0), 1)
        #     cv2.line(crop_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
        #              (255, 0, 0), 1)
        #     cv2.line(crop_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
        #              (255, 0, 0), 1)
        # plt.imshow(depth_img)
        # plt.imshow(crop_img,alpha=0.8)
        # # plt.imshow(pos_img,alpha=0.1)
        # plt.show()
        # #######################################################
        if self.is_training:
            ######################################################
            ang = 0
            count_ang = 0
            for b in bbs:
                ang += b.angle
                count_ang += 1
            ang = -ang / count_ang

            if np.random.randint(2, size=1)==1:
                rand_ang = np.random.uniform(-np.pi/2,np.pi/2,1)
                quality = np.exp(-2*abs(rand_ang - ang))
                pos_img, width_img = bbs.draw_rot((self.output_size, self.output_size),quality=quality)
                ang = rand_ang
            
            ######################################################
            # # plt.show()
            #
            # crop_img = depth_img.copy()
            # for gr in bbs:
            #     g = gr.as_grasp
            #     cv2.circle(crop_img, (g.center[1],g.center[0]),2,(0, 0, 255))
            #     cv2.line(crop_img, (int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)),
            #              (int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)),
            #              (255, 0, 0), 2)
            #     cv2.line(crop_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
            #              (255, 0, 0), 2)
            #     cv2.line(crop_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
            #              (255, 0, 0), 2)
            # plt.imshow(depth_img)
            # # plt.imshow(crop_img,alpha=0.8)
            # plt.imshow(pos_img,alpha=0.1)
            # plt.show()
            # #######################################################
            depth_img = rotate(depth_img, ang / np.pi * 180, center=None, mode='edge', preserve_range=True).astype(
                depth_img.dtype)

            pos_img = rotate(pos_img, ang / np.pi * 180, center=None, mode='edge', preserve_range=True).astype(
                pos_img.dtype)

            width_img = rotate(width_img, ang / np.pi * 180, center=None, mode='edge', preserve_range=True).astype(
                width_img.dtype)
            ######################################################
            # # plt.show()
            #
            # crop_img = depth_img.copy()
            # for gr in bbs:
            #     g = gr.as_grasp
            #     cv2.circle(crop_img, (g.center[1],g.center[0]),2,(0, 0, 255))
            #     cv2.line(crop_img, (int((gr.points[2][1] + gr.points[1][1]) * 0.5), int((gr.points[2][0] + gr.points[1][0]) * 0.5)),
            #              (int((gr.points[0][1] + gr.points[3][1]) * 0.5), int((gr.points[0][0] + gr.points[3][0]) * 0.5)),
            #              (255, 0, 0), 2)
            #     cv2.line(crop_img, (int(gr.points[1][1]), int(gr.points[1][0])), (int(gr.points[2][1]), int(gr.points[2][0])),
            #              (255, 0, 0), 2)
            #     cv2.line(crop_img, (int(gr.points[0][1]), int(gr.points[0][0])), (int(gr.points[3][1]), int(gr.points[3][0])),
            #              (255, 0, 0), 2)
            # plt.imshow(depth_img)
            # # plt.imshow(crop_img,alpha=0.8)
            # plt.imshow(pos_img,alpha=0.1)
            # plt.show()
            # #######################################################
            width_img = np.clip(width_img, 0.0, 150.0) / 150.0
            plt.clf()
            plt.imshow(pos_img)
            plt.colorbar()
            plt.savefig("save/"+str(idx)+".png")
            if self.include_depth and self.include_rgb:
                x = self.numpy_to_torch(
                    np.concatenate(
                        (np.expand_dims(depth_img, 0),
                         rgb_img),
                        0
                    )
                )
            elif self.include_depth:
                x = self.numpy_to_torch(depth_img)
            elif self.include_rgb:
                x = self.numpy_to_torch(rgb_img)

            pos = self.numpy_to_torch(pos_img)
            width = self.numpy_to_torch(width_img)

        else:
            d_img = []
            p_img = []
            w_img = []
            for i in range(12):
                # print(np.pi/2 - i * np.pi/9)
                depth_img_rot = rotate(depth_img, (np.pi/2 - i * np.pi/12) / np.pi * 180, center=None, mode='edge', preserve_range=True).astype(
                    depth_img.dtype)
                # plt.imshow(depth_img)
                # plt.show()

                pos_img_rot = rotate(pos_img, (np.pi/2 - i * np.pi/12)  / np.pi * 180, center=None, mode='edge', preserve_range=True).astype(
                    pos_img.dtype)

                width_img_rot = rotate(width_img, (np.pi/2 - i * np.pi/12)  / np.pi * 180, center=None, mode='edge', preserve_range=True).astype(
                    width_img.dtype)

                d_img.append(torch.tensor(depth_img_rot).unsqueeze(0).float())
                p_img.append(torch.tensor(pos_img_rot).unsqueeze(0).float())
                w_img.append(torch.tensor(width_img_rot).unsqueeze(0).float())

            return torch.cat(d_img), (torch.cat(p_img), torch.cat(w_img)), idx, rot, zoom_factor



        return x, (pos, width), idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)
