import os
import cv2
import random
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset

from augmentations import Rotation, Scale, Translate, Flip
from augmentations import White_Noise, Gray, add_light, contrast_image, saturation_image, Equalization

# mean and std in my own data
# for 6 channel: b, g, r, z, intensity, depth
mean = [0.3248, 0.3373, 0.3436, 0.2411, 0.2518, 0.1794]
std = [0.2365, 0.2438, 0.2502, 0.1624, 0.1484, 0.1807]

# for 8 channel: b, g, r, x, y, z, intensity, depth
# mean = [0.3248, 0.3373, 0.3436, 0.1720, 0.4633, 0.2411, 0.2518, 0.1794]
# std = [0.2365, 0.2438, 0.2502, 0.1721, 0.1515, 0.1624, 0.1484, 0.1807]

def quantization(image):
    q1 = np.where(image <= 0.1)
    q2 = np.where((image > 0.1) & (image <= 0.2))
    q3 = np.where((image > 0.2) & (image <= 0.3))
    q4 = np.where((image > 0.3) & (image <= 0.4))
    q5 = np.where((image > 0.4) & (image <= 0.5))
    q6 = np.where((image > 0.5) & (image <= 0.6))
    q7 = np.where((image > 0.6) & (image <= 0.7))
    q8 = np.where((image > 0.7) & (image <= 0.8))
    q9 = np.where((image > 0.8) & (image <= 0.9))
    q10 = np.where((image > 0.9) & (image <= 1.0))

    image[q1] = 0.1
    image[q2] = 0.2
    image[q3] = 0.3
    image[q4] = 0.4
    image[q5] = 0.5
    image[q6] = 0.6
    image[q7] = 0.7
    image[q8] = 0.8
    image[q9] = 0.9
    image[q10] = 1.0

    return image

def histogram_equalization(image):
    data = image.copy().flatten()
    hist, bins = np.histogram(data, 256, density=True)
    cdf = hist.cumsum()
    img_eq = np.interp(data, bins[:-1], cdf)

    return img_eq

class Train_DataSet(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder

        self.proj_H = 64
        # self.proj_H = 48
        self.proj_W = 512
        self.proj_C = 6
        # self.proj_C = 8

        # self.C_GT = 3
        self.C_GT = 1

        self.calib_dir = glob(os.path.join(self.data_folder, "training", "calib", "*.txt"))
        self.image_dir = glob(os.path.join(self.data_folder, "training", "image_2", "*.png"))
        self.velodyne_dir = glob(os.path.join(self.data_folder, "training", "velodyne", "*.bin"))
        self.gt_lidar_dir = glob(os.path.join(self.data_folder, "training", "gt_txt", "*.txt"))

    def __getitem__(self, item):
        # Build Data from Raw Camera & LiDAR data
        # Build a Camera
        img = cv2.imread(self.image_dir[item])
        height, width, channel = img.shape

        # Build a LiDAR
        velodyne = np.fromfile(self.velodyne_dir[item], dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity

        coordinate = velodyne[:, :3]
        scan_x = coordinate[:, 0]
        scan_y = coordinate[:, 1]
        scan_z = coordinate[:, 2]

        add_ones = np.ones((coordinate.shape[0]))
        add_ones = np.expand_dims(add_ones, axis=1)
        coordinate_add_one = np.append(coordinate, add_ones, axis=1)

        intensity = velodyne[:, 3]

        # Build a Calibration
        calib_file = open(self.calib_dir[item], 'r')
        lines = calib_file.readlines()

        P2_rect = np.array(lines[2].split(" ")[1:]).astype(np.float32).reshape(3, 4)
        R0_rect = np.array(lines[4].split(" ")[1:]).astype(np.float32).reshape(3, 3)
        velo_to_cam = np.array(lines[5].split(" ")[1:]).astype(np.float32).reshape(3, 4)

        add_r0 = np.array([0., 0., 0.])
        add_r0_1 = np.array([0., 0., 0., 1.])
        R0_rect = np.append(R0_rect, add_r0.reshape(3, 1), axis=1)
        R0_rect = np.append(R0_rect, add_r0_1.reshape(1, 4), axis=0)

        add_velo = np.array([0., 0., 0., 1.])
        velo_to_cam = np.append(velo_to_cam, add_velo.reshape(1, 4), axis=0)

        homogeneous_matrix = np.matmul(np.matmul(P2_rect, R0_rect), velo_to_cam)

        result = np.matmul(homogeneous_matrix, coordinate_add_one.transpose(1, 0))

        u = result[0] / result[2]
        v = result[1] / result[2]

        condition = np.where(
            (result[0] >= 0) & (result[2] >= 0) & (scan_x >= 0) & (v < height) & (v >= 0) & (u >= 0) & (u < width))

        u = u[condition]
        v = v[condition]

        scan_x = scan_x[condition]
        scan_y = scan_y[condition]
        scan_z = scan_z[condition]
        intensity = intensity[condition]

        u = np.floor(u)
        u = np.minimum(width - 1, u)
        u = np.maximum(0, u).astype(np.int32)

        v = np.floor(v)
        v = np.minimum(height - 1, v)
        v = np.maximum(0, v).astype(np.int32)

        b = img[v, u][:, 0]
        g = img[v, u][:, 1]
        r = img[v, u][:, 2]

        dtheta = np.radians(0.4)
        dphi = np.radians(90. / 512.0)

        depth_ = np.sqrt(scan_x ** 2 + scan_y ** 2 + scan_z ** 2)
        range_ = np.sqrt(scan_x ** 2 + scan_y ** 2)

        depth_[depth_ == 0] = 0.000001
        range_[range_ == 0] = 0.000001

        phi = np.radians(45.) - np.arcsin(scan_y / range_)
        phi_ = (phi / dphi).astype(int)
        phi_[phi_ < 0] = 0
        phi_[phi_ >= 512] = 511

        theta = np.radians(2.) - np.arcsin(scan_z / depth_)
        theta_ = (theta / dtheta).astype(int)
        theta_[theta_ < 0] = 0
        theta_[theta_ >= 64] = 63

        fusion_map = np.zeros((self.proj_H, self.proj_W, self.proj_C)
                              
        scan_z = 255.0 * (scan_z - scan_z.min()) / (scan_z.max() - scan_z.min())
        # intensity = 255.0 * (intensity - intensity.min()) / (intensity.max() - intensity.min())
        # depth_ = 255.0 * (depth_ - depth_.min()) / (depth_.max() - depth_.min())

        scan_z = histogram_equalization(scan_z)
        # intensity = histogram_equalization(intensity)
        # depth_ = histogram_equalization(depth_)

        # b = 1.0 * (b - b.min()) / (b.max() - b.min())
        # g = 1.0 * (g - g.min()) / (g.max() - g.min())
        # r = 1.0 * (r - r.min()) / (r.max() - r.min())
        # scan_z = 1.0 * (scan_z - scan_z) / (scan_z.max())
        scan_z = 1.0 * (scan_z - scan_z.min()) / (scan_z.max() - scan_z.min())
        intensity = 1.0 * (intensity - intensity.min()) / (intensity.max() - intensity.min())
        depth_ = 1.0 * (depth_ - depth_.min()) / (depth_.max() - depth_.min())

        fusion_map[theta_, phi_, 0] = b
        fusion_map[theta_, phi_, 1] = g
        fusion_map[theta_, phi_, 2] = r
        fusion_map[theta_, phi_, 3] = scan_z
        fusion_map[theta_, phi_, 4] = intensity
        fusion_map[theta_, phi_, 5] = depth_

        # Step 1 Color Augmentation
        selection_num = random.randint(0, 7)
        # for selection_num in range(0, 7):
        if selection_num == 0:
            fusion_map[:,:,:3] = White_Noise(fusion_map[:,:,:3].astype(np.uint8))
        elif selection_num == 1:
            fusion_map[:,:,:3] = Gray(fusion_map[:,:,:3].astype(np.uint8))
        elif selection_num == 2:
            fusion_map[:,:,:3] = add_light(fusion_map[:,:,:3].astype(np.uint8))
        elif selection_num == 3:
            fusion_map[:,:,:3] = contrast_image(fusion_map[:,:,:3].astype(np.uint8))
        elif selection_num == 4:
            fusion_map[:,:,:3] = saturation_image(fusion_map[:,:,:3].astype(np.uint8))
        elif selection_num == 5:
            fusion_map[:,:,:3] = Equalization(fusion_map[:,:,:3].astype(np.uint8))

        fusion_map[:, :, 0] = 1.0 * (fusion_map[:, :, 0] - fusion_map[:, :, 0].min()) / (
                    fusion_map[:, :, 0].max() - fusion_map[:, :, 0].min())
        fusion_map[:, :, 1] = 1.0 * (fusion_map[:, :, 1] - fusion_map[:, :, 1].min()) / (
                    fusion_map[:, :, 1].max() - fusion_map[:, :, 1].min())
        fusion_map[:, :, 2] = 1.0 * (fusion_map[:, :, 2] - fusion_map[:, :, 2].min()) / (
                    fusion_map[:, :, 2].max() - fusion_map[:, :, 2].min())

        fusion_map = (fusion_map - mean) / std

        # Build a GT
        velodyne = self.gt_lidar_dir[item]
        with open(velodyne) as velo_object:
            contents = velo_object.readlines()

        points = []
        for content in contents:
            temp = []
            point = content.split(" ")

            temp.append(float(point[0]))
            temp.append(float(point[1]))
            temp.append(float(point[2]))
            temp.append(int(point[3]))
            temp.append(int(point[4]))
            temp.append(int(point[5]))
            points.append(temp)

        final = np.array(points)

        dtheta = np.radians(0.4)
        dphi = np.radians(90. / 512.0)

        point = final[:, :3]
        rgb = final[:, 3:]
        scan_x = point[:, 0]
        scan_y = point[:, 1]
        scan_z = point[:, 2]
        scan_r = rgb[:, 0]
        scan_g = rgb[:, 1]
        scan_b = rgb[:, 2]

        d = np.sqrt(pow(scan_x, 2) + pow(scan_y, 2) + pow(scan_z, 2))
        r = np.sqrt(pow(scan_x, 2) + pow(scan_y, 2))

        d[d == 0] = 0.000001
        r[r == 0] = 0.000001

        phi = np.radians(45.) - np.arcsin(scan_y / r)
        phi_ = (phi / dphi).astype(int)
        phi_[phi_ < 0] = 0
        phi_[phi >= 512] = 511

        theta = np.radians(2.) - np.arcsin(scan_z / d)
        theta_ = (theta / dtheta).astype(int)
        theta_[theta_ < 0] = 0
        theta_[theta_ >= 64] = 63

        depth_map_gt = np.zeros((self.proj_H, self.proj_W, self.C_GT))

        scan_b[scan_b == 255.0] = 1.0
        scan_g[scan_g == 255.0] = 1.0
        scan_r[scan_r == 255.0] = 1.0

        depth_map_gt[theta_, phi_, 0] = scan_b
        # depth_map_gt[theta_, phi_, 1] = scan_r - scan_b
        # depth_map_gt[:, :, 2] = 1.
        # depth_map_gt[theta_, phi_, 2] = 1. - scan_r

        input_tensor = fusion_map
        label_tensor = depth_map_gt

        # Step 1
        # Rotation -10~10
        input_tensor, label_tensor = Rotation(input_tensor, label_tensor, 5)

        # Scale 0.9~1.3
        input_tensor, label_tensor = Scale(input_tensor, label_tensor)

        # Translate -50~50
        input_tensor, label_tensor = Translate(input_tensor, label_tensor, 5, 5)

        # Flip
        input_tensor, label_tensor = Flip(input_tensor, label_tensor, 0.5)


        # NHWC -> NCHW
        input_tensor = input_tensor.transpose(2, 0, 1)
        label_tensor = label_tensor.transpose(2, 0, 1)

        input_tensor = torch.tensor(input_tensor).float()
        label_tensor = torch.tensor(label_tensor).float()

        return input_tensor, label_tensor, self.image_dir[item]

    def __len__(self):
        return len(self.image_dir)
        # return 10

class Valid_DataSet(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder

        self.proj_H = 64
        # self.proj_H = 48
        self.proj_W = 512
        self.proj_C = 6

        # self.C_GT = 3
        self.C_GT = 1

        self.calib_dir = glob(os.path.join(self.data_folder, "validating", "calib", "*.txt"))
        self.image_dir = glob(os.path.join(self.data_folder, "validating", "image_2", "*.png"))
        self.velodyne_dir = glob(os.path.join(self.data_folder, "validating", "velodyne", "*.bin"))
        self.gt_lidar_dir = glob(os.path.join(self.data_folder, "validating", "gt_txt", "*.txt"))

    def __getitem__(self, item):
        # Build Data from Raw Camera & LiDAR data
        # Build a Camera
        img = cv2.imread(self.image_dir[item])
        height, width, channel = img.shape

        # Build a LiDAR
        velodyne = np.fromfile(self.velodyne_dir[item], dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity

        coordinate = velodyne[:, :3]
        scan_x = coordinate[:, 0]
        scan_y = coordinate[:, 1]
        scan_z = coordinate[:, 2]

        add_ones = np.ones((coordinate.shape[0]))
        add_ones = np.expand_dims(add_ones, axis=1)
        coordinate_add_one = np.append(coordinate, add_ones, axis=1)

        intensity = velodyne[:, 3]

        # Build a Calibration
        calib_file = open(self.calib_dir[item], 'r')
        lines = calib_file.readlines()

        P2_rect = np.array(lines[2].split(" ")[1:]).astype(np.float32).reshape(3, 4)
        R0_rect = np.array(lines[4].split(" ")[1:]).astype(np.float32).reshape(3, 3)
        velo_to_cam = np.array(lines[5].split(" ")[1:]).astype(np.float32).reshape(3, 4)

        add_r0 = np.array([0., 0., 0.])
        add_r0_1 = np.array([0., 0., 0., 1.])
        R0_rect = np.append(R0_rect, add_r0.reshape(3, 1), axis=1)
        R0_rect = np.append(R0_rect, add_r0_1.reshape(1, 4), axis=0)

        add_velo = np.array([0., 0., 0., 1.])
        velo_to_cam = np.append(velo_to_cam, add_velo.reshape(1, 4), axis=0)

        homogeneous_matrix = np.matmul(np.matmul(P2_rect, R0_rect), velo_to_cam)

        result = np.matmul(homogeneous_matrix, coordinate_add_one.transpose(1, 0))

        u = result[0] / result[2]
        v = result[1] / result[2]

        condition = np.where(
            (result[0] >= 0) & (result[2] >= 0) & (scan_x >= 0) & (v < height) & (v >= 0) & (u >= 0) & (u < width))

        u = u[condition]
        v = v[condition]

        scan_x = scan_x[condition]
        scan_y = scan_y[condition]
        scan_z = scan_z[condition]
        intensity = intensity[condition]

        u = np.floor(u)
        u = np.minimum(width - 1, u)
        u = np.maximum(0, u).astype(np.int32)

        v = np.floor(v)
        v = np.minimum(height - 1, v)
        v = np.maximum(0, v).astype(np.int32)

        b = img[v, u][:, 0]
        g = img[v, u][:, 1]
        r = img[v, u][:, 2]

        dtheta = np.radians(0.4)
        dphi = np.radians(90. / 512.0)

        depth_ = np.sqrt(scan_x ** 2 + scan_y ** 2 + scan_z ** 2)
        range_ = np.sqrt(scan_x ** 2 + scan_y ** 2)

        depth_[depth_ == 0] = 0.000001
        range_[range_ == 0] = 0.000001

        phi = np.radians(45.) - np.arcsin(scan_y / range_)
        phi_ = (phi / dphi).astype(int)
        phi_[phi_ < 0] = 0
        phi_[phi_ >= 512] = 511

        theta = np.radians(2.) - np.arcsin(scan_z / depth_)
        theta_ = (theta / dtheta).astype(int)
        theta_[theta_ < 0] = 0
        theta_[theta_ >= 64] = 63

        fusion_map = np.zeros((self.proj_H, self.proj_W, self.proj_C))

        # scan_z = 255.0 * (scan_z - scan_z.min()) / (scan_z.max())
        # intensity = 255.0 * (intensity - intensity.min()) / (intensity.max() - intensity.min())
        # depth_ = 255.0 * (depth_ - depth_.min()) / (depth_.max() - depth_.min())

        b = 1.0 * (b - b.min()) / (b.max() - b.min())
        g = 1.0 * (g - g.min()) / (g.max() - g.min())
        r = 1.0 * (r - r.min()) / (r.max() - r.min())
        scan_z = 1.0 * (scan_z - scan_z.min()) / (scan_z.max())
        scan_z = 1.0 * (scan_z - scan_z.min()) / (scan_z.max() - scan_z.min())
        intensity = 1.0 * (intensity - intensity.min()) / (intensity.max() - intensity.min())
        depth_ = 1.0 * (depth_ - depth_.min()) / (depth_.max() - depth_.min())

        fusion_map[theta_, phi_, 0] = b
        fusion_map[theta_, phi_, 1] = g
        fusion_map[theta_, phi_, 2] = r
        fusion_map[theta_, phi_, 3] = scan_z
        fusion_map[theta_, phi_, 4] = intensity
        fusion_map[theta_, phi_, 5] = depth_

        fusion_map = (fusion_map - mean) / std

        # Build a GT
        velodyne = self.gt_lidar_dir[item]
        with open(velodyne) as velo_object:
            contents = velo_object.readlines()

        points = []
        for content in contents:
            temp = []
            point = content.split(" ")

            temp.append(float(point[0]))
            temp.append(float(point[1]))
            temp.append(float(point[2]))
            temp.append(int(point[3]))
            temp.append(int(point[4]))
            temp.append(int(point[5]))
            points.append(temp)

        final = np.array(points)

        dtheta = np.radians(0.4)
        dphi = np.radians(90. / 512.0)

        point = final[:, :3]
        rgb = final[:, 3:]
        scan_x = point[:, 0]
        scan_y = point[:, 1]
        scan_z = point[:, 2]
        scan_r = rgb[:, 0]
        scan_g = rgb[:, 1]
        scan_b = rgb[:, 2]

        d = np.sqrt(pow(scan_x, 2) + pow(scan_y, 2) + pow(scan_z, 2))
        r = np.sqrt(pow(scan_x, 2) + pow(scan_y, 2))

        d[d == 0] = 0.000001
        r[r == 0] = 0.000001

        phi = np.radians(45.) - np.arcsin(scan_y / r)
        phi_ = (phi / dphi).astype(int)
        phi_[phi_ < 0] = 0
        phi_[phi >= 512] = 511

        theta = np.radians(2.) - np.arcsin(scan_z / d)
        theta_ = (theta / dtheta).astype(int)
        theta_[theta_ < 0] = 0
        theta_[theta_ >= 64] = 63

        depth_map_gt = np.zeros((self.proj_H, self.proj_W, self.C_GT))

        scan_b[scan_b == 255.0] = 1.0
        scan_g[scan_g == 255.0] = 1.0
        scan_r[scan_r == 255.0] = 1.0

        depth_map_gt[theta_, phi_, 0] = scan_b
        # depth_map_gt[theta_, phi_, 1] = scan_r - scan_b
        # depth_map_gt[:, :, 2] = 1.
        # depth_map_gt[theta_, phi_, 2] = 1. - scan_r

        input_tensor = fusion_map
        label_tensor = depth_map_gt
        # label_tensor = 1.0 * (label_tensor - label_tensor.min()) / (label_tensor.max() - label_tensor.min())

        # NHWC -> NCHW4
        input_tensor = input_tensor.transpose(2, 0, 1)
        label_tensor = label_tensor.transpose(2, 0, 1)

        input_tensor = torch.tensor(input_tensor).float()
        label_tensor = torch.tensor(label_tensor).float()

        return input_tensor, label_tensor, self.image_dir[item]

    def __len__(self):
        return len(self.image_dir)
        # return 10

class Test_DataSet(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder

        self.proj_H = 64
        # self.proj_H = 48
        self.proj_W = 512
        self.proj_C = 6
        # self.proj_C = 8

        self.C_GT = 1

        self.calib_dir = glob(os.path.join(self.data_folder, "testing", "calib", "*.txt"))
        self.image_dir = glob(os.path.join(self.data_folder, "testing", "image_2", "*.png"))
        self.velodyne_dir = glob(os.path.join(self.data_folder, "testing", "velodyne", "*.bin"))

    def __getitem__(self, item):
        # Build Data from Raw Camera & LiDAR data
        # Build a Camera
        img = cv2.imread(self.image_dir[item])
        height, width, channel = img.shape

        # Build a LiDAR
        velodyne = np.fromfile(self.velodyne_dir[item], dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity

        coordinate = velodyne[:, :3]
        scan_x = coordinate[:, 0]
        scan_y = coordinate[:, 1]
        scan_z = coordinate[:, 2]

        add_ones = np.ones((coordinate.shape[0]))
        add_ones = np.expand_dims(add_ones, axis=1)
        coordinate_add_one = np.append(coordinate, add_ones, axis=1)

        intensity = velodyne[:, 3]

        # Build a Calibration
        calib_file = open(self.calib_dir[item], 'r')
        lines = calib_file.readlines()

        P2_rect = np.array(lines[2].split(" ")[1:]).astype(np.float32).reshape(3, 4)
        R0_rect = np.array(lines[4].split(" ")[1:]).astype(np.float32).reshape(3, 3)
        velo_to_cam = np.array(lines[5].split(" ")[1:]).astype(np.float32).reshape(3, 4)

        add_r0 = np.array([0., 0., 0.])
        add_r0_1 = np.array([0., 0., 0., 1.])
        R0_rect = np.append(R0_rect, add_r0.reshape(3, 1), axis=1)
        R0_rect = np.append(R0_rect, add_r0_1.reshape(1, 4), axis=0)

        add_velo = np.array([0., 0., 0., 1.])
        velo_to_cam = np.append(velo_to_cam, add_velo.reshape(1, 4), axis=0)

        homogeneous_matrix = np.matmul(np.matmul(P2_rect, R0_rect), velo_to_cam)

        result = np.matmul(homogeneous_matrix, coordinate_add_one.transpose(1, 0))

        u = result[0] / result[2]
        v = result[1] / result[2]

        condition = np.where(
            (result[0] >= 0) & (result[2] >= 0) & (scan_x >= 0) & (v < height) & (v >= 0) & (u >= 0) & (u < width))

        u = u[condition]
        v = v[condition]

        scan_x = scan_x[condition]
        scan_y = scan_y[condition]
        scan_z = scan_z[condition]
        intensity = intensity[condition]

        u = np.floor(u)
        u = np.minimum(width - 1, u)
        u = np.maximum(0, u).astype(np.int32)

        v = np.floor(v)
        v = np.minimum(height - 1, v)
        v = np.maximum(0, v).astype(np.int32)

        b = img[v, u][:, 0]
        g = img[v, u][:, 1]
        r = img[v, u][:, 2]

        dtheta = np.radians(0.4)
        dphi = np.radians(90. / 512.0)

        depth_ = np.sqrt(scan_x ** 2 + scan_y ** 2 + scan_z ** 2)
        range_ = np.sqrt(scan_x ** 2 + scan_y ** 2)

        depth_[depth_ == 0] = 0.000001
        range_[range_ == 0] = 0.000001

        phi = np.radians(45.) - np.arcsin(scan_y / range_)
        phi_ = (phi / dphi).astype(int)
        phi_[phi_ < 0] = 0
        phi_[phi_ >= 512] = 511

        theta = np.radians(2.) - np.arcsin(scan_z / depth_)
        theta_ = (theta / dtheta).astype(int)
        theta_[theta_ < 0] = 0
        theta_[theta_ >= 64] = 63

        fusion_map = np.zeros((self.proj_H, self.proj_W, self.proj_C))

        scan_z = 255.0 * (scan_z - scan_z.min()) / (scan_z.max())
        # intensity = 255.0 * (intensity - intensity.min()) / (intensity.max() - intensity.min())
        # depth_ = 255.0 * (depth_ - depth_.min()) / (depth_.max() - depth_.min())

        scan_z = histogram_equalization(scan_z)

        b = 1.0 * (b - b.min()) / (b.max() - b.min())
        g = 1.0 * (g - g.min()) / (g.max() - g.min())
        r = 1.0 * (r - r.min()) / (r.max() - r.min())
        scan_z = 1.0 * (scan_z - scan_z.min()) / (scan_z.max())
        scan_z = 1.0 * (scan_z - scan_z.min()) / (scan_z.max() - scan_z.min())
        intensity = 1.0 * (intensity - intensity.min()) / (intensity.max() - intensity.min())
        depth_ = 1.0 * (depth_ - depth_.min()) / (depth_.max() - depth_.min())

        fusion_map[theta_, phi_, 0] = b
        fusion_map[theta_, phi_, 1] = g
        fusion_map[theta_, phi_, 2] = r
        fusion_map[theta_, phi_, 3] = scan_z
        fusion_map[theta_, phi_, 4] = intensity
        fusion_map[theta_, phi_, 5] = depth_

        fusion_map = (fusion_map - mean) / std

        input_tensor = fusion_map

        # NHWC -> NCHW4
        input_tensor = input_tensor.transpose(2, 0, 1)

        input_tensor = torch.tensor(input_tensor).float()

        return input_tensor, input_tensor, self.image_dir[item]

    def __len__(self):
        return len(self.image_dir)
        # return 10
