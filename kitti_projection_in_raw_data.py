#####################################################
# Loader code kitti's raw data for inference LBMNet #
#####################################################

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob

import torch
from torch.utils.data import Dataset

mean = [0.3248, 0.3373, 0.3436, 0.2411, 0.2518, 0.1794]
std = [0.2365, 0.2438, 0.2502, 0.1624, 0.1484, 0.1807]

def histogram_equalization(image):
    data = image.copy().flatten()
    hist, bins = np.histogram(data, 256, density=True)
    cdf = hist.cumsum()
    img_eq = np.interp(data, bins[:-1], cdf)

    return img_eq

def calib_velo2cam(filepath):
    """
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)
    return R, T


def calib_cam2cam(filepath, mode='02'):
    """
    If your image is 'rectified image' :
        get only Projection(P : 3x4) matrix is enough
    but if your image is 'distorted image'(not rectified image) :
        you need undistortion step using distortion coefficients(5 : D)

    in this code, I'll get P matrix since I'm using rectified image
    """
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == ('P_rect_' + mode):
                P_ = np.fromstring(val, sep=' ')
                P_ = P_.reshape(3, 4)
                # erase 4th column ([0,0,0])
                P_ = P_[:3, :3]
    return P_

class Raw_Dataset(Dataset):
    def __init__(self, data_folder, sequence_num):
        self.data_folder = data_folder

        self.proj_H = 64

        self.proj_W = 512
        self.proj_C = 6

        self.image_dir = glob(os.path.join(self.data_folder, "2011_09_26_drive_" + sequence_num + "_sync", "image_02", "data", "*.png"))
        self.velodyne_dir = glob(os.path.join(self.data_folder, "2011_09_26_drive_" + sequence_num + "_sync", "velodyne_points", "data", "*.bin"))
        self.v2c_filepath = self.data_folder + 'calib_velo_to_cam.txt'
        self.c2c_filepath = self.data_folder + 'calib_cam_to_cam.txt'


    def __getitem__(self, item):
        # Build an raw image
        image = cv2.imread(self.image_dir[item])
        height, width = image.shape[:2]

        # bin file -> numpy array
        velo_points = np.fromfile(self.velodyne_dir[item], dtype=np.float32).reshape(-1, 4)

        # R_vc = Rotation matrix (velodyne -> camera)
        # T_vc = Translation matrix (velodyne -> camera)
        R_vc, T_vc = calib_velo2cam(self.v2c_filepath)

        # P_ = Projection matrix (camera coordinates 3d points -> image plane 2d points)
        P_ = calib_cam2cam(self.c2c_filepath)

        scan_x = velo_points[:, 0]
        scan_y = velo_points[:, 1]
        scan_z = velo_points[:, 2]
        intensity = velo_points[:, 3]

        v_fov, h_fov = (-24.9, 2.0), (-90, 90)

        # RT_ = rotation matrix & translation matrix
        RT_ = np.concatenate((R_vc, T_vc), axis=1)

        xyz_ = np.vstack((scan_x, scan_y, scan_z))

        # stack (1, n) arrays filled with the number 1
        one_mat = np.full((1, xyz_.shape[1]), 1)

        xyz_ = np.concatenate((xyz_, one_mat), axis=0)

        homogeneous_matrix = np.matmul(P_, RT_)

        Result = np.matmul(homogeneous_matrix, xyz_)

        u = Result[0] / Result[2]
        v = Result[1] / Result[2]

        fov_condition = np.where(
            (Result[0] >= 0) & (Result[2] >= 0) & (scan_x >= 0) & (v < height) & (v >= 0) & (u >= 0) & (u < width))

        u = u[fov_condition]
        v = v[fov_condition]

        scan_x = scan_x[fov_condition]
        scan_y = scan_y[fov_condition]
        scan_z = scan_z[fov_condition]
        intensity = intensity[fov_condition]

        u = np.floor(u)
        u = np.minimum(width - 1, u)
        u = np.maximum(0, u).astype(np.int32)

        v = np.floor(v)
        v = np.minimum(height - 1, v)
        v = np.maximum(0, v).astype(np.int32)

        b = image[v, u][:, 0]
        g = image[v, u][:, 1]
        r = image[v, u][:, 2]

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

        scan_z = 255.0 * (scan_z - scan_z.min()) / (scan_z.max() - scan_z.min())
        histogram_z = histogram_equalization(scan_z)

        b_ = 1.0 * (b - b.min()) / (b.max() - b.min())
        g_ = 1.0 * (g - g.min()) / (g.max() - g.min())
        r_ = 1.0 * (r - r.min()) / (r.max() - r.min())
        scan_z_ = 1.0 * (scan_z - scan_z.min()) / (scan_z.max())
        scan_z_ = 1.0 * (scan_z_ - scan_z_.min()) / (scan_z_.max() - scan_z_.min())
        intensity = 1.0 * (intensity - intensity.min()) / (intensity.max() - intensity.min())
        depth_ = 1.0 * (depth_ - depth_.min()) / (depth_.max() - depth_.min())

        fusion_map[theta_, phi_, 0] = b_
        fusion_map[theta_, phi_, 1] = g_
        fusion_map[theta_, phi_, 2] = r_
        fusion_map[theta_, phi_, 3] = histogram_z
        fusion_map[theta_, phi_, 4] = intensity
        fusion_map[theta_, phi_, 5] = depth_

        fusion_map = (fusion_map - mean) / std

        input_tensor = fusion_map

        # NHWC -> NCHW4
        input_tensor = input_tensor.transpose(2, 0, 1)

        input_tensor = torch.tensor(input_tensor).float()

        return input_tensor, theta_, phi_, u, v, scan_x, scan_y, scan_z, b, g, r, self.image_dir[item]

    def __len__(self):
        return len(self.image_dir)
