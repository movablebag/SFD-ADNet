import os
import numpy as np
import random
import torch
from PIL import Image
import pyshtools as pysh
from math import sqrt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from cv2 import getGaussianKernel

# 画3D点云散点图
def plot_pc(pc, second_pc=None, s=4, o=0.6):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],)
    fig.add_trace(
        go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode='markers', marker=dict(size=s, opacity=o)),
        row=1, col=1
    )
    if second_pc is not None:
        fig.add_trace(
            go.Scatter3d(x=second_pc[:, 0], y=second_pc[:, 1], z=second_pc[:, 2], mode='markers', marker=dict(size=s, opacity=o)),
            row=1, col=2
        )
    fig.update_layout(scene_aspectmode='data')
    fig.show()

# 将点云投影到以质心为中心的单位球表面
def convert_pc_to_grid(pc, lmax, device="cuda"):
    pc = torch.from_numpy(pc).to(device)

    grid = pysh.SHGrid.from_zeros(lmax, grid='DH')
    nlon = grid.nlon
    nlat = grid.nlat
    ngrid = nlon * nlat
    grid_lon = torch.from_numpy(np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)).to(device)
    grid_lat = torch.from_numpy(np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)).to(device)
    grid_lon = torch.broadcast_to(grid_lon, (nlat, nlon))
    grid_lat = torch.broadcast_to(grid_lat.view(nlat, 1), (nlat, nlon))
    grid_lon = grid_lon.reshape(1, ngrid)
    grid_lat = grid_lat.reshape(1, ngrid)

    origin = torch.mean(pc, axis=0)  # 质心
    pc -= origin  # 相对于质心的坐标系
    npc = len(pc)
    origin = origin.to("cpu").numpy()

    pc_x, pc_y, pc_z = pc[:, 0], pc[:, 1], pc[:, 2]

    pc_r = torch.sqrt(pc_x * pc_x + pc_y * pc_y + pc_z * pc_z)
    pc_lat = torch.arcsin(pc_z / pc_r)
    pc_lon = torch.atan2(pc_y, pc_x)
    pc_r = pc_r.view(npc, 1)
    pc_lat = pc_lat.view(npc, 1)
    pc_lon = pc_lon.view(npc, 1)

    dist = -torch.cos(grid_lat) * torch.cos(pc_lat) * torch.cos(grid_lon - pc_lon) + torch.sin(grid_lat) * torch.sin(pc_lat)

    argmin = torch.argmin(dist, axis=0)
    grid_r = pc_r[argmin].view(nlat, nlon)
    grid.data = grid_r.to("cpu").numpy()  # 投影后的数据

    argmin = torch.argmin(dist, axis=1)
    flag = torch.zeros(ngrid, dtype=bool)
    flag[argmin] = True  # 投影过程中有效的极角标记
    flag = flag.to("cpu").numpy()

    return grid, flag, origin

# 低通滤波器，用于衰减高频分量
def low_pass_filter(grid, sigma):
    clm = grid.expand()

    weights = getGaussianKernel(clm.coeffs.shape[1] * 2 - 1, sigma)[clm.coeffs.shape[1] - 1:]
    weights /= weights[0]

    clm.coeffs *= weights
    low_passed_grid = clm.expand()

    return low_passed_grid

# 点云数据丢失时的补偿
def duplicate_randomly(pc, size):
    loss_cnt = size - pc.shape[0]
    if loss_cnt <= 0:
        return pc
    rand_indices = np.random.randint(0, pc.shape[0], size=loss_cnt)
    dup = pc[rand_indices]
    return np.concatenate((pc, dup))

# 核心处理方法
def our_method(pc, lmax, sigma, pc_size=1024, device="cuda"):
    grid, flag, origin = convert_pc_to_grid(pc, lmax, device)
    smooth_grid = low_pass_filter(grid, sigma)
    smooth_pc = convert_grid_to_pc(smooth_grid, flag, origin)
    smooth_pc = duplicate_randomly(smooth_pc, pc_size)
    return smooth_pc

# 由球面投影重建点云
def convert_grid_to_pc(grid, flag, origin):
    nlon = grid.nlon
    nlat = grid.nlat
    lon = np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)
    lat = np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)
    lon = np.broadcast_to(lon.reshape((1, nlon)), (nlat, nlon))
    lat = np.broadcast_to(lat.reshape((nlat, 1)), (nlat, nlon))
    r = grid.data

    z = np.sin(lat) * r
    t = np.cos(lat) * r
    x = t * np.cos(lon)
    y = t * np.sin(lon)

    pc = np.zeros(grid.data.shape + (3,))
    pc[:, :, 0] = x
    pc[:, :, 1] = y
    pc[:, :, 2] = -z  # 确保正确的方向
    pc = pc.reshape((-1, 3))
    pc = pc[flag, :]
    pc += origin

    return pc

class SphericalHarmonicTransform:
    def __init__(self, args, alpha=1.0, dataset_list=None, base_dir=None):
        self.alpha = alpha
        self.dataset_list = dataset_list
        self.base_dir = base_dir
        self.dataset = args.data

        self.filter_flag = args.freq_analyse
        self.filter_S = args.freq_analyse_S
        self.high_or_low = args.freq_analyse_high_or_low

        domain_path = os.path.join(self.base_dir, self.dataset_list[0])
        if self.dataset == "VLCS":
            domain_path = os.path.join(domain_path, "full")
        self.class_names = sorted(os.listdir(domain_path))

    def __call__(self, pc, domain_label):
        if self.filter_flag == 1:
            pc_s2o = our_method(pc, lmax=self.filter_S, sigma=5)
            domain_s = None
            lam = None
        else:
            pc_s, label_s, domain_s = self.sample_image(domain_label)
            pc_s2o, lam = self.spherical_mix(pc, pc_s, alpha=self.alpha)

        return pc_s2o, domain_s, lam

    def spherical_mix(self, pc1, pc2, alpha=1.0):
        # Perform spherical harmonic mixing between two point clouds
        pc1_smooth = our_method(pc1, lmax=20, sigma=5)
        pc2_smooth = our_method(pc2, lmax=20, sigma=5)
        lam = np.random.uniform(0, alpha)
        mixed_pc = lam * pc1_smooth + (1 - lam) * pc2_smooth
        return mixed_pc, lam

    def sample_image(self, domain_label):
        domain_idx = random.randint(0, len(self.dataset_list) - 1)
        other_domain_name = self.dataset_list[domain_idx]
        class_idx = random.randint(0, len(self.class_names)-1)
        other_class_name = self.class_names[class_idx]
        base_dir_domain = os.path.join(self.base_dir, other_domain_name)
        if self.dataset == "VLCS":
            base_dir_domain = os.path.join(base_dir_domain, "full")
        base_dir_domain_class = os.path.join(base_dir_domain, other_class_name)
        other_id = np.random.choice(os.listdir(base_dir_domain_class))
        other_pc = np.load(os.path.join(base_dir_domain_class, other_id))  # 假设是点云数据的.npy文件

        return other_pc, class_idx, domain_idx
