import numpy as np
import torch
import torch.nn.functional as nnf
from torch_harmonics import RealSHT, InverseRealSHT
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 可视化点云的函数
def plot_pc(pc, second_pc=None, s=4, o=0.6):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], )
    fig.add_trace(
        go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode='markers', marker=dict(size=s, opacity=o)),
        row=1, col=1
    )
    if second_pc is not None:
        fig.add_trace(
            go.Scatter3d(x=second_pc[:, 0], y=second_pc[:, 1], z=second_pc[:, 2], mode='markers',
                         marker=dict(size=s, opacity=o)),
            row=1, col=2
        )
    fig.update_layout(scene_aspectmode='data')
    fig.show()


# 将点云投影到球面网格上
def convert_pc_to_grid(pc, nlat, nlon, device="cuda"):
    pc = torch.from_numpy(pc).to(device)

    # 网格的经纬度计算
    lon = torch.linspace(0, 2 * np.pi, nlon, device=device)
    lat = torch.linspace(-np.pi / 2, np.pi / 2, nlat, device=device)

    lon, lat = torch.meshgrid(lon, lat, indexing='ij')

    # 将点云转换到球坐标系
    origin = torch.mean(pc, axis=0)  # 计算点云的质心
    pc -= origin
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    pc_lat = torch.arcsin(z / r)
    pc_lon = torch.atan2(y, x)

    # 根据距离最近的经纬度找到对应网格
    dist = -torch.cos(lat) * torch.cos(pc_lat) * torch.cos(lon - pc_lon) + torch.sin(lat) * torch.sin(pc_lat)
    argmin = torch.argmin(dist, dim=1)
    grid_r = r[argmin].reshape(nlat, nlon)

    return grid_r, origin


# 将球面上的数据转换回点云
def convert_grid_to_pc(grid, nlat, nlon, origin):
    # 经度和纬度
    lon = np.linspace(0, 2 * np.pi, nlon)
    lat = np.linspace(-np.pi / 2, np.pi / 2, nlat)
    lon, lat = np.meshgrid(lon, lat)

    # 球坐标系转换回笛卡尔坐标
    r = grid.data
    z = np.sin(lat) * r
    t = np.cos(lat) * r
    x = t * np.cos(lon)
    y = t * np.sin(lon)

    pc = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    pc += origin  # 加回原点偏移量

    return pc


# 将球面网格数据转换为频域数据并进行逆变换
def our_method(pc, nlat, nlon, lmax, sigma, pc_size=1024, device="cuda"):
    sht = RealSHT(nlat, nlon, lmax=lmax, mmax=lmax // 2).to(device)
    isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=lmax // 2).to(device)

    # 将点云转换为球面网格数据
    grid_r, origin = convert_pc_to_grid(pc, nlat, nlon, device)

    # 将网格数据转换为频域数据
    freq_data = sht(grid_r)

    # 进行频域滤波（高斯低通滤波）
    weights = torch.exp(-torch.arange(freq_data.size(-1), device=device) ** 2 / (2 * sigma ** 2))
    freq_data_filtered = freq_data * weights

    # 通过逆变换回到空间域
    grid_reconstructed = isht(freq_data_filtered)

    # 将重建后的网格数据转换回三维点云
    smooth_pc = convert_grid_to_pc(grid_reconstructed, nlat, nlon, origin.cpu().numpy())

    return smooth_pc
