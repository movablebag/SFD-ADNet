import numpy as np
import pywt
import torch
import plotly.graph_objects as go
from cv2 import getGaussianKernel
from plotly.subplots import make_subplots

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

def convert_pc_to_grid(pc, lmax, device="cuda"):
    pc = torch.from_numpy(pc).to(device)
    nlon = lmax
    nlat = lmax
    ngrid = nlon * nlat
    grid_lon = torch.from_numpy(np.linspace(0, np.pi * 2, num=nlon, endpoint=False)).to(device)
    grid_lat = torch.from_numpy(np.linspace(-np.pi / 2, np.pi / 2, num=nlat, endpoint=True)).to(device)
    grid_lon = torch.broadcast_to(grid_lon, (nlat, nlon))
    grid_lat = torch.broadcast_to(grid_lat.view(nlat, 1), (nlat, nlon))
    grid_lon = grid_lon.reshape(1, ngrid)
    grid_lat = grid_lat.reshape(1, ngrid)

    origin = torch.mean(pc, axis=0)
    pc -= origin
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

    return grid_r, origin

def convert_grid_to_pc(grid_r, origin):
    nlat, nlon = grid_r.shape
    lon = np.linspace(0, np.pi * 2, num=nlon, endpoint=False)
    lat = np.linspace(-np.pi / 2, np.pi / 2, num=nlat, endpoint=True)
    lon, lat = np.meshgrid(lon, lat)
    r = grid_r
    if isinstance(r, torch.Tensor):
        r = r.numpy()

    print("r的类型",type(r))
    print("lat的类型",type(lat))
    z = np.sin(lat) * r
    x = np.cos(lat) * np.cos(lon) * r
    y = np.cos(lat) * np.sin(lon) * r

    pc = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    pc += origin
    return pc
#
def low_pass_filter(grid_r, sigma):
    coeffs = pywt.dwt2(grid_r, 'haar')  # Use Haar wavelet for demonstration
    cA, (cH, cV, cD) = coeffs

    weights = getGaussianKernel(cA.shape[0], sigma)
    cA *= weights

    grid_r_smooth = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    return grid_r_smooth

def haar_wavelet_transform(grid_r,levels=3):
    # coeffs = pywt.dwt2(grid_r, 'haar')
    # cA, (cH, cV, cD) = coeffs
    # grid_r_transformed = pywt.idwt2((cA,(cH,cV,cD)), 'haar')
    # return grid_r_transformed
    #------------------------------------
    coeffs = pywt.wavedec2(grid_r, 'haar', level=levels)
    # Reconstruct the grid_r using inverse transform with all coefficients
    grid_r_transformed = pywt.waverec2(coeffs, 'haar')
    return grid_r_transformed

def duplicate_randomly(pc, size):
    loss_cnt = size - pc.shape[0]
    if loss_cnt <= 0:
        return pc
    rand_indices = np.random.randint(0, pc.shape[0], size=loss_cnt)
    dup = pc[rand_indices]
    return np.concatenate((pc, dup))

def our_method(pc, lmax, pc_size=1024, device="cuda"):
    grid_r, origin = convert_pc_to_grid(pc, lmax, device)
    # smooth_grid_r = low_pass_filter(grid_r,sigma)
    smooth_grid = haar_wavelet_transform(grid_r)
    smooth_pc = convert_grid_to_pc(smooth_grid, origin)
    # smooth_pc = duplicate_randomly(smooth_pc, pc_size)
    return smooth_pc
