import numpy as np
import pywt
import open3d as o3d

# 加载点云数据（适用于 .npy 格式）
def load_point_cloud_from_npy(data_path, sample_index=0):
    points = np.load(data_path)
    sample_points = points[sample_index]  # 提取第一个样本，形状为 (1024, 3)
    return sample_points

# 对点云数据的每一维 (x, y, z) 进行小波正变换
def wavelet_transform(points, wavelet='haar', level=2):
    coeffs_x = pywt.wavedec(points[:, 0], wavelet=wavelet, level=level)
    coeffs_y = pywt.wavedec(points[:, 1], wavelet=wavelet, level=level)
    coeffs_z = pywt.wavedec(points[:, 2], wavelet=wavelet, level=level)
    return coeffs_x, coeffs_y, coeffs_z

def keep_low_frequency_only(coeffs_x,coeffs_y,coeffs_z):
    low_freq_coeffs_x = [coeffs_x[0]] + [np.zeros_like(c) for c in coeffs_x[1:]]
    low_freq_coeffs_y = [coeffs_y[0]] + [np.zeros_like(c) for c in coeffs_y[1:]]
    low_freq_coeffs_z = [coeffs_z[0]] + [np.zeros_like(c) for c in coeffs_z[1:]]
    return low_freq_coeffs_x, low_freq_coeffs_y, low_freq_coeffs_z

# 保留高频系数，将低频系数设为零（与原代码中保留低频相反的操作）
def keep_high_frequency_only(coeffs_x, coeffs_y, coeffs_z):
    high_freq_coeffs_x = [np.zeros_like(coeffs_x[0])]
    high_freq_coeffs_y = [np.zeros_like(coeffs_y[0])]
    high_freq_coeffs_z = [np.zeros_like(coeffs_z[0])]
    for i in range(1, len(coeffs_x)):
        high_freq_coeffs_x.append(coeffs_x[i])
        high_freq_coeffs_y.append(coeffs_y[i])
        high_freq_coeffs_z.append(coeffs_z[i])
    return high_freq_coeffs_x, high_freq_coeffs_y, high_freq_coeffs_z

# 小波逆变换重构点云数据
def inverse_wavelet_transform(coeffs_x, coeffs_y, coeffs_z, wavelet='haar'):
    reconstructed_x = pywt.waverec(coeffs_x, wavelet=wavelet)
    reconstructed_y = pywt.waverec(coeffs_y, wavelet=wavelet)
    reconstructed_z = pywt.waverec(coeffs_z, wavelet=wavelet)
    return np.vstack((reconstructed_x, reconstructed_y, reconstructed_z)).T

# 主函数：加载点云 -> 小波变换 -> 逆变换 -> 展示
data_path = 'E:\\BJCWorkshop\\LPF-Defense-main\\model\\Data\\train_data.npy'  # 替换为你的点云数据路径
points = load_point_cloud_from_npy(data_path, sample_index=0)

# 小波正变换
coeffs_x, coeffs_y, coeffs_z = wavelet_transform(points)

# 保留低频系数，其他系数设为零
low_freq_coeffs_x, low_freq_coeffs_y, low_freq_coeffs_z = keep_low_frequency_only(coeffs_x, coeffs_y, coeffs_z)

high_freq_coeffs_x,high_freq_coeffs_y,high_freq_coffs_z = keep_high_frequency_only(coeffs_x,coeffs_y,coeffs_z)
# 小波逆变换，重构仅包含低频信息的点云数据
high_freq_points = inverse_wavelet_transform(high_freq_coeffs_x, high_freq_coeffs_y, high_freq_coffs_z)
low_freq_points = inverse_wavelet_transform(low_freq_coeffs_x,low_freq_coeffs_y,low_freq_coeffs_z)
# 小波逆变换
# reconstructed_points = inverse_wavelet_transform(coeffs_x, coeffs_y, coeffs_z)
reconstructed_points_low_freq = inverse_wavelet_transform(low_freq_coeffs_x,low_freq_coeffs_y,low_freq_coeffs_z)
reconstructed_points_high_freq = inverse_wavelet_transform(high_freq_coeffs_x, high_freq_coeffs_y,high_freq_coffs_z)

# 显示原始和重建点云
original_pcd = o3d.geometry.PointCloud()
original_pcd.points = o3d.utility.Vector3dVector(points)

#小波变换后的点云高频版本
reconstructed_pcd_high = o3d.geometry.PointCloud()
reconstructed_pcd_high.points = o3d.utility.Vector3dVector(reconstructed_points_high_freq)

#小波变化后的点云低频版本
reconstructed_pcd_low = o3d.geometry.PointCloud()
reconstructed_pcd_low.points = o3d.utility.Vector3dVector(reconstructed_points_low_freq)
# 低频点云版本
# low_freq_pcd = o3d.geometry.PointCloud()
# low_freq_pcd.points = o3d.utility.Vector3dVector(low_freq_points)
# 可视化
o3d.visualization.draw_geometries([original_pcd], window_name="Original Point Cloud")
o3d.visualization.draw_geometries([reconstructed_pcd_low], window_name="Reconstructed Point Cloud")
