import numpy as np
import spherical_harmonics_defense as shd1

# Load data
train_data = np.load(r'E:\BJCWorkshop\LPF-Defense-main\model\Data\train_data.npy')

# Parameters
lmax = 13
sigma = 20
pc_size = 1024
device = "cuda"

# Process all point clouds to extract low-frequency components
low_freq_data = []

for idx, pc in enumerate(train_data):
    smooth_pc = shd1.our_method(pc, lmax, sigma, pc_size, device)
    low_freq_data.append(smooth_pc)

# Save low-frequency data
low_freq_data = np.array(low_freq_data)
np.save(r'E:\BJCWorkshop\LPF-Defense-main\model\Data\low_frequency_data.npy', low_freq_data)
