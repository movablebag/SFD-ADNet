import torch
import numpy as np
from model.sfno.models.sfno import SphericalFourierNeuralOperatorNet

# Initialize SFNO model (assuming the model uses some default parameters)
sfno_model = SphericalFourierNeuralOperatorNet(
    in_chans=1,  # Input channels, adjust based on your grid structure
    out_chans=1,  # Output channels, same here
    embed_dim=64,  # Number of hidden channels in SFNO layers
    num_layers=4,  # Depth of the network, number of layers
    img_size=(64, 128),  # Adjust based on your spherical grid resolution
    spectral_transform="sht",  # Use spherical harmonics transform (SHT)
    hard_thresholding_fraction=1.0
).cuda()

def convert_pc_to_sfno_grid(pc, sfno_model, device="cuda"):
    """
    Convert a point cloud to a spherical grid suitable for SFNO.
    Args:
        pc: Input point cloud as a numpy array of shape (N, 3)
        sfno_model: Initialized SFNO model
        device: Device to perform computation on ("cuda" or "cpu")
    Returns:
        grid_frequency: Frequency domain representation of the grid
        origin: The origin used for centering the point cloud
    """
    # Project point cloud onto a unit sphere
    pc = torch.from_numpy(pc).to(device)
    origin = torch.mean(pc, axis=0)  # Center of the point cloud
    pc -= origin  # Translate to unit sphere

    # Compute spherical coordinates from the point cloud
    radius = torch.norm(pc, dim=-1, keepdim=True)
    spherical_pc = pc / radius  # Normalize points onto the unit sphere

    # Convert 3D Cartesian coordinates to spherical coordinates (lon, lat)
    x, y, z = spherical_pc[:, 0], spherical_pc[:, 1], spherical_pc[:, 2]
    lon = torch.atan2(y, x)  # Longitude
    lat = torch.asin(z)  # Latitude

    # Create a spherical grid and interpolate the point cloud onto the grid
    nlat, nlon = 64, 128
    grid = torch.zeros(1, 1, nlat, nlon).to(device)

    # Map point cloud to grid using bilinear interpolation or nearest neighbor (example below is nearest)
    lat_idx = ((lat + np.pi / 2) * (nlat - 1) / np.pi).long()
    lon_idx = ((lon + np.pi) * (nlon - 1) / (2 * np.pi)).long()

    # Populate the grid with some feature (e.g., point density)
    for i in range(pc.shape[0]):
        grid[0, 0, lat_idx[i], lon_idx[i]] += 1  # Simple accumulation, can be replaced with more complex interpolation

    # Apply SFNO model to convert to frequency domain
    grid_frequency = sfno_model(grid)

    return grid_frequency, origin

def convert_sfno_grid_to_pc(grid_frequency, sfno_model, origin):
    """
    Convert a frequency domain grid back to a point cloud.
    Args:
        grid_frequency: Frequency domain representation of the grid
        sfno_model: Initialized SFNO model
        origin: The origin used for centering the point cloud
    Returns:
        pc: Reconstructed point cloud as a numpy array
    """
    # Transform the frequency domain grid back to the spatial domain
    grid_spatial = sfno_model.forward_features(grid_frequency)

    # Convert spherical grid to point cloud
    nlat, nlon = grid_spatial.shape[-2], grid_spatial.shape[-1]
    lon = torch.linspace(0, 2 * np.pi, nlon, device=grid_spatial.device)
    lat = torch.linspace(-np.pi / 2, np.pi / 2, nlat, device=grid_spatial.device)

    lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing='ij')
    x = torch.cos(lat_grid) * torch.cos(lon_grid)
    y = torch.cos(lat_grid) * torch.sin(lon_grid)
    z = torch.sin(lat_grid)

    # Reshape the grid and add the origin to return to original coordinates
    pc = torch.stack([x, y, z], dim=-1).reshape(-1, 3) + origin
    return pc.cpu().numpy()

def our_method_sfno(pc, sfno_model, sigma, pc_size=1024, device="cuda"):
    """
    Example method that takes a point cloud, converts it to frequency domain, and back.
    Args:
        pc: Input point cloud as a numpy array (N, 3)
        sfno_model: Initialized SFNO model
        sigma: Some parameter for processing
        pc_size: Desired size of output point cloud
        device: Device to perform computation on ("cuda" or "cpu")
    Returns:
        processed_pc: Processed point cloud after frequency domain operations
    """
    # Convert point cloud to SFNO grid (frequency domain)
    grid_frequency, origin = convert_pc_to_sfno_grid(pc, sfno_model, device=device)

    # Perform any operations in the frequency domain (if needed, can involve sigma)
    # ...

    # Convert frequency domain back to point cloud
    processed_pc = convert_sfno_grid_to_pc(grid_frequency, sfno_model, origin)

    return processed_pc
