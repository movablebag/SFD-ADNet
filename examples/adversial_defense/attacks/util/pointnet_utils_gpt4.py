import numpy as np

def normalize_points_np(points):
    """
    Normalize the point cloud to have zero mean and unit variance.
    
    Args:
        points (np.ndarray): The input point cloud of shape [N, 3].
    
    Returns:
        np.ndarray: The normalized point cloud.
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=-1)))
    points = points / furthest_distance
    return points

def random_sample_points_np(points, num_points):
    """
    Randomly sample a fixed number of points from the point cloud.
    
    Args:
        points (np.ndarray): The input point cloud of shape [N, 3].
        num_points (int): The number of points to sample.
    
    Returns:
        np.ndarray: The sampled point cloud of shape [num_points, 3].
    """
    N = points.shape[0]
    if N < num_points:
        raise ValueError("The number of points in the point cloud is less than the number of points to sample.")
    indices = np.random.choice(N, num_points, replace=False)
    return points[indices]

def center_points_np(points):
    """
    Center the point cloud to have zero mean.
    
    Args:
        points (np.ndarray): The input point cloud of shape [N, 3].
    
    Returns:
        np.ndarray: The centered point cloud.
    """
    centroid = np.mean(points, axis=0)
    return points - centroid

def scale_points_np(points, scale=1.0):
    """
    Scale the point cloud to a specific range.
    
    Args:
        points (np.ndarray): The input point cloud of shape [N, 3].
        scale (float): The scale factor.
    
    Returns:
        np.ndarray: The scaled point cloud.
    """
    return points * scale

def rotate_points_np(points, angle):
    """
    Rotate the point cloud around the z-axis.
    
    Args:
        points (np.ndarray): The input point cloud of shape [N, 3].
        angle (float): The rotation angle in radians.
    
    Returns:
        np.ndarray: The rotated point cloud.
    """
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([
        [cosval, -sinval, 0],
        [sinval, cosval, 0],
        [0, 0, 1]
    ])
    return np.dot(points, rotation_matrix)

def jitter_points_np(points, sigma=0.01, clip=0.05):
    """
    Add random noise to the point cloud for data augmentation.
    
    Args:
        points (np.ndarray): The input point cloud of shape [N, 3].
        sigma (float): Standard deviation of the Gaussian noise.
        clip (float): Clipping value for the noise.
    
    Returns:
        np.ndarray: The jittered point cloud.
    """
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    return points + noise