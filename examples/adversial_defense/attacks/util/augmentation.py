import numpy as np

def rotate_point_cloud(points):
    """
    Randomly rotate the point cloud around the y-axis.
    
    Args:
        points (np.ndarray): The input point cloud of shape [N, 3].
    
    Returns:
        np.ndarray: The rotated point cloud.
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_points = np.dot(points, rotation_matrix)
    return rotated_points

def jitter_point_cloud(points, sigma=0.01, clip=0.05):
    """
    Randomly jitter points. Jittering is applied independently to each point.
    
    Args:
        points (np.ndarray): The input point cloud of shape [N, 3].
        sigma (float): Standard deviation of the jitter.
        clip (float): Clipping value for the jitter.
    
    Returns:
        np.ndarray: The jittered point cloud.
    """
    jitter = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    jittered_points = points + jitter
    return jittered_points 