import os, sys, h5py, pickle, numpy as np, logging, os.path as osp
import torch
from torch.utils.data import Dataset
# Remove the problematic imports
# from ..build import DATASETS
# from openpoints.models.layers import fps

# Add a simple FPS implementation
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3] or [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint] or [npoint]
    """
    if len(xyz.shape) == 2:
        # Handle single point cloud [N, 3]
        xyz = xyz.unsqueeze(0)  # [1, N, 3]
        single_cloud = True
    else:
        single_cloud = False
        
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    if single_cloud:
        return centroids.squeeze(0)  # [npoint]
    return centroids

def fps(data, number):
    '''
    data: [B, N, C] or [N, C]
    number: int
    '''
    if len(data.shape) == 2:
        # Handle single point cloud
        fps_idx = farthest_point_sample(data[:, :3].contiguous(), number)
        fps_data = data[fps_idx]
    else:
        # Handle batch
        fps_idx = farthest_point_sample(data[:, :, :3].contiguous(), number)
        fps_data = torch.gather(
            data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    return fps_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


# Remove the @DATASETS.register_module() decorator
class ScanObjectNNHardest(Dataset):
    """The hardest variant of ScanObjectNN. 
    The data we use is: `training_objectdataset_augmentedrot_scale75.h5`[1], 
    where there are 2048 points in training and testing.
    The number of training samples is: 11416, and the number of testing samples is 2882. 
    Args:
    """
    classes = [
        "bag",
        "bin",
        "box",
        "cabinet",
        "chair",
        "desk",
        "display",
        "door",
        "shelf",
        "table",
        "bed",
        "pillow",
        "sink",
        "sofa",
        "toilet",
    ]
    num_classes = 15
    gravity_dim = 1

    def __init__(self, data_dir, split,
                 num_points=2048,
                 uniform_sample=True,
                 transform=None,
                 **kwargs):
        super().__init__()
        self.partition = split
        self.transform = transform
        self.num_points = num_points
        slit_name = 'training' if split == 'train' else 'test'
        h5_name = os.path.join(
            data_dir, f'{slit_name}_objectdataset_augmentedrot_scale75.h5')

        if not osp.isfile(h5_name):
            raise FileExistsError(
                f'{h5_name} does not exist, please download dataset at first')
        with h5py.File(h5_name, 'r') as f:
            self.points = np.array(f['data']).astype(np.float32)
            self.labels = np.array(f['label']).astype(int)

        if slit_name == 'test' and uniform_sample:
            precomputed_path = os.path.join(
                data_dir, f'{slit_name}_objectdataset_augmentedrot_scale75_1024_fps.pkl')
            if not os.path.exists(precomputed_path):
                points = torch.from_numpy(self.points).to(torch.float32).cuda()
                self.points = fps(points, 1024).cpu().numpy()
                with open(precomputed_path, 'wb') as f:
                    pickle.dump(self.points, f)
                    print(f"{precomputed_path} saved successfully")
            else:
                with open(precomputed_path, 'rb') as f:
                    self.points = pickle.load(f)
                    print(f"{precomputed_path} load successfully")
        logging.info(f'Successfully load ScanObjectNN {split} '
                     f'size: {self.points.shape}, num_classes: {self.labels.max()+1}')

    @property
    def num_classes(self):
        return self.labels.max() + 1

    def __getitem__(self, idx):
        current_points = self.points[idx][:self.num_points]
        label = self.labels[idx]
        if self.partition == 'train':
            np.random.shuffle(current_points)
        data = {'pos': current_points,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        # height appending. @KPConv
        # TODO: remove pos here, and only use heights. 
        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            # 原始代码：
            # data['x'] = torch.cat((data['pos'],
            #                       torch.from_numpy(current_points[:, self.gravity_dim:self.gravity_dim+1] - current_points[:, self.gravity_dim:self.gravity_dim+1].min())), dim=1)
            
            # 修改后的代码：确保data['pos']是PyTorch张量
            if isinstance(data['pos'], np.ndarray):
                data['pos'] = torch.from_numpy(data['pos']).float()
            data['x'] = torch.cat((data['pos'],
                                   torch.from_numpy(current_points[:, self.gravity_dim:self.gravity_dim+1] - current_points[:, self.gravity_dim:self.gravity_dim+1].min()).float()), dim=1)
        return data

    def __len__(self):
        return self.points.shape[0]

    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """
