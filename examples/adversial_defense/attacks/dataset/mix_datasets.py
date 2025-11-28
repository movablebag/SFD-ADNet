import sys
sys.path.append('E:/BJCWorkshop/LPF-Defense-main/attacks')
print(sys.path)
import os
import numpy as np
from torch.utils.data import DataLoader
from attacks.dataset.ModelNet40 import ModelNet40Hybrid


def load_separate_npy_data(data_path, label_path):
    data = np.load(data_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    return data, labels


def save_npy_data(file_path, data, labels):
    np.savez(file_path, data=data, label=labels)

def main(ori_data_path,ori_label_path,def_data_path,def_label_path,output_path, num_points=1024):

    ori_data,ori_labels = load_separate_npy_data(ori_data_path,ori_label_path)
    def_data,def_labels = load_separate_npy_data(def_data_path,def_label_path)

    out_npz_path = 'Data\data_npz'
    os.makedirs(out_npz_path, exist_ok=True)
    
    ori_npz_path = os.path.join(out_npz_path,'ori_npz.npz')
    save_npy_data(ori_npz_path,ori_data,ori_labels)

    def_npz_path = os.path.join(out_npz_path,'def_npz.npz')
    save_npy_data(def_npz_path,def_data,def_labels)

    # Create the hybrid dataset
    hybrid_dataset = ModelNet40Hybrid(
        ori_data=ori_npz_path,
        def_data=def_npz_path,
        num_points=num_points,
        partition='train',
        augmentation=False
    )
    # Use DataLoader to iterate over the dataset
    data_loader = DataLoader(hybrid_dataset, batch_size=len(hybrid_dataset), shuffle=False)

    # Get the combined data and labels
    for data, labels in data_loader:
        combined_data = data.numpy()
        combined_labels = labels.numpy()

    # Save the combined dataset
    save_npy_data(output_path, combined_data, combined_labels)
    print(f"Combined dataset saved to {output_path}")

if __name__ == "__main__":
    ori_data_path = 'E:/BJCWorkshop/LPF-Defense-main/model/Data/train_data.npy'
    ori_label_path = 'E:/BJCWorkshop/LPF-Defense-main/model/Data/train_labels.npy'
    def_data_path = 'E:/BJCWorkshop/LPF-Defense-main/model/Data/low_frequency_data.npy'
    def_label_path = 'E:/BJCWorkshop/LPF-Defense-main/model/Data/train_labels.npy'
    output_path = 'E:/BJCWorkshop/LPF-Defense-main/model/Data/mixed_data.npy'
    main(ori_data_path,ori_label_path, def_data_path,def_data_path,output_path)