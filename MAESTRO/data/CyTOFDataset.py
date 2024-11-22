import os
import h5py
import torch
import random
from torch.utils.data import Dataset

class CyTOFDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_paths = self._get_file_paths()
        self.sample_names = list(self.file_paths.keys())
        self.raw_dict = {}
        self.fps_dict = {}

    def _get_file_paths(self):
        file_paths = {}
        files = os.listdir(self.data_dir)
        for filename in files:
            if filename.endswith('.h5'):
                sample_name = filename.replace('.h5', '')
                file_paths[sample_name] = os.path.join(self.data_dir, filename)

        return file_paths

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample_name = self.sample_names[idx]
        file_path = self.file_paths[sample_name]

        with h5py.File(file_path, 'r') as h5file:
            raw_data = torch.tensor(h5file['raw/data'][:], dtype=torch.bfloat16)
            fps_data = torch.tensor(h5file['fps/data'][:], dtype=torch.bfloat16)

        return raw_data, fps_data, sample_name