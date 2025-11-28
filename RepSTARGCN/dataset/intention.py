import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os


class intentionData(Dataset):
    def __init__(self, data_path, label_path, dir_flag):
        self.name_label ={'att': 0, 'dis': 1, 'sei': 2, 'de': 3, 'other': 4}
        self.path = data_path
        self.label_path = label_path
        self.dir_flag = dir_flag
        self.load_data()

    def load_data(self):
        if self.dir_flag:
            self.files = os.listdir(self.path)
            self.sample_name = self.files
        else:
            self.files = np.load(self.path)

            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.dir_flag:
            filename = self.files[index]
            data = np.load(os.path.join(self.path, self.files[index]))
            label = self.name_label[filename.split('\\')[-1][0:2]]
        else:
            data = self.files[index]
            label = self.label[index]

        return data, label, filename

    def noise(self):
        pass
