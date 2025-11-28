import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import json


class dataProcess:
    def __init__(self, num_sensors, params, save_path):
        self.save_path = save_path
        self.params = params  # acc,
        self.slices = {'acc': [0, 1, 2], 'gyro': [3, 4, 5], 'angle': [6, 7, 8]}
        self.num_sensors = num_sensors
        self.keyword = ['accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz', 'anglex', 'angley', 'anglez']
        self.all_params = ['acc', 'gyro', 'angle']

        self.name_label = {'do': 0, 'le': 1, 'ri': 2, 'st': 3, 'up': 4}
        # down,left,right,straight,up

    def check_format(self):
        pass

    def read(self, file_path):
        label = self.name_label[file_path.split('\\')[-1][0:2]]
        with open(file_path, 'r') as fcc_file:
            data = json.load(fcc_file)
            data = np.array(data)
        return data, label

    def norm(self, datas):
        pass


    def center(self, data):  


        if isinstance(data, np.ndarray):
            for i in range(len(self.params)):
                data[:, :, :] -= data[:, 0:1, :]
            return data
        elif isinstance(data, list):
            datas = []
            for index, d in enumerate(data):
                d[:, :, :] -= d[:, 0:1, :]
                datas.append(d)
            return datas

    def random_sample_np(self, data_numpy, size):
        T, V, C = data_numpy.shape
        if T == size:
            return data_numpy
        interval = T / size
        random_list = [int(i * interval + np.random.randint(interval * 10) / 10) for i in range(size)]
        return data_numpy[random_list]

    def uniform_sample_np(self, data_numpy, size):
        T, V, C = data_numpy.shape
        if T == size:
            return data_numpy
        interval = T / size
        uniform_list = [int(i * interval) for i in range(size)]  # 均匀采样
        return data_numpy[:, uniform_list]

    def continual_sample(self, data_numpy, size):
        T, V, C = data_numpy.shape
        data_numpys = []
        if T == size:
            return [data_numpy]
        for i in range(size, T):  # (size, T,size)
            data_numpys.append(data_numpy[i - size:i, :, :])
        return data_numpys

    def save(self, data, file):
        if isinstance(data, np.ndarray):
            np.save(os.path.join(self.save_path, file.split('.')[0]), data)
        elif isinstance(data, list):
            for index, d in enumerate(data):
                np.save(os.path.join(self.save_path, file.split('.')[0] + '-' + str(index)), d)


if __name__ == '__main__':

    size = 20
    parts = ['train', 'test']

    source_path = r'H:\zhang_data\{}'
    save_path = r'H:\zhang_data\{}_npy'
    for part in parts:
        dp = dataProcess(num_sensors=5, params=['angle'],
                         save_path=save_path.format(part))
        path_dir = source_path.format(part)
        files = os.listdir(path_dir)
        for file in tqdm(files):
            data, label = dp.read(file_path=os.path.join(path_dir, file))
            data = dp.continual_sample(data_numpy=data, size=size)
            if len(data) == 0:
                continue
            data = dp.center(data)
            dp.save(data, file)
