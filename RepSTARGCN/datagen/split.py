import os
import shutil
import random


def split_dataset(input_folder, output_train_folder, output_test_folder, split_ratio=0.8, seed=42):

    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(output_test_folder, exist_ok=True)

    file_list = os.listdir(input_folder)


    random.seed(seed)

    random.shuffle(file_list)


    split_index = int(len(file_list) * split_ratio)


    for file_name in file_list[:split_index]:
        source_path = os.path.join(input_folder, file_name)
        destination_path = os.path.join(output_train_folder, file_name)
        shutil.copy(source_path, destination_path)


    for file_name in file_list[split_index:]:
        source_path = os.path.join(input_folder, file_name)
        destination_path = os.path.join(output_test_folder, file_name)
        shutil.copy(source_path, destination_path)



input_folder = r'H:\zhang_data\all'
output_train_folder = r'H:\zhang_data\train'
output_test_folder = r'H:\zhang_data\test'

split_dataset(input_folder, output_train_folder, output_test_folder, split_ratio=0.7, seed=42)
