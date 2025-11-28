import os
import shutil
from tqdm import tqdm


def delete_files_in_folders(folders):
    for folder in folders:
        try:

            folder_path = os.path.abspath(folder)


            for filename in tqdm(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

            print(f"All files in {folder_path} have been deleted.")

        except Exception as e:
            print(f"Error while processing {folder}. Reason: {e}")



folders_to_delete = [r'H:\zhang_data\test_npy',r'H:\zhang_data\train_npy']

delete_files_in_folders(folders_to_delete)
