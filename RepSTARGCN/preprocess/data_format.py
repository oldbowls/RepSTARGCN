import os
import json
import random


def format(input_path, output_path, random_flag=False):
    for root, dirs, _ in os.walk(input_path):
        for index, dir in enumerate(dirs):
            files = os.listdir(os.path.join(root, dir))
            files = sorted(files, key=lambda x: int(x.split('.')[0]))
            sample = []
            for file in files:
                if file.split('.')[-1] == 'json':
                    with open(os.path.join(root, dir, file), 'r') as fcc_file:
                        # print(os.path.join(root, dir, file))
                        kps = json.load(fcc_file)['keypoints']
                        if random_flag:
                            for kp in kps:
                                kp[0] += random.randint(0, 1)
                                kp[1] += random.randint(0, 1)

                        sample.append(kps)
            sample_save(sample, output_path, root.split('\\')[-1], index)

            # print(kp)
            # os.path.join()


def sample_save(kp, output_path, classes, index):
    with open(os.path.join(output_path, classes + str(index) + '.json'), "w") as f:
        json.dump(kp, f)
    print(index)


if __name__ == '__main__':
    format(input_path=r'C:\Users\11761\Desktop\tracking_a\up_json',
           output_path=r'C:\Users\11761\Desktop\new_data\up_json', random_flag=False)
