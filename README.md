# RepSTARGCN
This repo is the official implementation for "An Attitude-centric and Cross-band Infrared Framework for Aerial Target Intention Recognition"

## :mega: News
- [2025.12.08]: We open source the code of SLPHM!
- [2025.11.28]: We open source the code of RepSTARGCN!

##  :sparkles: Overview

In terminal attack-defence scenarios, aerial target intention can be accurately recognized based on trajectory and attitude information. Currently, radar-based intention recognition methods hold a dominant position. However, radar is not adept at capturing target attitude information (e.g., pitch, roll, and yaw) and can only perform intention recognition based on trajectory information. To bridge this gap, this paper proposes an attitude-centric and cross-band infrared aerial target intention recognition framework that simultaneously extracts trajectory and attitude representations from multi-band infrared images. The framework comprises two steps: pose estimation for predicting keypoints from infrared images, followed by intention recognition based on these keypoints. For pose estimation, a cross-band invariant representation learning method is proposed to reduce the impact of band-bias, thereby improving the model's generalization on multi-band infrared images. For intention recognition, regularized adaptive adjacency matrix and parameter fusion mechanisms are designed to effectively capture aerial target attitude representations, forming an attitude-centric approach. Experiments demonstrate that the proposed method significantly enhances the generalization of various pose estimation models on multi-band infrared images. Additionally, with the introduction of attitude representations, the intention recognition accuracy increases from 90.12% to 96.64%.

# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- Windows 10, cuda 11.1
- We provide the dependency file of our experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running pip install -r requirements.txt 




# Training & Testing


## Train

- Change the config file depending on what you want. (The configuration file is located in config/config.yaml.)

```python train.py -config config/config.yaml ```


## TEST

- To test the trained models saved in <model_saved_name>, run the following command:

```python train.py -config config/config.yaml -eval True -pre_trained_model xxx.state ```


# Acknowledgement

This repository is built upon SGN and RepVGG.

## :envelope: Contact
If you have any questions about our code, please feel free to contact dlh@mail.nwpu.edu.cn.
