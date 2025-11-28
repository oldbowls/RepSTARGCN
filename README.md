# RepSTARGCN
This repo is the official implementation for "An Attitude-centric and Cross-band Infrared Framework for Aerial Target Intention Recognition"

# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- Windows 10, cuda 11.1
- We provide the dependency file of our experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running pip install -r requirements.txt 




# Training & Testing


## Train

- Change the config file depending on what you want. (The configuration file is located in config/UCI.yaml.)

```python train.py -config config/config.yaml ```


## TEST

- To test the trained models saved in <model_saved_name>, run the following command:

```python train.py -config config/config.yaml -eval True -pre_trained_model xxx.state ```


# Acknowledgement

This repository is built upon SGN.
