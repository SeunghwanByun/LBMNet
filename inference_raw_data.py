##############################################################
# Inference for road detection in kitti raw data(road scene) #
##############################################################

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

torch.cuda.empty_cache()

from LBM import LBMNet50_Improv as LBMNet50
from kitti_projection_in_raw_data import Raw_Dataset
from utils import inference_check

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1377)
if device == 'cuda':
    torch.cuda.manual_seed_all(1377)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1377)
random.seed(1377)

def main():
    print("Start Infernce...!")

    data_path = '../path/KITTI Datasets/2011_09_26/'
    raw_data = Raw_Dataset(data_path, sequence_num="0001")
    raw_data_loader = DataLoader(raw_data, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    model = LBMNet50().to(device)
    checkpoint_filename = './Result/model/model_epoch_xxxx.pth'
    if os.path.exists(checkpoint_filename):
        model.load_state_dict(torch.load(checkpoint_filename))
        print("Successfully load the model..!")
        inference_check(model, device, raw_data_loader, save_path="./Result/inference_result/")

    print("Finished Inference...!")

if __name__=='__main__':
    main()
