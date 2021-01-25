####################################
# Train LBMNet using kitti dataset #
####################################
import os
import cv2
import time
import random
import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torch.nn.utils as torch_utils
from torch.utils.data import DataLoader

from torchsummary import summary

import visdom

vis = visdom.Visdom()
vis.close(env="main")

torch.cuda.empty_cache()

from loader import Train_DataSet, Test_DataSet
from LBM import LBMNet50_Improv as LBMNet50
from utils import acc_check, value_tracker

batch_size = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1377)
if device == 'cuda':
    torch.cuda.manual_seed_all(1377)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1377)
random.seed(1377)

def main():
    dataset = Train_DataSet("../path/to/kitti/data_road")
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    dataset_test = Test_DataSet("../path/to/kitti/data_road")
    dataset_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=True, num_workers=0)

    model = LBMNet50().to(device)
    # checkpoint_filename = './Result/model54/model_epoch_1400.pth'
    # if os.path.exists(checkpoint_filename):
    #     model.load_state_dict(torch.load(checkpoint_filename))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=4e-6)

    loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='ResNeSt Test', legend=['loss Semantic Segmentation'], showlegend=True))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

    iters = len(dataset_loader)
    
    # 부득이하게 학습을 중간에 멈추고 다시 돌려야한다면, 학습을 멈췄을 당시의 epoch의 Learning Rate로 설정하기 위한 반복문
    # for epoch in range(1400):
    #     for i, data in enumerate(dataset_loader, 0):
    #         scheduler.step(epoch + i / iters)

    epochs = 3100
    start = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataset_loader, 0):
            images, labels, name = data

            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output, loss = model(images, labels)

            loss.backward()

            optimizer.step()

            scheduler.step(epoch + i / iters)

            # print statistics
            running_loss += loss.item()

            if i % 20 == 19:
                value_tracker(vis, torch.Tensor([i + epoch * len(dataset_loader)]), torch.Tensor([running_loss / 10]), loss_plt)
                print("[%d, %5d] loss: %.3f lr: %f, time: %.3f" % (epoch + 1, i + 1, running_loss / 10, optimizer.param_groups[0]['lr'], time.time() - start))
                start = time.time()

            running_loss = 0.0
            del loss, output

        # Check Accuracy
        if epoch == 0 or epoch % 100 == 99:
            save_test_path = "./Result/output/path"
            acc_check(model, device, dataset_test, dataset_loader_test, epoch, save_test_path)
            torch.save(model.state_dict(), "./Result/model/output/path/model_epoch_{}.pth".format(epoch + 1))

        start = time.time()
    print("Finished Training...!")

if __name__=='__main__':
    main()
