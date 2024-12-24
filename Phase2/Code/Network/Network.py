"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Mayank Deshpande (msdeshp4@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park

Vikram Setty (msdeshp4@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park

Vinay Lanka (msdeshp4@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park
"""

import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True


def SupLossFn(H4pt_pred, H4pt):
    criterion = nn.MSELoss()
    H4pt = H4pt.float()  
    loss = torch.sqrt(criterion(H4pt_pred, H4pt))
    return loss


class SupHomographyModel(pl.LightningModule):
    def __init__(self):
        super(SupHomographyModel, self).__init__()
        self.network = SupNet()

    def forward(self, Pa_Pb):
        return self.network(Pa_Pb)
    
    def training_step(self, Pa_Pb, H4pt):
        H4pt_pred = self.forward(Pa_Pb)
        loss = SupLossFn(H4pt_pred, H4pt)
        return loss

    def validation_step(self, Pa_Pb, H4pt):
        H4pt_pred = self.forward(Pa_Pb)
        loss = SupLossFn(H4pt_pred, H4pt)
        return loss

    @staticmethod
    def validation_epoch_end(outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class SupNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 16 * 128, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):

        mini_batch_size = x.shape[0]
        dim_x = x.shape[1]
        dim_y = x.shape[2]
        depth = x.shape[3]

        x = x.view(torch.Size([mini_batch_size, depth, dim_x, dim_y]))

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv4(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x