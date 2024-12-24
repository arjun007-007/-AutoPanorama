#!/usr/bin/evn python

"""
CMSC733 Spring 2024: Computer Processing of Pictorial Information
Project1: MyAutoPano: Phase 2 Code

Author(s):
Vinay Lanka (vlanka@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park

Mayank Deshpande (msdeshp4@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park

Vikram Setty (vikrams@umd.edu)
M.Eng. in Robotics,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
import torchvision.transforms as tf
import argparse
from Network.Network import *
from Network.Unsup_Network import *
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
from copy import deepcopy

# Don't generate pyc codes
sys.dont_write_bytecode = True

patch_size = 128 #square patch so (128,128)
p = 16

def generate_test_data(index):
    rgb_img = cv2.imread('Phase2/Data/Test/' + str(index) + '.jpg')
    img = deepcopy(rgb_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (320, 240))
    h,w = img.shape

    patch_x = random.randint(p, w-patch_size-p)
    patch_y = random.randint(p, h-patch_size-p)

    patch_A  =   img[patch_y : patch_y + patch_size, patch_x : patch_x + patch_size]    # patch_A extracted from the original Image

    corners_A =   np.array([[patch_x             , patch_y              ], 
                            [patch_x + patch_size, patch_y              ], 
                            [patch_x + patch_size, patch_y + patch_size ],
                            [patch_x             , patch_y + patch_size ],], dtype=np.float32)

    pert_x = np.array(random.sample(range(-p, p), 4)).reshape(4,1)
    pert_y = np.array(random.sample(range(-p, p), 4)).reshape(4,1)
    pert_mat = np.hstack([pert_x, pert_y])

    corners_B = corners_A + pert_mat
    corners_B = np.array(corners_B, dtype=np.float32)

    H4Pt = corners_B - corners_A

    H_AB = cv2.getPerspectiveTransform(corners_A, corners_B)

    warped_img = cv2.warpPerspective(img, H_AB, (w, h))
    patch_B = warped_img[patch_y : patch_y + patch_size, patch_x : patch_x + patch_size]

    ####DATA AUGMENTATION 
    patch_A=(np.float32(patch_A))/255
    patch_B=(np.float32(patch_B))/255
    
    return patch_A, patch_B, corners_A, corners_B, rgb_img, H4Pt

def concat(img1, img2):
    img1 = np.expand_dims(img1, 2)
    img2 = np.expand_dims(img2, 2)
    I = torch.stack([torch.from_numpy(np.concatenate((img1, img2), axis = 2))]).to(dtype=torch.float32)
    return I

def SupTest_operation(Pa_Pb, ModelPath):
    model = SupHomographyModel()
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    model.eval()
    H4PT = model.forward(Pa_Pb)
    return H4PT

def UnSupTest_operation(PaPbB, ModelPath):
    model = UnSupHomographyModel()
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    model.eval()
    Pred = model.forward(PaPbB)
    return Pred

def plot_corners(img, warped_corners, pred_warped_corners):
    warped_corners = warped_corners.astype('int32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    cv2.polylines(img, [warped_corners], isClosed = True, color = (255,0,0), thickness = 2)
    cv2.polylines(img, [pred_warped_corners], isClosed = True, color = (0,0,255), thickness = 2)
    plt.imshow(img)
    plt.show()

def main():
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--Epochs",
        dest="Epochs",
        default=25,
        help="Number of epochs run"
    )
    Args = Parser.parse_args()
    Epochs = Args.Epochs

    SupModelPath = "./Phase2/Checkpoints/Sup" + str(Epochs-1) + "model.ckpt"
    UnSupModelPath = "./Phase2/Checkpoints/UnSup" + str(Epochs-1) + "model.ckpt"

    for i in range(1,5):
        patch_A, patch_B, corners_A, corners_B, img, H4Pt = generate_test_data(i)
        H4Pt = torch.from_numpy(H4Pt)

        Pa_Pb = concat(patch_A, patch_B)
        #Supervised Testing
        SupH4PT_pred = SupTest_operation(Pa_Pb, SupModelPath).detach().numpy()
        SupH4PT_pred = np.reshape(SupH4PT_pred, (4,2))
        corners_B_pred = (corners_A + SupH4PT_pred).astype(int)
        corners_B = corners_B.astype(int)
        plot_corners(img, corners_B, corners_B_pred)

        SupH4PT_pred = torch.from_numpy(SupH4PT_pred)
        LossThisBatch = SupLossFn(SupH4PT_pred, H4Pt)
        print("Supervised RMSE Loss: ", LossThisBatch)

        UnSupH4PT_pred = UnSupTest_operation(Pa_Pb, UnSupModelPath).detach().numpy()
        UnSupH4PT_pred = np.reshape(UnSupH4PT_pred, (4,2))
        corners_B_pred = (corners_A + UnSupH4PT_pred).astype(int)
        corners_B = corners_B.astype(int)
        plot_corners(img, corners_B, corners_B_pred)
        UnSupH4PT_pred = torch.from_numpy(UnSupH4PT_pred)
        LossThisBatch = SupLossFn(UnSupH4PT_pred, H4Pt)
        print("Unsupervised RMSE Loss: ", LossThisBatch)

if __name__ == "__main__":
    main()