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

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as tf
from torch.optim import AdamW, SGD, Adam
from Network.Network import *
from Network.Unsup_Network import *
from Wrapper import *
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from termcolor import colored, cprint
import math as m
from random import choice
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")

def GenerateBatch(BasePath, MiniBatchSize, State, Model):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    ImageBatch - Batch of images
    GroundTruthBatch - Batch of coordinates
    """
    PaPbBatch = []
    GroundTruthBatch = []
    if Model == "UnSup":
        PaCornersBatch = []
        PaBatch = []
        PbBatch = []
    ImageNum = 0
    patches_a_path = os.path.join(BasePath, f"Synthetic{State}/patch_A")
    patches_b_path = os.path.join(BasePath, f"Synthetic{State}/patch_B")
    GTs_path = os.path.join(BasePath, f"Synthetic{State}/H4Pt/")
    if Model == "UnSup":
        Pa_base_corners_path = os.path.join(BasePath, f"Synthetic{State}/patch_A_corners/")
    if os.path.exists(patches_a_path): 
        image_list = os.listdir(patches_a_path)
    else:
        raise Exception ("Patch A Directory doesn't exist")
    
    while ImageNum < MiniBatchSize:
        ImageNum += 1
        RandIdx = random.randint(1, len(image_list)-1)
        # print(len(image_list))
        if State == "Train":
            patch_a_path = os.path.join(patches_a_path,image_list[RandIdx])
            patch_b_path = os.path.join(patches_b_path,image_list[RandIdx])
            GT_path = GTs_path + image_list[RandIdx][:-4] + f".txt"
            if Model == "UnSup":
                Pa_corners_path = Pa_base_corners_path + image_list[RandIdx][:-4] + f".txt"
        elif State == "Val":
            # print(ImageNum)
            patch_a_path = os.path.join(patches_a_path,image_list[ImageNum-1])
            patch_b_path = os.path.join(patches_b_path,image_list[ImageNum-1])
            GT_path = GTs_path + image_list[ImageNum-1][:-4] + f".txt"
            if Model == "UnSup":
                Pa_corners_path = Pa_base_corners_path + image_list[ImageNum-1][:-4] + f".txt"

        patch_A = cv2.imread(patch_a_path, cv2.IMREAD_GRAYSCALE)
        patch_B = cv2.imread(patch_b_path, cv2.IMREAD_GRAYSCALE)

        ####DATA AUGMENTATION 
        patch_A=(np.float32(patch_A))
        patch_B=(np.float32(patch_B))

        patch_A = np.expand_dims(patch_A, 2)
        patch_B = np.expand_dims(patch_B, 2)
        Pa_Pb = np.concatenate((patch_A, patch_B), axis = 2)
        GT = np.genfromtxt(GT_path, delimiter=',').reshape(-1).astype(np.float32)
        PaPbBatch.append(torch.from_numpy(Pa_Pb))
        GroundTruthBatch.append(torch.from_numpy(GT).to(device))

        if Model == "UnSup":
            Pa_corners = np.genfromtxt(Pa_corners_path, delimiter=',').flatten(order = 'F')
            PaCornersBatch.append(torch.from_numpy(Pa_corners).to(device))
            patch_A = np.squeeze(patch_A)
            patch_A = np.expand_dims(patch_A, axis=0)
            patch_B = np.squeeze(patch_B)
            patch_B = np.expand_dims(patch_B, axis=0)
            PaBatch.append(torch.from_numpy(patch_A))
            PbBatch.append(torch.from_numpy(patch_B))


    PaPbBatch = torch.stack(PaPbBatch)
    PaPbBatch = PaPbBatch.to(device,dtype=torch.float32)

    if Model == "UnSup":
        PaBatch = torch.stack(PaBatch)
        PaBatch = PaBatch.to(device,dtype=torch.float32)
        PbBatch = torch.stack(PbBatch)
        PbBatch = PbBatch.to(device,dtype=torch.float32)
        return PaPbBatch, torch.stack(PaCornersBatch), PaBatch, PbBatch
    return PaPbBatch, torch.stack(GroundTruthBatch)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    if ModelType == 'Sup':
        # Predict output with forward pass
        model = SupHomographyModel()
    else:
        model = UnSupHomographyModel()

    Optimizer = Adam(model.parameters(), lr=0.0001)
    # Optimizer = torch.optim.SGD(model.parameters(),lr = 0.0001)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    #Enable cuda
    model.to(device)

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):

            if ModelType == 'Sup':
                PaPbBatch, CoordinatesBatch = GenerateBatch(
                    BasePath, MiniBatchSize, "Train", "Sup"
                )
                PaPbBatch.to(device)
                CoordinatesBatch.to(device)

                LossThisBatch =  model.training_step(PaPbBatch, CoordinatesBatch)
            elif ModelType == "UnSup":
                PaPbBatch, CornersBatch, PaBatch, PbBatch = GenerateBatch(
                    BasePath, MiniBatchSize, "Train", "UnSup"
                )
                PaPbBatch.to(device)
                CornersBatch.to(device)
                PaBatch.to(device)
                PbBatch.to(device)

                LossThisBatch = model.training_step(PaBatch, PaPbBatch, CornersBatch, PbBatch)


            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            if PerEpochCounter % SaveCheckPoint == 0:
                # Write training losses to tensorboard
                Writer.add_scalar(
                    "Training Loss over iterations",
                    LossThisBatch,
                    Epochs * NumIterationsPerEpoch + PerEpochCounter,
                )
                Writer.flush()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            if ModelType == 'Sup':
                val_PaPbBatch, val_CoordinatesBatch = GenerateBatch(
                    BasePath, 1000, "Val", "Sup"
                )
                for val_PaPb,val_Coordinates in zip(val_PaPbBatch, val_CoordinatesBatch):
                    val_PaPb = torch.stack([val_PaPb]).to(device)
                    val_Coordinates = torch.stack([val_Coordinates]).to(device)
                    loss = model.validation_step(val_PaPb, val_Coordinates)
                    val_loss+=loss
                val_loss = val_loss / len(val_CoordinatesBatch)
            elif ModelType == 'UnSup':
                val_PaPbBatch, val_CornersBatch, val_PaBatch, val_PbBatch = GenerateBatch(
                    BasePath, 1000, "Val", "UnSup"
                )
                for val_PaPb, val_Corners, val_Pa, val_Pb in zip(val_PaPbBatch, val_CornersBatch, val_PaBatch, val_PbBatch):
                    val_PaPb = torch.stack([val_PaPb]).to(device)
                    val_Corners = torch.stack([val_Corners]).to(device)
                    val_Pa = torch.stack([val_Pa]).to(device)
                    val_Pb = torch.stack([val_Pb]).to(device)
                    loss = model.validation_step(val_Pa, val_PaPb, val_Corners, val_Pb)
                    val_loss+=loss
                val_loss = val_loss / len(val_PbBatch)
            
        print("Validation loss - ", val_loss)
        # Write validation losses to tensorboard
        Writer.add_scalar(
            "Validation Loss over epochs",
            val_loss,
            Epochs,
        )
        Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + ModelType + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")
 

def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)

def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default='./Phase2/Data/',
        help="Base path of images, Default:./Phase2/Data/",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="./Phase2/Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="UnSup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=25,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=32,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="./Phase2/Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    ModelType = Args.ModelType
    LogsPath = Args.LogsPath + ModelType + "/"

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        TrainCoordinates,
        NumTrainSamples,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
    )


if __name__ == "__main__":
    main()