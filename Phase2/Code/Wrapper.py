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

import numpy as np
import cv2
import os
from copy import deepcopy
import random
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

patch_size = 128 #square patch so (128,128)
p = 16

def generate_data(img):
    img = deepcopy(img)
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

    return warped_img, corners_A, corners_B, patch_A, patch_B, H4Pt

def generate_dataset(number_of_files, path, synthetic_data_path):
    #Make paths
    os.makedirs(synthetic_data_path+'Grayscale/', exist_ok=True)
    os.makedirs(synthetic_data_path+'H4Pt/', exist_ok=True)
    os.makedirs(synthetic_data_path+'patch_A/', exist_ok=True)
    os.makedirs(synthetic_data_path+'patch_B/', exist_ok=True)
    os.makedirs(synthetic_data_path+'patch_A_corners/', exist_ok=True)
    for i in range(1,number_of_files+1):
        img = cv2.imread(path+str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(synthetic_data_path+'Grayscale/'+str(i)+'.jpg', img)
        img = cv2.resize(img, (320,240))
        img_warped, patch_a_corners, patch_b_corners, patch_A, patch_B, H4Pt = generate_data(img)
        cv2.imwrite(synthetic_data_path+'patch_A/'+str(i)+'.jpg', patch_A)
        cv2.imwrite(synthetic_data_path+'patch_B/'+str(i)+'.jpg', patch_B)
        np.savetxt(synthetic_data_path + 'H4Pt/' + str(i) + ".txt", H4Pt, delimiter = ",")
        np.savetxt(synthetic_data_path + 'patch_A_corners/' + str(i) + ".txt", patch_a_corners, delimiter = ",")
    return


def main():
    train_path = './Phase2/Data/Train/'
    train_synthetic_data_path = './Phase2/Data/SyntheticTrain/'
    os.makedirs(train_synthetic_data_path, exist_ok=True)
    generate_dataset(5000, train_path, train_synthetic_data_path)

    val_path = './Phase2/Data/Val/'
    val_synthetic_data_path = './Phase2/Data/SyntheticVal/'
    os.makedirs(val_synthetic_data_path, exist_ok=True)
    generate_dataset(1000, val_path, val_synthetic_data_path)       
    
    test_path = './Phase2/Data/Test/'
    test_synthetic_data_path = './Phase2/Data/SyntheticTest/'
    os.makedirs(test_synthetic_data_path, exist_ok=True)
    generate_dataset(1000, test_path, test_synthetic_data_path)   

if __name__ == "__main__":
    main()