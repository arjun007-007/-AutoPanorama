#!/usr/bin/evn python

# Importing the required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
import copy

# Hyperparameters used in the pipeline
harris_max_ratio = 0.005
harris_block_size = 2
harris_sobel_aperture = 3
harris_kvalue = 0.04
anms_N_best = 300
descriptor_patch_size = 40
SSD_ratio_threshold = 0.99
num_ransac_iterations = 10000
homography_inlier_threshold = 6
final_img_shape = [1280,1024]

# Detecting and thresholding corner features using Harris Corner Detection
def detect_corners(rgb_img):
      gray_img = cv2.cvtColor(copy.deepcopy(rgb_img),cv2.COLOR_BGR2GRAY).astype(np.float32)
      corners_info = cv2.cornerHarris(gray_img,harris_block_size,harris_sobel_aperture,harris_kvalue)
      corners_info[corners_info < harris_max_ratio*corners_info.max()] = 0
      corner_locations = np.where(corners_info > 0)
      img_with_corners = copy.deepcopy(rgb_img)
      img_with_corners[corner_locations] = np.array([255,0,0])
      return (corners_info,corner_locations,img_with_corners)

# Adaptive Non-Maximal Suppression (ANMS) to remove duplicate representations of the same feature/corner
def adaptive_non_max_supression(rgb_img,corners_info):
      strong_corner_coords = peak_local_max(corners_info,min_distance=15)
      N_strong = strong_corner_coords.shape[0]
      r_values = np.inf*np.ones((N_strong))
      for i in range(N_strong):
            for j in range(N_strong):
                  x_i, y_i = strong_corner_coords[i]
                  x_j, y_j = strong_corner_coords[j]
                  ED = r_values[i]
                  if corners_info[x_i,y_i] < corners_info[x_j,y_j]:
                        ED = (strong_corner_coords[j,1]-strong_corner_coords[i,1])**2 + (strong_corner_coords[j,0]-strong_corner_coords[i,0])**2
                  if r_values[i] > ED:
                        r_values[i] = ED
      N_best_indices = np.flip(np.argsort(r_values))[0:min(anms_N_best,N_strong)]
      anms_corner_locations = np.array([(int(strong_corner_coords[i][0]),int(strong_corner_coords[i][1])) for i in N_best_indices])
      bad_corner_indices = []
      for i in range(len(anms_corner_locations)):
            x = anms_corner_locations[i][0]
            y = anms_corner_locations[i][1]
            if x<20 or x+20>=rgb_img.shape[0]:
                  continue
            if np.array_equal(rgb_img[x-20,y,:],np.array([0,0,0])) or np.array_equal(rgb_img[x+20,y,:],np.array([0,0,0])):
                  bad_corner_indices.append(i)
      new_anms_corners = []
      for i in range(len(anms_corner_locations)):
            if i not in bad_corner_indices:
                  new_anms_corners.append(anms_corner_locations[i])
      img_with_anms_corners = copy.deepcopy(rgb_img)
      for i in range(len(new_anms_corners)):
            cv2.circle(img_with_anms_corners,(new_anms_corners[i][1],new_anms_corners[i][0]),2,(255,0,0),-1)
      return (new_anms_corners,img_with_anms_corners)

# Generating feature descriptions for each corner as a vector
def corner_feature_descriptors(rgb_img,anms_corner_locations):
      gray_img = cv2.cvtColor(copy.deepcopy(rgb_img),cv2.COLOR_BGR2GRAY)
      feature_descriptor = []
      for loc in anms_corner_locations:
            if(loc[0]<=descriptor_patch_size//2 or loc[0]>=gray_img.shape[0]-descriptor_patch_size//2 or loc[1]<=descriptor_patch_size//2 or loc[1]>=gray_img.shape[1]-descriptor_patch_size//2):
                  continue
            feature_patch = gray_img[loc[0]-descriptor_patch_size//2:loc[0]+descriptor_patch_size//2,loc[1]-descriptor_patch_size//2:loc[1]+descriptor_patch_size//2]
            feature_patch = cv2.GaussianBlur(feature_patch,(3,3),0)
            feature_patch = cv2.resize(feature_patch,(descriptor_patch_size//5,descriptor_patch_size//5),interpolation=cv2.INTER_CUBIC).reshape(-1)
            feature_patch = (feature_patch-feature_patch.mean())/feature_patch.std()
            feature_descriptor.append([loc,feature_patch])
      return feature_descriptor

# Checking if a feature in the second image has already been matched with a feature from the first image
def in_matches(loc2,matches):
      for match in matches:
            if np.array_equal(np.array(match[1]),np.array([loc2[1],loc2[0]])):
                  return True
      return False

# Matching features using the SSD ratio test between two images
def match_features(img1_features,img2_features):
      matches = []
      for loc1, feature1 in img1_features:
            ssd_array = []
            for loc2, feature2 in img2_features:
                  ssd_array.append(np.sum(np.square(feature1-feature2)))
            best_matches = np.argsort(np.array(ssd_array))
            match_index = 0
            for idx in list(best_matches):
                  if not in_matches(img2_features[idx][0],matches):
                        break
                  match_index += 1
            if match_index == best_matches.shape[0]:
                  continue
            if ssd_array[best_matches[match_index]]/ssd_array[best_matches[match_index+1]] < SSD_ratio_threshold:
                  matches.append([[loc1[1],loc1[0]],[img2_features[best_matches[match_index]][0][1],img2_features[best_matches[match_index]][0][0]]])
      return matches

# Drawing the feature matches between two images
def draw_feature_matches2(rgb_img1,rgb_img2,matches):
      # Both images must be of the same sizes to concatenate them side by side
      img_concat = np.concatenate([copy.deepcopy(rgb_img1),copy.deepcopy(rgb_img2)],axis=1)
      for match in matches:
            loc1, loc2 = match
            loc2[0] += rgb_img1.shape[1]
            cv2.line(img_concat,[loc1[0],loc1[1]],[loc2[0],loc2[1]],(0,0,255),1)
      plt.imshow(img_concat)
      plt.show()

# Displaying the homoghraphy from img1 --> img2 to verify correct homography calculation
def display_homography(rgb_img1,rgb_img2,H):
      h_img = cv2.warpPerspective(copy.deepcopy(rgb_img1),H,(rgb_img2.shape[1],rgb_img2.shape[0]))
      img_concat = np.concatenate([h_img,copy.deepcopy(rgb_img2)],axis=1)
      plt.imshow(img_concat)
      plt.show()      

# Getting the inlier matches for a chosen homography
def get_inliers(img1_matches,img2_matches,homography):
      num_inlier_points = 0
      inlier_indices = []
      img1_points_transformed = np.vstack((img1_matches[:,0],img1_matches[:,1],np.ones((1,img1_matches.shape[0]))))
      img1_points_transformed = np.dot(homography,img1_points_transformed)
      x_transformed = img1_points_transformed[0,:]/(img1_points_transformed[2,:]+np.exp(-7))
      y_transformed = img1_points_transformed[1,:]/(img1_points_transformed[2,:]+np.exp(-7))
      img1_points_transformed = np.array([x_transformed,y_transformed]).T
      for i in range(img1_matches.shape[0]):
            if np.linalg.norm(img1_points_transformed[i]-img2_matches[i]) < homography_inlier_threshold:
                  num_inlier_points += 1
                  inlier_indices.append(i)
      return num_inlier_points, inlier_indices

# RANSAC wrapper function to get the best homography using a set of matched features between images
def RANSAC(matches):
      img1_matches = np.array(matches)[:,0,:]
      img2_matches = np.array(matches)[:,1,:]
      best_inlier_indices = []
      max_inliers = 0
      best_homography = None
      for i in range(num_ransac_iterations):
            random_points = np.random.choice(len(matches),size=4)
            img1_points = img1_matches[random_points].astype(np.float32)
            img2_points = img2_matches[random_points].astype(np.float32)
            homography = cv2.getPerspectiveTransform(img1_points,img2_points)
            num_inlier_points, inlier_indices = get_inliers(img1_matches,img2_matches,homography)
            if num_inlier_points > max_inliers or max_inliers == 0:
                  max_inliers = num_inlier_points
                  best_inlier_indices = inlier_indices
                  best_homography = homography
      old_best_homography = best_homography
      img1_points = img1_matches[best_inlier_indices].astype(np.float32).reshape(-1,1,2)
      img2_points = img2_matches[best_inlier_indices].astype(np.float32).reshape(-1,1,2)
      best_homography,_ = cv2.findHomography(img1_points,img2_points)
      if best_homography is None:
            best_homography = old_best_homography
      return best_homography, best_inlier_indices

# Wrapper function to take two images and compute the best homography between them
def homography_wrapper(img1,img2):
     rgb_img1 = copy.deepcopy(img1)
     rgb_img2 = copy.deepcopy(img2)
     corner_info1, corner_loc1, corner_img1 = detect_corners(rgb_img1)
     anms_corner_locs1, anms_corners_img1 = adaptive_non_max_supression(rgb_img1,corner_info1)
     features1 = corner_feature_descriptors(rgb_img1,anms_corner_locs1)
     corner_info2, corner_loc2, corner_img2 = detect_corners(rgb_img2)
     anms_corner_locs2, anms_corners_img2 = adaptive_non_max_supression(rgb_img2,corner_info2)
     features2 = corner_feature_descriptors(rgb_img2,anms_corner_locs2)
     matches = match_features(features1,features2)
     best_homography, best_inlier_indices = RANSAC(matches)
     ransac_matches = []
     for index in best_inlier_indices:
          ransac_matches.append(matches[index])
     anms_corners_img1 = cv2.resize(anms_corners_img1,(anms_corners_img2.shape[1],anms_corners_img2.shape[0]),interpolation=cv2.INTER_CUBIC)
     #draw_feature_matches2(anms_corners_img1,anms_corners_img2,matches)
     #draw_feature_matches2(anms_corners_img1,anms_corners_img2,ransac_matches)
     return best_homography

# Warping one image to the perspective of the other
def warp_images(rgb_img1,rgb_img2,best_homography):
      h1, w1, c1 = rgb_img1.shape
      h2, w2, c2 = rgb_img2.shape
      img1_corners = np.array([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2).astype(np.float32)
      img2_corners = np.array([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2).astype(np.float32)
      img1_corners_transformed = cv2.perspectiveTransform(img1_corners,best_homography)
      all_corners = np.concatenate((img1_corners_transformed,img2_corners),axis=0)
      w_min,h_min = all_corners.min(axis=0).ravel().astype(np.int32)
      w_max,h_max = all_corners.max(axis=0).ravel().astype(np.int32)
      warping_correction_matrix = np.array([[1,0,-w_min],[0,1,-h_min],[0,0,1]]).astype(np.float32)
      new_homography = np.dot(warping_correction_matrix,best_homography)
      new_w = w_max-w_min
      new_h = h_max-h_min
      transformed_img1 = cv2.warpPerspective(copy.deepcopy(rgb_img1),new_homography,dsize=(new_w,new_h))
      transformed_img1[-h_min:-h_min+h2,-w_min:-w_min+w2,:] = copy.deepcopy(rgb_img2)
      return transformed_img1

# Stitching images together using Approach 1
def stitch_images(images):
      base_img = copy.deepcopy(images[0])
      for img in images[1:]:
            base_img = cv2.resize(base_img,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_CUBIC)
            best_homography = homography_wrapper(base_img,img)
            #display_homography(base_img,img,best_homography)
            result = warp_images(base_img,img,best_homography)
            base_img = copy.deepcopy(result)
            #plt.imshow(base_img)
            #plt.show()
      base_img = cv2.resize(base_img,final_img_shape,interpolation=cv2.INTER_CUBIC)
      base_img = cv2.GaussianBlur(base_img,(3,3),0)
      cv2.imshow("test",base_img)
      #cv2.waitKey(0)
      return base_img

# Stitching images together using Approach 2
def stitch_images2(images):
      homographies = []
      for i in range(len(images)-1):
            img1 = images[i]
            img2 = images[i+1]
            best_homography = homography_wrapper(img1,img2)
            homographies.append(best_homography)
      running_homography = np.eye(3,dtype=np.float32)
      result = copy.deepcopy(images[0])
      i = 1
      for homography in homographies:
            running_homography = np.matmul(running_homography,homography)
            result = copy.deepcopy(warp_images(result,images[i],running_homography))
            i += 1
      result = cv2.resize(result,final_img_shape,interpolation=cv2.INTER_CUBIC)
      result = cv2.GaussianBlur(result,(3,3),0)
      cv2.imshow("test",result)
      #cv2.waitKey(0)
      return result

if __name__ == '__main__':
    
    # Creating a panormaic view of a set if images
    # Another set of images can be used by changing the path of the 'path_to_images' variable
    path_to_images = "../Data/Train/Set1/"
    imgs = []
    number_files = len(os.listdir(path_to_images))
    for i in range(number_files):
          imgs.append(cv2.imread(path_to_images+str(i+1)+'.jpg'))
    result = stitch_images(imgs)
    cv2.waitKey(0)
    