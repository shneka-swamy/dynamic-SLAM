import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
from scipy.spatial import ConvexHull

# Gets the coordinates of the bounding box
def get_bounding_box(x_coords, y_coords):
    minimum_x = np.min(x_coords)
    maximum_x = np.max(x_coords) 
    minimum_y = np.min(y_coords)
    maximum_y = np.max(y_coords)

    return maximum_x, maximum_y, minimum_x, minimum_y

# Get the homography mismatch
def get_homography(matches, src_pts, dst_pts, kp2):
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    outliers_homography = np.array(matches)[mask.ravel()==0]
    points2_homography = np.int32([kp2[m.trainIdx].pt for m in outliers_homography]).reshape(-1,1,2)
    pointh_2d = points2_homography.reshape(points2_homography.shape[0], 2)

    return pointh_2d

# Get the epipolar mismatch
def get_epipolar_mismatch(matches, src_pts, dst_pts, kp2):
      # Fundamental matrix
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
    # Convert to homogenous coordinates
    homogeneous_src = np.concatenate((src_pts, np.ones((src_pts.shape[0], 1, 1))), axis=2)
    homogeneous_dst = np.concatenate((dst_pts, np.ones((dst_pts.shape[0], 1, 1))), axis=2)

    src_squeeze = homogeneous_src.reshape(src_pts.shape[0], 3)
    dst_squeeze = homogeneous_dst.reshape(dst_pts.shape[0], 3)

    # Epipolar lines
    I  = F @ src_squeeze.T
    distance = np.abs(np.sum(I * dst_squeeze.T, axis=0)) / np.sqrt(I[0,:]**2 + I[1,:]**2)
    # Find the median distance
    mean_distance = np.mean(distance)
    outliers_epipolar = np.array(matches)[distance > mean_distance]

    # Get the points in image2
    points2_epipolar = np.int32([kp2[m.trainIdx].pt for m in outliers_epipolar]).reshape(-1,1,2)
    point2_2d = points2_epipolar.reshape(points2_epipolar.shape[0], 2)
    
    return point2_2d

# Draw the bounding box
def draw_bbox(point2_2d, pointh_2d, eps_value):
    eps = eps_value
    point_append = np.append(point2_2d, pointh_2d, axis=0)
    df = pd.DataFrame(point_append, columns=['xcoord', 'ycoord'])
    db = DBSCAN(eps=eps, min_samples=1).fit(point_append)
    
    cluster_labels = pd.Series(db.labels_).rename("cluster")
    cluster_labels[cluster_labels == -1] = np.nan
    dfnew=pd.concat([df,cluster_labels],axis=1,sort=False)

    MINIMUM_CLUSTER_SIZE = 3
    store_bbox = []
    for df_sub in dfnew.groupby('cluster'):
        if df_sub[1].shape[0] >= MINIMUM_CLUSTER_SIZE:
            x_coords = df_sub[1]['xcoord'].values
            y_coords = df_sub[1]['ycoord'].values
            maxx, maxy, minx, miny = get_bounding_box(x_coords, y_coords)
            store_bbox.append([minx, miny, maxx, maxy])
    return store_bbox

# ORB-based feature detection
def feature_detection(image1, image2):
    #print("Feature detection")
    # Initiate ORB detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1,None)
    kp2, des2 = orb.detectAndCompute(image2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    src_pts = np.int32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.int32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    return src_pts, dst_pts, matches, kp2


def geometry_evaluation(image1, image2, eps_value):
    src_pts, dst_pts, matches, kp2 = feature_detection(image1, image2)
    
    #pointh_2d = get_homography(matches, src_pts, dst_pts, kp2)
    point2_2d = get_epipolar_mismatch(matches, src_pts, dst_pts, kp2)
  
    #store_bbox = draw_bbox(point2_2d, pointh_2d, eps_value)
    return []
    #return store_bbox
    

