from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import random
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

np.random.seed(1428) # do not change this seed
random.seed(1428) # do not change this seed

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    # FLANN-based descriptor matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(desc_query, desc_model, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        return None, None, None, None
    
    # Get matched keypoints
    obj_pts = np.array([kp_model[m.trainIdx] for m in good_matches])
    img_pts = np.array([kp_query[m.queryIdx] for m in good_matches])
    
    # Solve PnP with RANSAC
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, cameraMatrix, distCoeffs,
        reprojectionError=8.0, confidence=0.99
    )
    
    return retval, rvec, tvec, inliers

def rotation_error(R1, R2):
    # R1 and R2 are quaternions in (x, y, z, w) format
    # Convert quaternions to rotation matrices
    rot1 = R.from_quat(R1.flatten())
    rot2 = R.from_quat(R2.flatten())
    
    # Calculate relative rotation
    relative_rot = rot1.inv() * rot2
    
    # Convert to angle in degrees
    angle = relative_rot.magnitude() * 180 / np.pi
    
    return angle

def translation_error(t1, t2):
    # Calculate Euclidean distance between translation vectors
    return np.linalg.norm(t1.flatten() - t2.flatten())

def visualization(Camera2World_Transform_Matrixs, points3D_df):
    # Create 3D visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Visualize 3D points
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB']) / 255.0
    
    # Sample points for better performance (if too many)
    if len(xyz) > 10000:
        indices = np.random.choice(len(xyz), 10000, replace=False)
        xyz = xyz[indices]
        rgb = rgb[indices]
    
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, marker='.', s=1, alpha=0.5)
    
    # Visualize camera poses
    for i, c2w in enumerate(Camera2World_Transform_Matrixs):
        # Extract camera position
        cam_pos = c2w[:3, 3]
        
        # Extract camera orientation (Z-axis direction)
        cam_z = c2w[:3, 2]
        
        # Plot camera position
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], c='r', marker='o', s=100)
        
        # Plot camera orientation
        ax.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
                 cam_z[0], cam_z[1], cam_z[2],
                 length=0.5, color='r', arrow_length_ratio=0.3)
        
        # Add camera label
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f'Cam {i}', fontsize=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scene and Camera Poses')
    
    plt.savefig('camera_poses_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Visualization saved as 'camera_poses_visualization.png'")

if __name__ == "__main__":
    # Load data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)


    IMAGE_ID_LIST = [200,201]
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []
    for idx in tqdm(IMAGE_ID_LIST):
        # Load quaery image
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        
        # Handle case where PnP solver fails
        if retval is None or not retval:
            print(f"PnP solver failed for image {idx}")
            r_list.append(None)
            t_list.append(None)
            rotation_error_list.append(None)
            translation_error_list.append(None)
            continue
            
        # Convert rotation vector to quaternion
        rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat()
        tvec = tvec.reshape(1,3)
        r_list.append(rvec)
        t_list.append(tvec)

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        # Calculate error - comparing estimated vs ground truth
        r_error = rotation_error(rotq_gt, rotq)
        t_error = translation_error(tvec_gt, tvec)
        rotation_error_list.append(r_error)
        translation_error_list.append(t_error)

    # Calculate median of rotation and translation errors
    valid_rotation_errors = [e for e in rotation_error_list if e is not None]
    valid_translation_errors = [e for e in translation_error_list if e is not None]
    
    if valid_rotation_errors:
        median_rotation_error = np.median(valid_rotation_errors)
        print(f"Median Rotation Error: {median_rotation_error:.4f} degrees")
    else:
        print("No valid rotation errors to calculate median")
    
    if valid_translation_errors:
        median_translation_error = np.median(valid_translation_errors)
        print(f"Median Translation Error: {median_translation_error:.4f}")
    else:
        print("No valid translation errors to calculate median")

    # Result visualization
    Camera2World_Transform_Matrixs = []
    for r, t in zip(r_list, t_list):
        # Skip if PnP failed
        if r is None or t is None:
            continue
            
        # Convert rotation vector to rotation matrix
        R_mat, _ = cv2.Rodrigues(r)
        
        # solvePnP returns camera pose in world coordinates
        # The transformation matrix is [R|t] where R*X + t = x
        # To get camera-to-world, we need to invert this
        # Camera-to-world: R_cw = R^T, t_cw = -R^T * t
        R_c2w = R_mat.T
        t_c2w = -R_c2w @ t.reshape(3, 1)
        
        # Build 4x4 transformation matrix
        c2w = np.eye(4)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = t_c2w.flatten()
        
        Camera2World_Transform_Matrixs.append(c2w)
    visualization(Camera2World_Transform_Matrixs, points3D_df)