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
    """
    Solve Perspective-n-Point (PnP) problem using FLANN-based matching and RANSAC.
    
    Args:
        query: Tuple of (keypoints, descriptors) from query image
        model: Tuple of (3D points, descriptors) from model
        cameraMatrix: Camera intrinsic matrix (default provided)
        distortion: Distortion coefficients (default provided)
    
    Returns:
        retval: Success flag
        rvec: Rotation vector
        tvec: Translation vector  
        inliers: Inlier indices
    """
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
    """
    Calculate rotation error between two quaternions.
    
    Args:
        R1: First quaternion (x, y, z, w) format
        R2: Second quaternion (x, y, z, w) format
    
    Returns:
        Rotation error in degrees
    """
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
    """
    Calculate Euclidean distance between two translation vectors.
    
    Args:
        t1: First translation vector
        t2: Second translation vector
    
    Returns:
        Euclidean distance
    """
    # Calculate Euclidean distance between translation vectors
    return np.linalg.norm(t1.flatten() - t2.flatten())

def visualization(Camera2World_Transform_Matrixs, points3D_df):
    """
    Visualize 3D scene points and camera poses.
    
    Args:
        Camera2World_Transform_Matrixs: List of 4x4 camera-to-world transformation matrices
        points3D_df: DataFrame containing 3D points with XYZ and RGB columns
    """
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

# ===== BONUS: Custom P3P + RANSAC Implementation =====

def p3p_solver(pts_3d, pts_2d, K):
    """
    Custom P3P (Perspective-3-Point) solver.
    
    Given 3 3D points and their 2D projections, solve for camera pose.
    This is a simplified implementation based on the geometric approach.
    
    Args:
        pts_3d: 3x3 array of 3D points
        pts_2d: 3x2 array of 2D image points
        K: 3x3 camera intrinsic matrix
    
    Returns:
        List of possible [R|t] solutions (can have up to 4 solutions)
    """
    # Normalize 2D points to get bearing vectors
    K_inv = np.linalg.inv(K)
    bearing_vectors = []
    for i in range(3):
        p_homo = np.array([pts_2d[i, 0], pts_2d[i, 1], 1.0])
        bearing = K_inv @ p_homo
        bearing = bearing / np.linalg.norm(bearing)
        bearing_vectors.append(bearing)
    
    # Get 3D points
    P1, P2, P3 = pts_3d[0], pts_3d[1], pts_3d[2]
    
    # Calculate distances between 3D points
    a = np.linalg.norm(P2 - P3)  # distance P2-P3
    b = np.linalg.norm(P1 - P3)  # distance P1-P3
    c = np.linalg.norm(P1 - P2)  # distance P1-P2
    
    # Calculate cosines of angles between bearing vectors
    f1, f2, f3 = bearing_vectors[0], bearing_vectors[1], bearing_vectors[2]
    cos_alpha = np.dot(f2, f3)  # angle between rays to P2 and P3
    cos_beta = np.dot(f1, f3)   # angle between rays to P1 and P3
    cos_gamma = np.dot(f1, f2)  # angle between rays to P1 and P2
    
    # Use law of cosines to solve for distances from camera to 3D points
    # This is a simplified solution - full P3P involves solving a quartic equation
    # Here we use an approximate geometric solution
    
    solutions = []
    
    # Try to solve using geometric constraints
    # Distance from camera to P3 (depth)
    for sign in [1, -1]:
        try:
            # Approximate solution using law of cosines
            # a^2 = d2^2 + d3^2 - 2*d2*d3*cos_alpha
            # b^2 = d1^2 + d3^2 - 2*d1*d3*cos_beta
            # c^2 = d1^2 + d2^2 - 2*d1*d2*cos_gamma
            
            # Simplified approach: assume d3 = 1 and solve for ratios
            d3 = b  # initial guess
            d2 = np.sqrt(max(0, a**2 + d3**2 - 2*a*d3*cos_alpha))
            d1 = np.sqrt(max(0, b**2 + d3**2 - 2*b*d3*cos_beta))
            
            # Get 3D points in camera frame
            P1_cam = d1 * f1
            P2_cam = d2 * f2
            P3_cam = d3 * f3
            
            # Estimate rotation using Procrustes/Kabsch algorithm
            pts_world = np.vstack([P1, P2, P3])
            pts_cam = np.vstack([P1_cam, P2_cam, P3_cam])
            
            # Center the points
            centroid_world = np.mean(pts_world, axis=0)
            centroid_cam = np.mean(pts_cam, axis=0)
            
            pts_world_centered = pts_world - centroid_world
            pts_cam_centered = pts_cam - centroid_cam
            
            # Compute rotation using SVD
            H = pts_cam_centered.T @ pts_world_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure proper rotation matrix (det = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Compute translation
            t = centroid_cam - R @ centroid_world
            
            solutions.append((R, t))
        except:
            continue
    
    return solutions

def p3p_ransac(pts_3d, pts_2d, K, num_iterations=1000, threshold=8.0):
    """
    RANSAC wrapper for P3P solver.
    
    Args:
        pts_3d: Nx3 array of 3D points
        pts_2d: Nx2 array of 2D image points
        K: 3x3 camera intrinsic matrix
        num_iterations: Number of RANSAC iterations
        threshold: Reprojection error threshold in pixels
    
    Returns:
        best_R: Best rotation matrix
        best_t: Best translation vector
        best_inliers: Indices of inlier points
    """
    n_points = len(pts_3d)
    best_inliers = []
    best_R = None
    best_t = None
    
    for iteration in range(num_iterations):
        # Randomly sample 3 points
        sample_indices = np.random.choice(n_points, 3, replace=False)
        sample_3d = pts_3d[sample_indices]
        sample_2d = pts_2d[sample_indices]
        
        # Solve P3P
        solutions = p3p_solver(sample_3d, sample_2d, K)
        
        # Evaluate each solution
        for R, t in solutions:
            # Reproject all points
            inliers = []
            for i in range(n_points):
                # Project 3D point to 2D
                p_3d = pts_3d[i]
                p_cam = R @ p_3d + t
                
                if p_cam[2] <= 0:  # Point behind camera
                    continue
                
                p_proj = K @ p_cam
                p_proj = p_proj[:2] / p_proj[2]
                
                # Calculate reprojection error
                error = np.linalg.norm(p_proj - pts_2d[i])
                
                if error < threshold:
                    inliers.append(i)
            
            # Update best solution
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_R = R
                best_t = t
    
    return best_R, best_t, best_inliers

def pnpsolver_p3p(query, model, cameraMatrix=0, distortion=0):
    """
    Alternative PnP solver using custom P3P + RANSAC implementation.
    
    This is a bonus implementation for educational purposes.
    Uses the same matching as pnpsolver() but with custom P3P+RANSAC.
    
    Args:
        query: Tuple of (keypoints, descriptors) from query image
        model: Tuple of (3D points, descriptors) from model
        cameraMatrix: Camera intrinsic matrix
        distortion: Distortion coefficients (not used in P3P)
    
    Returns:
        retval: Success flag
        rvec: Rotation vector
        tvec: Translation vector
        inliers: Inlier indices
    """
    kp_query, desc_query = query
    kp_model, desc_model = model
    K = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])
    
    # FLANN-based descriptor matching (same as pnpsolver)
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
    
    if len(good_matches) < 3:
        return None, None, None, None
    
    # Get matched keypoints
    obj_pts = np.array([kp_model[m.trainIdx] for m in good_matches])
    img_pts = np.array([kp_query[m.queryIdx] for m in good_matches])
    
    # Solve using custom P3P + RANSAC
    R, t, inliers = p3p_ransac(obj_pts, img_pts, K)
    
    if R is None:
        return None, None, None, None
    
    # Convert rotation matrix to rotation vector
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    
    return True, rvec, tvec, np.array(inliers)



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