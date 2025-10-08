[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/lyfclldM)
# Homework2

Dataset: [Download](https://drive.google.com/u/0/uc?export=download&confirm=qrVw&id=1GrCpYJFc8IZM_Uiisq6e8UxwVMFvr4AJ)

## 2D-3D Matching Implementation

This repository contains a comprehensive implementation of a 2D-3D matching system for camera pose estimation.

### Features Implemented

#### 1. PnP Solver (`pnpsolver`)
- **FLANN-based descriptor matching**: Uses KD-tree algorithm for efficient nearest neighbor search
- **Lowe's ratio test**: Filters matches using 0.7 ratio threshold for robust feature matching
- **cv2.solvePnPRansac**: Robust pose estimation with RANSAC outlier rejection
- Returns rotation vector (rvec), translation vector (tvec), and inlier indices

#### 2. Error Calculation Functions
- **`rotation_error(R1, R2)`**: 
  - Computes rotation error between two quaternions
  - Returns angular difference in degrees
  - Uses scipy's Rotation class for accurate quaternion operations
  
- **`translation_error(t1, t2)`**: 
  - Calculates Euclidean distance between translation vectors
  - Provides metric for camera position accuracy

#### 3. Main Processing Pipeline
- Loads pre-computed 3D points and descriptors
- Processes query images with 2D-3D matching
- Converts rotation vectors to quaternions for ground truth comparison
- Handles PnP solver failures gracefully
- Calculates median rotation and translation errors across all test images
- Prints comprehensive error statistics

#### 4. Camera Pose Transformation
- Converts world-to-camera pose (from solvePnP) to camera-to-world transformation
- Properly handles rotation matrix inversion: R_c2w = R^T
- Correctly transforms translation: t_c2w = -R^T * t
- Builds 4x4 homogeneous transformation matrices

#### 5. Visualization (`visualization`)
- Creates 3D matplotlib visualization of the scene
- Displays point cloud with RGB colors
- Shows camera positions as red markers
- Visualizes camera orientations using quiver arrows
- Saves output as PNG image
- Handles large point clouds by sampling

#### 6. Bonus: Custom P3P Implementation
Three additional functions for educational purposes:

- **`p3p_solver(pts_3d, pts_2d, K)`**: 
  - Custom Perspective-3-Point algorithm
  - Solves for camera pose from 3 point correspondences
  - Uses geometric approach with bearing vectors
  - Returns multiple possible solutions

- **`p3p_ransac(pts_3d, pts_2d, K)`**: 
  - RANSAC wrapper for P3P solver
  - Iteratively samples 3-point subsets
  - Evaluates solutions using reprojection error
  - Returns best pose with inlier set

- **`pnpsolver_p3p(query, model)`**: 
  - Alternative PnP solver using custom P3P+RANSAC
  - Same matching pipeline as main solver
  - Educational implementation showing P3P internals

### Usage

```python
python 2d3dmathcing.py
```

The script will:
1. Load dataset from `data/` directory
2. Process images 200 and 201
3. Estimate camera poses using PnP
4. Calculate and print median errors
5. Generate visualization saved as `camera_poses_visualization.png`

### Key Improvements Made
- Fixed error calculation bug (was comparing ground truth to itself)
- Implemented all TODO functions with comprehensive documentation
- Added robust error handling for edge cases
- Included detailed comments explaining mathematical operations
- Provided bonus P3P implementation for deeper understanding

### Mathematical Background

**PnP Problem**: Given n ≥ 3 correspondences between 3D points and 2D image points, estimate the camera pose (R, t) such that:
```
x_image = K * [R | t] * X_world
```

**Lowe's Ratio Test**: For each match m with nearest neighbor n:
```
if distance(m) < 0.7 * distance(n): accept match
```

**Camera-to-World Transform**: 
```
X_world = R_c2w * x_camera + t_c2w
where R_c2w = R^T and t_c2w = -R^T * t
```
