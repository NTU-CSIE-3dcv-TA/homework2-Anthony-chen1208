# System Architecture and Flow Diagram

## 2D-3D Matching Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT DATA LOADING                           │
├─────────────────────────────────────────────────────────────────┤
│ • images.pkl      - Image metadata & ground truth poses         │
│ • train.pkl       - Training 3D points & descriptors            │
│ • points3D.pkl    - 3D point cloud (XYZ, RGB)                   │
│ • point_desc.pkl  - Query image keypoints & descriptors         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  DESCRIPTOR PREPROCESSING                        │
├─────────────────────────────────────────────────────────────────┤
│ average_desc(): Group descriptors by 3D point, compute mean     │
│                                                                  │
│ Input:  Multiple descriptors per 3D point from different views  │
│ Output: Single averaged descriptor per 3D point                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FOR EACH QUERY IMAGE                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │         PNP SOLVER (Main Path)          │
        ├─────────────────────────────────────────┤
        │ 1. FLANN KD-Tree Matching               │
        │    - Query descriptors → Model descr.   │
        │    - Find k=2 nearest neighbors         │
        │                                         │
        │ 2. Lowe's Ratio Test                    │
        │    - Keep if dist(m) < 0.7*dist(n)     │
        │    - Filter unreliable matches          │
        │                                         │
        │ 3. Extract Correspondences              │
        │    - 3D points ← matched model pts      │
        │    - 2D points ← matched query pts      │
        │                                         │
        │ 4. cv2.solvePnPRansac                   │
        │    - Input: 3D-2D correspondences       │
        │    - Output: [R|t], inliers            │
        │    - Returns world-to-camera transform  │
        └─────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │      ALTERNATIVE: P3P + RANSAC          │
        │           (Bonus Implementation)         │
        ├─────────────────────────────────────────┤
        │ 1. Same FLANN matching                  │
        │                                         │
        │ 2. RANSAC Loop:                         │
        │    - Sample 3 correspondences           │
        │    - Solve P3P (geometric solution)     │
        │    - Count inliers                      │
        │    - Keep best solution                 │
        │                                         │
        │ 3. Output: [R|t]                        │
        └─────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   POSE REPRESENTATION CONVERSION                 │
├─────────────────────────────────────────────────────────────────┤
│ rvec (rotation vector) → quaternion (x,y,z,w)                   │
│ Using: scipy.spatial.transform.Rotation                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      ERROR CALCULATION                           │
├─────────────────────────────────────────────────────────────────┤
│ rotation_error(q_gt, q_est):                                    │
│   - Convert to Rotation objects                                 │
│   - Compute relative rotation                                   │
│   - Extract angle in degrees                                    │
│                                                                  │
│ translation_error(t_gt, t_est):                                 │
│   - Euclidean distance: ||t_gt - t_est||₂                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   AGGREGATE STATISTICS                           │
├─────────────────────────────────────────────────────────────────┤
│ • Collect rotation errors from all images                       │
│ • Collect translation errors from all images                    │
│ • Filter out None values (failed PnP)                           │
│ • Calculate median of each                                      │
│ • Print results                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              CAMERA-TO-WORLD TRANSFORMATION                      │
├─────────────────────────────────────────────────────────────────┤
│ Input: World-to-camera [R|t] from solvePnP                      │
│                                                                  │
│ Conversion:                                                     │
│   R_c2w = R^T        (transpose of rotation)                    │
│   t_c2w = -R^T * t   (transformed translation)                  │
│                                                                  │
│ Build 4x4 matrix:                                               │
│   ┌              ┐                                              │
│   │ R_c2w  t_c2w │                                              │
│   │   0      1   │                                              │
│   └              ┘                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      3D VISUALIZATION                            │
├─────────────────────────────────────────────────────────────────┤
│ matplotlib 3D plot:                                             │
│                                                                  │
│ • Point cloud (colored by RGB)                                  │
│   - Sample if >10k points                                       │
│   - Small markers (s=1)                                         │
│                                                                  │
│ • Camera positions (red spheres)                                │
│   - Position from t_c2w                                         │
│   - Size = 100                                                  │
│                                                                  │
│ • Camera orientations (red arrows)                              │
│   - Direction: Z-axis of R_c2w                                  │
│   - Length: 0.5 units                                           │
│                                                                  │
│ • Save as PNG (150 DPI)                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                         [COMPLETE]
```

## Key Mathematical Transformations

### 1. World-to-Camera (from solvePnP)
```
x_cam = R_w2c * X_world + t_w2c
```

### 2. Camera-to-World (for visualization)
```
X_world = R_c2w * x_cam + t_c2w

where:
  R_c2w = R_w2c^T
  t_c2w = -R_w2c^T * t_w2c
```

### 3. Rotation Representations
```
Rotation Vector (rvec) → Rotation Matrix (R)
    cv2.Rodrigues(rvec) → R

Rotation Vector (rvec) → Quaternion (q)
    scipy.Rotation.from_rotvec(rvec).as_quat() → q
```

### 4. Error Metrics
```
Rotation Error:
    relative_rot = rot_gt^(-1) * rot_est
    error = magnitude(relative_rot) * 180/π [degrees]

Translation Error:
    error = ||t_gt - t_est||₂ [units]
```

## Function Call Graph

```
main()
  │
  ├─ average_desc()
  │   └─ average()
  │
  ├─ [for each image]
  │   │
  │   ├─ pnpsolver()  [or pnpsolver_p3p()]
  │   │   ├─ cv2.FlannBasedMatcher()
  │   │   ├─ flann.knnMatch()
  │   │   ├─ Lowe's ratio test
  │   │   └─ cv2.solvePnPRansac()
  │   │       [or]
  │   │       └─ p3p_ransac()
  │   │           └─ p3p_solver()
  │   │
  │   ├─ R.from_rotvec().as_quat()
  │   │
  │   ├─ rotation_error()
  │   │   ├─ R.from_quat()
  │   │   └─ relative_rot.magnitude()
  │   │
  │   └─ translation_error()
  │       └─ np.linalg.norm()
  │
  ├─ np.median() [statistics]
  │
  ├─ [for each pose]
  │   └─ cv2.Rodrigues()
  │   └─ build c2w matrix
  │
  └─ visualization()
      ├─ plt.figure()
      ├─ ax.scatter() [points]
      ├─ ax.scatter() [cameras]
      ├─ ax.quiver() [orientations]
      └─ plt.savefig()
```

## Data Flow Summary

```
Input Files → Load → Process Descriptors → Match → PnP Solve → 
Convert Pose → Calculate Errors → Aggregate Stats → Transform → 
Visualize → Save Output
```

## Error Handling

- PnP failure: Returns None values, skipped in statistics
- Insufficient matches: Returns None from pnpsolver
- Invalid quaternions: Would raise exception (not explicitly handled)
- Large point clouds: Automatically sampled in visualization
