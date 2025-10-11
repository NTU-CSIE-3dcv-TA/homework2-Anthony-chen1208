from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import pandas as pd
import numpy as np
import random
import cv2
import time
from pathlib import Path


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

def p3p(p_w, p_i, K):
    """
    Solves the P3P problem.
    Args:
        p_w: 3x3 numpy array of 3D points in world coordinates.
        p_i: 2x3 numpy array of 2D projections in image coordinates.
        K: 3x3 camera intrinsic matrix.
    Returns:
        A list of possible (R, t) tuples.
    """
    eps = 1e-9
    # Convert image points to normalized camera coordinates
    Kinv = np.linalg.inv(K)
    p_c = Kinv @ np.vstack((p_i, np.ones(3)))
    # Normalize rays (guard divide-by-zero)
    norms = np.linalg.norm(p_c, axis=0)
    if np.any(norms < eps) or not np.all(np.isfinite(norms)):
        return []
    v = p_c / norms

    # Distances between 3D points (guard degeneracy)
    d_12_sq = float(np.sum((p_w[:, 0] - p_w[:, 1])**2))
    d_23_sq = float(np.sum((p_w[:, 1] - p_w[:, 2])**2))
    d_31_sq = float(np.sum((p_w[:, 2] - p_w[:, 0])**2))
    # Triangle area check
    area2 = np.linalg.norm(np.cross(p_w[:, 1] - p_w[:, 0], p_w[:, 2] - p_w[:, 0]))**2
    if d_12_sq < 1e-8 or d_23_sq < 1e-8 or d_31_sq < 1e-8 or area2 < 1e-12:
        return []

    # Cosines of angles between rays
    cos_alpha = float(np.clip(v[:, 1].T @ v[:, 2], -1.0, 1.0))
    cos_beta  = float(np.clip(v[:, 0].T @ v[:, 2], -1.0, 1.0))
    cos_gamma = float(np.clip(v[:, 0].T @ v[:, 1], -1.0, 1.0))

    # Coefficients of the quartic equation
    A4 = ((d_12_sq - d_31_sq) / d_23_sq - 1)**2 - 4 * d_31_sq / d_23_sq * cos_gamma**2
    A3 = 4 * (((d_12_sq - d_31_sq) / d_23_sq) * (1 - (d_12_sq - d_31_sq) / d_23_sq) * cos_beta - (1 - (d_12_sq + d_31_sq) / d_23_sq) * cos_alpha * cos_gamma + 2 * d_31_sq / d_23_sq * cos_gamma**2 * cos_beta)
    A2 = 2 * (((d_12_sq - d_31_sq) / d_23_sq)**2 - 1 + 2 * ((d_12_sq - d_31_sq) / d_23_sq)**2 * cos_beta**2 + 2 * ((d_23_sq - d_31_sq) / d_23_sq) * cos_alpha**2 - 4 * ((d_12_sq + d_31_sq) / d_23_sq) * cos_alpha * cos_beta * cos_gamma + 2 * ((d_23_sq - d_12_sq) / d_23_sq) * cos_gamma**2)
    A1 = 4 * (-(d_12_sq - d_31_sq) / d_23_sq * (1 + (d_12_sq - d_31_sq) / d_23_sq) * cos_beta + 2 * d_12_sq / d_23_sq * cos_alpha**2 * cos_beta - (1 - (d_12_sq + d_31_sq) / d_23_sq) * cos_alpha * cos_gamma)
    A0 = (1 + (d_12_sq - d_31_sq) / d_23_sq)**2 - 4 * d_12_sq / d_23_sq * cos_alpha**2

    coeffs = np.array([A4, A3, A2, A1, A0], dtype=float)
    # Guard invalid/inf coefficients (degenerate sample)
    if not np.all(np.isfinite(coeffs)):
        return []
    # If leading term ~0, fall back to lower degree
    if abs(coeffs[0]) < 1e-12:
        coeffs = coeffs[1:]
        if coeffs.size < 2 or not np.any(np.abs(coeffs) > 0):
            return []
    # Solve for x = d1/d2 (catch numeric failures)
    try:
        x_roots = np.roots(coeffs)
    except Exception:
        return []
    x_reals = x_roots[np.isreal(x_roots)].real

    solutions = []
    for x in x_reals:
        # Solve for y = d3/d2
        y = ((d_12_sq - d_31_sq) / d_23_sq - 1) * x**2 - 2 * ((d_12_sq - d_31_sq) / d_23_sq) * cos_beta * x + 1 + (d_12_sq - d_31_sq) / d_23_sq
        
        if y <= 0:
            continue
        y = np.sqrt(y)

        # Solve for d2
        d2_sq_num = d_12_sq
        d2_sq_den = x**2 - 2 * x * cos_gamma + 1
        if d2_sq_den <= 1e-8:
            continue
        d2 = np.sqrt(d2_sq_num / d2_sq_den)

        # Get d1, d3
        d1 = x * d2
        d3 = y * d2

        # Two possible solutions for y (sqrt)
        for sign in [1, -1]:
            d3_sol = sign * d3
            
            # Check if distances are positive
            if d1 > 0 and d2 > 0 and d3_sol > 0:
                # 3D points in camera coordinates
                p_c_calc = np.zeros((3, 3))
                p_c_calc[:, 0] = d1 * v[:, 0]
                p_c_calc[:, 1] = d2 * v[:, 1]
                p_c_calc[:, 2] = d3_sol * v[:, 2]

                # Check distances
                if not np.allclose(np.sum((p_c_calc[:, 1] - p_c_calc[:, 2])**2), d_23_sq, atol=1e-3):
                    continue
                if not np.allclose(np.sum((p_c_calc[:, 0] - p_c_calc[:, 2])**2), d_31_sq, atol=1e-3):
                    continue

                # Find R, t using Horn's method (absolute orientation)
                p_w_centroid = np.mean(p_w, axis=1, keepdims=True)
                p_c_centroid = np.mean(p_c_calc, axis=1, keepdims=True)
                
                H = (p_w - p_w_centroid) @ (p_c_calc - p_c_centroid).T
                
                U, S, Vt = np.linalg.svd(H)
                
                R = Vt.T @ U.T
                
                if np.linalg.det(R) < 0:
                    Vt[2, :] *= -1
                    R = Vt.T @ U.T
                
                t = p_c_centroid - R @ p_w_centroid
                solutions.append((R, t))

    # If quartic produced valid solutions, return them
    if len(solutions) > 0:
        return solutions

    # Fallback: numeric P3P via three scale variables on unit rays + Kabsch
    f = v  # 3x3 unit rays corresponding to p_i columns
    d12 = np.sqrt(d_12_sq); d23 = np.sqrt(d_23_sq); d13 = np.sqrt(d_31_sq)
    c12 = float(np.dot(f[:,0], f[:,1]))
    c23 = float(np.dot(f[:,1], f[:,2]))
    c13 = float(np.dot(f[:,0], f[:,2]))

    def residuals(s):
        s1, s2, s3 = s
        r1 = (s1*s1 + s2*s2 - 2*s1*s2*c12) - d12*d12
        r2 = (s2*s2 + s3*s3 - 2*s2*s3*c23) - d23*d23
        r3 = (s1*s1 + s3*s3 - 2*s1*s3*c13) - d13*d13
        return np.array([r1, r2, r3], dtype=float)

    def jac(s):
        s1, s2, s3 = s
        return np.array([
            [2*(s1 - s2*c12), 2*(s2 - s1*c12), 0.0],
            [0.0, 2*(s2 - s3*c23), 2*(s3 - s2*c23)],
            [2*(s1 - s3*c13), 0.0, 2*(s3 - s1*c13)],
        ], dtype=float)

    d_mean = max(1e-3, (d12 + d23 + d13)/3.0)
    s0 = np.array([d_mean, d_mean, d_mean], dtype=float)
    try:
        res = least_squares(residuals, s0, jac=jac, method='lm', max_nfev=40)
    except Exception:
        res = least_squares(residuals, s0, method='trf', loss='soft_l1', f_scale=1.0, max_nfev=60)
    if not res.success:
        return []
    s = res.x
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        return []
    X_cam = (f * s.reshape(3,1))  # 3x3
    # Kabsch to align world->camera: minimize ||X_cam - R*p_w - t||
    Pw = p_w
    Pw_c = Pw - Pw.mean(axis=1, keepdims=True)
    Xc_c = X_cam - X_cam.mean(axis=1, keepdims=True)
    H = Pw_c @ Xc_c.T
    U, Sg, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[2,:] *= -1
        Rm = Vt.T @ U.T
    t = X_cam.mean(axis=1, keepdims=True) - Rm @ Pw.mean(axis=1, keepdims=True)
    return [(Rm, t)]

def ransac_p3p(p_w, p_i, K, threshold, iterations):
    """
    P3P with RANSAC.
    Args:
        p_w: 3xN numpy array of 3D points in world coordinates.
        p_i: 2xN numpy array of 2D projections in image coordinates.
        K: 3x3 camera intrinsic matrix.
        threshold: Inlier threshold for reprojection error.
        iterations: Number of RANSAC iterations.
    Returns:
        Best (R, t) tuple and inlier indices.
    """
    best_inliers_count = 0
    best_R, best_t = None, None
    best_inliers_indices = None
    num_points = p_w.shape[1]

    for i in range(iterations):
        # Randomly sample 3 points
        sample_indices = np.random.choice(num_points, 3, replace=False)
        p_w_sample = p_w[:, sample_indices]
        p_i_sample = p_i[:, sample_indices]

        # Solve P3P for the sample
        poses = p3p(p_w_sample, p_i_sample, K)

        for R_est, t_est in poses:
            # Project all points; use only points in front of camera
            Pc = (R_est @ p_w) + t_est
            valid = Pc[2, :] > 1e-6
            if valid.sum() < 3:
                continue
            p_i_proj_h = K @ Pc[:, valid]
            p_i_proj = p_i_proj_h[:2, :] / np.clip(p_i_proj_h[2, :], 1e-8, None)

            # Calculate reprojection error only on valid
            err = np.linalg.norm(p_i[:, valid] - p_i_proj, axis=0)
            inliers_local = np.where(err < threshold)[0]
            inliers_indices = np.where(valid)[0][inliers_local]
            inliers_count = len(inliers_indices)

            # Update best model if current one is better
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_R, best_t = R_est, t_est
                best_inliers_indices = inliers_indices
                
                # Optional: LM refine on inliers (reprojection error)
                if len(inliers_indices) >= 4:
                    Pw_in = p_w[:, inliers_indices]
                    uv_in = p_i[:, inliers_indices]

                    def reproj_resid(x):
                        rvec = x[:3]
                        tvec = x[3:]
                        Rm = R.from_rotvec(rvec).as_matrix()
                        Pc2 = (Rm @ Pw_in) + tvec.reshape(3,1)
                        z = Pc2[2, :]
                        mask = z > 1e-6
                        if mask.sum() == 0:
                            return np.ones(uv_in.size) * 1e6
                        uvp_h = K @ Pc2[:, mask]
                        uvp = uvp_h[:2, :] / np.clip(uvp_h[2, :], 1e-8, None)
                        r = (uvp - uv_in[:, mask]).reshape(-1)
                        return r

                    x0 = np.zeros(6, dtype=float)
                    x0[:3] = R.from_matrix(best_R).as_rotvec()
                    x0[3:] = best_t.reshape(3)
                    try:
                        opt = least_squares(reproj_resid, x0, method='trf', loss='huber', f_scale=4.0, max_nfev=50)
                        best_R = R.from_rotvec(opt.x[:3]).as_matrix()
                        best_t = opt.x[3:].reshape(3,1)
                    except Exception:
                        pass


    return best_R, best_t, best_inliers_indices

def rotation_error(R1, R2):
    R1 = np.asarray(R1)
    R2 = np.asarray(R2)
    def to_rotation(x):
        x = np.asarray(x)
        if x.ndim == 1:
            if x.size == 3:
                return R.from_rotvec(x)
            if x.size == 4:
                return R.from_quat(x)
            if x.size == 9:
                return R.from_matrix(x.reshape(3, 3))
        elif x.ndim == 2:
            if x.shape == (3, 3):
                return R.from_matrix(x)
            if x.shape[1] == 3:
                return R.from_rotvec(x)
            if x.shape[1] == 4:
                return R.from_quat(x)
        elif x.ndim == 3 and x.shape[1:] == (3, 3):
            return R.from_matrix(x)
        raise ValueError("Unsupported rotation representation")
    rot1 = to_rotation(R1)
    rot2 = to_rotation(R2)
    # Relative rotation between R1 and R2
    rel = rot1 * rot2.inv()
    # Angle magnitude in radians; convert to degrees
    ang = rel.magnitude()
    return np.degrees(ang)

def translation_error(t1, t2):
    t1 = np.asarray(t1, dtype=float).ravel()
    t2 = np.asarray(t2, dtype=float).ravel()
    if t1.size % 3 != 0 or t2.size % 3 != 0:
        raise ValueError("Translation vectors must be multiples of 3 in length")
    t1 = t1.reshape(-1, 3)
    t2 = t2.reshape(-1, 3)
    # Broadcast along rows if one is a single vector
    if t1.shape[0] == 1 and t2.shape[0] > 1:
        t1 = np.repeat(t1, t2.shape[0], axis=0)
    elif t2.shape[0] == 1 and t1.shape[0] > 1:
        t2 = np.repeat(t2, t1.shape[0], axis=0)
    elif t1.shape[0] != t2.shape[0]:
        raise ValueError("Mismatched batch sizes for translation vectors")
    err = np.linalg.norm(t1 - t2, axis=1)
    return err.item() if err.shape[0] == 1 else err

def visualization(
    Camera2World_Transform_Matrixs,
    points3D_df,
    close_loop=False,
    draw_circle=False,
    draw_circle_cameras=False,
    n_circle_cams=24,
    draw_axes=False,
    draw_center_spheres=True,
    cam_scale="auto",
    frustum_stride=1,
    intrinsics=None,
    img_wh=None,
    edge_width_frac_sides=0.5,
    edge_width_frac_base=0.5,
    traj_width_frac=0.2,
    draw_frustum_sides=True,
    draw_frustum_base=True,
    diag_mode="none",  # 'none' | 'one' | 'both'
    frustum_size_gain=1.0,
    sphere_size_gain=1.0,
):
    import open3d as o3d
    
    def camera_pyramid_mesh(Tcw_inv, scale=0.15, color=(1.0, 0.2, 0.0)):
        """Create a quadrangular pyramid representing a camera frustum.
        Apex at optical center; base normal aligns with camera +Z (bearing).
        Tcw_inv is camera->world (4x4).
        """
        Rw = Tcw_inv[:3, :3]
        tw = Tcw_inv[:3, 3]
        d = float(scale)              # depth of base from the apex in camera coords
        half_w = 0.3 * scale          # half width of base (x)
        half_h = 0.1 * scale         # half height of base (y)
        # Camera coordinates: +Z forward, X right, Y down (OpenCV). We still form a pyramid.
        pts_cam = np.array([
            [0.0,     0.0,    0.0],              # 0 apex (optical center)
            [-half_w,-half_h, d],                # 1 base corners (quad)
            [ half_w,-half_h, d],                # 2
            [ half_w, half_h, d],                # 3
            [-half_w, half_h, d],                # 4
        ], dtype=float)
        pts_world = (Rw @ pts_cam.T).T + tw
        faces = np.array([
            [0,1,2], [0,2,3], [0,3,4], [0,4,1],  # sides
            [1,2,3], [1,3,4]                      # base (two triangles)
        ], dtype=np.int32)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(pts_world)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh

    def camera_pyramid_lines(
        Tcw_inv,
        depth=0.15,
        color_sides=(0.0, 0.0, 0.0),
        color_base=(0.0, 0.4, 1.0),
        K=None,
        img_wh=None,
        fill_base=False,
        add_sides=True,
        diag_mode="none",
    ):
        # Build vertices for a quadrangular pyramid that matches FOV at given depth.
        Rw = Tcw_inv[:3, :3]
        tw = Tcw_inv[:3, 3]
        d = float(depth)
        if K is not None and img_wh is not None:
            K = np.asarray(K, dtype=float)
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            W, H = int(img_wh[0]), int(img_wh[1])
            # Pixel corners (u,v) at image borders
            uv = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=float)
            # Back-project to 3D at depth d: X = (u-cx)/fx * d, Y = (v-cy)/fy * d, Z = d
            x = (uv[:, 0] - cx) / max(1e-12, fx) * d
            y = (uv[:, 1] - cy) / max(1e-12, fy) * d
            base = np.stack([x, y, np.full_like(x, d)], axis=1)
            pts_cam = np.vstack([
                np.array([[0.0, 0.0, 0.0]], dtype=float),  # apex
                base
            ])
        else:
            # Fallback: symmetric rectangle around optical axis
            half_w = 0.3 * d
            half_h = 0.1 * d
            pts_cam = np.array([
                [0.0,     0.0,    0.0],
                [-half_w,-half_h, d],
                [ half_w,-half_h, d],
                [ half_w, half_h, d],
                [-half_w, half_h, d],
            ], dtype=float)
        pts_world = (Rw @ pts_cam.T).T + tw
        # Separate LineSets so base can be drawn clearly on top
        # Base loop (draw later to be on top)
        base_lines = [[1,2],[2,3],[3,4],[4,1]]
        if isinstance(diag_mode, str):
            dm = diag_mode.lower()
        else:
            dm = "none"
        if dm == "one":
            base_lines += [[1,3]]
        elif dm == "both":
            base_lines += [[1,3],[2,4]]
        base_lines = np.asarray(base_lines, dtype=np.int32)
        ls_base = o3d.geometry.LineSet()
        ls_base.points = o3d.utility.Vector3dVector(pts_world)
        ls_base.lines  = o3d.utility.Vector2iVector(base_lines)
        ls_base.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color_base,float), (base_lines.shape[0],1)))

        # Sides: apex->corners (optional)
        geoms = []
        if add_sides:
            side_lines = np.array([[0,1],[0,2],[0,3],[0,4]], dtype=np.int32)
            ls_sides = o3d.geometry.LineSet()
            ls_sides.points = o3d.utility.Vector3dVector(pts_world)
            ls_sides.lines  = o3d.utility.Vector2iVector(side_lines)
            ls_sides.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color_sides,float), (side_lines.shape[0],1)))
            geoms.append(ls_sides)

        if fill_base:
            # Base fill with double-sided triangles to avoid back-face culling
            base_faces = np.array([[1,2,3],[1,3,4]], dtype=np.int32)
            base_faces_rev = np.array([[3,2,1],[4,3,1]], dtype=np.int32)
            base_mesh = o3d.geometry.TriangleMesh()
            base_mesh.vertices  = o3d.utility.Vector3dVector(pts_world)
            base_mesh.triangles = o3d.utility.Vector3iVector(np.vstack([base_faces, base_faces_rev]))
            base_mesh.compute_vertex_normals()
            base_mesh.paint_uniform_color([0.80, 0.90, 1.0])  # very light blue fill
            geoms.append(base_mesh)

        # Add base outline last for crisp edges
        geoms.append(ls_base)

        return geoms
    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB']) / 255.0
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    geoms = [pcd]
    # Precompute centers and auto scale
    centers = [T[:3, 3] for T in Camera2World_Transform_Matrixs]
    centers_np_full = np.asarray(centers, dtype=float) if centers else np.empty((0,3))
    if isinstance(cam_scale, str) and cam_scale.lower() == "auto" and len(centers_np_full) >= 2:
        diffs = np.linalg.norm(np.diff(centers_np_full, axis=0), axis=1)
        diffs = diffs[diffs > 1e-9]
        med_step = float(np.median(diffs)) if diffs.size else 1.0
        # Make depth small relative to path spacing to avoid overlap
        scale_val = max(0.02, 0.25 * med_step)
    else:
        # If a fixed number is given, respect it; if a string (legacy), fallback to 0.25
        scale_val = float(cam_scale) if not isinstance(cam_scale, str) else 0.25
    # Allow user to enlarge/shrink all frustums at once
    try:
        scale_val = float(scale_val) * float(frustum_size_gain)
    except Exception:
        pass
    # Draw camera coordinate frames and centers
    for i, T in enumerate(Camera2World_Transform_Matrixs):
        Rw = T[:3, :3]
        tw = T[:3, 3]
        # Optional axis frame (disabled by default)
        if draw_axes:
            cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
            cam_axis.rotate(Rw, center=(0, 0, 0))
            cam_axis.translate(tw)
            geoms.append(cam_axis)
        # Camera frustum: wireframe pyramid like reference figure
        if frustum_stride and (i % int(max(1, frustum_stride)) == 0):
            geoms.extend(
                camera_pyramid_lines(
                    T,
                    depth=scale_val,
                    color_sides=(0.3, 0.3, 0.3),
                    color_base=(1.0, 0.0, 0.0),
                    K=intrinsics,
                    img_wh=img_wh,
                    fill_base=False,
                    add_sides=bool(draw_frustum_sides),
                    diag_mode=diag_mode,
                )
            )
        # Optional sphere to mark the apex (optical center) in red
        if draw_center_spheres:
            # Size sphere relative to frustum depth; adjustable with sphere_size_gain
            base = max(0.002, 0.08 * float(scale_val))
            sph_r = base * float(sphere_size_gain)
            s = o3d.geometry.TriangleMesh.create_sphere(radius=sph_r)
            s.paint_uniform_color([1.0, 0.0, 0.0])
            s.translate(tw)
            geoms.append(s)
    # Add a trajectory that follows a smooth order (PCA angle sort) and draw dashed green + thin black line
    if len(centers_np_full) >= 2:
        C = centers_np_full
        cmean = C.mean(axis=0)
        X = C - cmean
        # PCA to get a stable 2D plane ordering
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        axes = Vt[:2]  # two principal directions
        uv = X @ axes.T
        ang = np.arctan2(uv[:, 1], uv[:, 0])
        order = np.argsort(ang)
        C_ord = C[order]
        n = C_ord.shape[0]
        # black polyline
        lines = [[i, i + 1] for i in range(n - 1)]
        if close_loop and n > 2:
            lines.append([n - 1, 0])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(C_ord)
        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, 0.0]]), (len(lines), 1)))
        geoms.append(ls)
        # dashed green points along trajectory
        dash_idx = np.arange(0, n, max(1, n // 100 + 1))  # adaptive density
        pts_dash = C_ord[dash_idx]
        pcd_dash = o3d.geometry.PointCloud()
        pcd_dash.points = o3d.utility.Vector3dVector(pts_dash)
        pcd_dash.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.8, 0.0]]), (pts_dash.shape[0], 1)))
        geoms.append(pcd_dash)

    # Optionally draw a fitted circle on the XY plane to illustrate a full loop
    if draw_circle and len(centers) >= 3:
        cnp = np.asarray(centers, dtype=float)
        x, y = cnp[:, 0], cnp[:, 1]
        A = np.c_[2 * x, 2 * y, np.ones_like(x)]
        b = (x ** 2 + y ** 2)
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            xc, yc, c = sol
            r = float(max(1e-6, np.sqrt(max(1e-8, c + xc * xc + yc * yc))))
            z0 = float(np.median(cnp[:, 2]))
            thetas = np.linspace(0.0, 2 * np.pi, 181)
            circ_pts = np.stack([xc + r * np.cos(thetas),
                                 yc + r * np.sin(thetas),
                                 np.full_like(thetas, z0)], axis=1)
            circ_lines = np.array([[i, i + 1] for i in range(circ_pts.shape[0] - 1)] + [[circ_pts.shape[0] - 1, 0]], dtype=np.int32)
            cls = o3d.geometry.LineSet()
            cls.points = o3d.utility.Vector3dVector(circ_pts)
            cls.lines = o3d.utility.Vector2iVector(circ_lines)
            cls.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.6, 1.0]]), (circ_lines.shape[0], 1)))
            geoms.append(cls)

            # Optionally place small camera frames evenly along the circle
            if draw_circle_cameras and n_circle_cams > 1:
                center3 = np.array([xc, yc, z0], dtype=float)
                # helper to create orientation that looks at center
                def look_at_rotation(forward, up=np.array([0.0, 0.0, 1.0])):
                    f = np.asarray(forward, dtype=float)
                    f = f / (np.linalg.norm(f) + 1e-12)
                    u = up
                    # In case forward || up, pick another up
                    if abs(np.dot(f, u)) > 0.99:
                        u = np.array([0.0, 1.0, 0.0])
                    r = np.cross(u, f)
                    r = r / (np.linalg.norm(r) + 1e-12)
                    u2 = np.cross(f, r)
                    Rm = np.stack([r, u2, f], axis=1)
                    return Rm

                for t in np.linspace(0.0, 2 * np.pi, int(n_circle_cams), endpoint=False):
                    pos = np.array([xc + r * np.cos(t), yc + r * np.sin(t), z0], dtype=float)
                    fwd = center3 - pos
                    Rm = look_at_rotation(fwd)
                    cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
                    cam.rotate(Rm, center=(0, 0, 0))
                    cam.translate(pos)
                    geoms.append(cam)
        except Exception:
            pass
    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    # Load data
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    images_df = pd.read_pickle(DATA_DIR / "images.pkl")
    train_df = pd.read_pickle(DATA_DIR / "train.pkl")
    points3D_df = pd.read_pickle(DATA_DIR / "points3D.pkl")
    point_desc_df = pd.read_pickle(DATA_DIR / "point_desc.pkl")

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)
    # Normalize once for stable matching
    norms = np.linalg.norm(desc_model, axis=1, keepdims=True) + 1e-8
    desc_model = desc_model / norms


    # Only use validation images (filename contains 'valid') and ensure descriptors exist
    valid_mask = images_df["NAME"].astype(str).str.contains("valid", case=False, regex=False)
    valid_ids = set(images_df.loc[valid_mask, "IMAGE_ID"].tolist())
    desc_ids = set(point_desc_df["IMAGE_ID"].unique().tolist())
    IMAGE_ID_LIST = sorted(valid_ids.intersection(desc_ids))
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []
    img_wh = None

    # Camera intrinsics, matching the (now removed) pnpsolver
    cameraMatrix = np.array([[1868.27, 0, 540],
                             [0, 1869.18, 960],
                             [0, 0, 1]], dtype=np.float64)

    # FLANN matcher for descriptor matching
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=16))

    for idx in tqdm(IMAGE_ID_LIST):
        # Load query image
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        rimg = cv2.imread(str(DATA_DIR / "frames" / fname), cv2.IMREAD_GRAYSCALE)
        if rimg is not None and img_wh is None:
            h, w = rimg.shape[:2]
            img_wh = (w, h)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)
        
        # Normalize query descriptors
        desc_query_norm = np.linalg.norm(desc_query, axis=1, keepdims=True) + 1e-8
        desc_query = desc_query / desc_query_norm

        # Find correspondences using FLANN
        try:
            matches = flann.knnMatch(desc_query, desc_model, k=2)
        except cv2.error:
            continue

        good = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
        
        if len(good) < 4: # Need at least 4 points for PnP, but RANSAC needs more to be robust
            continue

        # Prepare points for our custom ransac_p3p
        # p_w: 3xN, p_i: 2xN
        obj_pts = np.array([kp_model[m.trainIdx] for m in good], dtype=np.float64).T
        img_pts = np.array([kp_query[m.queryIdx] for m in good], dtype=np.float64).T

        # Call our own RANSAC P3P implementation
        R_est, t_est, inliers = ransac_p3p(
            p_w=obj_pts,
            p_i=img_pts,
            K=cameraMatrix,
            threshold=4.0,  # Reprojection error threshold in pixels
            iterations=1000
        )

        if R_est is None or t_est is None:
            continue

        # Convert rotation matrix to rotation vector for error calculation
        rvec = R.from_matrix(R_est).as_rotvec()
        tvec = t_est

        r_list.append(rvec)
        t_list.append(tvec)

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        # Calculate error: compare estimate vs GT
        rotq_est = R.from_rotvec(rvec.reshape(3)).as_quat().reshape(1,4)
        tvec_est = tvec.reshape(1,3)
        r_error = rotation_error(rotq_est, rotq_gt)
        t_error = translation_error(tvec_est, tvec_gt)
        rotation_error_list.append(r_error)
        translation_error_list.append(t_error)

    if rotation_error_list and translation_error_list:
        print("\n=== Pose Error (median over images) [CUSTOM P3P+RANSAC] ===")
        print(f"Rotation (deg):   {np.median(rotation_error_list):.10f}")
        print(f"Translation (L2): {np.median(translation_error_list):.10f}")
    else:
        print("[WARN] No valid pose estimated by custom P3P+RANSAC.")

    Camera2World_Transform_Matrixs = []
    for r, t in zip(r_list, t_list):
        # r: (3,) rotvec ; t: (3,1)
        R_cw = R.from_rotvec(r.reshape(3)).as_matrix()    # world -> camera 的 R
        t_cw = t.reshape(3)                                # world -> camera 的 t
        # camera center in world: C = -R^T t
        C_w  = -R_cw.T @ t_cw
        # Twc = [ R^T | C ; 0 0 0 1 ]  (camera->world)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3,:3] = R_cw.T
        c2w[:3, 3] = C_w
        Camera2World_Transform_Matrixs.append(c2w)

    print(f"Estimated poses: {len(Camera2World_Transform_Matrixs)} (from {len(IMAGE_ID_LIST)} images)")
    # Intrinsics used in PnP (match pnpsolver)
    K_vis = np.array([[1868.27, 0.0, 540.0],
                      [0.0, 1869.18, 960.0],
                      [0.0,    0.0,   1.0]], dtype=np.float64)
    visualization(
        Camera2World_Transform_Matrixs,
        points3D_df,
        close_loop=False,
        draw_circle=False,
        draw_circle_cameras=False,
        draw_axes=False,
        cam_scale="auto",
        frustum_stride=1,
        intrinsics=K_vis,
        img_wh=img_wh,
        draw_frustum_sides=True,
        draw_frustum_base=True,
        diag_mode="none",
        frustum_size_gain=1.8,
        sphere_size_gain=1.8,
    )
