from scipy.spatial.transform import Rotation as R
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

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model
    cameraMatrix = np.array([[1868.27, 0, 540],
                             [0, 1869.18, 960],
                             [0, 0, 1]], dtype=np.float64)
    distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352], dtype=np.float64)
    # Descriptor matching with FLANN (KD-Tree) and ratio test
    if desc_query is None or len(desc_query) == 0 or desc_model is None or len(desc_model) == 0:
        return False, None, None, None
    
    # L2 normalize descriptors (helps after averaging and for robust matching)
    desc_query = np.asarray(desc_query, dtype=np.float32)
    n = np.linalg.norm(desc_query, axis=1, keepdims=True) + 1e-8
    desc_query = desc_query / n
    desc_model = np.asarray(desc_model, dtype=np.float32)
    # FLANN match + ratio test (fewer checks for speed)
    # KDTree for float descriptors: algorithm=1
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=16))
    try:
        matches = flann.knnMatch(desc_query, desc_model, k=2)
    except cv2.error:
        return False, None, None, None
    
    good = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < 0.70 * n.distance:
            good.append(m)
    # Keep only top-N matches to limit RANSAC per-iteration cost
    if good:
        good.sort(key=lambda m: m.distance)
        good = good[:500]
    if len(good) < 4:
        return False, None, None, None
    # Use float32 to reduce memory and improve speed
    img_pts = np.array([kp_query[m.queryIdx] for m in good], dtype=np.float32).reshape(-1,1,2)
    obj_pts = np.array([kp_model[m.trainIdx] for m in good], dtype=np.float32).reshape(-1,1,3)
    # RANSAC PnP
    # Prefer AP3P (fast minimal solver) if available; fallback to EPNP
    pnp_flag = getattr(cv2, "SOLVEPNP_AP3P", cv2.SOLVEPNP_EPNP)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs,
        iterationsCount=500,
        reprojectionError=4.0,
        confidence=0.99,
        flags=pnp_flag
    )
    if not success or inliers is None or len(inliers) < 4:
        return False, None, None, None
    # Flatten inliers and refine
    inliers = inliers.ravel()
    obj_in = obj_pts[inliers].reshape(-1,1,3)
    img_in = img_pts[inliers].reshape(-1,1,2)
    success2, rvec2, tvec2 = cv2.solvePnP(
        objectPoints=obj_in,
        imagePoints=img_in,
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs,
        rvec=rvec, tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success2:
        return False, None, None, None
    # Optional Levenberg-Marquardt refine if available (OpenCV >= 4.1)
    refine_fn = getattr(cv2, "solvePnPRefineLM", None)
    if refine_fn is not None:
        rvec2, tvec2 = refine_fn(objectPoints=obj_in,
                                 imagePoints=img_in,
                                 cameraMatrix=cameraMatrix,
                                 distCoeffs=distCoeffs,
                                 rvec=rvec2,
                                 tvec=tvec2)
    return True, rvec2, tvec2, inliers

def build_cube_points(grid=13):
    """Create a unit cube (local frame) as a colored point set on its 6 faces.
    Returns (P_local Nx3 float64, C_bgr Nx3 uint8).
    """
    g = max(2, int(grid))
    lin = np.linspace(-0.5, 0.5, g)
    uu, vv = np.meshgrid(lin, lin)
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1)

    pts = []
    cols = []
    face_cols = [
        (255, 0, 0),    # +X blue
        (0, 255, 0),    # -X green
        (0, 0, 255),    # +Y red
        (0, 255, 255),  # -Y yellow
        (255, 0, 255),  # +Z magenta
        (255, 255, 0),  # -Z cyan
    ]
    pts.append(np.stack([np.full(uv.shape[0], 0.5), uv[:, 0], uv[:, 1]], axis=1)); cols.append(np.tile(face_cols[0], (uv.shape[0], 1)))
    pts.append(np.stack([np.full(uv.shape[0],-0.5), uv[:, 0], uv[:, 1]], axis=1)); cols.append(np.tile(face_cols[1], (uv.shape[0], 1)))
    pts.append(np.stack([uv[:, 0], np.full(uv.shape[0], 0.5), uv[:, 1]], axis=1)); cols.append(np.tile(face_cols[2], (uv.shape[0], 1)))
    pts.append(np.stack([uv[:, 0], np.full(uv.shape[0],-0.5), uv[:, 1]], axis=1)); cols.append(np.tile(face_cols[3], (uv.shape[0], 1)))
    pts.append(np.stack([uv[:, 0], uv[:, 1], np.full(uv.shape[0], 0.5)], axis=1)); cols.append(np.tile(face_cols[4], (uv.shape[0], 1)))
    pts.append(np.stack([uv[:, 0], uv[:, 1], np.full(uv.shape[0],-0.5)], axis=1)); cols.append(np.tile(face_cols[5], (uv.shape[0], 1)))

    P = np.concatenate(pts, axis=0).astype(np.float64)
    C = np.concatenate(cols, axis=0).astype(np.uint8)
    return P, C

def apply_object_transform(P_local, T_obj_world):
    """Apply a 3x4 transform [S*R|t] to local cube points to get world points."""
    return (T_obj_world @ np.c_[P_local, np.ones((P_local.shape[0], 1))].T).T

def project_world(Pw, rvec, tvec, K):
    """Project world points using world->camera pose and intrinsics K. Returns (uv Nx2, z Nx)."""
    R_cw = R.from_rotvec(rvec.reshape(3)).as_matrix()
    Pc = (R_cw @ Pw.T).T + tvec.reshape(1, 3)
    z = Pc[:, 2]
    xy = Pc[:, :2] / np.clip(z[:, None], 1e-12, None)
    uv = np.empty_like(xy)
    uv[:, 0] = K[0,0] * xy[:, 0] + K[0,2]
    uv[:, 1] = K[1,1] * xy[:, 1] + K[1,2]
    return uv, z

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
if False and __name__ == "__main__":
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


    # Use all available query/validation images from point_desc_df
    IMAGE_ID_LIST = sorted(point_desc_df["IMAGE_ID"].unique().tolist())
    r_list = []
    t_list = []
    rotation_error_list = []
    translation_error_list = []
    img_wh = None
    for idx in tqdm(IMAGE_ID_LIST):
        # Load quaery image
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        rimg = cv2.imread(str(DATA_DIR / "frames" / fname), cv2.IMREAD_GRAYSCALE)
        if rimg is not None and img_wh is None:
            h, w = rimg.shape[:2]
            img_wh = (w, h)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondance and solve pnp
        retval, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))
        # rotq = R.from_rotvec(rvec.reshape(1,3)).as_quat() # Convert rotation vector to quaternion
        # tvec = tvec.reshape(1,3) # Reshape translation vector
        if not retval:
            continue

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
        print("\n=== Pose Error (median over images) ===")
        print(f"Rotation (deg):   {np.median(rotation_error_list):.10f}")
        print(f"Translation (L2): {np.median(translation_error_list):.10f}")
    else:
        print("[WARN] No valid pose estimated.")

    Camera2World_Transform_Matrixs = []
    for r, t in zip(r_list, t_list):
        # r: (3,1) rodrigues ; t: (3,1)
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

if __name__ == "__main__":
    # Data paths
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"

    # Load COLMAP-derived data
    images_df = pd.read_pickle(DATA_DIR / "images.pkl")
    train_df = pd.read_pickle(DATA_DIR / "train.pkl")
    points3D_df = pd.read_pickle(DATA_DIR / "points3D.pkl")
    point_desc_df = pd.read_pickle(DATA_DIR / "point_desc.pkl")

    # Prepare model descriptors (average per 3D point)
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)
    desc_model /= (np.linalg.norm(desc_model, axis=1, keepdims=True) + 1e-8)

    # Intrinsics (consistent with pnpsolver)
    K = np.array([[1868.27, 0.0, 540.0],
                  [0.0, 1869.18, 960.0],
                  [0.0,    0.0,   1.0]], dtype=np.float64)

    # Load cube transform saved from transform_cube.py; else identity
    T_path = BASE_DIR / "cube_transform_mat.npy"
    if T_path.exists():
        T_obj_world = np.load(T_path)
        if T_obj_world.shape != (3, 4):
            raise ValueError("cube_transform_mat.npy must be 3x4 matrix (scale*R | t)")
    else:
        T_obj_world = np.hstack([np.eye(3), np.zeros((3, 1))])

    # Build cube points once and transform to world
    P_local, C_bgr = build_cube_points(grid=13)
    Pw = apply_object_transform(P_local, T_obj_world)

    # Output setup
    OUT_DIR = BASE_DIR / "output" / "ar_frames"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = BASE_DIR / "output" / "ar_cube.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    # Iterate only over validation images (filter by filename pattern in images_df)
    valid_mask = images_df["NAME"].astype(str).str.contains("valid", case=False, regex=False)
    valid_ids = images_df.loc[valid_mask, "IMAGE_ID"].tolist()
    # Ensure descriptors exist for these images
    desc_ids = set(point_desc_df["IMAGE_ID"].unique().tolist())
    IMAGE_ID_LIST = sorted([i for i in valid_ids if i in desc_ids])
    for idx in tqdm(IMAGE_ID_LIST):
        # Load frame
        fname = (images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values[0]
        frame_path = DATA_DIR / "frames" / fname
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        if writer is None:
            writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (frame.shape[1], frame.shape[0]))

        # Query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Pose estimation via the same RANSAC method as 2d3dmathcing.py
        ok, rvec, tvec, inliers = pnpsolver((kp_query, desc_query), (kp_model, desc_model))

        if ok:
            uv, depth = project_world(Pw, rvec, tvec, K)
            # Painter: draw far -> near, only points in front of camera
            valid = depth > 1e-6
            order = np.argsort(depth)[::-1]  # far to near
            order = order[valid[order]]
            rad = max(1, int(0.004 * max(frame.shape[0], frame.shape[1])))
            for i in order:
                u, v = uv[i]
                color = tuple(int(c) for c in C_bgr[i])
                cv2.circle(frame, (int(round(u)), int(round(v))), rad, color, thickness=-1, lineType=cv2.LINE_AA)

        # Save frame and video
        cv2.imwrite(str(OUT_DIR / fname), frame)
        if writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()
    print("Saved AR frames to", OUT_DIR)
    print("Saved AR video to", video_path)
