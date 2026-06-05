import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from trimesh.transformations import quaternion_from_matrix


def load_sdf_metadata(path):
    data = np.load(Path(path).resolve(), allow_pickle=True)
    required = ("sdf", "x", "y", "z")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"SDF npz missing required keys: {missing}")

    return {
        "sdf": np.asarray(data["sdf"], dtype=np.float64),
        "axes": (
            np.asarray(data["x"], dtype=np.float64),
            np.asarray(data["y"], dtype=np.float64),
            np.asarray(data["z"], dtype=np.float64),
        ),
    }


def build_sdf_interpolator(sdf_meta):
    return RegularGridInterpolator(
        sdf_meta["axes"],
        sdf_meta["sdf"],
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def inside_bounds(points, axes):
    x, y, z = axes
    lower = np.array([x[0], y[0], z[0]], dtype=np.float64)
    upper = np.array([x[-1], y[-1], z[-1]], dtype=np.float64)
    return np.all((points >= lower) & (points <= upper), axis=1)


def farthest_point_sample_numpy(points, sample_count, seed=0):
    if sample_count >= len(points):
        return np.arange(len(points), dtype=np.int64)

    rng = np.random.default_rng(seed)
    selected = np.empty(sample_count, dtype=np.int64)
    selected[0] = int(rng.integers(0, len(points)))
    distances = np.full(len(points), np.inf, dtype=np.float64)

    for i in range(1, sample_count):
        last = points[selected[i - 1]]
        current = np.sum((points - last) ** 2, axis=1)
        distances = np.minimum(distances, current)
        selected[i] = int(np.argmax(distances))
    return selected


def load_point_cloud(path):
    path = Path(path).resolve()
    suffix = path.suffix.lower()

    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        point_key = None
        # Prefer world-frame ROI outputs when both local-frame and world-frame
        # point clouds are stored in the same npz.
        for key in ("point_cloud", "cropped_world_points", "cropped_points_world_m", "raw_mesh_points_world_m"):
            if key in data:
                point_key = key
                break
        if point_key is None:
            for key in ("points", "point_cloud", "xyz", "vertices", "cropped_local_points", "cropped_points_start_tcp_m"):
                if key in data:
                    point_key = key
                    break
        if point_key is None:
            raise ValueError(f"No point array found in {path}")
        points = np.asarray(data[point_key], dtype=np.float64)
        normals = np.asarray(data["normals"], dtype=np.float64) if "normals" in data else None
        metadata = {
            "point_key": point_key,
            "start_xyz_world_m": np.asarray(data["start_xyz_world_m"], dtype=np.float64).reshape(3)
            if "start_xyz_world_m" in data
            else None,
            "goal_xyz_world_m": np.asarray(data["goal_xyz_world_m"], dtype=np.float64).reshape(3)
            if "goal_xyz_world_m" in data
            else None,
            "radius_m": float(np.asarray(data["radius_m"], dtype=np.float64).reshape(-1)[0]) if "radius_m" in data else None,
            "height_m": float(np.asarray(data["height_m"], dtype=np.float64).reshape(-1)[0]) if "height_m" in data else None,
            "z_min": float(np.min(np.asarray(data["cropped_points_world_m"], dtype=np.float64)[:, 2]))
            if "cropped_points_world_m" in data
            else None,
        }
    elif suffix == ".npy":
        points = np.asarray(np.load(path), dtype=np.float64)
        normals = None
        metadata = {"point_key": "npy", "start_xyz_world_m": None, "goal_xyz_world_m": None, "radius_m": None, "height_m": None, "z_min": None}
    else:
        loaded = trimesh.load(path, process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        if isinstance(loaded, trimesh.points.PointCloud):
            points = np.asarray(loaded.vertices, dtype=np.float64)
            normals = np.asarray(loaded.metadata.get("vertex_normals"), dtype=np.float64) if loaded.metadata.get("vertex_normals") is not None else None
        elif isinstance(loaded, trimesh.Trimesh):
            points = np.asarray(loaded.vertices, dtype=np.float64)
            normals = np.asarray(loaded.vertex_normals, dtype=np.float64) if len(loaded.vertex_normals) == len(points) else None
        else:
            raise ValueError(f"Unsupported point cloud format: {path}")
        metadata = {"point_key": suffix, "start_xyz_world_m": None, "goal_xyz_world_m": None, "radius_m": None, "height_m": None, "z_min": None}

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point cloud must have shape (N, 3)")
    if normals is not None and (normals.ndim != 2 or normals.shape != points.shape):
        normals = None
    return points, normals, metadata


def estimate_point_normals(points, k):
    if len(points) < 3:
        raise ValueError("At least 3 points are required to estimate normals.")

    k = max(3, min(int(k), len(points)))
    tree = cKDTree(points)
    _, nn_idx = tree.query(points, k=k)
    centroid = points.mean(axis=0)
    normals = np.zeros_like(points, dtype=np.float64)

    for idx, neighbors in enumerate(nn_idx):
        local = points[neighbors] - points[idx]
        cov = local.T @ local
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, np.argmin(eigvals)]
        if np.dot(normal, points[idx] - centroid) < 0:
            normal = -normal
        normals[idx] = normal

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(norms, 1e-12, None)


def choose_outward_normals(points, normals, interpolator, probe_eps):
    plus_probe = points + normals * probe_eps
    minus_probe = points - normals * probe_eps
    plus_sdf = interpolator(plus_probe)
    minus_sdf = interpolator(minus_probe)

    outward = normals.copy()
    finite_plus = np.isfinite(plus_sdf)
    finite_minus = np.isfinite(minus_sdf)
    flip = np.zeros(len(points), dtype=bool)
    flip[finite_plus & finite_minus] = plus_sdf[finite_plus & finite_minus] < minus_sdf[finite_plus & finite_minus]
    flip[finite_plus & ~finite_minus] = False
    flip[~finite_plus & finite_minus] = True
    outward[flip] *= -1.0
    return outward


def sample_surface_anchors(points, normals, count, seed):
    if len(points) < count:
        raise ValueError(f"Point cloud only has {len(points)} points, but {count} TCP anchors are required.")
    keep = farthest_point_sample_numpy(points, count, seed=seed)
    return points[keep], normals[keep]


def generate_tcp_points_from_point_cloud(
    points,
    normals,
    sdf_meta,
    num_points,
    near_range,
    probe_eps,
    seed,
):
    rng = np.random.default_rng(seed)
    interpolator = build_sdf_interpolator(sdf_meta)
    outward_normals = choose_outward_normals(points, normals, interpolator, probe_eps)

    accepted_tcp = []
    accepted_anchor = []
    accepted_normals = []
    accepted_distance = []

    batch_factor = max(6, int(np.ceil(num_points / max(len(points), 1))))
    while len(accepted_tcp) < num_points:
        candidate_count = min(len(points), max(num_points * batch_factor, num_points))
        candidate_idx = farthest_point_sample_numpy(points, candidate_count, seed=int(rng.integers(0, 1_000_000)))
        anchor_points = points[candidate_idx]
        anchor_normals = outward_normals[candidate_idx]
        distances = rng.uniform(near_range[0], near_range[1], size=len(anchor_points))
        tcp_points = anchor_points + anchor_normals * distances[:, None]

        in_bounds = inside_bounds(tcp_points, sdf_meta["axes"])
        tcp_points = tcp_points[in_bounds]
        kept_anchor = anchor_points[in_bounds]
        kept_normals = anchor_normals[in_bounds]
        kept_distances = distances[in_bounds]
        if len(tcp_points) == 0:
            batch_factor += 2
            continue

        sdf_values = interpolator(tcp_points)
        valid = np.isfinite(sdf_values) & (sdf_values >= -1e-4)
        tcp_points = tcp_points[valid]
        kept_anchor = kept_anchor[valid]
        kept_normals = kept_normals[valid]
        kept_distances = kept_distances[valid]
        if len(tcp_points) == 0:
            batch_factor += 2
            continue

        need = num_points - len(accepted_tcp)
        tcp_points = tcp_points[:need]
        kept_anchor = kept_anchor[:need]
        kept_normals = kept_normals[:need]
        kept_distances = kept_distances[:need]

        accepted_tcp.extend(tcp_points.tolist())
        accepted_anchor.extend(kept_anchor.tolist())
        accepted_normals.extend(kept_normals.tolist())
        accepted_distance.extend(kept_distances.tolist())

    tcp_points = np.asarray(accepted_tcp, dtype=np.float32)
    anchor_points = np.asarray(accepted_anchor, dtype=np.float32)
    surface_normals = np.asarray(accepted_normals, dtype=np.float32)
    surface_distance = np.asarray(accepted_distance, dtype=np.float32)
    order = rng.permutation(len(tcp_points))

    return {
        "tcp_points": tcp_points[order],
        "anchor_points": anchor_points[order],
        "surface_normals": surface_normals[order],
        "surface_distance": surface_distance[order],
        "band": np.array(["near"] * len(tcp_points), dtype=object),
        "num_points": int(len(tcp_points)),
        "near_count": int(len(tcp_points)),
        "far_count": 0,
        "sampling_mode": "near_surface",
        "point_cloud_bounds": np.array([points.min(axis=0), points.max(axis=0)], dtype=np.float32),
    }


def point_to_segment_distance_2d(points_xy, start_xy, goal_xy):
    segment = goal_xy - start_xy
    segment_norm_sq = float(np.dot(segment, segment))
    if segment_norm_sq < 1e-12:
        return np.linalg.norm(points_xy - start_xy[None, :], axis=1)

    rel_points = points_xy - start_xy[None, :]
    t = np.sum(rel_points * segment[None, :], axis=1) / segment_norm_sq
    t = np.clip(t, 0.0, 1.0)
    projection = start_xy[None, :] + t[:, None] * segment[None, :]
    return np.linalg.norm(points_xy - projection, axis=1)


def point_inside_roi_capsule(point, start, goal, radius, z_min, height):
    point = np.asarray(point, dtype=np.float64).reshape(3)
    planar_dist = point_to_segment_distance_2d(point[:2].reshape(1, 2), start[:2], goal[:2])[0]
    z_max = z_min + float(height)
    return bool(planar_dist <= radius + 1e-9 and z_min - 1e-9 <= point[2] <= z_max + 1e-9)


def compute_capsule_boundary_distance(anchor_point, outward_normal, start, goal, radius, z_min, height, max_search):
    if not point_inside_roi_capsule(anchor_point, start, goal, radius, z_min, height):
        return 0.0

    lo = 0.0
    hi = max_search
    if point_inside_roi_capsule(anchor_point + outward_normal * hi, start, goal, radius, z_min, height):
        return hi

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        probe = anchor_point + outward_normal * mid
        if point_inside_roi_capsule(probe, start, goal, radius, z_min, height):
            lo = mid
        else:
            hi = mid
    return float(lo)


def generate_tcp_points_from_roi_capsule(
    points,
    normals,
    sdf_meta,
    roi_meta,
    num_points,
    far_min,
    probe_eps,
    seed,
):
    if roi_meta["start_xyz_world_m"] is None or roi_meta["goal_xyz_world_m"] is None:
        raise ValueError("ROI capsule far sampling requires start_xyz_world_m and goal_xyz_world_m in the point cloud npz.")
    if roi_meta["radius_m"] is None or roi_meta["height_m"] is None:
        raise ValueError("ROI capsule far sampling requires radius_m and height_m in the point cloud npz.")

    start = np.asarray(roi_meta["start_xyz_world_m"], dtype=np.float64)
    goal = np.asarray(roi_meta["goal_xyz_world_m"], dtype=np.float64)
    radius = float(roi_meta["radius_m"])
    height = float(roi_meta["height_m"])
    z_min = float(roi_meta["z_min"] if roi_meta["z_min"] is not None else np.min(points[:, 2]))
    max_search = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)) + radius + height)

    rng = np.random.default_rng(seed)
    interpolator = build_sdf_interpolator(sdf_meta)
    outward_normals = choose_outward_normals(points, normals, interpolator, probe_eps)

    accepted_tcp = []
    accepted_anchor = []
    accepted_normals = []
    accepted_distance = []
    accepted_boundary = []

    batch_factor = max(6, int(np.ceil(num_points / max(len(points), 1))))
    while len(accepted_tcp) < num_points:
        candidate_count = min(len(points), max(num_points * batch_factor, num_points))
        candidate_idx = farthest_point_sample_numpy(points, candidate_count, seed=int(rng.integers(0, 1_000_000)))
        anchor_points = points[candidate_idx]
        anchor_normals = outward_normals[candidate_idx]

        boundary_distances = np.array(
            [
                compute_capsule_boundary_distance(anchor_points[i], anchor_normals[i], start, goal, radius, z_min, height, max_search)
                for i in range(len(anchor_points))
            ],
            dtype=np.float64,
        )
        valid_anchor = boundary_distances > far_min
        anchor_points = anchor_points[valid_anchor]
        anchor_normals = anchor_normals[valid_anchor]
        boundary_distances = boundary_distances[valid_anchor]
        if len(anchor_points) == 0:
            batch_factor += 2
            continue

        distances = rng.uniform(far_min, boundary_distances)
        tcp_points = anchor_points + anchor_normals * distances[:, None]

        in_capsule = np.array(
            [point_inside_roi_capsule(tcp_points[i], start, goal, radius, z_min, height) for i in range(len(tcp_points))],
            dtype=bool,
        )
        tcp_points = tcp_points[in_capsule]
        kept_anchor = anchor_points[in_capsule]
        kept_normals = anchor_normals[in_capsule]
        kept_distances = distances[in_capsule]
        kept_boundary = boundary_distances[in_capsule]
        if len(tcp_points) == 0:
            batch_factor += 2
            continue

        in_bounds = inside_bounds(tcp_points, sdf_meta["axes"])
        tcp_points = tcp_points[in_bounds]
        kept_anchor = kept_anchor[in_bounds]
        kept_normals = kept_normals[in_bounds]
        kept_distances = kept_distances[in_bounds]
        kept_boundary = kept_boundary[in_bounds]
        if len(tcp_points) == 0:
            batch_factor += 2
            continue

        sdf_values = interpolator(tcp_points)
        valid = np.isfinite(sdf_values) & (sdf_values >= -1e-4)
        tcp_points = tcp_points[valid]
        kept_anchor = kept_anchor[valid]
        kept_normals = kept_normals[valid]
        kept_distances = kept_distances[valid]
        kept_boundary = kept_boundary[valid]
        if len(tcp_points) == 0:
            batch_factor += 2
            continue

        need = num_points - len(accepted_tcp)
        tcp_points = tcp_points[:need]
        kept_anchor = kept_anchor[:need]
        kept_normals = kept_normals[:need]
        kept_distances = kept_distances[:need]
        kept_boundary = kept_boundary[:need]

        accepted_tcp.extend(tcp_points.tolist())
        accepted_anchor.extend(kept_anchor.tolist())
        accepted_normals.extend(kept_normals.tolist())
        accepted_distance.extend(kept_distances.tolist())
        accepted_boundary.extend(kept_boundary.tolist())

    tcp_points = np.asarray(accepted_tcp, dtype=np.float32)
    anchor_points = np.asarray(accepted_anchor, dtype=np.float32)
    surface_normals = np.asarray(accepted_normals, dtype=np.float32)
    surface_distance = np.asarray(accepted_distance, dtype=np.float32)
    boundary_distance = np.asarray(accepted_boundary, dtype=np.float32)
    order = rng.permutation(len(tcp_points))

    return {
        "tcp_points": tcp_points[order],
        "anchor_points": anchor_points[order],
        "surface_normals": surface_normals[order],
        "surface_distance": surface_distance[order],
        "capsule_boundary_distance": boundary_distance[order],
        "band": np.array(["roi_capsule_far"] * len(tcp_points), dtype=object),
        "num_points": int(len(tcp_points)),
        "near_count": 0,
        "far_count": int(len(tcp_points)),
        "sampling_mode": "roi_capsule_far",
        "capsule_radius_m": np.array(radius, dtype=np.float32),
        "capsule_height_m": np.array(height, dtype=np.float32),
        "point_cloud_bounds": np.array([points.min(axis=0), points.max(axis=0)], dtype=np.float32),
    }


def generate_tcp_orientations(
    num_points,
    base_tilt_deg=(40.0, 45.0, 50.0),
    base_yaw_deg=(0.0, 90.0, 180.0, 270.0),
    tilt_jitter_range_deg=(-5.0, 5.0),
    yaw_jitter_range_deg=(-20.0, 20.0),
    seed=0,
):
    rng = np.random.default_rng(seed)
    base_tilt_deg = np.asarray(base_tilt_deg, dtype=np.float64)
    base_yaw_deg = np.asarray(base_yaw_deg, dtype=np.float64)
    if base_tilt_deg.ndim != 1 or base_yaw_deg.ndim != 1:
        raise ValueError("base_tilt_deg and base_yaw_deg must be 1D sequences.")

    base_tilt_grid = np.repeat(base_tilt_deg, len(base_yaw_deg))
    base_yaw_grid = np.tile(base_yaw_deg, len(base_tilt_deg))
    num_orientations = len(base_tilt_grid)

    tilt_jitter_deg = rng.uniform(tilt_jitter_range_deg[0], tilt_jitter_range_deg[1], size=(num_points, num_orientations))
    yaw_jitter_deg = rng.uniform(yaw_jitter_range_deg[0], yaw_jitter_range_deg[1], size=(num_points, num_orientations))

    tilt_deg = base_tilt_grid[None, :] + tilt_jitter_deg
    yaw_deg = base_yaw_grid[None, :] + yaw_jitter_deg
    tilt_rad = np.deg2rad(tilt_deg)
    yaw_rad = np.deg2rad(yaw_deg)

    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    cos_tilt = np.cos(tilt_rad)
    sin_tilt = np.sin(tilt_rad)
    z_axis = np.stack([cos_yaw * cos_tilt, sin_yaw * cos_tilt, sin_tilt], axis=-1)
    y_axis = np.stack([-sin_yaw, cos_yaw, np.zeros_like(yaw_rad)], axis=-1)
    x_axis = np.cross(y_axis, z_axis, axis=-1)

    x_axis /= np.clip(np.linalg.norm(x_axis, axis=-1, keepdims=True), 1e-12, None)
    y_axis = np.cross(z_axis, x_axis, axis=-1)
    y_axis /= np.clip(np.linalg.norm(y_axis, axis=-1, keepdims=True), 1e-12, None)
    z_axis /= np.clip(np.linalg.norm(z_axis, axis=-1, keepdims=True), 1e-12, None)

    rot_mats = np.stack([x_axis, y_axis, z_axis], axis=-1).astype(np.float32)
    quats = np.empty((num_points, num_orientations, 4), dtype=np.float32)
    for i in range(num_points):
        for j in range(num_orientations):
            tf = np.eye(4, dtype=np.float64)
            tf[:3, :3] = rot_mats[i, j]
            quats[i, j] = quaternion_from_matrix(tf).astype(np.float32)

    return {
        "orientation_matrices": rot_mats,
        "orientation_quaternions_xyzw": quats,
        "orientation_base_tilt_deg": np.broadcast_to(base_tilt_grid[None, :], (num_points, num_orientations)).astype(np.float32),
        "orientation_base_yaw_deg": np.broadcast_to(base_yaw_grid[None, :], (num_points, num_orientations)).astype(np.float32),
        "orientation_tilt_jitter_deg": tilt_jitter_deg.astype(np.float32),
        "orientation_yaw_jitter_deg": yaw_jitter_deg.astype(np.float32),
        "orientation_tilt_deg": tilt_deg.astype(np.float32),
        "orientation_yaw_deg": yaw_deg.astype(np.float32),
    }


def save_npz(output_path, sample_dict, sdf_meta):
    save_kwargs = {
        "tcp_points": sample_dict["tcp_points"],
        "anchor_points": sample_dict["anchor_points"],
        "surface_normals": sample_dict["surface_normals"],
        "surface_distance": sample_dict["surface_distance"],
        "band": sample_dict["band"],
        "num_points": np.array(sample_dict["num_points"], dtype=np.int64),
        "near_count": np.array(sample_dict["near_count"], dtype=np.int64),
        "far_count": np.array(sample_dict["far_count"], dtype=np.int64),
        "sampling_mode": np.array(sample_dict["sampling_mode"], dtype=object),
        "point_cloud_bounds": sample_dict["point_cloud_bounds"],
        "orientation_matrices": sample_dict["orientation_matrices"],
        "orientation_quaternions_xyzw": sample_dict["orientation_quaternions_xyzw"],
        "orientation_base_tilt_deg": sample_dict["orientation_base_tilt_deg"],
        "orientation_base_yaw_deg": sample_dict["orientation_base_yaw_deg"],
        "orientation_tilt_jitter_deg": sample_dict["orientation_tilt_jitter_deg"],
        "orientation_yaw_jitter_deg": sample_dict["orientation_yaw_jitter_deg"],
        "orientation_tilt_deg": sample_dict["orientation_tilt_deg"],
        "orientation_yaw_deg": sample_dict["orientation_yaw_deg"],
        "sdf_x": np.asarray(sdf_meta["axes"][0], dtype=np.float32),
        "sdf_y": np.asarray(sdf_meta["axes"][1], dtype=np.float32),
        "sdf_z": np.asarray(sdf_meta["axes"][2], dtype=np.float32),
    }
    if "capsule_boundary_distance" in sample_dict:
        save_kwargs["capsule_boundary_distance"] = sample_dict["capsule_boundary_distance"]
    if "capsule_radius_m" in sample_dict:
        save_kwargs["capsule_radius_m"] = np.array(sample_dict["capsule_radius_m"], dtype=np.float32)
    if "capsule_height_m" in sample_dict:
        save_kwargs["capsule_height_m"] = np.array(sample_dict["capsule_height_m"], dtype=np.float32)
    np.savez_compressed(output_path, **save_kwargs)


def main():
    default_near_points = 40
    default_far_points = 20
    parser = argparse.ArgumentParser("sample_tcp_points_from_workpiece")
    parser.add_argument("--point-cloud", type=str, required=True, help="Point cloud path (.npz/.npy/.ply)")
    parser.add_argument("--sdf", type=str, required=True, help="Matching workpiece SDF npz path")
    parser.add_argument("--output", type=str, required=True, help="Output tcp sample npz path")
    parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help="Total number of TCP points per point cloud; defaults to 40 for near_surface and 20 for roi_capsule_far",
    )
    parser.add_argument("--num-orientations", type=int, default=12, help="Number of discrete TCP orientations per point; must be 12")
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="near_surface",
        choices=["near_surface", "roi_capsule_far"],
        help="TCP sampling mode",
    )
    parser.add_argument("--near-min", type=float, default=0.0, help="Near band min distance in meters")
    parser.add_argument("--near-max", type=float, default=0.02, help="Near band max distance in meters")
    parser.add_argument("--far-min", type=float, default=0.02, help="Far-mode min distance in meters")
    parser.add_argument("--normal-k", type=int, default=30, help="Neighborhood size for PCA normal estimation when the point cloud has no normals")
    parser.add_argument("--probe-eps", type=float, default=0.003, help="Small probe distance used to decide which normal direction is outward")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    if not (0.0 <= args.near_min <= args.near_max):
        raise ValueError("Near range is invalid.")
    if args.far_min <= 0:
        raise ValueError("--far-min must be positive.")
    if args.num_orientations != 12:
        raise ValueError("--num-orientations must be 12 for the fixed 3x4 orientation grid.")
    if args.num_points is None:
        args.num_points = default_near_points if args.sampling_mode == "near_surface" else default_far_points

    sdf_meta = load_sdf_metadata(args.sdf)
    points, normals, point_cloud_meta = load_point_cloud(args.point_cloud)
    if normals is None:
        normals = estimate_point_normals(points, args.normal_k)

    if args.sampling_mode == "near_surface":
        samples = generate_tcp_points_from_point_cloud(
            points=points,
            normals=normals,
            sdf_meta=sdf_meta,
            num_points=args.num_points,
            near_range=(args.near_min, args.near_max),
            probe_eps=args.probe_eps,
            seed=args.seed,
        )
    else:
        samples = generate_tcp_points_from_roi_capsule(
            points=points,
            normals=normals,
            sdf_meta=sdf_meta,
            roi_meta=point_cloud_meta,
            num_points=args.num_points,
            far_min=args.far_min,
            probe_eps=args.probe_eps,
            seed=args.seed,
        )
    samples.update(
        generate_tcp_orientations(
            num_points=samples["num_points"],
            seed=args.seed + 1,
        )
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_path, samples, sdf_meta)

    print(f"Saved TCP samples to: {output_path}")
    print(f"Sampling mode: {samples['sampling_mode']}")
    print(f"Total points: {samples['num_points']}")
    if samples["sampling_mode"] == "near_surface":
        print(f"Near-band points [0-2cm]: {samples['near_count']}")
    else:
        print(f"Far-band points [2cm-capsule boundary]: {samples['far_count']}")
        print(
            "Capsule boundary distance range (m): "
            f"[{float(np.min(samples['capsule_boundary_distance'])):.6f}, {float(np.max(samples['capsule_boundary_distance'])):.6f}]"
        )
    print(f"Orientations per point: {args.num_orientations}")
    print("Base tilt set (deg): [40.0, 45.0, 50.0]")
    print("Base yaw set (deg): [0.0, 90.0, 180.0, 270.0]")
    print(f"Point cloud bounds: {samples['point_cloud_bounds'].tolist()}")


if __name__ == "__main__":
    main()
