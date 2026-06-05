import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


DEFAULT_POINT_CLOUD_SIZE = 1024
DEFAULT_MESH_SAMPLE_POINTS = 100000


@dataclass
class WorldROICropResult:
    point_cloud: np.ndarray
    raw_mesh_points_world_m: np.ndarray
    cropped_points_world_m: np.ndarray
    start_xyz_world_m: np.ndarray
    goal_xyz_world_m: np.ndarray


def load_transition_data(path):
    path = Path(path).resolve()
    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "array_file" not in payload:
            raise KeyError(f"Transition json does not contain array_file: {path}")
        npz_path = (path.parent / payload["array_file"]).resolve()
        return load_transition_data(npz_path)

    data = np.load(path, allow_pickle=True)
    required = ("start_xyz", "end_xyz")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Transition npz missing required keys {missing}: {path}")
    return {
        "path": path,
        "start_xyz": np.asarray(data["start_xyz"], dtype=np.float32).reshape(3),
        "end_xyz": np.asarray(data["end_xyz"], dtype=np.float32).reshape(3),
    }


def load_sdf_transform(job_dir):
    sdf_path = Path(job_dir).resolve() / "workpiece_sdf.npz"
    data = np.load(sdf_path, allow_pickle=True)
    required = ("workpiece_scale", "workpiece_offset", "workpiece_z_offset")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"SDF file missing transform keys {missing}: {sdf_path}")

    scale = float(np.asarray(data["workpiece_scale"], dtype=np.float64).reshape(-1)[0])
    offset = np.asarray(data["workpiece_offset"], dtype=np.float64).reshape(3)
    z_offset = float(np.asarray(data["workpiece_z_offset"], dtype=np.float64).reshape(-1)[0])
    return scale, offset, z_offset


def load_transformed_mesh_world(stl_path, job_dir):
    mesh = trimesh.load_mesh(Path(stl_path).resolve(), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh = mesh.copy()

    scale, offset, z_offset = load_sdf_transform(job_dir)
    mesh.apply_scale(scale)
    translation = offset.copy()
    translation[2] += z_offset
    mesh.apply_translation(translation)
    return mesh


def sample_mesh_surface(mesh, num_points, use_even=False, seed=0):
    if use_even:
        points, _ = trimesh.sample.sample_surface_even(mesh, count=num_points, seed=seed)
    else:
        points, _ = trimesh.sample.sample_surface(mesh, count=num_points, seed=seed)
    return np.asarray(points, dtype=np.float32)


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


def crop_xy_radius_height_point_cloud(points, start, goal, radius, height):
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")

    start = np.asarray(start, dtype=np.float32).reshape(3)
    goal = np.asarray(goal, dtype=np.float32).reshape(3)
    points_xy = points[:, :2].astype(np.float32)
    planar_distances = point_to_segment_distance_2d(points_xy, start[:2], goal[:2])

    z_min = float(np.min(points[:, 2]))
    z_max = z_min + float(height)
    z_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    xy_mask = planar_distances <= radius
    cropped = points[xy_mask & z_mask]
    if len(cropped) == 0:
        raise ValueError(
            "XY-radius/height ROI contains zero points. "
            "Increase radius/height or check the mesh transform."
        )
    return cropped.astype(np.float32)


def farthest_point_sampling_numpy(points, num_points, seed=0):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    n_points = points.shape[0]
    if n_points == 0:
        raise ValueError("Cannot sample from an empty point cloud.")
    if n_points == 1:
        return np.repeat(points, num_points, axis=0)
    if n_points <= num_points:
        rng = np.random.default_rng(seed)
        repeat_count = num_points - n_points
        if repeat_count == 0:
            return points.astype(np.float32)
        repeat_indices = rng.integers(0, n_points, size=repeat_count)
        return np.concatenate([points, points[repeat_indices]], axis=0).astype(np.float32)

    rng = np.random.default_rng(seed)
    selected_indices = np.zeros(num_points, dtype=np.int64)
    distances = np.full(n_points, np.inf, dtype=np.float32)
    selected_indices[0] = int(rng.integers(0, n_points))

    for i in range(1, num_points):
        last_point = points[selected_indices[i - 1]]
        current_dist = np.sum((points - last_point[None, :]) ** 2, axis=1)
        distances = np.minimum(distances, current_dist)
        selected_indices[i] = int(np.argmax(distances))

    return points[selected_indices].astype(np.float32)


def extract_world_xy_radius_height_roi_from_stl_and_transition(
    stl_path,
    transition_path,
    job_dir,
    radius_m=0.1,
    height_m=0.1,
    num_output_points=DEFAULT_POINT_CLOUD_SIZE,
    num_mesh_sample_points=DEFAULT_MESH_SAMPLE_POINTS,
    use_even_sampling=False,
    seed=0,
):
    transition = load_transition_data(transition_path)
    mesh = load_transformed_mesh_world(stl_path, job_dir)
    raw_mesh_points_world_m = sample_mesh_surface(
        mesh=mesh,
        num_points=num_mesh_sample_points,
        use_even=use_even_sampling,
        seed=seed,
    )
    cropped_points_world_m = crop_xy_radius_height_point_cloud(
        points=raw_mesh_points_world_m,
        start=transition["start_xyz"],
        goal=transition["end_xyz"],
        radius=radius_m,
        height=height_m,
    )
    sampled_points = farthest_point_sampling_numpy(cropped_points_world_m, num_output_points, seed=seed)
    return WorldROICropResult(
        point_cloud=sampled_points.astype(np.float32),
        raw_mesh_points_world_m=raw_mesh_points_world_m.astype(np.float32),
        cropped_points_world_m=cropped_points_world_m.astype(np.float32),
        start_xyz_world_m=transition["start_xyz"].astype(np.float32),
        goal_xyz_world_m=transition["end_xyz"].astype(np.float32),
    )


def infer_job_dir_from_transition(transition_path, jobs_root):
    transition_path = Path(transition_path).resolve()
    job_name = transition_path.parent.name
    job_dir = Path(jobs_root).resolve() / job_name
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Cannot find matching job directory for {transition_path}: {job_dir}")
    return job_dir


def collect_transition_files(results_root):
    results_root = Path(results_root).resolve()
    transition_files = []
    for job_dir in sorted(results_root.glob("job_*")):
        npz_files = sorted(job_dir.glob("transition_*.npz"))
        if npz_files:
            transition_files.extend(npz_files)
            continue
        transition_files.extend(sorted(job_dir.glob("transition_*.json")))
    return transition_files


def save_result_npz(output_path, result, metadata):
    np.savez_compressed(
        output_path,
        # Recommended downstream input for sample_tcp_points_from_workpiece.py.
        point_cloud=result.point_cloud,
        raw_mesh_points_world_m=result.raw_mesh_points_world_m,
        cropped_points_world_m=result.cropped_points_world_m,
        start_xyz_world_m=result.start_xyz_world_m,
        goal_xyz_world_m=result.goal_xyz_world_m,
        radius_m=np.array(metadata["radius_m"], dtype=np.float32),
        height_m=np.array(metadata["height_m"], dtype=np.float32),
        source_transition=np.array(str(metadata["transition_path"]), dtype=object),
        source_stl=np.array(str(metadata["stl_path"]), dtype=object),
    )


def main():
    parser = argparse.ArgumentParser("extract_transition_pointcloud_roi")
    parser.add_argument(
        "--results-root",
        type=str,
        default="/Users/ycj/Desktop/Research/Warmup/DiffusionPolicyPathplanning/3D-Diffusion-Policy/data/raw_data/results",
        help="Root directory containing results/job_xxx/transition_*.npz",
    )
    parser.add_argument(
        "--jobs-root",
        type=str,
        default="/Users/ycj/Desktop/Research/Warmup/DiffusionPolicyPathplanning/3D-Diffusion-Policy/data/raw_data/jobs",
        help="Root directory containing jobs/job_xxx/workpiece.stl",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/transition_pointcloud_roi_world",
        help="Output root for extracted ROI point clouds",
    )
    parser.add_argument("--radius-m", type=float, default=0.1, help="XY capsule radius in world coordinates")
    parser.add_argument("--height-m", type=float, default=0.1, help="Height measured from the workpiece minimum z")
    parser.add_argument("--num-output-points", type=int, default=DEFAULT_POINT_CLOUD_SIZE, help="Fixed ROI point count")
    parser.add_argument("--num-mesh-sample-points", type=int, default=DEFAULT_MESH_SAMPLE_POINTS, help="Dense mesh sampling count before ROI crop")
    parser.add_argument("--use-even-sampling", action="store_true", default=False, help="Use trimesh sample_surface_even before ROI crop")
    parser.add_argument("--job-name", type=str, default=None, help="Only process one job, e.g. job_003")
    parser.add_argument("--transition-name", type=str, default=None, help="Only process one transition stem, e.g. transition_0012_0053")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    transition_files = collect_transition_files(args.results_root)
    if args.job_name is not None:
        transition_files = [path for path in transition_files if path.parent.name == args.job_name]
    if args.transition_name is not None:
        transition_files = [path for path in transition_files if path.stem == args.transition_name]
    if not transition_files:
        raise FileNotFoundError("No transition files matched the provided filters.")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    success = 0
    failures = []
    for transition_path in transition_files:
        try:
            job_dir = infer_job_dir_from_transition(transition_path, args.jobs_root)
            stl_path = job_dir / "workpiece.stl"
            result = extract_world_xy_radius_height_roi_from_stl_and_transition(
                stl_path=stl_path,
                transition_path=transition_path,
                job_dir=job_dir,
                radius_m=args.radius_m,
                height_m=args.height_m,
                num_output_points=args.num_output_points,
                num_mesh_sample_points=args.num_mesh_sample_points,
                use_even_sampling=args.use_even_sampling,
                seed=args.seed,
            )
            out_dir = output_root / job_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{transition_path.stem}_roi_world.npz"
            save_result_npz(
                output_path=output_path,
                result=result,
                metadata={
                    "radius_m": args.radius_m,
                    "height_m": args.height_m,
                    "transition_path": transition_path,
                    "stl_path": stl_path,
                },
            )
            success += 1
            print(f"saved: {output_path}")
        except Exception as exc:
            failures.append((str(transition_path), str(exc)))
            print(f"failed: {transition_path} | {exc}")

    print(f"processed: {len(transition_files)}")
    print(f"succeeded: {success}")
    print(f"failed: {len(failures)}")
    if failures:
        print("failure_examples:")
        for path, reason in failures[:10]:
            print(f"{path} -> {reason}")


if __name__ == "__main__":
    main()
