import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    from .numpy_npz_compat import install_numpy_pickle_compat
    from .query_current_robot_workpiece_distance import RobotWorkpieceDistanceQuery
    from .extract_transition_pointcloud_roi import (
        crop_xy_radius_height_point_cloud,
        farthest_point_sampling_numpy,
        load_transformed_mesh_world,
        sample_mesh_surface,
    )
except ImportError:
    from numpy_npz_compat import install_numpy_pickle_compat
    from query_current_robot_workpiece_distance import RobotWorkpieceDistanceQuery
    from extract_transition_pointcloud_roi import (
        crop_xy_radius_height_point_cloud,
        farthest_point_sampling_numpy,
        load_transformed_mesh_world,
        sample_mesh_surface,
    )

install_numpy_pickle_compat()


DEFAULT_RESULTS_ROOT = "/Users/ycj/Desktop/Research/Warmup/DiffusionPolicyPathplanning/3D-Diffusion-Policy/data/raw_data/results"
DEFAULT_JOBS_ROOT = "/Users/ycj/Desktop/Research/Warmup/DiffusionPolicyPathplanning/3D-Diffusion-Policy/data/raw_data/jobs"
DEFAULT_URDF = "config/robot-model/ur5e_with_pen.urdf"
DEFAULT_LOCAL_POINTS = "config/robot-model/ur5e_surface_points_local.npz"
DEFAULT_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
TRANSITION_RE = re.compile(r"(transition_\d+_\d+)")


def scalar_string(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    raise ValueError(f"Expected scalar string-compatible value, got shape {arr.shape}")


def infer_transition_stem(path):
    match = TRANSITION_RE.search(Path(path).stem)
    if match is None:
        raise ValueError(f"Cannot infer transition stem from path: {path}")
    return match.group(1)


def find_job_dir(root, job_name):
    root = Path(root).resolve()
    if job_name is None:
        return root
    nested = root / job_name
    if nested.is_dir():
        return nested
    if root.name == job_name:
        return root
    return nested


def collect_transition_files(results_root, job_name=None, transition_name=None):
    results_root = Path(results_root).resolve()
    if job_name is not None:
        job_dirs = [find_job_dir(results_root, job_name)]
    else:
        job_dirs = sorted(path for path in results_root.glob("job_*") if path.is_dir())

    transition_files = []
    for job_dir in job_dirs:
        if not job_dir.is_dir():
            continue
        if transition_name is None:
            transition_files.extend(sorted(job_dir.glob("transition_*.npz")))
        else:
            transition_path = job_dir / f"{transition_name}.npz"
            if transition_path.is_file():
                transition_files.append(transition_path)

    if not transition_files:
        raise FileNotFoundError("No transition npz files matched the provided filters.")
    return transition_files


def canonicalize_axis_symmetric_tcp_transform(tcp_transform):
    tcp_transform = np.asarray(tcp_transform, dtype=np.float64)
    if tcp_transform.shape != (4, 4):
        raise ValueError(f"start_tf must have shape [4, 4], got {tcp_transform.shape}")

    canonical = tcp_transform.copy()
    z_axis = canonical[:3, 2].astype(np.float64)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-12:
        raise ValueError("start_tf has a degenerate TCP z axis.")
    z_axis = z_axis / z_norm

    y_axis = np.array([-z_axis[1], z_axis[0], 0.0], dtype=np.float64)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        original_y = canonical[:3, 1].astype(np.float64)
        y_axis = np.array([original_y[0], original_y[1], 0.0], dtype=np.float64)
        y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_norm = 1.0
    y_axis = y_axis / y_norm

    x_axis = np.cross(y_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("Cannot build a right-handed TCP frame from start_tf.")
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    canonical[:3, 0] = x_axis
    canonical[:3, 1] = y_axis
    canonical[:3, 2] = z_axis
    return canonical.astype(np.float32)


def world_to_tcp_points(points_world, tcp_transform):
    points_world = np.asarray(points_world, dtype=np.float32)
    tcp_transform = np.asarray(tcp_transform, dtype=np.float32)
    rotation = tcp_transform[:3, :3]
    translation = tcp_transform[:3, 3]
    return ((points_world - translation[None, :]) @ rotation).astype(np.float32)


def parse_urdf_joint_limits(urdf_path, joint_names=DEFAULT_JOINT_NAMES):
    urdf_path = Path(urdf_path).resolve()
    root = ET.parse(urdf_path).getroot()
    joint_lookup = {joint.get("name"): joint for joint in root.findall("joint")}

    lower = []
    upper = []
    for joint_name in joint_names:
        joint = joint_lookup.get(joint_name)
        if joint is None:
            raise KeyError(f"Joint '{joint_name}' not found in URDF: {urdf_path}")
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise KeyError(f"Joint '{joint_name}' is missing lower/upper limits in URDF: {urdf_path}")
        lower.append(float(limit.get("lower")))
        upper.append(float(limit.get("upper")))

    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    if np.any(upper <= lower):
        raise ValueError(f"Invalid joint limits parsed from URDF: lower={lower}, upper={upper}")
    return lower, upper


def normalize_with_joint_limits(values, lower, upper):
    values = np.asarray(values, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    return (2.0 * (values - lower[None, :]) / (upper - lower)[None, :] - 1.0).astype(np.float32)


def build_joint_features(q_start, q_samples, lower, upper):
    q_start = np.asarray(q_start, dtype=np.float32).reshape(6)
    q_samples = np.asarray(q_samples, dtype=np.float32)
    if q_samples.ndim != 2 or q_samples.shape[1] != 6:
        raise ValueError(f"q_samples must have shape [N, 6], got {q_samples.shape}")

    q_start_tiled = np.broadcast_to(q_start[None, :], q_samples.shape).astype(np.float32)
    delta_q = (q_samples - q_start_tiled).astype(np.float32)
    q_start_norm = normalize_with_joint_limits(q_start_tiled, lower, upper)
    q_sample_norm = normalize_with_joint_limits(q_samples, lower, upper)
    delta_q_norm = normalize_with_joint_limits(delta_q, lower, upper)
    joint_features = np.concatenate([q_start_norm, q_sample_norm, delta_q_norm], axis=1).astype(np.float32)
    return joint_features, q_start_tiled, delta_q


def load_transition_npz(transition_path):
    transition_path = Path(transition_path).resolve()
    with np.load(transition_path, allow_pickle=True) as data:
        required = ("start_tf", "q_start", "start_xyz", "end_xyz")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Transition npz missing required fields {missing}: {transition_path}")
        return {
            "path": transition_path,
            "start_tf": np.asarray(data["start_tf"], dtype=np.float32),
            "q_start": np.asarray(data["q_start"], dtype=np.float32).reshape(6),
            "start_xyz": np.asarray(data["start_xyz"], dtype=np.float32).reshape(3),
            "end_xyz": np.asarray(data["end_xyz"], dtype=np.float32).reshape(3),
        }


def candidate_dir_for_root(root, job_name):
    root = Path(root).resolve()
    if job_name is None:
        return root
    nested = root / job_name
    if nested.is_dir():
        return nested
    return root


def source_file_matches_kind(path, data, source_kind):
    if source_kind not in ("ik_near", "ik_far"):
        return True

    markers = [Path(path).stem.lower()]
    if "source_tcp_npz" in data:
        markers.append(scalar_string(data["source_tcp_npz"]).lower())
    joined = " ".join(markers)
    looks_far = "far" in joined
    return looks_far if source_kind == "ik_far" else not looks_far


def find_source_files(source_root, job_name, transition_stem, required_key, source_kind):
    if source_root is None:
        return []
    source_dir = candidate_dir_for_root(source_root, job_name)
    if not source_dir.is_dir():
        return []

    matches = []
    for path in sorted(source_dir.glob(f"{transition_stem}*.npz")):
        try:
            with np.load(path, allow_pickle=True) as data:
                if required_key in data and source_file_matches_kind(path, data, source_kind):
                    matches.append(path.resolve())
        except Exception:
            continue
    return matches


def load_joint_samples_from_source(source_path, source_kind):
    with np.load(source_path, allow_pickle=True) as data:
        if source_kind in ("ik_near", "ik_far"):
            key = "ik_solutions"
            if "joint_names" not in data:
                raise KeyError(f"IK source {source_path} missing required key 'joint_names'")
            joint_names = [str(name) for name in data["joint_names"].tolist()]
        elif source_kind == "noisy_playback":
            key = "q_playback_noisy"
            joint_names = list(DEFAULT_JOINT_NAMES)
        else:
            raise ValueError(f"Unsupported source kind: {source_kind}")

        if key not in data:
            raise KeyError(f"Joint source {source_path} missing required key '{key}'")
        q_samples = np.asarray(data[key], dtype=np.float32)
    if q_samples.ndim != 2 or q_samples.shape[1] != 6:
        raise ValueError(f"{key} must have shape [N, 6], got {q_samples.shape} in {source_path}")
    return q_samples, joint_names


def resolve_job_assets(jobs_root, transition_path):
    job_name = Path(transition_path).resolve().parent.name
    job_dir = find_job_dir(jobs_root, job_name)
    stl_path = job_dir / "workpiece.stl"
    sdf_path = job_dir / "workpiece_sdf.npz"
    if not stl_path.is_file():
        raise FileNotFoundError(f"Cannot find workpiece STL for {transition_path}: {stl_path}")
    if not sdf_path.is_file():
        raise FileNotFoundError(f"Cannot find workpiece SDF for {transition_path}: {sdf_path}")
    return job_dir.resolve(), stl_path.resolve(), sdf_path.resolve()


def load_job_surface_points(job_dir, stl_path, args):
    mesh = load_transformed_mesh_world(stl_path=stl_path, job_dir=job_dir)
    return sample_mesh_surface(
        mesh=mesh,
        num_points=args.num_mesh_sample_points,
        use_even=args.use_even_sampling,
        seed=args.seed,
    )


def build_distance_targets(min_signed_distance):
    min_signed_distance = np.asarray(min_signed_distance, dtype=np.float32)
    collision_labels = (min_signed_distance < 0.0).astype(np.int64)
    min_distance_norm = (np.clip(min_signed_distance, -0.05, 0.05) / 0.05).astype(np.float32)
    return collision_labels, min_distance_norm


def process_one_transition(
    transition_path,
    job_surface_points,
    stl_path,
    sdf_path,
    args,
    joint_limits_lower,
    joint_limits_upper,
    distance_query,
):
    transition = load_transition_npz(transition_path)
    transition_stem = infer_transition_stem(transition_path)
    job_name = Path(transition_path).resolve().parent.name
    source_specs = [
        ("ik_near", args.ik_near_root, "ik_solutions"),
        ("ik_far", args.ik_far_root, "ik_solutions"),
        ("noisy_playback", args.noisy_root, "q_playback_noisy"),
    ]
    matched_sources = []
    for source_kind, source_root, required_key in source_specs:
        for source_path in find_source_files(source_root, job_name, transition_stem, required_key, source_kind):
            matched_sources.append((source_kind, source_path))

    if not matched_sources:
        return {
            "transition_path": str(transition_path),
            "stl_path": str(stl_path),
            "blocks": [],
            "failures": [],
            "skipped_invalid_distance": 0,
        }

    cropped_points_world = crop_xy_radius_height_point_cloud(
        points=job_surface_points,
        start=transition["start_xyz"],
        goal=transition["end_xyz"],
        radius=args.radius_m,
        height=args.height_m,
    )
    points_world_512 = farthest_point_sampling_numpy(
        cropped_points_world,
        num_points=args.num_points,
        seed=args.seed,
    )
    canonical_start_tf = canonicalize_axis_symmetric_tcp_transform(transition["start_tf"])
    points_tcp = world_to_tcp_points(points_world_512, canonical_start_tf)
    point_cloud_input = (points_tcp / float(args.point_scale)).astype(np.float32)

    blocks = []
    failures = []
    skipped_invalid_distance = 0
    for source_kind, source_path in matched_sources:
        try:
            q_samples, source_joint_names = load_joint_samples_from_source(source_path, source_kind)
            distance_result = distance_query.query_joint_values(
                joint_values=q_samples,
                sdf_path=sdf_path,
                joint_names=source_joint_names,
                outside_mode=args.outside_mode,
                progress_every=args.distance_progress_every,
            )
            min_signed_distance = distance_result["min_signed_distance"]
            valid_mask = np.isfinite(min_signed_distance)
            skipped_invalid_distance += int(np.sum(~valid_mask))
            if not np.any(valid_mask):
                continue

            source_indices = np.flatnonzero(valid_mask).astype(np.int32)
            q_samples = q_samples[valid_mask]
            min_signed_distance = min_signed_distance[valid_mask]
            joint_features, q_start_tiled, delta_q = build_joint_features(
                q_start=transition["q_start"],
                q_samples=q_samples,
                lower=joint_limits_lower,
                upper=joint_limits_upper,
            )
            collision_labels, min_distance_norm = build_distance_targets(min_signed_distance)
            count = q_samples.shape[0]
            block = {
                "joint_features": joint_features.astype(np.float32),
                "point_clouds": np.repeat(point_cloud_input[None, :, :], count, axis=0).astype(np.float32),
                "collision_labels": collision_labels,
                "min_distance_norm": min_distance_norm,
            }
            if args.save_metadata:
                block.update(
                    {
                        "source_kind": source_kind,
                        "source_path": str(source_path),
                        "q_sample": q_samples.astype(np.float32),
                        "q_start": q_start_tiled.astype(np.float32),
                        "delta_q": delta_q.astype(np.float32),
                        "canonical_start_tf": np.repeat(canonical_start_tf[None, :, :], count, axis=0).astype(np.float32),
                        "source_indices": source_indices,
                    }
                )
            blocks.append(block)
        except Exception as exc:
            failures.append((str(source_path), str(exc)))

    return {
        "transition_path": str(transition_path),
        "stl_path": str(stl_path),
        "blocks": blocks,
        "failures": failures,
        "skipped_invalid_distance": skipped_invalid_distance,
    }


def concatenate_or_empty(arrays, shape, dtype):
    if arrays:
        return np.concatenate(arrays, axis=0).astype(dtype)
    return np.zeros(shape, dtype=dtype)


def build_dataset(args):
    joint_limits_lower, joint_limits_upper = parse_urdf_joint_limits(args.urdf)
    transition_files = collect_transition_files(args.results_root, args.job_name, args.transition_name)
    distance_query = RobotWorkpieceDistanceQuery(
        urdf_path=args.urdf,
        local_points_path=args.local_points,
        num_points=args.distance_num_points,
        min_points_per_link=args.distance_min_points_per_link,
        seed=args.seed,
    )

    point_clouds = []
    joint_features = []
    collision_labels = []
    min_distance_norm = []
    if args.save_metadata:
        q_start = []
        q_sample = []
        delta_q = []
        canonical_start_tf = []
        joint_source = []
        source_transition_npz = []
        source_joint_npz = []
        source_workpiece_stl = []
        transition_index = []
        joint_index_in_source = []
    failures = []
    skipped_invalid_distance = 0
    job_surface_cache = {}

    processed_transitions = 0
    for transition_index_value, transition_path in enumerate(transition_files):
        try:
            job_dir, stl_path, sdf_path = resolve_job_assets(args.jobs_root, transition_path)
            if job_dir not in job_surface_cache:
                job_surface_cache[job_dir] = load_job_surface_points(job_dir, stl_path, args)
                print(
                    f"sampled_workpiece_surface: {job_dir.name} | "
                    f"points: {len(job_surface_cache[job_dir])}"
                )

            result = process_one_transition(
                transition_path=transition_path,
                job_surface_points=job_surface_cache[job_dir],
                stl_path=stl_path,
                sdf_path=sdf_path,
                args=args,
                joint_limits_lower=joint_limits_lower,
                joint_limits_upper=joint_limits_upper,
                distance_query=distance_query,
            )
            processed_transitions += 1
            failures.extend(result["failures"])
            skipped_invalid_distance += result["skipped_invalid_distance"]
            for block in result["blocks"]:
                count = block["joint_features"].shape[0]
                point_clouds.append(block["point_clouds"])
                joint_features.append(block["joint_features"])
                collision_labels.append(block["collision_labels"])
                min_distance_norm.append(block["min_distance_norm"])
                if args.save_metadata:
                    q_start.append(block["q_start"])
                    q_sample.append(block["q_sample"])
                    delta_q.append(block["delta_q"])
                    canonical_start_tf.append(block["canonical_start_tf"])
                    joint_source.extend([block["source_kind"]] * count)
                    source_transition_npz.extend([result["transition_path"]] * count)
                    source_joint_npz.extend([block["source_path"]] * count)
                    source_workpiece_stl.extend([result["stl_path"]] * count)
                    transition_index.extend([transition_index_value] * count)
                    joint_index_in_source.extend(block["source_indices"].tolist())
        except Exception as exc:
            failures.append((str(transition_path), str(exc)))

    dataset = {
        "point_clouds": concatenate_or_empty(point_clouds, (0, args.num_points, 3), np.float32),
        "joint_features": concatenate_or_empty(joint_features, (0, 18), np.float32),
        "collision_labels": concatenate_or_empty(collision_labels, (0,), np.int64),
        "min_distance_norm": concatenate_or_empty(min_distance_norm, (0,), np.float32),
    }
    if args.save_metadata:
        dataset.update(
            {
                "q_start": concatenate_or_empty(q_start, (0, 6), np.float32),
                "q_sample": concatenate_or_empty(q_sample, (0, 6), np.float32),
                "delta_q": concatenate_or_empty(delta_q, (0, 6), np.float32),
                "joint_source": np.asarray(joint_source, dtype=str),
                "source_transition_npz": np.asarray(source_transition_npz, dtype=str),
                "source_joint_npz": np.asarray(source_joint_npz, dtype=str),
                "source_workpiece_stl": np.asarray(source_workpiece_stl, dtype=str),
                "joint_limits_lower": joint_limits_lower.astype(np.float32),
                "joint_limits_upper": joint_limits_upper.astype(np.float32),
                "canonical_start_tf": concatenate_or_empty(canonical_start_tf, (0, 4, 4), np.float32),
                "transition_index": np.asarray(transition_index, dtype=np.int32),
                "joint_index_in_source": np.asarray(joint_index_in_source, dtype=np.int32),
                "num_points": np.array(args.num_points, dtype=np.int32),
                "point_scale": np.array(args.point_scale, dtype=np.float32),
                "seed": np.array(args.seed, dtype=np.int32),
            }
        )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)

    print(f"saved: {output_path}")
    print(f"transition_files_found: {len(transition_files)}")
    print(f"transition_files_processed: {processed_transitions}")
    print(f"workpiece_surfaces_sampled: {len(job_surface_cache)}")
    print(f"samples: {dataset['point_clouds'].shape[0]}")
    print(f"point_clouds: {dataset['point_clouds'].shape}")
    print(f"joint_features: {dataset['joint_features'].shape}")
    print(f"collision_labels: {dataset['collision_labels'].shape}")
    print(f"min_distance_norm: {dataset['min_distance_norm'].shape}")
    print(f"skipped_invalid_distance: {skipped_invalid_distance}")
    print(f"save_metadata: {args.save_metadata}")
    print(f"failures: {len(failures)}")
    if failures:
        print("failure_examples:")
        for path, reason in failures[:10]:
            print(f"{path} -> {reason}")


def main():
    parser = argparse.ArgumentParser("build_pointcloud_joint_input_dataset")
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT, help="Root containing raw transition npz files")
    parser.add_argument("--jobs-root", type=str, default=DEFAULT_JOBS_ROOT, help="Root containing job_xxx/workpiece.stl and workpiece_sdf.npz")
    parser.add_argument("--job-name", type=str, default=None, help="Optional job name, e.g. job_003")
    parser.add_argument("--transition-name", type=str, default=None, help="Optional transition stem, e.g. transition_0012_0053")
    parser.add_argument("--ik-near-root", type=str, default=None, help="Root containing near-surface IK npz files")
    parser.add_argument("--ik-far-root", type=str, default=None, help="Root containing far-field IK npz files")
    parser.add_argument("--noisy-root", type=str, default=None, help="Root containing q_playback_noisy npz files")
    parser.add_argument("--output", type=str, required=True, help="Output dataset npz path")
    parser.add_argument("--urdf", type=str, default=DEFAULT_URDF, help="Robot URDF path used for joint limits")
    parser.add_argument("--local-points", type=str, default=DEFAULT_LOCAL_POINTS, help="Link-local robot surface point cache; sampled in memory if missing")
    parser.add_argument("--radius-m", type=float, default=0.1, help="XY capsule radius used for in-memory ROI cropping")
    parser.add_argument("--height-m", type=float, default=0.1, help="ROI height measured from the workpiece minimum z")
    parser.add_argument("--num-mesh-sample-points", type=int, default=100000, help="Dense workpiece surface points sampled once per job")
    parser.add_argument("--use-even-sampling", action="store_true", help="Use trimesh even surface sampling for the workpiece")
    parser.add_argument("--num-points", type=int, default=512, help="Point count per sample after downsampling/padding")
    parser.add_argument("--point-scale", type=float, default=0.1, help="TCP-frame point cloud normalization divisor in meters")
    parser.add_argument("--distance-num-points", type=int, default=8192, help="Robot surface point count if local cache must be sampled in memory")
    parser.add_argument("--distance-min-points-per-link", type=int, default=32, help="Minimum robot surface points per link when sampling in memory")
    parser.add_argument("--outside-mode", type=str, default="project", choices=["ignore", "project", "error"], help="How robot points outside the SDF grid are handled")
    parser.add_argument("--distance-progress-every", type=int, default=100, help="Print progress every N joint distance queries; 0 disables progress")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible point downsampling/padding")
    parser.add_argument("--save-metadata", action="store_true", help="Also save source paths, raw joints, joint limits, and canonical TCP transforms for debugging")
    args = parser.parse_args()

    if args.num_points <= 0:
        raise ValueError("--num-points must be positive.")
    if args.radius_m <= 0:
        raise ValueError("--radius-m must be positive.")
    if args.height_m <= 0:
        raise ValueError("--height-m must be positive.")
    if args.num_mesh_sample_points <= 0:
        raise ValueError("--num-mesh-sample-points must be positive.")
    if args.point_scale <= 0:
        raise ValueError("--point-scale must be positive.")
    if args.distance_num_points <= 0:
        raise ValueError("--distance-num-points must be positive.")
    if args.distance_min_points_per_link < 0:
        raise ValueError("--distance-min-points-per-link must be non-negative.")
    if args.distance_progress_every < 0:
        raise ValueError("--distance-progress-every must be non-negative.")
    if args.ik_near_root is None and args.ik_far_root is None and args.noisy_root is None:
        raise ValueError("Provide at least one of --ik-near-root, --ik-far-root, or --noisy-root.")

    build_dataset(args)


if __name__ == "__main__":
    main()
