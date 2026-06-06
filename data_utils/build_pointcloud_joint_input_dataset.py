import argparse
import json
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
    from .extract_noisy_transition_joint_trajectories import build_noisy_trajectory
    from .sample_tcp_points_from_workpiece import (
        estimate_point_normals,
        generate_tcp_orientations,
        generate_tcp_points_from_point_cloud,
        generate_tcp_points_from_roi_capsule,
        load_sdf_metadata,
    )
    from .solve_tcp_ik_from_samples import PyBulletIKSolver
except ImportError:
    from numpy_npz_compat import install_numpy_pickle_compat
    from query_current_robot_workpiece_distance import RobotWorkpieceDistanceQuery
    from extract_transition_pointcloud_roi import (
        crop_xy_radius_height_point_cloud,
        farthest_point_sampling_numpy,
        load_transformed_mesh_world,
        sample_mesh_surface,
    )
    from extract_noisy_transition_joint_trajectories import build_noisy_trajectory
    from sample_tcp_points_from_workpiece import (
        estimate_point_normals,
        generate_tcp_orientations,
        generate_tcp_points_from_point_cloud,
        generate_tcp_points_from_roi_capsule,
        load_sdf_metadata,
    )
    from solve_tcp_ik_from_samples import PyBulletIKSolver

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


REQUIRED_DATASET_FIELDS = (
    "point_clouds",
    "joint_features",
    "collision_labels",
    "min_distance_norm",
)


def write_dataset_npz(output_path, dataset):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    return output_path


def dataset_sample_count(dataset):
    return int(dataset["point_clouds"].shape[0])


def empty_dataset_dict(args):
    dataset = {
        "point_clouds": np.zeros((0, args.num_points, 3), dtype=np.float32),
        "joint_features": np.zeros((0, 18), dtype=np.float32),
        "collision_labels": np.zeros((0,), dtype=np.int64),
        "min_distance_norm": np.zeros((0,), dtype=np.float32),
    }
    if args.save_metadata:
        dataset.update(
            {
                "q_start": np.zeros((0, 6), dtype=np.float32),
                "q_sample": np.zeros((0, 6), dtype=np.float32),
                "delta_q": np.zeros((0, 6), dtype=np.float32),
                "joint_source": np.asarray([], dtype=str),
                "source_transition_npz": np.asarray([], dtype=str),
                "source_joint_npz": np.asarray([], dtype=str),
                "source_workpiece_stl": np.asarray([], dtype=str),
                "joint_limits_lower": np.zeros((6,), dtype=np.float32),
                "joint_limits_upper": np.zeros((6,), dtype=np.float32),
                "canonical_start_tf": np.zeros((0, 4, 4), dtype=np.float32),
                "transition_index": np.asarray([], dtype=np.int32),
                "joint_index_in_source": np.asarray([], dtype=np.int32),
                "num_points": np.array(args.num_points, dtype=np.int32),
                "point_scale": np.array(args.point_scale, dtype=np.float32),
                "seed": np.array(args.seed, dtype=np.int32),
            }
        )
    return dataset


def build_dataset_from_transition_results(args, joint_limits_lower, joint_limits_upper, transition_results):
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

    for transition_result in transition_results:
        for block in transition_result["blocks"]:
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
                source_transition_npz.extend([transition_result["transition_path"]] * count)
                source_joint_npz.extend([block["source_path"]] * count)
                source_workpiece_stl.extend([transition_result["stl_path"]] * count)
                transition_index.extend([transition_result["transition_index"]] * count)
                joint_index_in_source.extend(block["source_indices"].tolist())

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
    return dataset


def sanitize_transition_name(transition_path):
    path = Path(transition_path)
    return f"{path.parent.name}__{path.stem}"


def write_transition_shard(shard_output_dir, transition_result, dataset):
    shard_output_dir = Path(shard_output_dir).resolve()
    shard_output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_output_dir / f"{transition_result['transition_index']:06d}_{sanitize_transition_name(transition_result['transition_path'])}.npz"
    write_dataset_npz(shard_path, dataset)
    return shard_path


def write_shard_manifest(manifest_path, args, transition_files, shard_records, totals):
    manifest_path = Path(manifest_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "pointcloud_joint_dataset_manifest_v1",
        "required_fields": list(REQUIRED_DATASET_FIELDS),
        "save_metadata": bool(args.save_metadata),
        "results_root": str(Path(args.results_root).resolve()),
        "jobs_root": str(Path(args.jobs_root).resolve()),
        "transition_files_found": int(len(transition_files)),
        "shards": shard_records,
        "totals": totals,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path
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
            "q_playback": np.asarray(data["q_playback"], dtype=np.float32) if "q_playback" in data else None,
        }


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
    sdf_meta,
    args,
    joint_limits_lower,
    joint_limits_upper,
    distance_query,
    ik_solver,
    transition_seed,
):
    transition = load_transition_npz(transition_path)
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
        seed=transition_seed,
    )
    canonical_start_tf = canonicalize_axis_symmetric_tcp_transform(transition["start_tf"])
    points_tcp = world_to_tcp_points(points_world_512, canonical_start_tf)
    point_cloud_input = (points_tcp / float(args.point_scale)).astype(np.float32)

    failures = []
    generated_sources = []
    normals = None

    if not args.skip_near_ik or not args.skip_far_ik:
        normals = estimate_point_normals(points_world_512.astype(np.float64), args.normal_k)

    def solve_tcp_source(source_kind, tcp_samples, seed_offset):
        orientations = generate_tcp_orientations(
            num_points=len(tcp_samples["tcp_points"]),
            seed=transition_seed + seed_offset,
        )
        ik_result = ik_solver.solve(
            tcp_points=tcp_samples["tcp_points"],
            orientation_quaternions_xyzw=orientations["orientation_quaternions_xyzw"],
            max_orientations=args.ik_max_orientations,
            max_solutions=args.ik_max_solutions,
            num_random_seeds=args.ik_num_random_seeds,
            max_random_retries=args.ik_max_random_retries,
            seed=transition_seed + seed_offset + 1,
            joint_tol=args.ik_joint_tol,
            pos_tol=args.ik_pos_tol,
            orn_tol_deg=args.ik_orn_tol_deg,
            max_iterations=args.ik_max_iterations,
            residual_threshold=args.ik_residual_threshold,
            progress_every=args.ik_progress_every,
        )
        generated_sources.append(
            (
                source_kind,
                np.asarray(ik_result["ik_solutions"], dtype=np.float32),
                np.arange(len(ik_result["ik_solutions"]), dtype=np.int32),
            )
        )

    if not args.skip_near_ik:
        try:
            near_samples = generate_tcp_points_from_point_cloud(
                points=points_world_512.astype(np.float64),
                normals=normals,
                sdf_meta=sdf_meta,
                num_points=args.near_tcp_points,
                near_range=(args.near_min, args.near_max),
                probe_eps=args.probe_eps,
                seed=transition_seed + 101,
            )
            solve_tcp_source("ik_near", near_samples, 201)
        except Exception as exc:
            failures.append((f"{transition_path}:ik_near", str(exc)))

    if not args.skip_far_ik:
        try:
            roi_meta = {
                "start_xyz_world_m": transition["start_xyz"],
                "goal_xyz_world_m": transition["end_xyz"],
                "radius_m": args.radius_m,
                "height_m": args.height_m,
                "z_min": float(np.min(job_surface_points[:, 2])),
            }
            far_samples = generate_tcp_points_from_roi_capsule(
                points=points_world_512.astype(np.float64),
                normals=normals,
                sdf_meta=sdf_meta,
                roi_meta=roi_meta,
                num_points=args.far_tcp_points,
                far_min=args.far_min,
                probe_eps=args.probe_eps,
                seed=transition_seed + 301,
            )
            solve_tcp_source("ik_far", far_samples, 401)
        except Exception as exc:
            failures.append((f"{transition_path}:ik_far", str(exc)))

    if not args.skip_noisy_playback:
        try:
            q_playback = np.asarray(transition["q_playback"], dtype=np.float32)
            if q_playback.ndim != 2 or q_playback.shape[1] != 6:
                raise ValueError(f"q_playback must have shape [T, 6], got {q_playback.shape}")
            q_noisy, _ = build_noisy_trajectory(
                trajectory=q_playback,
                sigma=args.noise_sigma,
                clip_min=-args.noise_clip,
                clip_max=args.noise_clip,
                rng=np.random.default_rng(transition_seed + 501),
            )
            generated_sources.append(
                ("noisy_playback", q_noisy, np.arange(len(q_noisy), dtype=np.int32))
            )
        except Exception as exc:
            failures.append((f"{transition_path}:noisy_playback", str(exc)))

    blocks = []
    skipped_invalid_distance = 0
    for source_kind, q_samples, source_indices in generated_sources:
        if len(q_samples) == 0:
            continue
        try:
            distance_result = distance_query.query_joint_values(
                joint_values=q_samples,
                sdf_path=sdf_path,
                joint_names=DEFAULT_JOINT_NAMES,
                outside_mode=args.outside_mode,
                progress_every=args.distance_progress_every,
            )
            min_signed_distance = distance_result["min_signed_distance"]
            valid_mask = np.isfinite(min_signed_distance)
            skipped_invalid_distance += int(np.sum(~valid_mask))
            if not np.any(valid_mask):
                continue

            source_indices = source_indices[valid_mask]
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
                "source_kind": source_kind,
                "joint_features": joint_features.astype(np.float32),
                "point_clouds": np.repeat(point_cloud_input[None, :, :], count, axis=0).astype(np.float32),
                "collision_labels": collision_labels,
                "min_distance_norm": min_distance_norm,
            }
            if args.save_metadata:
                block.update(
                    {
                        "source_path": f"in_memory:{source_kind}",
                        "q_sample": q_samples.astype(np.float32),
                        "q_start": q_start_tiled.astype(np.float32),
                        "delta_q": delta_q.astype(np.float32),
                        "canonical_start_tf": np.repeat(canonical_start_tf[None, :, :], count, axis=0).astype(np.float32),
                        "source_indices": source_indices,
                    }
                )
            blocks.append(block)
        except Exception as exc:
            failures.append((f"{transition_path}:{source_kind}:distance", str(exc)))

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
    if args.max_transitions is not None:
        transition_files = transition_files[: args.max_transitions]
    print(f"transition_files_found: {len(transition_files)}")
    if args.shard_output_dir:
        print(f"shard_output_dir: {Path(args.shard_output_dir).resolve()}")
    if args.output:
        print(f"output_target: {Path(args.output).resolve()}")
    print("initializing_distance_query...")
    distance_query = RobotWorkpieceDistanceQuery(
        urdf_path=args.urdf,
        local_points_path=args.local_points,
        num_points=args.distance_num_points,
        min_points_per_link=args.distance_min_points_per_link,
        seed=args.seed,
    )
    print("distance_query_ready")

    stream_to_shards = args.shard_output_dir is not None
    point_clouds = [] if not stream_to_shards else None
    joint_features = [] if not stream_to_shards else None
    collision_labels = [] if not stream_to_shards else None
    min_distance_norm = [] if not stream_to_shards else None
    if args.save_metadata and not stream_to_shards:
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
    source_sample_counts = {"ik_near": 0, "ik_far": 0, "noisy_playback": 0}
    job_surface_cache = {}
    sdf_meta_cache = {}
    shard_records = []
    skipped_existing_shards = 0
    ik_solver = None
    if not args.skip_near_ik or not args.skip_far_ik:
        print("initializing_ik_solver...")
        ik_solver = PyBulletIKSolver(args.urdf, tcp_link=args.tcp_link)
        print("ik_solver_ready")

    processed_transitions = 0
    try:
        for transition_index_value, transition_path in enumerate(transition_files):
            if stream_to_shards:
                existing_shard_path = (
                    Path(args.shard_output_dir).resolve()
                    / f"{transition_index_value:06d}_{sanitize_transition_name(transition_path)}.npz"
                )
                if args.resume and existing_shard_path.is_file():
                    with np.load(existing_shard_path, allow_pickle=True) as existing_data:
                        shard_sample_count = int(existing_data["point_clouds"].shape[0])
                    shard_records.append(
                        {
                            "transition_index": transition_index_value,
                            "transition_path": str(Path(transition_path).resolve()),
                            "shard_path": str(existing_shard_path),
                            "samples": shard_sample_count,
                            "status": "reused",
                        }
                    )
                    skipped_existing_shards += 1
                    print(
                        f"reused_shard: {existing_shard_path.name} | "
                        f"transition {transition_index_value + 1}/{len(transition_files)} | "
                        f"samples: {shard_sample_count}"
                    )
                    continue

            print(
                f"processing_transition: {transition_index_value + 1}/{len(transition_files)} | "
                f"{Path(transition_path).parent.name}/{Path(transition_path).name}"
            )
            try:
                job_dir, stl_path, sdf_path = resolve_job_assets(args.jobs_root, transition_path)
                if job_dir not in job_surface_cache:
                    job_surface_cache[job_dir] = load_job_surface_points(job_dir, stl_path, args)
                    sdf_meta_cache[job_dir] = load_sdf_metadata(sdf_path)
                    print(
                        f"sampled_workpiece_surface: {job_dir.name} | "
                        f"points: {len(job_surface_cache[job_dir])}"
                    )

                result = process_one_transition(
                    transition_path=transition_path,
                    job_surface_points=job_surface_cache[job_dir],
                    stl_path=stl_path,
                    sdf_path=sdf_path,
                    sdf_meta=sdf_meta_cache[job_dir],
                    args=args,
                    joint_limits_lower=joint_limits_lower,
                    joint_limits_upper=joint_limits_upper,
                    distance_query=distance_query,
                    ik_solver=ik_solver,
                    transition_seed=args.seed + transition_index_value * 1009,
                )
                result["transition_index"] = transition_index_value
                processed_transitions += 1
                failures.extend(result["failures"])
                skipped_invalid_distance += result["skipped_invalid_distance"]
                for block in result["blocks"]:
                    source_sample_counts[block["source_kind"]] += block["joint_features"].shape[0]

                if stream_to_shards:
                    shard_dataset = build_dataset_from_transition_results(
                        args=args,
                        joint_limits_lower=joint_limits_lower,
                        joint_limits_upper=joint_limits_upper,
                        transition_results=[result],
                    )
                    shard_sample_count = dataset_sample_count(shard_dataset)
                    shard_path = write_transition_shard(args.shard_output_dir, result, shard_dataset)
                    shard_records.append(
                        {
                            "transition_index": transition_index_value,
                            "transition_path": result["transition_path"],
                            "shard_path": str(shard_path),
                            "samples": shard_sample_count,
                            "status": "written",
                        }
                    )
                    print(
                        f"saved_shard: {shard_path.name} | "
                        f"transition {transition_index_value + 1}/{len(transition_files)} | "
                        f"samples: {shard_sample_count}"
                    )
                else:
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
    finally:
        if ik_solver is not None:
            ik_solver.close()

    if stream_to_shards:
        total_samples = int(sum(record["samples"] for record in shard_records))
        totals = {
            "processed_transitions": int(processed_transitions),
            "reused_shards": int(skipped_existing_shards),
            "workpiece_surfaces_sampled": int(len(job_surface_cache)),
            "samples": total_samples,
            "source_samples": {key: int(value) for key, value in source_sample_counts.items()},
            "skipped_invalid_distance": int(skipped_invalid_distance),
            "failures": int(len(failures)),
        }
        manifest_output = (
            Path(args.output).resolve()
            if args.output
            else Path(args.shard_output_dir).resolve() / "manifest.json"
        )
        manifest_path = write_shard_manifest(
            manifest_output,
            args=args,
            transition_files=transition_files,
            shard_records=shard_records,
            totals=totals,
        )
        dataset = empty_dataset_dict(args)
        print(f"saved_manifest: {manifest_path}")
        print(f"shards_written_or_reused: {len(shard_records)}")
        print(f"samples: {total_samples}")
    else:
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

        output_path = write_dataset_npz(args.output, dataset)
        print(f"saved: {output_path}")

    print(f"transition_files_processed: {processed_transitions}")
    print(f"workpiece_surfaces_sampled: {len(job_surface_cache)}")
    print(f"samples: {dataset_sample_count(dataset) if not stream_to_shards else total_samples}")
    print(f"source_samples: {source_sample_counts}")
    if not stream_to_shards:
        print(f"point_clouds: {dataset['point_clouds'].shape}")
        print(f"joint_features: {dataset['joint_features'].shape}")
        print(f"collision_labels: {dataset['collision_labels'].shape}")
        print(f"min_distance_norm: {dataset['min_distance_norm'].shape}")
    print(f"skipped_invalid_distance: {skipped_invalid_distance}")
    print(f"save_metadata: {args.save_metadata}")
    if stream_to_shards:
        print(f"resume_enabled: {args.resume}")
        print(f"reused_shards: {skipped_existing_shards}")
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
    parser.add_argument("--max-transitions", type=int, default=None, help="Only process the first N transitions for debugging")
    parser.add_argument("--output", type=str, default=None, help="Output dataset npz path, or manifest path when using --shard-output-dir")
    parser.add_argument("--shard-output-dir", type=str, default=None, help="Optional directory to stream one dataset shard per transition and write a manifest")
    parser.add_argument("--resume", action="store_true", help="When using --shard-output-dir, reuse existing shard files and continue writing the manifest")
    parser.add_argument("--urdf", type=str, default=DEFAULT_URDF, help="Robot URDF path used for joint limits")
    parser.add_argument("--local-points", type=str, default=DEFAULT_LOCAL_POINTS, help="Link-local robot surface point cache; sampled in memory if missing")
    parser.add_argument("--radius-m", type=float, default=0.1, help="XY capsule radius used for in-memory ROI cropping")
    parser.add_argument("--height-m", type=float, default=0.1, help="ROI height measured from the workpiece minimum z")
    parser.add_argument("--num-mesh-sample-points", type=int, default=100000, help="Dense workpiece surface points sampled once per job")
    parser.add_argument("--use-even-sampling", action="store_true", help="Use trimesh even surface sampling for the workpiece")
    parser.add_argument("--num-points", type=int, default=512, help="Point count per sample after downsampling/padding")
    parser.add_argument("--point-scale", type=float, default=0.1, help="TCP-frame point cloud normalization divisor in meters")
    parser.add_argument("--near-tcp-points", type=int, default=40, help="Near-surface TCP points generated per transition")
    parser.add_argument("--far-tcp-points", type=int, default=20, help="ROI far-field TCP points generated per transition")
    parser.add_argument("--near-min", type=float, default=0.0, help="Minimum near-surface TCP offset in meters")
    parser.add_argument("--near-max", type=float, default=0.02, help="Maximum near-surface TCP offset in meters")
    parser.add_argument("--far-min", type=float, default=0.02, help="Minimum far-field TCP offset in meters")
    parser.add_argument("--normal-k", type=int, default=30, help="Neighbors used to estimate ROI point normals")
    parser.add_argument("--probe-eps", type=float, default=0.003, help="SDF probe distance used to orient surface normals")
    parser.add_argument("--tcp-link", type=str, default="tool0", help="URDF link whose pose is constrained by IK")
    parser.add_argument("--ik-max-orientations", type=int, default=None, help="Only solve the first M of the 12 orientations for debugging")
    parser.add_argument("--ik-max-solutions", type=int, default=8, help="Maximum unique IK solutions per TCP pose")
    parser.add_argument("--ik-num-random-seeds", type=int, default=6, help="Random seeds added to the deterministic IK seed bank")
    parser.add_argument("--ik-max-random-retries", type=int, default=6, help="Extra random retries after the IK seed bank")
    parser.add_argument("--ik-joint-tol", type=float, default=0.15, help="Joint-space tolerance used to merge duplicate IK solutions")
    parser.add_argument("--ik-pos-tol", type=float, default=0.005, help="Maximum accepted IK TCP position error in meters")
    parser.add_argument("--ik-orn-tol-deg", type=float, default=5.0, help="Maximum accepted IK orientation error in degrees")
    parser.add_argument("--ik-max-iterations", type=int, default=240, help="PyBullet iterations per IK solve")
    parser.add_argument("--ik-residual-threshold", type=float, default=1e-5, help="PyBullet IK residual threshold")
    parser.add_argument("--ik-progress-every", type=int, default=0, help="Print progress every N TCP poses inside each IK source")
    parser.add_argument("--noise-sigma", type=float, default=0.05 / 3.0, help="Gaussian sigma for q_playback perturbation")
    parser.add_argument("--noise-clip", type=float, default=0.05, help="Absolute clipping bound for Gaussian joint noise")
    parser.add_argument("--skip-near-ik", action="store_true", help="Do not generate near-surface TCP/IK samples")
    parser.add_argument("--skip-far-ik", action="store_true", help="Do not generate far-field TCP/IK samples")
    parser.add_argument("--skip-noisy-playback", action="store_true", help="Do not generate noisy q_playback samples")
    parser.add_argument("--distance-num-points", type=int, default=8192, help="Robot surface point count if local cache must be sampled in memory")
    parser.add_argument("--distance-min-points-per-link", type=int, default=32, help="Minimum robot surface points per link when sampling in memory")
    parser.add_argument("--outside-mode", type=str, default="project", choices=["ignore", "project", "error"], help="How robot points outside the SDF grid are handled")
    parser.add_argument("--distance-progress-every", type=int, default=100, help="Print progress every N joint distance queries; 0 disables progress")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible point downsampling/padding")
    parser.add_argument("--save-metadata", action="store_true", help="Also save source paths, raw joints, joint limits, and canonical TCP transforms for debugging")
    args = parser.parse_args()

    if args.num_points <= 0:
        raise ValueError("--num-points must be positive.")
    if args.output is None and args.shard_output_dir is None:
        raise ValueError("Provide --output for single-file mode, or --shard-output-dir for streaming mode.")
    if args.max_transitions is not None and args.max_transitions <= 0:
        raise ValueError("--max-transitions must be positive.")
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
    if args.near_tcp_points <= 0 or args.far_tcp_points <= 0:
        raise ValueError("--near-tcp-points and --far-tcp-points must be positive.")
    if not 0.0 <= args.near_min <= args.near_max:
        raise ValueError("Require 0 <= --near-min <= --near-max.")
    if args.far_min < 0.0 or args.probe_eps <= 0.0:
        raise ValueError("--far-min must be non-negative and --probe-eps must be positive.")
    if args.normal_k < 3:
        raise ValueError("--normal-k must be at least 3.")
    if args.ik_max_orientations is not None and not 1 <= args.ik_max_orientations <= 12:
        raise ValueError("--ik-max-orientations must be in [1, 12].")
    if args.ik_max_solutions <= 0 or args.ik_num_random_seeds < 0 or args.ik_max_random_retries < 0:
        raise ValueError("IK solution count must be positive and IK retry counts must be non-negative.")
    if args.noise_sigma < 0.0 or args.noise_clip < 0.0:
        raise ValueError("--noise-sigma and --noise-clip must be non-negative.")
    if args.skip_near_ik and args.skip_far_ik and args.skip_noisy_playback:
        raise ValueError("At least one sample source must remain enabled.")
    if args.resume and args.shard_output_dir is None:
        raise ValueError("--resume requires --shard-output-dir.")

    build_dataset(args)


if __name__ == "__main__":
    main()
