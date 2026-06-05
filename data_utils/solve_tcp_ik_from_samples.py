import argparse
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


PYBULLET_PYTHON = "/Users/ycj/miniconda3/envs/pybullet/bin/python"


def import_pybullet():
    try:
        import pybullet as p  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pybullet is not available in the current interpreter. "
            f"Run this script with {PYBULLET_PYTHON}."
        ) from exc
    return p


def resolve_package_uri(urdf_path, filename):
    robot_root = Path(urdf_path).resolve().parent
    if filename.startswith("package://urdf-pen/"):
        return str((robot_root / filename.replace("package://urdf-pen/", "")).resolve())
    if filename.startswith("file://"):
        return filename
    return str((robot_root / filename).resolve())


def make_pybullet_ready_urdf(urdf_path):
    urdf_path = Path(urdf_path).resolve()
    root = ET.parse(urdf_path).getroot()

    for mesh_node in root.findall(".//mesh"):
        filename = mesh_node.get("filename")
        if filename:
            mesh_node.set("filename", resolve_package_uri(urdf_path, filename))

    temp_dir = Path(tempfile.mkdtemp(prefix="pybullet_urdf_"))
    temp_urdf = temp_dir / urdf_path.name
    ET.ElementTree(root).write(temp_urdf, encoding="utf-8", xml_declaration=True)
    return temp_urdf


def load_tcp_sample_file(path):
    path = Path(path).resolve()
    data = np.load(path, allow_pickle=True)
    required = ("tcp_points", "orientation_quaternions_xyzw")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"TCP sample npz missing required fields: {missing}")

    tcp_points = np.asarray(data["tcp_points"], dtype=np.float64)
    quats = np.asarray(data["orientation_quaternions_xyzw"], dtype=np.float64)
    if tcp_points.ndim != 2 or tcp_points.shape[1] != 3:
        raise ValueError("tcp_points must have shape (N, 3)")
    if quats.ndim != 3 or quats.shape[0] != len(tcp_points) or quats.shape[2] != 4:
        raise ValueError("orientation_quaternions_xyzw must have shape (N, M, 4)")

    sample = {
        "path": path,
        "tcp_points": tcp_points,
        "orientation_quaternions_xyzw": quats,
        "orientation_matrices": np.asarray(data["orientation_matrices"], dtype=np.float64)
        if "orientation_matrices" in data
        else None,
        "band": np.asarray(data["band"]) if "band" in data else None,
    }
    return sample


def wrap_to_pi(values):
    return (np.asarray(values, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def shortest_angle_diff(a, b):
    return wrap_to_pi(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))


def normalize_quaternion_xyzw(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("Encountered a zero-norm quaternion.")
    return quat / norm


def quaternion_angle_error(quat_a, quat_b):
    qa = normalize_quaternion_xyzw(quat_a)
    qb = normalize_quaternion_xyzw(quat_b)
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return 2.0 * math.acos(dot)


def canonicalize_solution(solution, joint_types, lower_limits, upper_limits):
    q = np.asarray(solution, dtype=np.float64).copy()
    for idx, joint_type in enumerate(joint_types):
        if joint_type in ("revolute", "continuous"):
            q[idx] = wrap_to_pi(q[idx])
        q[idx] = float(np.clip(q[idx], lower_limits[idx], upper_limits[idx]))
    return q


def solution_is_duplicate(candidate, solutions, joint_tol):
    for existing in solutions:
        diff = shortest_angle_diff(candidate, existing)
        if float(np.max(np.abs(diff))) <= joint_tol:
            return True
    return False


def extract_robot_info(p, robot_id, tcp_link_name):
    movable_joint_indices = []
    movable_joint_names = []
    joint_types = []
    lower_limits = []
    upper_limits = []
    link_name_to_index = {}

    joint_type_map = {
        p.JOINT_REVOLUTE: "revolute",
        p.JOINT_PRISMATIC: "prismatic",
        p.JOINT_SPHERICAL: "spherical",
        p.JOINT_PLANAR: "planar",
        p.JOINT_FIXED: "fixed",
    }

    for joint_index in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, joint_index)
        link_name = info[12].decode("utf-8")
        link_name_to_index[link_name] = joint_index
        if info[2] != p.JOINT_FIXED:
            movable_joint_indices.append(joint_index)
            movable_joint_names.append(info[1].decode("utf-8"))
            joint_types.append(joint_type_map.get(info[2], f"type_{info[2]}"))
            lower_limits.append(float(info[8]))
            upper_limits.append(float(info[9]))

    if tcp_link_name not in link_name_to_index:
        raise ValueError(f"TCP link '{tcp_link_name}' not found in URDF.")

    lower_limits = np.asarray(lower_limits, dtype=np.float64)
    upper_limits = np.asarray(upper_limits, dtype=np.float64)
    joint_ranges = np.maximum(upper_limits - lower_limits, 1e-6)

    return {
        "movable_joint_indices": movable_joint_indices,
        "movable_joint_names": movable_joint_names,
        "joint_types": joint_types,
        "lower_limits": lower_limits,
        "upper_limits": upper_limits,
        "joint_ranges": joint_ranges,
        "tcp_link_index": link_name_to_index[tcp_link_name],
    }


def reset_arm_joints(p, robot_id, joint_indices, joint_values):
    for joint_index, joint_value in zip(joint_indices, joint_values):
        p.resetJointState(robot_id, joint_index, float(joint_value))


def get_tcp_pose(p, robot_id, tcp_link_index):
    state = p.getLinkState(robot_id, tcp_link_index, computeForwardKinematics=True)
    return np.asarray(state[4], dtype=np.float64), np.asarray(state[5], dtype=np.float64)


def build_seed_bank(target_position, lower_limits, upper_limits, num_random_seeds, rng):
    azimuth = math.atan2(float(target_position[1]), float(target_position[0]))
    clipped_pi = np.minimum(upper_limits, math.pi)
    clipped_neg_pi = np.maximum(lower_limits, -math.pi)

    deterministic = [
        np.array([azimuth, -math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, 0.0], dtype=np.float64),
        np.array([azimuth, -math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2, 0.0], dtype=np.float64),
        np.array([azimuth + math.pi, -math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi], dtype=np.float64),
        np.array([azimuth + math.pi, -math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2, math.pi], dtype=np.float64),
        np.array([azimuth, 0.0, math.pi / 2, 0.0, math.pi / 2, 0.0], dtype=np.float64),
        np.array([azimuth, 0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0], dtype=np.float64),
        np.zeros(6, dtype=np.float64),
        np.array([0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, 0.0], dtype=np.float64),
    ]

    seeds = []
    seen = set()

    def add_seed(seed):
        clipped = np.clip(wrap_to_pi(seed), clipped_neg_pi, clipped_pi)
        key = tuple(np.round(clipped, 4).tolist())
        if key not in seen:
            seen.add(key)
            seeds.append(clipped)

    for seed in deterministic:
        add_seed(seed)

    for _ in range(num_random_seeds):
        add_seed(rng.uniform(clipped_neg_pi, clipped_pi))

    return seeds


def solve_one_pose(
    p,
    robot_id,
    robot_info,
    target_position,
    target_quaternion,
    base_seeds,
    carryover_seeds,
    max_solutions,
    max_random_retries,
    joint_tol,
    pos_tol,
    orn_tol_rad,
    max_iterations,
    residual_threshold,
    rng,
):
    joint_indices = robot_info["movable_joint_indices"]
    lower_limits = robot_info["lower_limits"]
    upper_limits = robot_info["upper_limits"]
    joint_ranges = robot_info["joint_ranges"]
    joint_types = robot_info["joint_types"]
    tcp_link_index = robot_info["tcp_link_index"]

    solutions = []
    position_errors = []
    orientation_errors = []
    seeds_to_try = list(carryover_seeds) + list(base_seeds)
    random_misses = 0

    while seeds_to_try or (len(solutions) < max_solutions and random_misses < max_random_retries):
        if seeds_to_try:
            seed = seeds_to_try.pop(0)
        else:
            clipped_pi = np.minimum(upper_limits, math.pi)
            clipped_neg_pi = np.maximum(lower_limits, -math.pi)
            seed = rng.uniform(clipped_neg_pi, clipped_pi)

        reset_arm_joints(p, robot_id, joint_indices, seed)
        raw_solution = np.asarray(
            p.calculateInverseKinematics(
                robot_id,
                tcp_link_index,
                target_position.tolist(),
                targetOrientation=target_quaternion.tolist(),
                lowerLimits=lower_limits.tolist(),
                upperLimits=upper_limits.tolist(),
                jointRanges=joint_ranges.tolist(),
                restPoses=np.asarray(seed, dtype=np.float64).tolist(),
                maxNumIterations=max_iterations,
                residualThreshold=residual_threshold,
            ),
            dtype=np.float64,
        )
        candidate = canonicalize_solution(raw_solution[: len(joint_indices)], joint_types, lower_limits, upper_limits)
        reset_arm_joints(p, robot_id, joint_indices, candidate)
        actual_position, actual_quaternion = get_tcp_pose(p, robot_id, tcp_link_index)

        position_error = float(np.linalg.norm(actual_position - target_position))
        orientation_error = float(quaternion_angle_error(actual_quaternion, target_quaternion))
        valid = position_error <= pos_tol and orientation_error <= orn_tol_rad

        if valid and not solution_is_duplicate(candidate, solutions, joint_tol):
            solutions.append(candidate)
            position_errors.append(position_error)
            orientation_errors.append(orientation_error)
            random_misses = 0
            if len(solutions) >= max_solutions:
                break
        elif not seeds_to_try:
            random_misses += 1

    return solutions, position_errors, orientation_errors


def save_results(output_path, input_sample, robot_info, payload):
    save_kwargs = {
        "source_tcp_npz": np.array(str(input_sample["path"]), dtype=object),
        "joint_names": np.asarray(robot_info["movable_joint_names"], dtype=object),
        "joint_lower_limits": robot_info["lower_limits"].astype(np.float32),
        "joint_upper_limits": robot_info["upper_limits"].astype(np.float32),
        "tcp_points": input_sample["tcp_points"].astype(np.float32),
        "orientation_quaternions_xyzw": input_sample["orientation_quaternions_xyzw"].astype(np.float32),
        "pose_tcp_index": payload["pose_tcp_index"].astype(np.int32),
        "pose_orientation_index": payload["pose_orientation_index"].astype(np.int32),
        "pose_target_positions": payload["pose_target_positions"].astype(np.float32),
        "pose_target_quaternions_xyzw": payload["pose_target_quaternions_xyzw"].astype(np.float32),
        "pose_solution_count": payload["pose_solution_count"].astype(np.int32),
        "ik_solutions": payload["ik_solutions"].astype(np.float32),
        "ik_solution_pose_index": payload["ik_solution_pose_index"].astype(np.int32),
        "ik_solution_tcp_index": payload["ik_solution_tcp_index"].astype(np.int32),
        "ik_solution_orientation_index": payload["ik_solution_orientation_index"].astype(np.int32),
        "ik_position_error": payload["ik_position_error"].astype(np.float32),
        "ik_orientation_error_rad": payload["ik_orientation_error_rad"].astype(np.float32),
        "num_tcp_points": np.array(payload["num_tcp_points"], dtype=np.int32),
        "num_orientations": np.array(payload["num_orientations"], dtype=np.int32),
        "num_poses": np.array(payload["num_poses"], dtype=np.int32),
        "num_solved_poses": np.array(payload["num_solved_poses"], dtype=np.int32),
        "num_failed_poses": np.array(payload["num_failed_poses"], dtype=np.int32),
    }
    if input_sample["band"] is not None:
        save_kwargs["tcp_band"] = input_sample["band"]
    if input_sample["orientation_matrices"] is not None:
        save_kwargs["orientation_matrices"] = input_sample["orientation_matrices"].astype(np.float32)
    np.savez_compressed(output_path, **save_kwargs)


def main():
    parser = argparse.ArgumentParser("solve_tcp_ik_from_samples")
    parser.add_argument("--tcp-samples", type=str, required=True, help="TCP sample npz from sample_tcp_points_from_workpiece.py")
    parser.add_argument("--output", type=str, required=True, help="Output npz path for IK solutions")
    parser.add_argument("--urdf", type=str, default="config/robot-model/ur5e_with_pen.urdf", help="Robot URDF path")
    parser.add_argument("--tcp-link", type=str, default="tool0", help="Target TCP link name in the URDF")
    parser.add_argument("--max-points", type=int, default=None, help="Only solve the first N TCP points for debugging")
    parser.add_argument("--max-orientations", type=int, default=None, help="Only solve the first M orientations per TCP point")
    parser.add_argument("--max-solutions", type=int, default=8, help="Stop searching once this many unique IK branches are found")
    parser.add_argument("--num-random-seeds", type=int, default=6, help="Random seeds added to the deterministic IK seed bank")
    parser.add_argument("--max-random-retries", type=int, default=6, help="Extra random rest-pose retries after the seed bank is exhausted")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for IK multi-start")
    parser.add_argument("--joint-tol", type=float, default=0.15, help="Solutions closer than this joint-angle threshold are merged")
    parser.add_argument("--pos-tol", type=float, default=0.005, help="Maximum TCP position error in meters")
    parser.add_argument("--orn-tol-deg", type=float, default=5.0, help="Maximum TCP orientation error in degrees")
    parser.add_argument("--max-iterations", type=int, default=240, help="PyBullet IK max iterations per solve")
    parser.add_argument("--residual-threshold", type=float, default=1e-5, help="PyBullet IK residual threshold")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every K poses")
    args = parser.parse_args()

    p = import_pybullet()
    sample = load_tcp_sample_file(args.tcp_samples)

    tcp_points = sample["tcp_points"]
    quats = sample["orientation_quaternions_xyzw"]
    if args.max_points is not None:
        tcp_points = tcp_points[: args.max_points]
        quats = quats[: args.max_points]
        if sample["orientation_matrices"] is not None:
            sample["orientation_matrices"] = sample["orientation_matrices"][: args.max_points]
        if sample["band"] is not None:
            sample["band"] = sample["band"][: args.max_points]
    if args.max_orientations is not None:
        quats = quats[:, : args.max_orientations]
        if sample["orientation_matrices"] is not None:
            sample["orientation_matrices"] = sample["orientation_matrices"][:, : args.max_orientations]
    sample["tcp_points"] = tcp_points
    sample["orientation_quaternions_xyzw"] = quats

    temp_urdf = make_pybullet_ready_urdf(args.urdf)
    client = p.connect(p.DIRECT)
    try:
        robot_id = p.loadURDF(str(temp_urdf), useFixedBase=True)
        robot_info = extract_robot_info(p, robot_id, args.tcp_link)
        orn_tol_rad = math.radians(args.orn_tol_deg)
        rng = np.random.default_rng(args.seed)

        pose_tcp_index = []
        pose_orientation_index = []
        pose_target_positions = []
        pose_target_quaternions = []
        pose_solution_count = []

        ik_solutions = []
        ik_solution_pose_index = []
        ik_solution_tcp_index = []
        ik_solution_orientation_index = []
        ik_position_error = []
        ik_orientation_error = []

        carryover_seeds = []
        pose_index = 0
        solved_pose_count = 0
        total_pose_count = int(tcp_points.shape[0] * quats.shape[1])

        for tcp_index in range(tcp_points.shape[0]):
            point = np.asarray(tcp_points[tcp_index], dtype=np.float64)
            point_carryover = list(carryover_seeds)
            for orientation_index in range(quats.shape[1]):
                quat = normalize_quaternion_xyzw(quats[tcp_index, orientation_index])
                base_seeds = build_seed_bank(
                    target_position=point,
                    lower_limits=robot_info["lower_limits"],
                    upper_limits=robot_info["upper_limits"],
                    num_random_seeds=args.num_random_seeds,
                    rng=rng,
                )
                solutions, pos_errors, orn_errors = solve_one_pose(
                    p=p,
                    robot_id=robot_id,
                    robot_info=robot_info,
                    target_position=point,
                    target_quaternion=quat,
                    base_seeds=base_seeds,
                    carryover_seeds=point_carryover,
                    max_solutions=args.max_solutions,
                    max_random_retries=args.max_random_retries,
                    joint_tol=args.joint_tol,
                    pos_tol=args.pos_tol,
                    orn_tol_rad=orn_tol_rad,
                    max_iterations=args.max_iterations,
                    residual_threshold=args.residual_threshold,
                    rng=rng,
                )

                pose_tcp_index.append(tcp_index)
                pose_orientation_index.append(orientation_index)
                pose_target_positions.append(point)
                pose_target_quaternions.append(quat)
                pose_solution_count.append(len(solutions))

                if solutions:
                    solved_pose_count += 1
                    point_carryover = [np.asarray(solution, dtype=np.float64) for solution in solutions]
                    for solution, pos_error, orn_error in zip(solutions, pos_errors, orn_errors):
                        ik_solutions.append(solution)
                        ik_solution_pose_index.append(pose_index)
                        ik_solution_tcp_index.append(tcp_index)
                        ik_solution_orientation_index.append(orientation_index)
                        ik_position_error.append(pos_error)
                        ik_orientation_error.append(orn_error)

                pose_index += 1
                if args.progress_every > 0 and pose_index % args.progress_every == 0:
                    print(
                        f"Processed poses: {pose_index}/{total_pose_count} | "
                        f"solved: {solved_pose_count} | "
                        f"flattened IK solutions: {len(ik_solutions)}"
                    )

            carryover_seeds = point_carryover

        payload = {
            "pose_tcp_index": np.asarray(pose_tcp_index, dtype=np.int32),
            "pose_orientation_index": np.asarray(pose_orientation_index, dtype=np.int32),
            "pose_target_positions": np.asarray(pose_target_positions, dtype=np.float64),
            "pose_target_quaternions_xyzw": np.asarray(pose_target_quaternions, dtype=np.float64),
            "pose_solution_count": np.asarray(pose_solution_count, dtype=np.int32),
            "ik_solutions": np.asarray(ik_solutions, dtype=np.float64).reshape(-1, len(robot_info["movable_joint_indices"])),
            "ik_solution_pose_index": np.asarray(ik_solution_pose_index, dtype=np.int32),
            "ik_solution_tcp_index": np.asarray(ik_solution_tcp_index, dtype=np.int32),
            "ik_solution_orientation_index": np.asarray(ik_solution_orientation_index, dtype=np.int32),
            "ik_position_error": np.asarray(ik_position_error, dtype=np.float64),
            "ik_orientation_error_rad": np.asarray(ik_orientation_error, dtype=np.float64),
            "num_tcp_points": int(tcp_points.shape[0]),
            "num_orientations": int(quats.shape[1]),
            "num_poses": int(pose_index),
            "num_solved_poses": int(solved_pose_count),
            "num_failed_poses": int(pose_index - solved_pose_count),
        }

        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_results(output_path, sample, robot_info, payload)

        print(f"Saved IK solutions to: {output_path}")
        print(f"TCP points processed: {payload['num_tcp_points']}")
        print(f"Orientations per point: {payload['num_orientations']}")
        print(f"Target poses: {payload['num_poses']}")
        print(f"Solved poses: {payload['num_solved_poses']}")
        print(f"Failed poses: {payload['num_failed_poses']}")
        print(f"Flattened IK solutions: {len(payload['ik_solutions'])}")
        print(f"Joint names: {robot_info['movable_joint_names']}")
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    main()
