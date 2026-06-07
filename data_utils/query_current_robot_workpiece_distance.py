import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    from .numpy_npz_compat import install_numpy_pickle_compat
except ImportError:
    from numpy_npz_compat import install_numpy_pickle_compat

install_numpy_pickle_compat()

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from robot_surface_sampler import (
    convert_link_local_sample_to_world,
    ensure_local_surface_points,
    generate_current_robot_surface_points,
    load_sample_dict,
    parse_joint_values,
    sample_surface_points_per_link,
    URDFSurfaceSampler,
)


def load_robot_points(path):
    sample = load_sample_dict(path)
    return sample


def load_sdf_grid(path):
    data = np.load(Path(path).resolve(), allow_pickle=True)
    if "sdf" not in data:
        raise ValueError("SDF npz must contain 'sdf'.")
    sdf = np.asarray(data["sdf"], dtype=np.float64)
    if sdf.ndim != 3:
        raise ValueError(f"Expected 3D sdf grid, got shape {sdf.shape}.")

    if all(k in data for k in ("x", "y", "z")):
        x = np.asarray(data["x"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.float64)
        z = np.asarray(data["z"], dtype=np.float64)
        return sdf, (x, y, z)

    if all(k in data for k in ("origin", "voxel_size")):
        origin = np.asarray(data["origin"], dtype=np.float64)
        voxel_size = np.asarray(data["voxel_size"], dtype=np.float64)
        if voxel_size.ndim == 0:
            voxel_size = np.repeat(voxel_size.item(), 3)
        x = origin[0] + np.arange(sdf.shape[0], dtype=np.float64) * voxel_size[0]
        y = origin[1] + np.arange(sdf.shape[1], dtype=np.float64) * voxel_size[1]
        z = origin[2] + np.arange(sdf.shape[2], dtype=np.float64) * voxel_size[2]
        return sdf, (x, y, z)

    if all(k in data for k in ("min_bound", "max_bound")):
        min_bound = np.asarray(data["min_bound"], dtype=np.float64)
        max_bound = np.asarray(data["max_bound"], dtype=np.float64)
        x = np.linspace(min_bound[0], max_bound[0], sdf.shape[0], dtype=np.float64)
        y = np.linspace(min_bound[1], max_bound[1], sdf.shape[1], dtype=np.float64)
        z = np.linspace(min_bound[2], max_bound[2], sdf.shape[2], dtype=np.float64)
        return sdf, (x, y, z)

    raise ValueError("Unsupported sdf npz format.")


def clamp_points_to_bounds(points, axes):
    x, y, z = axes
    lower = np.array([x[0], y[0], z[0]], dtype=np.float64)
    upper = np.array([x[-1], y[-1], z[-1]], dtype=np.float64)
    clamped = points.copy()
    clamped[:, 0] = np.clip(clamped[:, 0], lower[0], upper[0])
    clamped[:, 1] = np.clip(clamped[:, 1], lower[1], upper[1])
    clamped[:, 2] = np.clip(clamped[:, 2], lower[2], upper[2])
    outside_offset = np.linalg.norm(points - clamped, axis=1)
    inside_mask = np.all(clamped == points, axis=1)
    return clamped, outside_offset, inside_mask


def build_interpolator(sdf, axes):
    return RegularGridInterpolator(
        axes,
        sdf,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def evaluate_robot_points_distance(points, interpolator, axes, outside_mode="project"):
    query_points, outside_offset, inside_mask = clamp_points_to_bounds(points, axes)
    if outside_mode == "error" and not np.all(inside_mask):
        raise ValueError(f"{np.sum(~inside_mask)} robot points are outside the sdf volume.")

    sdf_values = interpolator(query_points)
    if outside_mode == "project":
        sdf_values = sdf_values + outside_offset
        valid_mask = np.isfinite(sdf_values)
    else:
        valid_mask = np.isfinite(sdf_values) & inside_mask

    if not np.any(valid_mask):
        return {
            "valid_mask": valid_mask,
            "inside_mask": inside_mask,
            "sdf_values": sdf_values,
            "valid_idx": np.empty((0,), dtype=np.int64),
            "valid_sdf": np.empty((0,), dtype=np.float64),
            "min_signed_idx": None,
            "min_abs_idx": None,
            "min_signed_distance": np.nan,
            "min_abs_distance": np.nan,
        }

    valid_idx = np.where(valid_mask)[0]
    valid_sdf = sdf_values[valid_mask]
    min_signed_local = int(np.argmin(valid_sdf))
    min_abs_local = int(np.argmin(np.abs(valid_sdf)))
    return {
        "valid_mask": valid_mask,
        "inside_mask": inside_mask,
        "sdf_values": sdf_values,
        "valid_idx": valid_idx,
        "valid_sdf": valid_sdf,
        "min_signed_idx": int(valid_idx[min_signed_local]),
        "min_abs_idx": int(valid_idx[min_abs_local]),
        "min_signed_distance": float(valid_sdf[min_signed_local]),
        "min_abs_distance": float(np.abs(valid_sdf[min_abs_local])),
    }


def joint_dict_from_solution(joint_names, joint_values):
    return {str(name): float(value) for name, value in zip(joint_names, joint_values)}


def movable_joint_names_from_sampler(sampler):
    return [
        joint.name
        for joint in sampler.joints.values()
        if joint.joint_type in ("revolute", "continuous", "prismatic")
    ]


def load_or_sample_local_surface_points(
    sampler,
    local_points_path,
    num_points,
    min_points_per_link,
    include_links=None,
    seed=0,
    force_resample=False,
):
    local_points_path = Path(local_points_path).resolve()
    if local_points_path.exists() and not force_resample:
        sample = load_sample_dict(local_points_path)
        if sample["frame"] != "link_local":
            raise ValueError(
                f"Existing local points file {local_points_path} has frame={sample['frame']}, expected link_local."
            )
        return sample

    link_meshes = sampler.build_link_local_meshes(include_links=include_links)
    sample = sample_surface_points_per_link(
        link_meshes,
        count=num_points,
        min_points_per_link=min_points_per_link,
        seed=seed,
    )
    sample["frame"] = "link_local"
    return sample


class RobotWorkpieceDistanceQuery:
    """Reusable in-memory robot surface to workpiece SDF distance query."""

    def __init__(
        self,
        urdf_path,
        local_points_path,
        num_points=8192,
        min_points_per_link=32,
        include_links=None,
        seed=0,
        force_resample=False,
    ):
        self.sampler = URDFSurfaceSampler(urdf_path)
        self.joint_names = movable_joint_names_from_sampler(self.sampler)
        self.local_sample = load_or_sample_local_surface_points(
            sampler=self.sampler,
            local_points_path=local_points_path,
            num_points=num_points,
            min_points_per_link=min_points_per_link,
            include_links=include_links,
            seed=seed,
            force_resample=force_resample,
        )
        self._sdf_cache = {}

    def _get_sdf_query(self, sdf_path):
        sdf_path = Path(sdf_path).resolve()
        if sdf_path not in self._sdf_cache:
            sdf, axes = load_sdf_grid(sdf_path)
            self._sdf_cache[sdf_path] = (build_interpolator(sdf, axes), axes)
        return self._sdf_cache[sdf_path]

    def query_joint_values(
        self,
        joint_values,
        sdf_path,
        joint_names=None,
        outside_mode="project",
        progress_every=0,
    ):
        joint_values = np.asarray(joint_values, dtype=np.float64)
        if joint_values.ndim != 2:
            raise ValueError(f"joint_values must have shape [N, D], got {joint_values.shape}")

        joint_names = list(self.joint_names if joint_names is None else joint_names)
        if joint_values.shape[1] != len(joint_names):
            raise ValueError(
                f"joint_values has {joint_values.shape[1]} columns but {len(joint_names)} joint names were provided."
            )
        unknown = sorted(set(joint_names) - set(self.joint_names))
        if unknown:
            raise ValueError(f"Joint names are not movable joints in the URDF: {unknown}")

        interpolator, axes = self._get_sdf_query(sdf_path)
        count = len(joint_values)
        min_signed_distance = np.full(count, np.nan, dtype=np.float32)
        min_abs_distance = np.full(count, np.nan, dtype=np.float32)
        valid_point_count = np.zeros(count, dtype=np.int32)
        penetrating_point_count = np.zeros(count, dtype=np.int32)

        for idx, solution in enumerate(joint_values):
            joint_dict = joint_dict_from_solution(joint_names, solution)
            robot = convert_link_local_sample_to_world(self.local_sample, self.sampler, joint_dict)
            result = query_single_robot_sample(robot, interpolator, axes, outside_mode)
            valid_point_count[idx] = int(len(result["valid_idx"]))
            penetrating_point_count[idx] = int(np.sum(result["valid_sdf"] < 0))
            min_signed_distance[idx] = result["min_signed_distance"]
            min_abs_distance[idx] = result["min_abs_distance"]

            if progress_every > 0 and (idx + 1) % progress_every == 0:
                print(f"distance_queries: {idx + 1}/{count}", flush=True)

        return {
            "min_signed_distance": min_signed_distance,
            "min_abs_distance": min_abs_distance,
            "valid_point_count": valid_point_count,
            "penetrating_point_count": penetrating_point_count,
        }


def query_single_robot_sample(robot, interpolator, axes, outside_mode):
    points = np.asarray(robot["points"], dtype=np.float64)
    link_name = np.asarray(robot["link_name"], dtype=object)
    result = evaluate_robot_points_distance(points, interpolator, axes, outside_mode=outside_mode)
    result["points"] = points
    result["link_name"] = link_name
    return result


def query_ik_solution_set(
    ik_path,
    sdf_path,
    urdf_path,
    local_points_path,
    num_points,
    min_points_per_link,
    include_links,
    seed,
    force_resample,
    outside_mode,
    output_path=None,
):
    ik_data = np.load(Path(ik_path).resolve(), allow_pickle=True)
    required = ("ik_solutions", "joint_names")
    missing = [name for name in required if name not in ik_data]
    if missing:
        raise ValueError(f"IK npz missing required fields: {missing}")

    joint_names = [str(name) for name in ik_data["joint_names"].tolist()]
    ik_solutions = np.asarray(ik_data["ik_solutions"], dtype=np.float64)
    pose_index = np.asarray(ik_data["ik_solution_pose_index"], dtype=np.int32) if "ik_solution_pose_index" in ik_data else None
    tcp_index = np.asarray(ik_data["ik_solution_tcp_index"], dtype=np.int32) if "ik_solution_tcp_index" in ik_data else None
    orientation_index = (
        np.asarray(ik_data["ik_solution_orientation_index"], dtype=np.int32)
        if "ik_solution_orientation_index" in ik_data
        else None
    )

    sampler = URDFSurfaceSampler(urdf_path)
    local_sample = ensure_local_surface_points(
        sampler=sampler,
        local_points_path=local_points_path,
        num_points=num_points,
        min_points_per_link=min_points_per_link,
        include_links=include_links,
        seed=seed,
        force_resample=force_resample,
    )
    sdf, axes = load_sdf_grid(sdf_path)
    interpolator = build_interpolator(sdf, axes)

    min_signed_distance = np.full(len(ik_solutions), np.nan, dtype=np.float32)
    min_abs_distance = np.full(len(ik_solutions), np.nan, dtype=np.float32)
    valid_point_count = np.zeros(len(ik_solutions), dtype=np.int32)
    inside_point_count = np.zeros(len(ik_solutions), dtype=np.int32)
    penetrating_point_count = np.zeros(len(ik_solutions), dtype=np.int32)
    min_signed_point = np.full((len(ik_solutions), 3), np.nan, dtype=np.float32)
    min_abs_point = np.full((len(ik_solutions), 3), np.nan, dtype=np.float32)
    min_signed_link = np.full(len(ik_solutions), "", dtype=object)
    min_abs_link = np.full(len(ik_solutions), "", dtype=object)

    for idx, solution in enumerate(ik_solutions):
        joint_values = joint_dict_from_solution(joint_names, solution)
        robot = convert_link_local_sample_to_world(local_sample, sampler, joint_values)
        result = query_single_robot_sample(robot, interpolator, axes, outside_mode)
        valid_point_count[idx] = int(len(result["valid_idx"]))
        inside_point_count[idx] = int(np.sum(result["inside_mask"]))
        penetrating_point_count[idx] = int(np.sum(result["valid_sdf"] < 0))
        min_signed_distance[idx] = result["min_signed_distance"]
        min_abs_distance[idx] = result["min_abs_distance"]

        if result["min_signed_idx"] is not None:
            point_idx = int(result["min_signed_idx"])
            min_signed_point[idx] = result["points"][point_idx].astype(np.float32)
            min_signed_link[idx] = str(result["link_name"][point_idx])
        if result["min_abs_idx"] is not None:
            point_idx = int(result["min_abs_idx"])
            min_abs_point[idx] = result["points"][point_idx].astype(np.float32)
            min_abs_link[idx] = str(result["link_name"][point_idx])

    payload = {
        "source_ik_npz": np.array(str(Path(ik_path).resolve()), dtype=object),
        "source_sdf_npz": np.array(str(Path(sdf_path).resolve()), dtype=object),
        "joint_names": np.asarray(joint_names, dtype=object),
        "ik_solutions": ik_solutions.astype(np.float32),
        "min_signed_distance": min_signed_distance,
        "min_abs_distance": min_abs_distance,
        "valid_point_count": valid_point_count,
        "inside_point_count": inside_point_count,
        "penetrating_point_count": penetrating_point_count,
        "min_signed_point": min_signed_point,
        "min_abs_point": min_abs_point,
        "min_signed_link": min_signed_link,
        "min_abs_link": min_abs_link,
    }
    if pose_index is not None:
        payload["ik_solution_pose_index"] = pose_index
    if tcp_index is not None:
        payload["ik_solution_tcp_index"] = tcp_index
    if orientation_index is not None:
        payload["ik_solution_orientation_index"] = orientation_index

    if output_path is not None:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **payload)

    return payload


def main():
    parser = argparse.ArgumentParser("query_current_robot_workpiece_distance")
    parser.add_argument("--sdf", type=str, required=True, help="Workpiece sdf .npz file")
    parser.add_argument("--robot-points", type=str, default=None, help="Current robot sampled points .npz file")
    parser.add_argument("--ik-solutions", type=str, default=None, help="IK solution npz from solve_tcp_ik_from_samples.py")
    parser.add_argument("--ik-distance-output", type=str, default=None, help="Optional output npz for per-IK distance results")
    parser.add_argument("--urdf", type=str, default="config/robot-model/ur5e_with_pen.urdf", help="URDF path")
    parser.add_argument("--joint-values", type=str, default=None, help="Joint values for current robot pose")
    parser.add_argument(
        "--local-points",
        type=str,
        default="config/robot-model/ur5e_surface_points_local.npz",
        help="Cached link-local sampled points .npz",
    )
    parser.add_argument(
        "--current-points-output",
        type=str,
        default="/private/tmp/ur5e_surface_points_current_for_distance.npz",
        help="Output .npz path when generating current world-frame robot points from joints",
    )
    parser.add_argument("--num-points", type=int, default=8192, help="Sample count used when creating local cache")
    parser.add_argument(
        "--min-points-per-link",
        type=int,
        default=32,
        help="Minimum sampled points per link when creating local cache",
    )
    parser.add_argument("--include-links", type=str, default=None, help="Comma-separated subset of links to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for cache generation")
    parser.add_argument(
        "--force-resample",
        action="store_true",
        default=False,
        help="Regenerate the local sampled cache even if it already exists",
    )
    parser.add_argument(
        "--outside-mode",
        type=str,
        default="project",
        choices=["ignore", "project", "error"],
        help="How to handle robot points outside the sdf volume",
    )
    args = parser.parse_args()

    include_links = args.include_links.split(",") if args.include_links else None
    if args.ik_solutions is not None:
        payload = query_ik_solution_set(
            ik_path=args.ik_solutions,
            sdf_path=args.sdf,
            urdf_path=args.urdf,
            local_points_path=args.local_points,
            num_points=args.num_points,
            min_points_per_link=args.min_points_per_link,
            include_links=include_links,
            seed=args.seed,
            force_resample=args.force_resample,
            outside_mode=args.outside_mode,
            output_path=args.ik_distance_output,
        )
        finite_mask = np.isfinite(payload["min_signed_distance"])
        print(f"IK solutions evaluated: {len(payload['ik_solutions'])}")
        print(f"Solutions with valid sdf points: {int(np.sum(finite_mask))}")
        if args.ik_distance_output is not None:
            print(f"IK distance results saved to: {Path(args.ik_distance_output).resolve()}")
        if np.any(finite_mask):
            best_idx = int(np.nanargmin(payload["min_signed_distance"]))
            print(
                "Global minimum signed distance: "
                f"{float(payload['min_signed_distance'][best_idx]):.6f} "
                f"at solution {best_idx}, point {payload['min_signed_point'][best_idx].tolist()}, "
                f"link {payload['min_signed_link'][best_idx]}"
            )
            print(
                "Corresponding absolute distance: "
                f"{float(payload['min_abs_distance'][best_idx]):.6f}"
            )
        return

    if args.robot_points is not None:
        robot = load_robot_points(args.robot_points)
        if robot["frame"] == "link_local":
            sampler = URDFSurfaceSampler(args.urdf)
            joint_values = parse_joint_values(args.joint_values, sampler)
            robot = convert_link_local_sample_to_world(robot, sampler, joint_values)
    else:
        robot = generate_current_robot_surface_points(
            urdf_path=args.urdf,
            joint_values_text=args.joint_values,
            output_path=args.current_points_output,
            local_points_path=args.local_points,
            num_points=args.num_points,
            min_points_per_link=args.min_points_per_link,
            include_links=include_links,
            seed=args.seed,
            force_resample=args.force_resample,
        )

    points = np.asarray(robot["points"], dtype=np.float64)
    sdf, axes = load_sdf_grid(args.sdf)
    interpolator = build_interpolator(sdf, axes)
    result = query_single_robot_sample(robot, interpolator, axes, args.outside_mode)
    if len(result["valid_idx"]) == 0:
        raise ValueError("No robot points could be evaluated against the sdf grid.")

    if args.robot_points is None:
        print(f"Current robot points saved to: {Path(args.current_points_output).resolve()}")
    print(f"Evaluated points: {len(result['valid_idx'])} / {len(points)}")
    print(f"Outside points: {int(np.sum(~result['inside_mask']))}")
    if args.outside_mode == "project":
        print("Outside handling: projected to sdf volume boundary with added outside offset")
    print(f"Penetrating points (sdf < 0): {int(np.sum(result['valid_sdf'] < 0))}")
    print(
        "Minimum signed distance: "
        f"{float(result['min_signed_distance']):.6f} "
        f"at point {points[result['min_signed_idx']].tolist()}"
        + f" on link {robot['link_name'][result['min_signed_idx']]}"
    )
    print(
        "Minimum absolute distance: "
        f"{float(result['min_abs_distance']):.6f} "
        f"(signed {float(result['sdf_values'][result['min_abs_idx']]):.6f}) "
        f"at point {points[result['min_abs_idx']].tolist()}"
        + f" on link {robot['link_name'][result['min_abs_idx']]}"
    )


if __name__ == "__main__":
    main()
