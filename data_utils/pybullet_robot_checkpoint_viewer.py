import argparse
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from robot_surface_sampler import URDFSurfaceSampler, parse_joint_values


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


def parse_checkpoint_string(text):
    points = []
    labels = []
    for idx, item in enumerate(text.split(";")):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            label, coords = item.split(":", 1)
            labels.append(label.strip())
        else:
            coords = item
            labels.append(f"checkpoint_{idx}")
        points.append([float(x) for x in coords.split(",")])
    if not points:
        raise ValueError("No checkpoint coordinates were parsed.")
    return np.asarray(points, dtype=np.float64), labels


def load_checkpoint_file(path):
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    labels = None

    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "points" in data:
            points = data["points"]
        elif "checkpoints" in data:
            points = data["checkpoints"]
        else:
            raise ValueError(f"No supported point array found in {path}")
        if "labels" in data:
            labels = [str(x) for x in data["labels"].tolist()]
    else:
        points = np.loadtxt(path, delimiter=None)

    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] != 3:
        raise ValueError("Checkpoint file must contain Nx3 coordinates.")

    if labels is None:
        labels = [f"checkpoint_{i}" for i in range(len(points))]
    return points, labels


def load_sampled_points_file(path):
    path = Path(path).resolve()
    data = np.load(path, allow_pickle=True)
    if "points" not in data:
        raise ValueError(f"No 'points' array found in sampled points file: {path}")

    points = np.asarray(data["points"], dtype=np.float64)
    link_names = None
    if "link_name" in data:
        link_names = [str(x) for x in data["link_name"].tolist()]
    return points, link_names


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


def color_for_index(index):
    palette = [
        [0.89, 0.34, 0.18, 0.95],
        [0.12, 0.47, 0.71, 0.95],
        [0.17, 0.63, 0.17, 0.95],
        [0.58, 0.40, 0.74, 0.95],
        [1.00, 0.50, 0.05, 0.95],
    ]
    return palette[index % len(palette)]


def build_point_colors(link_names):
    if link_names is None:
        return None
    unique_names = sorted(set(link_names))
    color_map = {name: color_for_index(i)[:3] for i, name in enumerate(unique_names)}
    return np.asarray([color_map[name] for name in link_names], dtype=np.float64)


def add_checkpoint_marker(p, point, label, radius, color):
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, basePosition=point.tolist())
    text_id = p.addUserDebugText(label, point.tolist(), textColorRGB=color[:3], textSize=1.3)
    return body, text_id


def update_checkpoint_marker(p, body_id, text_id, point, label, color):
    p.resetBasePositionAndOrientation(body_id, point.tolist(), [0, 0, 0, 1])
    p.removeUserDebugItem(text_id)
    new_text = p.addUserDebugText(label, point.tolist(), textColorRGB=color[:3], textSize=1.3)
    return new_text


def add_frame_axes(p, origin, rotation, axis_length, replace_ids=None):
    colors = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    ids = []
    for i, color in enumerate(colors):
        end = origin + rotation[:, i] * axis_length
        replace = replace_ids[i] if replace_ids is not None else -1
        ids.append(
            p.addUserDebugLine(
                origin.tolist(),
                end.tolist(),
                color,
                lineWidth=3.0,
                replaceItemUniqueId=replace,
            )
        )
    return ids


def main():
    parser = argparse.ArgumentParser("pybullet_robot_checkpoint_viewer")
    parser.add_argument(
        "--urdf",
        type=str,
        default="config/robot-model/ur5e_with_pen.urdf",
        help="URDF path",
    )
    parser.add_argument(
        "--joint-values",
        type=str,
        default=None,
        help="Either comma-separated values in movable-joint order, or name=value pairs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="goal:0.45,0.10,0.25",
        help='Single or multiple checkpoints, e.g. "0.4,0.1,0.2;goal:0.5,0.0,0.3"',
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Path to .txt/.npy/.npz containing checkpoint coordinates",
    )
    parser.add_argument(
        "--sampled-points-file",
        type=str,
        default=None,
        help="Path to .npz exported by robot_surface_sampler.py",
    )
    parser.add_argument(
        "--sampled-point-size",
        type=int,
        default=4,
        help="PyBullet debug point size for sampled point cloud",
    )
    parser.add_argument(
        "--sampled-point-limit",
        type=int,
        default=12000,
        help="Maximum number of sampled points to render in PyBullet",
    )
    parser.add_argument("--checkpoint-radius", type=float, default=0.018, help="Checkpoint sphere radius")
    parser.add_argument("--axis-length", type=float, default=0.08, help="Coordinate axis length")
    parser.add_argument(
        "--connection-mode",
        type=str,
        default="gui",
        choices=["gui", "direct"],
        help="PyBullet connection mode",
    )
    parser.add_argument("--camera-distance", type=float, default=1.35, help="Initial camera distance")
    parser.add_argument("--camera-yaw", type=float, default=50.0, help="Initial camera yaw")
    parser.add_argument("--camera-pitch", type=float, default=-30.0, help="Initial camera pitch")
    parser.add_argument(
        "--camera-target",
        type=str,
        default="0.35,0.0,0.25",
        help="Initial camera target x,y,z",
    )
    parser.add_argument("--dt", type=float, default=1.0 / 120.0, help="GUI update interval")
    args = parser.parse_args()

    p = import_pybullet()
    sampler = URDFSurfaceSampler(args.urdf)
    joint_values = parse_joint_values(args.joint_values, sampler)

    checkpoint_points = None
    checkpoint_labels = None
    if args.checkpoint_file is not None:
        checkpoint_points, checkpoint_labels = load_checkpoint_file(args.checkpoint_file)
    elif args.checkpoint is not None:
        checkpoint_points, checkpoint_labels = parse_checkpoint_string(args.checkpoint)

    sampled_points = None
    sampled_point_colors = None
    if args.sampled_points_file is not None:
        sampled_points, sampled_link_names = load_sampled_points_file(args.sampled_points_file)
        if len(sampled_points) > args.sampled_point_limit:
            keep = np.linspace(0, len(sampled_points) - 1, args.sampled_point_limit, dtype=np.int64)
            sampled_points = sampled_points[keep]
            if sampled_link_names is not None:
                sampled_link_names = [sampled_link_names[i] for i in keep.tolist()]
        sampled_point_colors = build_point_colors(sampled_link_names)

    temp_urdf = make_pybullet_ready_urdf(args.urdf)

    client = p.connect(p.GUI if args.connection_mode == "gui" else p.DIRECT)
    if args.connection_mode == "gui":
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    p.setGravity(0, 0, 0)

    camera_target = [float(x) for x in args.camera_target.split(",")]
    if args.connection_mode == "gui":
        p.resetDebugVisualizerCamera(
            cameraDistance=args.camera_distance,
            cameraYaw=args.camera_yaw,
            cameraPitch=args.camera_pitch,
            cameraTargetPosition=camera_target,
        )

    robot_id = p.loadURDF(str(temp_urdf), useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

    name_to_joint_idx = {}
    movable_joint_info = []
    name_to_link_idx = {"base_link": -1}

    for joint_idx in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, joint_idx)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]
        link_name = info[12].decode("utf-8")
        name_to_link_idx[link_name] = joint_idx
        name_to_joint_idx[joint_name] = joint_idx
        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            lower = info[8]
            upper = info[9]
            if lower > upper:
                lower, upper = -3.14, 3.14
            initial = joint_values.get(joint_name, 0.0)
            p.resetJointState(robot_id, joint_idx, initial)
            movable_joint_info.append((joint_name, joint_idx, lower, upper, initial))

    if "tool0" not in name_to_link_idx or "ee_link" not in name_to_link_idx:
        raise ValueError("URDF must contain tool0 and ee_link links for relative pose visualization.")

    joint_sliders = {}
    if args.connection_mode == "gui":
        for joint_name, joint_idx, lower, upper, initial in movable_joint_info:
            joint_sliders[joint_name] = p.addUserDebugParameter(joint_name, lower, upper, initial)

    if sampled_points is not None:
        if sampled_point_colors is None:
            sampled_point_colors = np.tile(np.array([[0.85, 0.85, 0.9]], dtype=np.float64), (len(sampled_points), 1))
        p.addUserDebugPoints(
            sampled_points.tolist(),
            sampled_point_colors.tolist(),
            pointSize=args.sampled_point_size,
        )

    checkpoint_bodies = []
    checkpoint_text_ids = []
    checkpoint_colors = []
    if checkpoint_points is not None:
        for i, (point, label) in enumerate(zip(checkpoint_points, checkpoint_labels)):
            color = color_for_index(i)
            body, text_id = add_checkpoint_marker(p, point, label, args.checkpoint_radius, color)
            checkpoint_bodies.append(body)
            checkpoint_text_ids.append(text_id)
            checkpoint_colors.append(color)

    checkpoint_sliders = None
    if args.connection_mode == "gui" and checkpoint_points is not None and len(checkpoint_points) == 1:
        point = checkpoint_points[0]
        checkpoint_sliders = {
            "x": p.addUserDebugParameter("checkpoint_x", -1.5, 1.5, point[0]),
            "y": p.addUserDebugParameter("checkpoint_y", -1.5, 1.5, point[1]),
            "z": p.addUserDebugParameter("checkpoint_z", -0.2, 1.5, point[2]),
        }

    status_text_id = -1
    line_ids = [-1] * (0 if checkpoint_points is None else len(checkpoint_points))
    tool0_axes = None
    base_axes = None

    print(f"PyBullet {args.connection_mode.upper()} launched.")
    print(f"Use this interpreter to reopen later: {PYBULLET_PYTHON}")

    while p.isConnected(client):
        if args.connection_mode == "gui":
            for joint_name, slider_id in joint_sliders.items():
                value = p.readUserDebugParameter(slider_id)
                p.resetJointState(robot_id, name_to_joint_idx[joint_name], value)

        if checkpoint_sliders is not None:
            checkpoint_points[0, 0] = p.readUserDebugParameter(checkpoint_sliders["x"])
            checkpoint_points[0, 1] = p.readUserDebugParameter(checkpoint_sliders["y"])
            checkpoint_points[0, 2] = p.readUserDebugParameter(checkpoint_sliders["z"])

        if checkpoint_points is not None:
            for i, (point, label, body_id, text_id, color) in enumerate(
                zip(checkpoint_points, checkpoint_labels, checkpoint_bodies, checkpoint_text_ids, checkpoint_colors)
            ):
                checkpoint_text_ids[i] = update_checkpoint_marker(p, body_id, text_id, point, label, color)

        tool0_state = p.getLinkState(robot_id, name_to_link_idx["tool0"], computeForwardKinematics=True)
        ee_state = p.getLinkState(robot_id, name_to_link_idx["ee_link"], computeForwardKinematics=True)
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)

        tool0_pos = np.asarray(tool0_state[4], dtype=np.float64)
        tool0_rot = np.asarray(p.getMatrixFromQuaternion(tool0_state[5]), dtype=np.float64).reshape(3, 3)
        ee_pos = np.asarray(ee_state[4], dtype=np.float64)
        base_pos = np.asarray(base_pos, dtype=np.float64)
        base_rot = np.asarray(p.getMatrixFromQuaternion(base_orn), dtype=np.float64).reshape(3, 3)

        tool0_axes = add_frame_axes(p, tool0_pos, tool0_rot, args.axis_length, replace_ids=tool0_axes)
        base_axes = add_frame_axes(p, base_pos, base_rot, args.axis_length, replace_ids=base_axes)

        summary_lines = [
            f"tool0: [{tool0_pos[0]:.4f}, {tool0_pos[1]:.4f}, {tool0_pos[2]:.4f}]",
            f"ee_link: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]",
        ]

        if checkpoint_points is not None:
            for i, (point, label, color) in enumerate(zip(checkpoint_points, checkpoint_labels, checkpoint_colors)):
                delta_tool0 = point - tool0_pos
                line_ids[i] = p.addUserDebugLine(
                    tool0_pos.tolist(),
                    point.tolist(),
                    color[:3],
                    lineWidth=2.5,
                    replaceItemUniqueId=line_ids[i],
                )
                summary_lines.append(
                    f"{label}: d_tool0=[{delta_tool0[0]:.4f}, {delta_tool0[1]:.4f}, {delta_tool0[2]:.4f}] "
                    f"| dist={np.linalg.norm(delta_tool0):.4f}"
                )

        status_text = "\n".join(summary_lines)
        if args.connection_mode == "gui":
            status_text_id = p.addUserDebugText(
                status_text,
                [0.02, -0.55, 0.95],
                textColorRGB=[1, 1, 1],
                textSize=1.2,
                replaceItemUniqueId=status_text_id,
                parentObjectUniqueId=robot_id,
                parentLinkIndex=-1,
            )
        else:
            print(status_text)
            break

        time.sleep(args.dt)


if __name__ == "__main__":
    main()
