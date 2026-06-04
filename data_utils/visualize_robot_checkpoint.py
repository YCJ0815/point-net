import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from robot_surface_sampler import URDFSurfaceSampler, parse_joint_values, sample_surface_points


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


def make_axes_trace(origin, rotation, axis_length, name_prefix):
    colors = ["red", "green", "blue"]
    axis_names = ["x", "y", "z"]
    traces = []
    for i in range(3):
        end = origin + rotation[:, i] * axis_length
        traces.append(
            go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines",
                line=dict(color=colors[i], width=6),
                name=f"{name_prefix}_{axis_names[i]}",
                showlegend=True,
            )
        )
    return traces


def main():
    parser = argparse.ArgumentParser("visualize_robot_checkpoint")
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
        default=None,
        help='Single or multiple checkpoints, e.g. "0.4,0.1,0.2; goal:0.5,0.0,0.3"',
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Path to .txt/.npy/.npz containing checkpoint coordinates",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=4096,
        help="Number of robot surface points used for visualization",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config/robot-model/robot_checkpoint_view.html",
        help="Output html path",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.08,
        help="Coordinate frame axis length",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    args = parser.parse_args()

    if args.checkpoint is None and args.checkpoint_file is None:
        raise ValueError("One of --checkpoint or --checkpoint-file is required.")

    sampler = URDFSurfaceSampler(args.urdf)
    joint_values = parse_joint_values(args.joint_values, sampler)
    robot_mesh, face_link_names = sampler.build_robot_mesh(joint_values=joint_values)
    sampled = sample_surface_points(
        robot_mesh,
        face_link_names,
        count=args.num_points,
        seed=args.seed,
    )

    if args.checkpoint_file is not None:
        checkpoint_points, checkpoint_labels = load_checkpoint_file(args.checkpoint_file)
    else:
        checkpoint_points, checkpoint_labels = parse_checkpoint_string(args.checkpoint)

    cache = {}
    ee_tf = sampler.link_transform("ee_link", joint_values, cache)
    tool0_tf = sampler.link_transform("tool0", joint_values, cache)
    base_tf = sampler.link_transform("base_link", joint_values, cache)

    ee_origin = ee_tf[:3, 3]
    tool0_origin = tool0_tf[:3, 3]
    base_origin = base_tf[:3, 3]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=sampled["points"][:, 0],
            y=sampled["points"][:, 1],
            z=sampled["points"][:, 2],
            mode="markers",
            marker=dict(size=2, color="#9aa4b2", opacity=0.72),
            name="robot_surface",
            text=sampled["link_name"],
            hovertemplate="link=%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=checkpoint_points[:, 0],
            y=checkpoint_points[:, 1],
            z=checkpoint_points[:, 2],
            mode="markers+text",
            marker=dict(size=8, color="#e4572e", symbol="diamond"),
            text=checkpoint_labels,
            textposition="top center",
            name="checkpoints",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[ee_origin[0]],
            y=[ee_origin[1]],
            z=[ee_origin[2]],
            mode="markers+text",
            marker=dict(size=8, color="#1f77b4"),
            text=["ee_link"],
            textposition="bottom center",
            name="ee_link",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[tool0_origin[0]],
            y=[tool0_origin[1]],
            z=[tool0_origin[2]],
            mode="markers+text",
            marker=dict(size=8, color="#2ca02c"),
            text=["tool0"],
            textposition="bottom center",
            name="tool0",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[base_origin[0]],
            y=[base_origin[1]],
            z=[base_origin[2]],
            mode="markers+text",
            marker=dict(size=7, color="#9467bd"),
            text=["base_link"],
            textposition="bottom center",
            name="base_link",
        )
    )

    fig.add_traces(make_axes_trace(base_origin, base_tf[:3, :3], args.axis_length, "base"))
    fig.add_traces(make_axes_trace(tool0_origin, tool0_tf[:3, :3], args.axis_length, "tool0"))

    for point, label in zip(checkpoint_points, checkpoint_labels):
        fig.add_trace(
            go.Scatter3d(
                x=[tool0_origin[0], point[0]],
                y=[tool0_origin[1], point[1]],
                z=[tool0_origin[2], point[2]],
                mode="lines",
                line=dict(color="#ff7f0e", width=4, dash="dash"),
                name=f"tool0_to_{label}",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Robot and Checkpoint Relative Pose",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.01, y=0.99),
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")

    print(f"Saved visualization to: {output_path}")
    print(f"tool0 position: {tool0_origin.tolist()}")
    print(f"ee_link position: {ee_origin.tolist()}")
    for point, label in zip(checkpoint_points, checkpoint_labels):
        delta_tool0 = point - tool0_origin
        delta_ee = point - ee_origin
        print(f"{label}:")
        print(f"  world position: {point.tolist()}")
        print(f"  relative to tool0: {delta_tool0.tolist()} | distance={float(np.linalg.norm(delta_tool0)):.6f}")
        print(f"  relative to ee_link: {delta_ee.tolist()} | distance={float(np.linalg.norm(delta_ee)):.6f}")


if __name__ == "__main__":
    main()
