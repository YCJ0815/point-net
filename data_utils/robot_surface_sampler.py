import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


def parse_xyz(text):
    if text is None:
        return np.zeros(3, dtype=np.float64)
    return np.array([float(x) for x in text.strip().split()], dtype=np.float64)


def parse_scale(text):
    if text is None:
        return np.ones(3, dtype=np.float64)
    return np.array([float(x) for x in text.strip().split()], dtype=np.float64)


def transform_from_xyz_rpy(xyz, rpy):
    mat = trimesh.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], axes="sxyz")
    mat[:3, 3] = xyz
    return mat


def rotation_about_axis(angle, axis):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(4, dtype=np.float64)
    axis = axis / norm
    return trimesh.transformations.rotation_matrix(angle, axis)


@dataclass
class CollisionGeometry:
    link_name: str
    kind: str
    origin: np.ndarray
    mesh_path: Path | None = None
    scale: np.ndarray | None = None
    size: np.ndarray | None = None


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class URDFSurfaceSampler:
    def __init__(self, urdf_path):
        self.urdf_path = Path(urdf_path).resolve()
        self.robot_root = self.urdf_path.parent
        self.links = {}
        self.joints = {}
        self.child_to_joint = {}
        self._parse_urdf()

    def _resolve_mesh_path(self, filename):
        if filename.startswith("package://urdf-pen/"):
            return self.robot_root / filename.replace("package://urdf-pen/", "")
        return (self.robot_root / filename).resolve()

    def _parse_collision(self, link_name, collision_node):
        origin_node = collision_node.find("origin")
        origin = transform_from_xyz_rpy(
            parse_xyz(origin_node.get("xyz") if origin_node is not None else None),
            parse_xyz(origin_node.get("rpy") if origin_node is not None else None),
        )

        geometry = collision_node.find("geometry")
        if geometry is None:
            return None

        mesh_node = geometry.find("mesh")
        if mesh_node is not None:
            filename = mesh_node.get("filename")
            if filename is None:
                return None
            return CollisionGeometry(
                link_name=link_name,
                kind="mesh",
                origin=origin,
                mesh_path=self._resolve_mesh_path(filename),
                scale=parse_scale(mesh_node.get("scale")),
            )

        box_node = geometry.find("box")
        if box_node is not None:
            return CollisionGeometry(
                link_name=link_name,
                kind="box",
                origin=origin,
                size=parse_xyz(box_node.get("size")),
            )

        return None

    def _parse_urdf(self):
        root = ET.parse(self.urdf_path).getroot()

        for link_node in root.findall("link"):
            link_name = link_node.get("name")
            collisions = []
            for collision_node in link_node.findall("collision"):
                collision = self._parse_collision(link_name, collision_node)
                if collision is not None:
                    collisions.append(collision)
            self.links[link_name] = collisions

        for joint_node in root.findall("joint"):
            joint_name = joint_node.get("name")
            joint_type = joint_node.get("type", "fixed")
            parent = joint_node.find("parent").get("link")
            child = joint_node.find("child").get("link")
            origin_node = joint_node.find("origin")
            origin = transform_from_xyz_rpy(
                parse_xyz(origin_node.get("xyz") if origin_node is not None else None),
                parse_xyz(origin_node.get("rpy") if origin_node is not None else None),
            )
            axis_node = joint_node.find("axis")
            axis = parse_xyz(axis_node.get("xyz") if axis_node is not None else "0 0 1")
            spec = JointSpec(
                name=joint_name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin=origin,
                axis=axis,
            )
            self.joints[joint_name] = spec
            self.child_to_joint[child] = spec

    def _joint_motion_transform(self, joint, joint_values):
        value = joint_values.get(joint.name, 0.0)
        if joint.joint_type in ("revolute", "continuous"):
            return rotation_about_axis(value, joint.axis)
        if joint.joint_type == "prismatic":
            motion = np.eye(4, dtype=np.float64)
            motion[:3, 3] = joint.axis * value
            return motion
        return np.eye(4, dtype=np.float64)

    def link_transform(self, link_name, joint_values, cache):
        if link_name in cache:
            return cache[link_name]
        joint = self.child_to_joint.get(link_name)
        if joint is None:
            cache[link_name] = np.eye(4, dtype=np.float64)
            return cache[link_name]
        parent_transform = self.link_transform(joint.parent, joint_values, cache)
        cache[link_name] = parent_transform @ joint.origin @ self._joint_motion_transform(joint, joint_values)
        return cache[link_name]

    def _primitive_mesh(self, collision):
        if collision.kind == "mesh":
            loaded = trimesh.load_mesh(collision.mesh_path, process=False)
            if isinstance(loaded, trimesh.Scene):
                mesh = loaded.dump(concatenate=True)
            else:
                mesh = loaded
            mesh = mesh.copy()
            mesh.apply_scale(collision.scale)
            return mesh
        if collision.kind == "box":
            return trimesh.creation.box(extents=collision.size)
        raise ValueError(f"Unsupported collision geometry: {collision.kind}")

    def build_robot_mesh(self, joint_values=None, include_links=None):
        joint_values = joint_values or {}
        include_links = set(include_links) if include_links else None
        cache = {}
        meshes = []
        face_link_names = []

        for link_name, collisions in self.links.items():
            if include_links is not None and link_name not in include_links:
                continue
            link_tf = self.link_transform(link_name, joint_values, cache)
            for collision in collisions:
                mesh = self._primitive_mesh(collision)
                mesh.apply_transform(link_tf @ collision.origin)
                meshes.append(mesh)
                face_link_names.extend([link_name] * len(mesh.faces))

        if not meshes:
            raise ValueError("No collision geometry found in the selected links.")

        merged = trimesh.util.concatenate(meshes)
        return merged, np.array(face_link_names, dtype=object)


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


def estimate_count_from_spacing(area, spacing):
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    return max(1, int(math.ceil(area / (spacing ** 2))))


def sample_surface_points(mesh, face_link_names, count=None, spacing=None, seed=0):
    if count is None and spacing is None:
        raise ValueError("Either count or spacing must be provided.")

    area = float(mesh.area)
    if count is None:
        count = estimate_count_from_spacing(area, spacing)

    if spacing is not None:
        sampled_points = np.empty((0, 3), dtype=np.float64)
        sampled_faces = np.empty((0,), dtype=np.int64)
        for factor in (1.00, 0.95, 0.90, 0.85, 0.80, 0.75):
            sampled_points, sampled_faces = trimesh.sample.sample_surface_even(
                mesh, count=count, radius=spacing * factor, seed=seed
            )
            if len(sampled_points) >= 0.9 * count:
                break
        if len(sampled_points) == 0:
            raise RuntimeError("No points sampled; try a smaller spacing.")
        normals = mesh.face_normals[sampled_faces]
        return {
            "points": sampled_points.astype(np.float32),
            "normals": normals.astype(np.float32),
            "face_index": sampled_faces.astype(np.int64),
            "link_name": face_link_names[sampled_faces],
            "target_count": int(count),
            "actual_count": int(len(sampled_points)),
            "surface_area": area,
        }

    candidate_count = max(count * 20, count + 4096)
    candidate_points, candidate_faces = trimesh.sample.sample_surface(mesh, count=candidate_count, seed=seed)
    keep = farthest_point_sample_numpy(candidate_points, count, seed=seed)
    sampled_points = candidate_points[keep]
    sampled_faces = candidate_faces[keep]
    normals = mesh.face_normals[sampled_faces]
    return {
        "points": sampled_points.astype(np.float32),
        "normals": normals.astype(np.float32),
        "face_index": sampled_faces.astype(np.int64),
        "link_name": face_link_names[sampled_faces],
        "target_count": int(count),
        "actual_count": int(len(sampled_points)),
        "surface_area": area,
    }


def parse_joint_values(text, robot):
    if text is None:
        return {}

    text = text.strip()
    if not text:
        return {}

    if "=" not in text:
        values = [float(x) for x in text.split(",")]
        movable = [
            joint.name
            for joint in robot.joints.values()
            if joint.joint_type in ("revolute", "continuous", "prismatic")
        ]
        if len(values) != len(movable):
            raise ValueError(
                f"Expected {len(movable)} joint values, got {len(values)}."
            )
        return dict(zip(movable, values))

    joint_values = {}
    for item in text.split(","):
        name, value = item.split("=")
        joint_values[name.strip()] = float(value)
    return joint_values


def save_npz(output_path, sample_dict):
    np.savez_compressed(
        output_path,
        points=sample_dict["points"],
        normals=sample_dict["normals"],
        face_index=sample_dict["face_index"],
        link_name=sample_dict["link_name"],
        target_count=np.array(sample_dict["target_count"], dtype=np.int64),
        actual_count=np.array(sample_dict["actual_count"], dtype=np.int64),
        surface_area=np.array(sample_dict["surface_area"], dtype=np.float64),
    )


def save_ply(output_path, sample_dict):
    cloud = trimesh.points.PointCloud(
        vertices=sample_dict["points"],
        metadata={"vertex_normals": sample_dict["normals"]},
    )
    cloud.export(output_path)


def main():
    parser = argparse.ArgumentParser("robot_surface_sampler")
    parser.add_argument(
        "--urdf",
        type=str,
        default="config/robot-model/ur5e_with_pen.urdf",
        help="URDF path",
    )
    parser.add_argument("--num-points", type=int, default=8192, help="Exact sampled point count")
    parser.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="Minimum spacing on the surface. If set, actual point count is approximate.",
    )
    parser.add_argument(
        "--joint-values",
        type=str,
        default=None,
        help="Either comma-separated values in URDF movable-joint order, or name=value pairs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config/robot-model/ur5e_surface_points.npz",
        help="Output .npz or .ply path",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--include-links",
        type=str,
        default=None,
        help="Comma-separated subset of links to sample",
    )
    args = parser.parse_args()

    sampler = URDFSurfaceSampler(args.urdf)
    joint_values = parse_joint_values(args.joint_values, sampler)
    include_links = args.include_links.split(",") if args.include_links else None
    robot_mesh, face_link_names = sampler.build_robot_mesh(
        joint_values=joint_values,
        include_links=include_links,
    )

    sample_dict = sample_surface_points(
        robot_mesh,
        face_link_names,
        count=args.num_points if args.spacing is None else None,
        spacing=args.spacing,
        seed=args.seed,
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".npz":
        save_npz(output_path, sample_dict)
    elif output_path.suffix.lower() == ".ply":
        save_ply(output_path, sample_dict)
    else:
        raise ValueError("Output must end with .npz or .ply")

    unique_links, counts = np.unique(sample_dict["link_name"], return_counts=True)
    print(f"Saved surface samples to: {output_path}")
    print(f"Surface area: {sample_dict['surface_area']:.6f}")
    print(f"Target count: {sample_dict['target_count']}")
    print(f"Actual count: {sample_dict['actual_count']}")
    print("Per-link counts:")
    for link_name, link_count in zip(unique_links, counts):
        print(f"  {link_name}: {int(link_count)}")


if __name__ == "__main__":
    main()
