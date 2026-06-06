import json
from pathlib import Path

import numpy as np

try:
    from .numpy_npz_compat import install_numpy_pickle_compat
except ImportError:
    from numpy_npz_compat import install_numpy_pickle_compat

install_numpy_pickle_compat()

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    class Dataset:
        pass


class PointCloudJointInputDataset(Dataset):
    def __init__(self, dataset_npz, return_metadata=False):
        self.dataset_npz = dataset_npz
        self.return_metadata = return_metadata
        self.required = ("point_clouds", "joint_features", "collision_labels", "min_distance_norm")
        self.data = None
        self.shard_data = []
        self.shard_offsets = []
        self.shard_metadata = []

        dataset_path = Path(dataset_npz).resolve()
        if dataset_path.suffix.lower() == ".json":
            self._load_manifest(dataset_path)
        else:
            self.data = np.load(dataset_path, allow_pickle=True)
            self._validate_required(self.data, dataset_path)
            self.point_clouds = np.asarray(self.data["point_clouds"], dtype=np.float32)
            self.joint_features = np.asarray(self.data["joint_features"], dtype=np.float32)
            self.collision_labels = np.asarray(self.data["collision_labels"], dtype=np.int64)
            self.min_distance_norm = np.asarray(self.data["min_distance_norm"], dtype=np.float32)
            self._validate_shapes(
                self.point_clouds,
                self.joint_features,
                self.collision_labels,
                self.min_distance_norm,
                dataset_path,
            )

    def _validate_required(self, data, dataset_path):
        missing = [key for key in self.required if key not in data]
        if missing:
            raise KeyError(f"Dataset npz missing required fields {missing}: {dataset_path}")

    def _validate_shapes(self, point_clouds, joint_features, collision_labels, min_distance_norm, dataset_path):
        if point_clouds.ndim != 3 or point_clouds.shape[2] != 3:
            raise ValueError(f"point_clouds must have shape [N, P, 3], got {point_clouds.shape} from {dataset_path}")
        if joint_features.ndim != 2 or joint_features.shape[1] != 18:
            raise ValueError(f"joint_features must have shape [N, 18], got {joint_features.shape} from {dataset_path}")
        if len(point_clouds) != len(joint_features):
            raise ValueError(
                "point_clouds and joint_features must have the same sample count, "
                f"got {len(point_clouds)} and {len(joint_features)} from {dataset_path}"
            )
        if collision_labels.shape != (len(point_clouds),):
            raise ValueError(
                f"collision_labels must have shape [{len(point_clouds)}], got {collision_labels.shape} from {dataset_path}"
            )
        if min_distance_norm.shape != (len(point_clouds),):
            raise ValueError(
                f"min_distance_norm must have shape [{len(point_clouds)}], got {min_distance_norm.shape} from {dataset_path}"
            )

    def _load_manifest(self, manifest_path):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != "pointcloud_joint_dataset_manifest_v1":
            raise ValueError(f"Unsupported manifest format: {manifest_path}")
        shard_records = payload.get("shards", [])
        if not shard_records:
            raise ValueError(f"Manifest contains no shards: {manifest_path}")

        total_samples = 0
        for record in shard_records:
            shard_path = Path(record["shard_path"]).resolve()
            shard = np.load(shard_path, allow_pickle=True)
            self._validate_required(shard, shard_path)
            point_clouds = np.asarray(shard["point_clouds"], dtype=np.float32)
            joint_features = np.asarray(shard["joint_features"], dtype=np.float32)
            collision_labels = np.asarray(shard["collision_labels"], dtype=np.int64)
            min_distance_norm = np.asarray(shard["min_distance_norm"], dtype=np.float32)
            self._validate_shapes(
                point_clouds,
                joint_features,
                collision_labels,
                min_distance_norm,
                shard_path,
            )
            self.shard_offsets.append(total_samples)
            self.shard_data.append(
                {
                    "point_clouds": point_clouds,
                    "joint_features": joint_features,
                    "collision_labels": collision_labels,
                    "min_distance_norm": min_distance_norm,
                    "raw": shard,
                }
            )
            self.shard_metadata.append(record)
            total_samples += len(point_clouds)

        self.total_samples = total_samples

    def __len__(self):
        if self.data is not None:
            return len(self.point_clouds)
        return self.total_samples

    def _locate_shard(self, index):
        if index < 0 or index >= self.total_samples:
            raise IndexError(index)
        for shard_idx in range(len(self.shard_offsets) - 1, -1, -1):
            if index >= self.shard_offsets[shard_idx]:
                local_index = index - self.shard_offsets[shard_idx]
                return self.shard_data[shard_idx], local_index
        raise IndexError(index)

    def __getitem__(self, index):
        if self.data is not None:
            point_cloud = self.point_clouds[index]
            joint_feature = self.joint_features[index]
            collision_label = self.collision_labels[index]
            min_distance_norm = self.min_distance_norm[index]
            data_ref = self.data
        else:
            shard, local_index = self._locate_shard(index)
            point_cloud = shard["point_clouds"][local_index]
            joint_feature = shard["joint_features"][local_index]
            collision_label = shard["collision_labels"][local_index]
            min_distance_norm = shard["min_distance_norm"][local_index]
            data_ref = shard["raw"]
            index = local_index
        if not self.return_metadata:
            return point_cloud, joint_feature, collision_label, min_distance_norm

        metadata = {}
        for key in (
            "joint_source",
            "source_transition_npz",
            "source_joint_npz",
            "source_workpiece_stl",
            "transition_index",
            "joint_index_in_source",
        ):
            if key in data_ref:
                value = data_ref[key][index]
                metadata[key] = value.item() if hasattr(value, "item") else value
        return point_cloud, joint_feature, collision_label, min_distance_norm, metadata
