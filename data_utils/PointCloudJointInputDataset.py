import numpy as np

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    class Dataset:
        pass


class PointCloudJointInputDataset(Dataset):
    def __init__(self, dataset_npz, return_metadata=False):
        self.dataset_npz = dataset_npz
        self.return_metadata = return_metadata
        self.data = np.load(dataset_npz, allow_pickle=True)

        required = ("point_clouds", "joint_features", "collision_labels", "min_distance_norm")
        missing = [key for key in required if key not in self.data]
        if missing:
            raise KeyError(f"Dataset npz missing required fields {missing}: {dataset_npz}")

        self.point_clouds = np.asarray(self.data["point_clouds"], dtype=np.float32)
        self.joint_features = np.asarray(self.data["joint_features"], dtype=np.float32)
        self.collision_labels = np.asarray(self.data["collision_labels"], dtype=np.int64)
        self.min_distance_norm = np.asarray(self.data["min_distance_norm"], dtype=np.float32)
        if self.point_clouds.ndim != 3 or self.point_clouds.shape[2] != 3:
            raise ValueError(f"point_clouds must have shape [N, P, 3], got {self.point_clouds.shape}")
        if self.joint_features.ndim != 2 or self.joint_features.shape[1] != 18:
            raise ValueError(f"joint_features must have shape [N, 18], got {self.joint_features.shape}")
        if len(self.point_clouds) != len(self.joint_features):
            raise ValueError(
                "point_clouds and joint_features must have the same sample count, "
                f"got {len(self.point_clouds)} and {len(self.joint_features)}"
            )
        if self.collision_labels.shape != (len(self.point_clouds),):
            raise ValueError(
                f"collision_labels must have shape [{len(self.point_clouds)}], got {self.collision_labels.shape}"
            )
        if self.min_distance_norm.shape != (len(self.point_clouds),):
            raise ValueError(
                f"min_distance_norm must have shape [{len(self.point_clouds)}], got {self.min_distance_norm.shape}"
            )

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, index):
        point_cloud = self.point_clouds[index]
        joint_feature = self.joint_features[index]
        collision_label = self.collision_labels[index]
        min_distance_norm = self.min_distance_norm[index]
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
            if key in self.data:
                value = self.data[key][index]
                metadata[key] = value.item() if hasattr(value, "item") else value
        return point_cloud, joint_feature, collision_label, min_distance_norm, metadata
