import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.projection = nn.Linear(1024, 64)
        self.projection_norm = nn.LayerNorm(64)

    @staticmethod
    def _to_channel_first(point_cloud):
        if point_cloud.ndim != 3:
            raise ValueError(
                "point_cloud must have shape [B, N, 3] or [B, 3, N], "
                f"got {tuple(point_cloud.shape)}"
            )
        if point_cloud.shape[-1] == 3:
            point_cloud = point_cloud.transpose(1, 2)
        elif point_cloud.shape[1] != 3:
            raise ValueError(
                "point_cloud must have coordinate dimension 3, "
                f"got {tuple(point_cloud.shape)}"
            )
        if point_cloud.shape[2] == 0:
            raise ValueError("point_cloud must contain at least one point")
        return point_cloud.contiguous()

    def forward(self, point_cloud):
        x = self._to_channel_first(point_cloud)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, dim=2).values
        return F.relu(self.projection_norm(self.projection(x)))


class JointFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(18, 64)
        self.norm1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, joint_feature):
        if joint_feature.ndim != 2 or joint_feature.shape[1] != 18:
            raise ValueError(
                "joint_feature must have shape [B, 18], "
                f"got {tuple(joint_feature.shape)}"
            )
        x = F.relu(self.norm1(self.fc1(joint_feature)))
        return F.relu(self.fc2(x))


class FusionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.norm1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, point_feature, joint_feature):
        x = torch.cat([point_feature, joint_feature], dim=1)
        x = F.relu(self.norm1(self.fc1(x)))
        return F.relu(self.fc2(x))


class PredictionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, feature):
        return self.fc2(F.relu(self.fc1(feature)))


class PointNetJointCollisionDistanceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_encoder = PointNetFeatureEncoder()
        self.joint_encoder = JointFeatureEncoder()
        self.fusion_encoder = FusionEncoder()
        self.collision_head = PredictionHead()
        self.distance_head = PredictionHead()

    def forward(self, point_cloud, joint_feature):
        if point_cloud.ndim != 3:
            raise ValueError(
                "point_cloud must have shape [B, N, 3] or [B, 3, N], "
                f"got {tuple(point_cloud.shape)}"
            )
        if joint_feature.ndim != 2 or joint_feature.shape[1] != 18:
            raise ValueError(
                "joint_feature must have shape [B, 18], "
                f"got {tuple(joint_feature.shape)}"
            )
        if point_cloud.shape[0] != joint_feature.shape[0]:
            raise ValueError(
                "point_cloud and joint_feature must have the same batch size, "
                f"got {point_cloud.shape[0]} and {joint_feature.shape[0]}"
            )

        z_pc = self.point_encoder(point_cloud)
        z_q = self.joint_encoder(joint_feature)
        z_fusion = self.fusion_encoder(z_pc, z_q)
        return {
            "unsafe_logit": self.collision_head(z_fusion),
            "d_min_norm": self.distance_head(z_fusion),
            "z_pc": z_pc,
            "z_q": z_q,
            "z_fusion": z_fusion,
        }


class CollisionDistanceLoss(nn.Module):
    def __init__(self, collision_weight=1.0, distance_weight=1.0, smooth_l1_beta=1.0):
        super().__init__()
        self.collision_weight = float(collision_weight)
        self.distance_weight = float(distance_weight)
        self.smooth_l1_beta = float(smooth_l1_beta)
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")
        self.collision_loss = nn.BCEWithLogitsLoss()

    def distance_loss(self, prediction, target):
        error = torch.abs(prediction - target)
        beta = self.smooth_l1_beta
        loss = torch.where(error < beta, 0.5 * error * error / beta, error - 0.5 * beta)
        return loss.mean()

    @staticmethod
    def _column_target(target, name, dtype, device):
        target = torch.as_tensor(target, dtype=dtype, device=device)
        if target.ndim == 1:
            target = target.unsqueeze(1)
        if target.ndim != 2 or target.shape[1] != 1:
            raise ValueError(f"{name} must have shape [B] or [B, 1], got {tuple(target.shape)}")
        return target

    def forward(self, outputs, collision_target, distance_target):
        required = ("unsafe_logit", "d_min_norm")
        missing = [key for key in required if key not in outputs]
        if missing:
            raise KeyError(f"Model outputs missing required keys: {missing}")

        unsafe_logit = outputs["unsafe_logit"]
        d_min_norm = outputs["d_min_norm"]
        collision_target = self._column_target(
            collision_target,
            "collision_target",
            unsafe_logit.dtype,
            unsafe_logit.device,
        )
        distance_target = self._column_target(
            distance_target,
            "distance_target",
            d_min_norm.dtype,
            d_min_norm.device,
        )
        if unsafe_logit.shape != collision_target.shape:
            raise ValueError(
                f"unsafe_logit and collision_target shapes differ: "
                f"{tuple(unsafe_logit.shape)} vs {tuple(collision_target.shape)}"
            )
        if d_min_norm.shape != distance_target.shape:
            raise ValueError(
                f"d_min_norm and distance_target shapes differ: "
                f"{tuple(d_min_norm.shape)} vs {tuple(distance_target.shape)}"
            )

        collision_loss = self.collision_loss(unsafe_logit, collision_target)
        distance_loss = self.distance_loss(d_min_norm, distance_target)
        total_loss = self.collision_weight * collision_loss + self.distance_weight * distance_loss
        return {
            "loss": total_loss,
            "collision_loss": collision_loss,
            "distance_loss": distance_loss,
        }


def get_model():
    return PointNetJointCollisionDistanceModel()


def get_loss(collision_weight=1.0, distance_weight=1.0, smooth_l1_beta=1.0):
    return CollisionDistanceLoss(
        collision_weight=collision_weight,
        distance_weight=distance_weight,
        smooth_l1_beta=smooth_l1_beta,
    )
