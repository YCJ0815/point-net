import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from data_utils.PointCloudJointInputDataset import PointCloudJointInputDataset
from models.pointnet_joint_collision_distance import get_loss, get_model


def parse_args():
    parser = argparse.ArgumentParser("train_joint_collision_distance")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset .npz or shard manifest .json produced by build_pointcloud_joint_input_dataset.py")
    parser.add_argument("--output-dir", type=str, default="log/joint_collision_distance", help="Checkpoint and log output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of samples used for validation")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--collision-weight", type=float, default=1.0, help="Collision BCE loss weight")
    parser.add_argument("--distance-weight", type=float, default=1.0, help="Distance SmoothL1 loss weight")
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0, help="SmoothL1 beta")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Training device")
    parser.add_argument("--log-every", type=int, default=500, help="Print batch progress every N steps; 0 disables batch logging")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mps_is_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "mps":
        if not mps_is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_dataset(dataset, val_ratio, seed):
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least two samples for train/validation splitting.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")

    val_size = max(1, int(round(len(dataset) * val_ratio)))
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def move_batch(batch, device):
    point_cloud, joint_feature, collision_label, min_distance_norm = batch
    return (
        point_cloud.to(device=device, dtype=torch.float32, non_blocking=True),
        joint_feature.to(device=device, dtype=torch.float32, non_blocking=True),
        collision_label.to(device=device, dtype=torch.float32, non_blocking=True),
        min_distance_norm.to(device=device, dtype=torch.float32, non_blocking=True),
    )


def run_epoch(model, criterion, loader, device, optimizer=None, log_every=0, epoch=None, stage="train"):
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "collision_loss": 0.0,
        "distance_loss": 0.0,
        "collision_correct": 0,
        "distance_abs_error": 0.0,
        "samples": 0,
    }
    total_steps = len(loader)
    start_time = time.time()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, batch in enumerate(loader, start=1):
            point_cloud, joint_feature, collision_target, distance_target = move_batch(batch, device)
            if training:
                optimizer.zero_grad()

            outputs = model(point_cloud, joint_feature)
            losses = criterion(outputs, collision_target, distance_target)
            if training:
                losses["loss"].backward()
                optimizer.step()

            batch_size = point_cloud.shape[0]
            collision_prediction = (outputs["unsafe_logit"].squeeze(1) >= 0).to(torch.int64)
            collision_truth = collision_target.to(torch.int64)
            distance_prediction = outputs["d_min_norm"].squeeze(1)

            totals["samples"] += batch_size
            totals["loss"] += float(losses["loss"].detach()) * batch_size
            totals["collision_loss"] += float(losses["collision_loss"].detach()) * batch_size
            totals["distance_loss"] += float(losses["distance_loss"].detach()) * batch_size
            totals["collision_correct"] += int((collision_prediction == collision_truth).sum().item())
            totals["distance_abs_error"] += float(torch.abs(distance_prediction - distance_target).sum().item())

            if log_every > 0 and (step % log_every == 0 or step == total_steps):
                elapsed = max(time.time() - start_time, 1e-6)
                steps_per_sec = step / elapsed
                eta_seconds = (total_steps - step) / max(steps_per_sec, 1e-6)
                running_loss = totals["loss"] / totals["samples"]
                prefix = f"epoch {epoch:03d} " if epoch is not None else ""
                print(
                    f"{prefix}{stage} step {step}/{total_steps} | "
                    f"loss {running_loss:.6f} | "
                    f"steps/s {steps_per_sec:.2f} | "
                    f"eta_min {eta_seconds / 60.0:.1f}",
                    flush=True,
                )

    samples = totals["samples"]
    return {
        "loss": totals["loss"] / samples,
        "collision_loss": totals["collision_loss"] / samples,
        "distance_loss": totals["distance_loss"] / samples,
        "collision_accuracy": totals["collision_correct"] / samples,
        "distance_mae_norm": totals["distance_abs_error"] / samples,
    }


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset = PointCloudJointInputDataset(args.dataset)
    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio, args.seed)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = get_model().to(device)
    criterion = get_loss(
        collision_weight=args.collision_weight,
        distance_weight=args.distance_weight,
        smooth_l1_beta=args.smooth_l1_beta,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pth"
    best_val_loss = float("inf")

    print(f"device: {device}")
    print(f"dataset: {Path(args.dataset).resolve()}")
    print(f"train_samples: {len(train_dataset)}")
    print(f"val_samples: {len(val_dataset)}")
    print(f"train_steps_per_epoch: {len(train_loader)}")
    print(f"val_steps_per_epoch: {len(val_loader)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            criterion,
            train_loader,
            device,
            optimizer=optimizer,
            log_every=args.log_every,
            epoch=epoch,
            stage="train",
        )
        val_metrics = run_epoch(
            model,
            criterion,
            val_loader,
            device,
            log_every=args.log_every,
            epoch=epoch,
            stage="val",
        )
        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train_loss {train_metrics['loss']:.6f} | "
            f"val_loss {val_metrics['loss']:.6f} | "
            f"val_collision_acc {val_metrics['collision_accuracy']:.4f} | "
            f"val_distance_mae_norm {val_metrics['distance_mae_norm']:.6f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                best_path,
            )

    print(f"best_checkpoint: {best_path}")
    print(f"best_val_loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
