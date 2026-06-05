import argparse
from pathlib import Path

import numpy as np


DEFAULT_RESULTS_ROOT = "/Users/ycj/Desktop/Research/Warmup/DiffusionPolicyPathplanning/3D-Diffusion-Policy/data/raw_data/results"
DEFAULT_OUTPUT_ROOT = "data/noisy_transition_joint_trajectories"
DEFAULT_TRAJECTORY_KEY = "q_playback"
DEFAULT_NOISE_CLIP = 0.05
DEFAULT_NOISE_SIGMA = DEFAULT_NOISE_CLIP / 3.0


def collect_transition_files(results_root, job_name):
    job_dir = Path(results_root).resolve() / job_name
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Job directory does not exist: {job_dir}")
    transition_files = sorted(job_dir.glob("transition_*.npz"))
    if not transition_files:
        raise FileNotFoundError(f"No transition_*.npz files found under: {job_dir}")
    return transition_files


def load_transition_trajectory(npz_path, trajectory_key):
    npz_path = Path(npz_path).resolve()
    data = np.load(npz_path, allow_pickle=True)
    if trajectory_key not in data:
        raise KeyError(f"Missing trajectory key '{trajectory_key}' in {npz_path}")

    trajectory = np.asarray(data[trajectory_key], dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 6:
        raise ValueError(
            f"Expected {trajectory_key} to have shape [T, 6], got {trajectory.shape} in {npz_path}"
        )

    payload = {
        "trajectory": trajectory,
        "q_start": np.asarray(data["q_start"], dtype=np.float32) if "q_start" in data else None,
        "q_goal": np.asarray(data["q_goal"], dtype=np.float32) if "q_goal" in data else None,
    }
    return payload


def generate_clipped_gaussian_noise(shape, sigma, clip_min, clip_max, rng):
    noise = rng.normal(loc=0.0, scale=sigma, size=shape).astype(np.float32)
    noise = np.clip(noise, clip_min, clip_max)
    return noise.astype(np.float32)


def build_noisy_trajectory(trajectory, sigma, clip_min, clip_max, rng):
    trajectory = np.asarray(trajectory, dtype=np.float32)
    noise = generate_clipped_gaussian_noise(
        shape=trajectory.shape,
        sigma=sigma,
        clip_min=clip_min,
        clip_max=clip_max,
        rng=rng,
    )
    noisy_trajectory = trajectory + noise
    return noisy_trajectory.astype(np.float32), noise.astype(np.float32)


def save_noisy_transition(output_path, trajectory_payload, noisy_trajectory, noise, sigma, clip_min, clip_max, source_npz):
    save_kwargs = {
        "q_playback_original": trajectory_payload["trajectory"].astype(np.float32),
        "q_playback_noisy": noisy_trajectory.astype(np.float32),
        "noise": noise.astype(np.float32),
        "noise_sigma": np.array(sigma, dtype=np.float32),
        "noise_clip_min": np.array(clip_min, dtype=np.float32),
        "noise_clip_max": np.array(clip_max, dtype=np.float32),
        "source_transition_npz": np.array(str(Path(source_npz).resolve()), dtype=object),
    }
    if trajectory_payload["q_start"] is not None:
        save_kwargs["q_start"] = trajectory_payload["q_start"].astype(np.float32)
    if trajectory_payload["q_goal"] is not None:
        save_kwargs["q_goal"] = trajectory_payload["q_goal"].astype(np.float32)

    np.savez_compressed(output_path, **save_kwargs)


def process_job(results_root, job_name, output_root, trajectory_key, sigma, clip_value, seed):
    transition_files = collect_transition_files(results_root, job_name)
    output_job_dir = Path(output_root).resolve() / job_name
    output_job_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    succeeded = 0
    failures = []

    for transition_path in transition_files:
        try:
            trajectory_payload = load_transition_trajectory(transition_path, trajectory_key)
            noisy_trajectory, noise = build_noisy_trajectory(
                trajectory=trajectory_payload["trajectory"],
                sigma=sigma,
                clip_min=-clip_value,
                clip_max=clip_value,
                rng=rng,
            )
            output_path = output_job_dir / f"{transition_path.stem}_{trajectory_key}_noisy.npz"
            save_noisy_transition(
                output_path=output_path,
                trajectory_payload=trajectory_payload,
                noisy_trajectory=noisy_trajectory,
                noise=noise,
                sigma=sigma,
                clip_min=-clip_value,
                clip_max=clip_value,
                source_npz=transition_path,
            )
            succeeded += 1
            print(f"saved: {output_path}")
        except Exception as exc:
            failures.append((str(transition_path), str(exc)))
            print(f"failed: {transition_path} | {exc}")

    print(f"processed: {len(transition_files)}")
    print(f"succeeded: {succeeded}")
    print(f"failed: {len(failures)}")
    if failures:
        print("failure_examples:")
        for path, reason in failures[:10]:
            print(f"{path} -> {reason}")


def main():
    parser = argparse.ArgumentParser("extract_noisy_transition_joint_trajectories")
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT, help="Root directory containing results/job_xxx/transition_*.npz")
    parser.add_argument("--job-name", type=str, required=True, help="Target job name, e.g. job_003")
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Output root for noisy trajectory npz files")
    parser.add_argument("--trajectory-key", type=str, default=DEFAULT_TRAJECTORY_KEY, help="Trajectory key to extract from transition npz")
    parser.add_argument("--noise-sigma", type=float, default=DEFAULT_NOISE_SIGMA, help="Gaussian sigma before clipping")
    parser.add_argument("--noise-clip", type=float, default=DEFAULT_NOISE_CLIP, help="Absolute clipping value applied to sampled noise")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    if args.noise_sigma <= 0:
        raise ValueError("--noise-sigma must be positive.")
    if args.noise_clip <= 0:
        raise ValueError("--noise-clip must be positive.")

    process_job(
        results_root=args.results_root,
        job_name=args.job_name,
        output_root=args.output_root,
        trajectory_key=args.trajectory_key,
        sigma=args.noise_sigma,
        clip_value=args.noise_clip,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
