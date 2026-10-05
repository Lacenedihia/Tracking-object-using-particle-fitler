"""Object tracking in video with a particle filter (OpenCV + NumPy).

Tracks a person (blue top) walking through a noisy, shaky video, including a
partial occlusion behind a tree. Each particle is a hypothesis [x, y, vx, vy];
its weight depends on how close the pixel colour under it is to the target colour.

Usage:
    python tracking_objects.py                # headless, writes tracking_demo.mp4
    python tracking_objects.py --show         # also opens a window (Esc to quit)
"""
import argparse
import os
import sys

import cv2
import numpy as np

VIDEO = "walking.mp4"
NUM_PARTICLES = 500
VEL_RANGE = 0.5            # initial velocity spread (pixels / frame)
POS_SIGMA, VEL_SIGMA = 1.0, 0.5   # process noise
WEIGHT_POWER = 4           # sharpens the weights; too high -> too sensitive to colour
TARGET_COLOUR = np.array((189, 105, 82))   # BGR pixel colour of the blue top


def get_frames(filename):
    video = cv2.VideoCapture(filename)
    if not video.isOpened():
        sys.exit(f"Cannot open '{filename}'. Put the video next to this script "
                 f"(current folder: {os.getcwd()}).")
    while True:
        ok, frame = video.read()
        if not ok:
            break
        yield frame
    video.release()


def initialize_particles(width, height):
    particles = np.random.rand(NUM_PARTICLES, 4) * np.array((width, height, VEL_RANGE, VEL_RANGE))
    particles[:, 2:4] -= VEL_RANGE / 2.0       # centre velocities on zero
    return particles


def apply_velocity(particles):
    particles[:, 0] += particles[:, 2]
    particles[:, 1] += particles[:, 3]
    return particles


def enforce_edges(particles, width, height):
    particles[:, 0] = np.clip(particles[:, 0], 0, width - 1)
    particles[:, 1] = np.clip(particles[:, 1], 0, height - 1)
    return particles


def compute_errors(particles, frame):
    x = particles[:, 0].astype(int)
    y = particles[:, 1].astype(int)
    return np.sum((TARGET_COLOUR - frame[y, x, :]) ** 2, axis=1).astype(float)


def compute_weights(particles, errors, width, height):
    weights = np.max(errors) - errors                   # invert: low error -> high weight
    on_edge = ((particles[:, 0] == 0) | (particles[:, 0] == width - 1) |
               (particles[:, 1] == 0) | (particles[:, 1] == height - 1))
    weights[on_edge] = 0.0                              # (the old code compared x twice, never y)
    return weights ** WEIGHT_POWER


def resample(particles, weights):
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):            # avoid division by zero / NaN
        probabilities = np.full(NUM_PARTICLES, 1.0 / NUM_PARTICLES)
    else:
        probabilities = weights / total
    idx = np.random.choice(NUM_PARTICLES, size=NUM_PARTICLES, p=probabilities)
    particles = particles[idx, :]
    return particles, (int(particles[:, 0].mean()), int(particles[:, 1].mean()))


def apply_noise(particles):
    noise = np.concatenate((np.random.normal(0, POS_SIGMA, (NUM_PARTICLES, 2)),
                            np.random.normal(0, VEL_SIGMA, (NUM_PARTICLES, 2))), axis=1)
    return particles + noise


def draw(frame, particles, location):
    for x, y in particles[:, :2].astype(int):
        cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)
    cv2.circle(frame, location, 15, (0, 0, 255), 5)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=VIDEO)
    parser.add_argument("--show", action="store_true", help="display a window while running")
    parser.add_argument("--out", default="tracking_demo.mp4", help="output video ('' to disable)")
    args = parser.parse_args()

    np.random.seed(0)                                   # repeatability
    cap = cv2.VideoCapture(args.video)
    width, height = int(cap.get(3)), int(cap.get(4))    # read size from the file (no hard-coding)
    fps = cap.get(5) or 30
    cap.release()

    writer = (cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
              if args.out else None)
    particles = initialize_particles(width, height)

    for frame in get_frames(args.video):
        particles = apply_velocity(particles)
        particles = enforce_edges(particles, width, height)
        errors = compute_errors(particles, frame)
        weights = compute_weights(particles, errors, width, height)
        particles, location = resample(particles, weights)
        particles = apply_noise(particles)
        out = draw(frame, particles, location)
        if writer:
            writer.write(out)
        if args.show:
            cv2.imshow("frame", out)
            if cv2.waitKey(30) == 27:                   # Esc
                break

    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
