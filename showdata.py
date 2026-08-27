import argparse
from pathlib import Path

import cv2
import numpy as np


MAP_KEYS = ("force_map", "position_map")
STAT_NAMES = ("total", "max", "area", "entropy", "diff")


def parse_args():
    parser = argparse.ArgumentParser(description="View Piper tactile dataset episodes")
    parser.add_argument("path", nargs="?", default="dataset/auto")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--map", choices=MAP_KEYS, default="force_map")
    parser.add_argument("--delay", type=int, default=30)
    parser.add_argument("--scale", type=int, default=12)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument(
        "--fixed-scale",
        action="store_true",
        help="display values on the raw 0..1 scale instead of per-frame max scaling",
    )
    args = parser.parse_args()

    if args.index < 0:
        parser.error("--index must be >= 0")
    if args.delay < 1:
        parser.error("--delay must be >= 1")
    if args.scale < 2:
        parser.error("--scale must be >= 2")
    return args


def find_episodes(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(path.rglob("*.npz"))


def scalar(data, key, default=None):
    if key not in data.files:
        return default
    value = data[key]
    if value.shape == ():
        return value.item()
    return value


def result_text(data):
    success = scalar(data, "success")
    if success is not None:
        return "success" if bool(success) else "fail"

    result = scalar(data, "result")
    if result is not None:
        return "success" if int(result) == 1 else "fail"

    return "unknown"


def display_path(path):
    if path.parent.name.startswith("run_"):
        return f"{path.parent.name}/{path.name}"
    return path.name


def print_summary(paths):
    rows = []
    success_count = 0
    fail_count = 0
    total_frames = 0

    for path in paths:
        with np.load(path) as data:
            map_key = "force_map" if "force_map" in data.files else data.files[0]
            frames = int(data[map_key].shape[0])
            duration = float(data["time"][-1]) if "time" in data.files and frames else 0.0
            result = result_text(data)
            if result == "success":
                success_count += 1
            elif result == "fail":
                fail_count += 1
            total_frames += frames

            peak_force = None
            if "force_stats" in data.files and data["force_stats"].size:
                peak_force = float(data["force_stats"][:, 0].max())

            rows.append((display_path(path), result, frames, duration, peak_force))

    print(f"episodes: {len(paths)}")
    print(f"success: {success_count}")
    print(f"fail: {fail_count}")
    print(f"frames: {total_frames}")
    if paths:
        print(f"avg frames: {total_frames / len(paths):.1f}")
    print()

    for name, result, frames, duration, peak_force in rows:
        force_text = "n/a" if peak_force is None else f"{peak_force:.3f}"
        print(
            f"{name}  result={result:<7} frames={frames:<4} "
            f"duration={duration:.2f}s peak_total_force={force_text}"
        )


def map_for_display(array, fixed_scale):
    array = np.nan_to_num(array, copy=False)
    if fixed_scale:
        return np.clip(array, 0.0, 1.0)

    max_value = float(np.max(array))
    if max_value <= 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip(array / max_value, 0.0, 1.0)


def render_finger(array, scale, fixed_scale):
    display = map_for_display(array.astype(np.float32, copy=True), fixed_scale)
    gray = (display * 255).astype(np.uint8)
    height, width = gray.shape
    resized = cv2.resize(
        gray,
        (width * scale, height * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.applyColorMap(resized, cv2.COLORMAP_TURBO)


def draw_text(image, lines):
    y = 24
    for line in lines:
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24


def render_frame(data, path, map_key, frame_id, scale, fixed_scale):
    maps = data[map_key]
    frame = maps[frame_id]
    left = render_finger(frame[0], scale, fixed_scale)
    right = render_finger(frame[1], scale, fixed_scale)

    gap = np.full((left.shape[0], 12, 3), 30, dtype=np.uint8)
    tactile = np.hstack([left, gap, right])
    header = np.full((110, tactile.shape[1], 3), 20, dtype=np.uint8)
    image = np.vstack([header, tactile])

    lines = [
        f"{path.name}",
        f"{map_key}  frame {frame_id + 1}/{len(maps)}  result={result_text(data)}",
    ]

    if "time" in data.files:
        lines.append(f"time={float(data['time'][frame_id]):.3f}s")

    if "force_stats" in data.files:
        stats = data["force_stats"][frame_id]
        stat_text = "  ".join(
            f"{name}={float(value):.3f}"
            for name, value in zip(STAT_NAMES, stats)
        )
        lines.append(stat_text)

    draw_text(image, lines)
    return image


def print_controls():
    print("controls:")
    print("  q/esc: quit")
    print("  space: pause/play")
    print("  n/p: next/previous episode")
    print("  d/a: next/previous frame")
    print("  ]/[: jump 10 frames")
    print("  f: switch force_map/position_map")
    print("  r: restart current episode")
    print()


def view_dataset(paths, start_index, map_key, delay, scale, fixed_scale):
    episode_id = min(start_index, len(paths) - 1)
    frame_id = 0
    paused = False
    window_name = "Piper tactile dataset"

    print_controls()

    while True:
        path = paths[episode_id]
        with np.load(path) as data:
            if map_key not in data.files:
                available = ", ".join(data.files)
                raise KeyError(f"{map_key!r} not found in {path}; available: {available}")

            num_frames = data[map_key].shape[0]
            frame_id = max(0, min(frame_id, num_frames - 1))
            image = render_frame(data, path, map_key, frame_id, scale, fixed_scale)
            cv2.imshow(window_name, image)

            key = cv2.waitKey(0 if paused else delay) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("n"):
                episode_id = (episode_id + 1) % len(paths)
                frame_id = 0
            elif key == ord("p"):
                episode_id = (episode_id - 1) % len(paths)
                frame_id = 0
            elif key == ord("d"):
                frame_id = min(frame_id + 1, num_frames - 1)
                paused = True
            elif key == ord("a"):
                frame_id = max(frame_id - 1, 0)
                paused = True
            elif key == ord("]"):
                frame_id = min(frame_id + 10, num_frames - 1)
                paused = True
            elif key == ord("["):
                frame_id = max(frame_id - 10, 0)
                paused = True
            elif key == ord("f"):
                map_key = "position_map" if map_key == "force_map" else "force_map"
            elif key == ord("r"):
                frame_id = 0
            elif not paused:
                frame_id += 1
                if frame_id >= num_frames:
                    episode_id = (episode_id + 1) % len(paths)
                    frame_id = 0

    cv2.destroyAllWindows()


def main():
    args = parse_args()
    paths = find_episodes(args.path)
    if not paths:
        raise SystemExit(f"No .npz files found in {args.path}")

    if args.summary:
        print_summary(paths)
        return

    view_dataset(
        paths=paths,
        start_index=args.index,
        map_key=args.map,
        delay=args.delay,
        scale=args.scale,
        fixed_scale=args.fixed_scale,
    )


if __name__ == "__main__":
    main()
