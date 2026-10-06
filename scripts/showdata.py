"""查看触觉采集片段并汇总数据集统计信息。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


MAP_KEYS = ("force_map", "position_map")
STAT_NAMES = ("total", "max", "area", "entropy", "diff")
DEFAULT_STATE_NAMES = {
    0: "no_contact",
    1: "weak_contact",
    2: "stable_grasp",
    3: "slip_or_drop",
}


def parse_args():
    """解析触觉数据集查看器的运行参数。

    输入：可选数据路径、起始片段索引、触觉图类型、播放延迟、
    放大倍率，以及只打印摘要或固定热图范围的开关。
    流程：填充默认值并检查索引非负、延迟及放大倍率有效。
    输出：argparse.Namespace；无效选项由解析器提示后退出。
    """
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
    """定位待查看的触觉 NPZ 片段。

    输入：单个文件或数据集目录 path。
    流程：文件直接加入列表，目录下递归查找并排序 NPZ 文件。
    输出：路径列表；路径不存在时抛 FileNotFoundError。
    """
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(path.rglob("*.npz"))


def scalar(data, key, default=None):
    """安全读取 NPZ 中可能不存在的单值字段。

    输入：已打开的 NPZ、字段 key 与缺省值 default。
    流程：字段不存在时返回默认值；零维数组转为 Python 标量。
    输出：解析后的标量或原始非标量数组。
    """
    if key not in data.files:
        return default
    value = data[key]
    if value.shape == ():
        return value.item()
    return value


def result_text(data):
    """将片段的不同版本结果字段统一为文本标签。

    输入：NPZ 中可选的 success 和旧版 result 字段。
    流程：优先读取 success 布尔值，再尝试 result 数值。
    输出：success、fail 或无标签时的 unknown 字符串。
    """
    success = scalar(data, "success")
    if success is not None:
        return "success" if bool(success) else "fail"

    result = scalar(data, "result")
    if result is not None:
        return "success" if int(result) == 1 else "fail"

    return "unknown"


def display_path(path):
    """生成摘要输出中可辨认批次的文件名。

    输入：采集片段路径 path。
    流程：若父目录为 run_*，将批次名与文件名拼接，否则只取文件名。
    输出：用于控制台显示的相对名称字符串。
    """
    if path.parent.name.startswith("run_"):
        return f"{path.parent.name}/{path.name}"
    return path.name


def load_metadata(path):
    """加载与 NPZ 同名的元信息文件。

    输入：采集片段 NPZ 路径。
    流程：改用 .json 扩展名读取 UTF-8；缺失或文件/JSON 损坏时忽略。
    输出：成功时为元数据字典，否则为空字典。
    """
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def names_by_id(metadata, key, defaults=None):
    """建立状态或阶段 ID 对应的可读名称。

    输入：元数据、名称字段 key、可选默认映射。
    流程：复制默认值，把 JSON 的字符串键转换成整数 ID；
    跳过无法转为整数的键。
    输出：整数 ID 到字符串名称的字典。
    """
    names = dict(defaults or {})
    for item_id, name in metadata.get(key, {}).items():
        try:
            names[int(item_id)] = str(name)
        except (TypeError, ValueError):
            continue
    return names


def print_summary(paths):
    """输出所选数据集的整体及逐回合摘要。

    输入：NPZ 片段路径列表。
    流程：逐个读取帧数、时长、最终标签、结束原因、峰值总力和
    状态标签计数；累计成功/失败数量与总帧数，再格式化打印。
    输出：无返回值；统计结果写到标准输出，不修改采集文件。
    """
    rows = []
    success_count = 0
    fail_count = 0
    total_frames = 0

    for path in paths:
        metadata = load_metadata(path)
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

            state_counts = "n/a"
            if "state_label" in data.files:
                state_names = names_by_id(
                    metadata,
                    "state_names",
                    DEFAULT_STATE_NAMES,
                )
                labels, counts = np.unique(data["state_label"], return_counts=True)
                state_counts = ",".join(
                    f"{state_names.get(int(label), str(int(label)))}:{int(count)}"
                    for label, count in zip(labels, counts)
                )

            rows.append(
                (
                    display_path(path),
                    result,
                    str(metadata.get("finish_reason", "n/a")),
                    frames,
                    duration,
                    peak_force,
                    state_counts,
                )
            )

    print(f"episodes: {len(paths)}")
    print(f"success: {success_count}")
    print(f"fail: {fail_count}")
    print(f"frames: {total_frames}")
    if paths:
        print(f"avg frames: {total_frames / len(paths):.1f}")
    print()

    for name, result, reason, frames, duration, peak_force, state_counts in rows:
        force_text = "n/a" if peak_force is None else f"{peak_force:.3f}"
        print(
            f"{name}  result={result:<7} reason={reason:<12} "
            f"frames={frames:<4} duration={duration:.2f}s "
            f"peak_total_force={force_text} states={state_counts}"
        )


def map_for_display(array, fixed_scale):
    """将触觉矩阵规范化到可渲染的强度范围。

    输入：二维数组 array 与 fixed_scale 开关。
    流程：将非有限值替换为零；固定缩放时钳至 [0,1]，动态
    缩放时除以当前矩阵最大值，全零图返回零矩阵。
    输出：范围 [0,1] 的图像强度数组。
    """
    array = np.nan_to_num(array, copy=False)
    if fixed_scale:
        return np.clip(array, 0.0, 1.0)

    # 逐帧归一化突出弱接触；全零帧单独处理，避免除零。
    max_value = float(np.max(array))
    if max_value <= 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip(array / max_value, 0.0, 1.0)


def render_finger(array, scale, fixed_scale):
    """渲染单根手指的触觉热力图。

    输入：单指二维力图、整数放大倍数和固定范围开关。
    流程：复制并归一化触觉值，转 uint8 后最近邻放大每个 taxel，
    再应用 OpenCV TURBO 色表。
    输出：三通道 BGR 热图数组。
    """
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
    """把多行片段信息写入图像抬头区域。

    输入：待修改的 OpenCV 图像及文本行序列。
    流程：从 y=24 开始按固定行距以抗锯齿白字逐行绘制。
    输出：无返回值；直接修改传入图像。
    """
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
    """生成当前片段某一帧的完整查看画面。

    输入：NPZ 数据、路径、力/位置图键、帧号、缩放及色阶设置。
    流程：并排渲染双指热图；读取同名元数据，把结果、时刻、
    力度统计、阶段状态及放置证据绘制到顶部信息区域。
    输出：包含图像和文字的 BGR 数组，供 cv2.imshow 展示。
    """
    maps = data[map_key]
    frame = maps[frame_id]
    left = render_finger(frame[0], scale, fixed_scale)
    right = render_finger(frame[1], scale, fixed_scale)

    gap = np.full((left.shape[0], 12, 3), 30, dtype=np.uint8)
    tactile = np.hstack([left, gap, right])
    header = np.full((164, tactile.shape[1], 3), 20, dtype=np.uint8)
    image = np.vstack([header, tactile])

    metadata = load_metadata(path)
    reason = metadata.get("finish_reason", "n/a")
    lines = [
        f"{path.name}",
        f"{map_key}  frame {frame_id + 1}/{len(maps)}  "
        f"result={result_text(data)}  reason={reason}",
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

    labels = []
    if "stage_id" in data.files:
        stage_names = names_by_id(metadata, "stage_names")
        stage_id = int(data["stage_id"][frame_id])
        labels.append(f"stage={stage_names.get(stage_id, stage_id)}")
    if "state_label" in data.files:
        state_names = names_by_id(metadata, "state_names", DEFAULT_STATE_NAMES)
        state_id = int(data["state_label"][frame_id])
        labels.append(f"state={state_names.get(state_id, state_id)}")
    if "contact_mask" in data.files:
        mask = np.asarray(data["contact_mask"][frame_id], dtype=bool)
        labels.append(f"contact=L{int(mask[0])}/R{int(mask[1])}")
    if labels:
        lines.append("  ".join(labels))

    placement = []
    for key, name in (
        ("object_on_b", "object_on_B"),
        ("gripper_object_contact", "gripper_contact"),
        ("gripper_released", "released"),
    ):
        if key in data.files:
            placement.append(f"{name}={int(bool(data[key][frame_id]))}")
    if placement:
        lines.append("  ".join(placement))

    draw_text(image, lines)
    return image


def print_controls():
    """向终端列出数据集查看器的操作键。

    输入：无。
    流程：打印退出、暂停、片段切换、逐帧前后移动、跳帧及换图说明。
    输出：无返回值；仅写标准输出。
    """
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
    """循环播放采集片段并处理键盘交互。

    输入：非空片段列表、起始索引、图类型、播放延迟、比例和缩放模式。
    流程：逐帧加载并渲染当前片段；暂停时阻塞等待按键，否则按
    延迟自动推进；按键可切换片段、调整帧位置或切换力/位置图。
    输出：无返回值；退出后销毁 OpenCV 窗口，不修改数据文件。
    """
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

            # 暂停时阻塞等待按键，播放时按设定间隔推进到下一帧。
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
    """执行触觉数据集查看器入口。

    输入：命令行指定的文件/目录及摘要或播放选项。
    流程：发现 NPZ 片段；无片段时结束并提示，否则摘要模式
    打印统计，交互模式打开逐帧浏览窗口。
    输出：无返回值；结果显示在终端或 OpenCV 窗口中。
    """
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
