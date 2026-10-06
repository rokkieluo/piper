from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset


WINDOW_POLICIES = ("first_contact", "episode_start", "episode_end")


def tactile_validity_mask():
    """构造触觉输入的有效传感器掩码。

    输入：无，使用固定双指 30×15 的触觉布局。
    流程：先填充为 1，再把两个指段前 15 行末 3 列的补齐位置设为 0。
    输出：形状 [2, 30, 15] 的 float32 张量，用于屏蔽无效 taxel。
    """
    mask = torch.ones((2, 30, 15), dtype=torch.float32)
    mask[:, :15, 12:] = 0.0
    return mask


@dataclass(frozen=True)
class EpisodeRecord:
    """记录单次抓取的数据路径及二分类标签。"""
    path: Path
    label: int


def _scalar_label(data, path):
    """从 NPZ 片段读取最终二分类标签。

    输入：已打开的 data 和用于错误定位的 path。
    流程：优先读取 success 布尔值，旧文件则读取 result 数值。
    输出：失败为 0、成功为 1；两个字段都缺失时抛 ValueError。
    """
    if "success" in data.files:
        return int(bool(data["success"].item()))
    if "result" in data.files:
        return int(data["result"].item())
    raise ValueError(f"missing success/result label in {path}")


def discover_episodes(data_dir, window_size=20):
    """发现并校验可用于训练的采集片段。

    输入：NPZ 数据根目录和最少需要的窗口帧数。
    流程：递归遍历文件，验证力图为 [N, 2, 30, 15]；跳过短片段，
    读取剩余片段的标签；无目录、坏格式或无有效片段时抛错。
    输出：按路径排序的 EpisodeRecord 列表。
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    records = []
    for path in sorted(data_dir.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "force_map" not in data.files:
                raise ValueError(f"missing force_map in {path}")
            shape = data["force_map"].shape
            if len(shape) != 4 or tuple(shape[1:]) != (2, 30, 15):
                raise ValueError(
                    f"invalid force_map shape {shape} in {path}; expected [N, 2, 30, 15]"
                )
            if shape[0] < window_size:
                continue
            records.append(EpisodeRecord(path=path, label=_scalar_label(data, path)))

    if not records:
        raise ValueError(f"no usable .npz episodes found below {data_dir}")
    return records


def stratified_split(records, seed=42, train_ratio=0.70, val_ratio=0.15):
    """按最终标签对片段进行可复现的分层划分。

    输入：EpisodeRecord 列表、随机种子和训练/验证比例。
    流程：按类别分组并各自打乱，每类至少保留一个训练、验证和测试
    样本；最后打乱每个划分，非法比例或类别数量不足时抛错。
    输出：键为 train、val、test 的片段列表字典，片段不会跨集合重复。
    """
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("split ratios must be positive and sum to less than 1")

    groups = {}
    for record in records:
        groups.setdefault(record.label, []).append(record)

    if len(groups) < 2:
        raise ValueError("at least two classes are required")

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for label, group in sorted(groups.items()):
        group = list(group)
        rng.shuffle(group)
        if len(group) < 3:
            raise ValueError(f"class {label} needs at least 3 episodes")

        num_train = max(1, int(round(len(group) * train_ratio)))
        num_val = max(1, int(round(len(group) * val_ratio)))
        if num_train + num_val >= len(group):
            num_train = len(group) - 2
            num_val = 1

        splits["train"].extend(group[:num_train])
        splits["val"].extend(group[num_train:num_train + num_val])
        splits["test"].extend(group[num_train + num_val:])

    for split_records in splits.values():
        rng.shuffle(split_records)
    return splits


class EpisodeDataset(Dataset):
    """按指定窗口抽取一次抓取，并在训练时增强触觉序列。"""

    def __init__(
        self,
        records,
        window_size=20,
        window_policy="first_contact",
        augment=False,
        force_scale=0.1,
        noise_std=0.005,
        taxel_dropout=0.01,
    ):
        """配置按片段读取的触觉数据集。

        输入：记录列表、窗口长度/选取策略、增强开关及缩放噪声与丢点幅度。
        流程：保存配置，验证窗口策略并创建有效传感器位置掩码。
        输出：无返回值；初始化后的实例供 DataLoader 按索引取样。
        """
        self.records = list(records)
        self.window_size = window_size
        if window_policy not in WINDOW_POLICIES:
            raise ValueError(
                f"invalid window policy {window_policy!r}; "
                f"expected one of {WINDOW_POLICIES}"
            )
        self.window_policy = window_policy
        self.augment = augment
        self.force_scale = force_scale
        self.noise_std = noise_std
        self.taxel_dropout = taxel_dropout
        self.validity_mask = tactile_validity_mask()

    def __len__(self):
        """读取本数据集的片段数。

        输入：实例中已保存的 EpisodeRecord 列表。
        流程：计算列表长度，不读取 NPZ 文件。
        输出：整数片段数量，供 DataLoader 确定采样范围。
        """
        return len(self.records)

    def __getitem__(self, index):
        """根据索引生成一个可供模型训练的样本。

        输入：片段索引 index 及初始化时配置的窗口和增强参数。
        流程：从 NPZ 选取固定长度窗口并拷贝；训练集按需随机缩放、
        加噪声和丢点，最后清零无效 taxel。
        输出：(float32 触觉张量 [T, 2, 30, 15], long 类型二分类标签)。
        """
        record = self.records[index]
        with np.load(record.path, allow_pickle=False) as data:
            start = window_start(data, self.window_size, self.window_policy)
            force_map = np.array(
                data["force_map"][start:start + self.window_size],
                dtype=np.float32,
                copy=True,
            )

        # 数据增强只用于训练；无效 taxel 最后统一清零，避免产生伪接触。
        sequence = torch.from_numpy(force_map)
        if self.augment:
            scale = 1.0 + (torch.rand(()) * 2.0 - 1.0) * self.force_scale
            sequence = sequence * scale
            if self.noise_std > 0:
                sequence = sequence + torch.randn_like(sequence) * self.noise_std
            if self.taxel_dropout > 0:
                keep = torch.rand_like(sequence) >= self.taxel_dropout
                sequence = sequence * keep
            sequence = sequence.clamp_(0.0, 1.0)
        sequence = sequence * self.validity_mask

        return sequence, torch.tensor(record.label, dtype=torch.long)


def window_start(data, window_size, policy="first_contact"):
    """计算固定长度触觉窗口在一个片段中的起始帧。

    输入：已加载的 NPZ、窗口长度和 first_contact/episode_start/episode_end 策略。
    流程：校验帧数与策略；首次接触策略寻找首个正状态标签，
    无接触时回退末尾，旧片段无状态标签时从头开始；过晚接触钳到末尾。
    输出：非负起点整数；数据长度不足或标签形状错误时抛 ValueError。
    """
    if policy not in WINDOW_POLICIES:
        raise ValueError(
            f"invalid window policy {policy!r}; expected one of {WINDOW_POLICIES}"
        )

    num_frames = int(data["force_map"].shape[0])
    if num_frames < window_size:
        raise ValueError(
            f"episode has {num_frames} frames, fewer than window size {window_size}"
        )
    last_start = num_frames - window_size

    if policy == "episode_start":
        return 0
    if policy == "episode_end":
        return last_start

    # 旧版数据从首次接触才开始记录，因此缺少状态标签时从第 0 帧取。
    if "state_label" not in data.files:
        return 0

    state_labels = np.asarray(data["state_label"])
    if state_labels.shape != (num_frames,):
        raise ValueError(
            f"state_label shape {state_labels.shape} does not match "
            f"{num_frames} force-map frames"
        )
    contact_frames = np.flatnonzero(state_labels > 0)
    if contact_frames.size == 0:
        return last_start
    return min(int(contact_frames[0]), last_start)


def records_from_manifest(data_dir, entries):
    """将训练保存的划分清单恢复为片段记录。

    输入：数据根目录和包含相对 path、label 的清单条目。
    流程：逐项解析并规范化绝对路径，拒绝逃逸出数据根目录的条目。
    输出：与清单顺序一致的 EpisodeRecord 列表。
    """
    root = Path(data_dir).resolve()
    records = []
    for entry in entries:
        path = (root / entry["path"]).resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"manifest path escapes data directory: {entry['path']}")
        records.append(EpisodeRecord(path=path, label=int(entry["label"])))
    return records


def manifest_from_splits(splits, data_dir):
    """把片段划分转换为可保存的相对路径清单。

    输入：按 train/val/test 分组的记录及数据根目录。
    流程：将每个片段的绝对路径相对于根目录表示，保留其标签。
    输出：以划分名为键、以 path/label 字典列表为值的 JSON 兼容结构。
    """
    root = Path(data_dir).resolve()
    manifest = {}
    for split_name, records in splits.items():
        manifest[split_name] = [
            {"path": str(record.path.resolve().relative_to(root)), "label": record.label}
            for record in records
        ]
    return manifest
