"""从多个采集目录构建成功/失败各半的去重数据集。"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np


def parse_args():
    """读取构建均衡数据集的命令行参数。

    输入：一个或多个来源批次目录、必须指定的输出目录及随机种子。
    流程：由 argparse 解析参数并应用默认种子。
    输出：供 main 使用的 argparse.Namespace。
    """
    parser = argparse.ArgumentParser(
        description="Build a deduplicated 1:1 success/fail episode dataset"
    )
    parser.add_argument("sources", nargs="+", help="source run directories")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def file_sha256(path):
    """计算片段文件的 SHA-256 内容指纹。

    输入：NPZ 文件路径 path。
    流程：按每次 1 MiB 分块读取文件，持续更新摘要，避免整文件加载。
    输出：十六进制哈希字符串，供跨批次精确去重。
    """
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_label(path):
    """读取一个 NPZ 片段的成功/失败标签。

    输入：采集片段文件路径。
    流程：优先读取 success 布尔字段，旧文件回退到 result 字段。
    输出：成功为 1、失败为 0；标签缺失时抛 ValueError。
    """
    with np.load(path, allow_pickle=False) as data:
        if "success" in data.files:
            return int(bool(data["success"].item()))
        if "result" in data.files:
            return int(data["result"].item())
    raise ValueError(f"missing success/result label in {path}")


def load_reason(path, label):
    """确定片段的最终结束原因。

    输入：NPZ 路径及已解析的二分类 label。
    流程：读取同名 JSON 的 finish_reason；缺失 JSON 时成功样本
    使用 success，失败样本使用 unknown。
    输出：原因字符串，用于按原因抽样及清单追溯。
    """
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        return "success" if label == 1 else "unknown"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return str(metadata.get("finish_reason", "unknown"))


def discover_unique_episodes(source_dirs):
    """收集多个批次中内容不重复的采集片段。

    输入：各来源目录的 Path 列表，直接扫描其中的 NPZ 文件。
    流程：按 SHA-256 指纹只保留首个文件，同时读取类别与结束原因；
    重复文件保存被丢弃路径及保留来源，空来源目录直接报错。
    输出：(唯一片段记录列表, 重复来源记录列表)。
    """
    unique = []
    duplicates = []
    by_digest = {}

    for source_dir in source_dirs:
        paths = sorted(source_dir.glob("*.npz"))
        if not paths:
            raise ValueError(f"no NPZ episodes found directly in {source_dir}")
        for path in paths:
            digest = file_sha256(path)
            if digest in by_digest:
                duplicates.append(
                    {
                        "duplicate": str(path.resolve()),
                        "kept": str(by_digest[digest]["source"]),
                        "sha256": digest,
                    }
                )
                continue

            label = episode_label(path)
            record = {
                "source": path.resolve(),
                "sha256": digest,
                "label": label,
                "reason": load_reason(path, label),
            }
            by_digest[digest] = record
            unique.append(record)
    return unique, duplicates


def sample_by_reason(records, count, rng):
    """按结束原因保留稀有样本并抽取指定数量。

    输入：同一类别的记录列表、目标数量 count 和已播种的 rng。
    流程：按原因分组并打乱，优先全部纳入样本数较小的组；
    再从其余记录中随机补齐，最终重新打乱顺序。
    输出：不重复的选中记录列表；样本不足时抛 ValueError。
    """
    groups = defaultdict(list)
    for record in records:
        groups[record["reason"]].append(record)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    remaining = count
    # 小组优先入选，防止均衡抽样时把罕见失败原因全部丢掉。
    small_group_limit = max(2, count // 50)
    for reason in sorted(groups):
        group = groups[reason]
        if len(group) <= small_group_limit:
            take = min(len(group), remaining)
            selected.extend(group[:take])
            groups[reason] = group[take:]
            remaining -= take

    pool = [record for group in groups.values() for record in group]
    if len(pool) < remaining:
        raise ValueError(f"cannot select {count} records from a pool of {len(records)}")
    rng.shuffle(pool)
    selected.extend(pool[:remaining])
    rng.shuffle(selected)
    return selected


def link_or_copy(source, destination):
    """将源片段放入目标数据集而不改动源文件。

    输入：源路径 source 和目标路径 destination。
    流程：先尝试创建硬链接；不支持时以 copy2 保留文件元信息复制。
    输出：表示实际传输方式的 hardlink 或 copy 字符串。
    """
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def build_dataset(source_dirs, output_dir, seed):
    """从多个来源生成成功/失败数量相同的新数据集。

    输入：来源目录、尚不存在的目标目录及抽样随机种子。
    流程：按文件内容去重并按标签分组；以较少类数量为基准，
    多数类优先保留少见结束原因；创建目标目录并链接或复制 NPZ
    及同名 JSON，保存来源、哈希和分布统计清单。
    输出：balance_manifest.json 对应的字典；缺类别或目录撞名时抛错。
    """
    source_dirs = [Path(path).resolve() for path in source_dirs]
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    records, duplicates = discover_unique_episodes(source_dirs)
    by_label = {
        label: [record for record in records if record["label"] == label]
        for label in (0, 1)
    }
    if not by_label[0] or not by_label[1]:
        raise ValueError("both fail and success episodes are required")

    per_class = min(len(by_label[0]), len(by_label[1]))
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        label_records = by_label[label]
        if len(label_records) == per_class:
            chosen = list(label_records)
            rng.shuffle(chosen)
        else:
            chosen = sample_by_reason(label_records, per_class, rng)
        selected.extend(chosen)
    rng.shuffle(selected)

    output_dir.mkdir(parents=True)
    samples = []
    transfer_modes = Counter()
    for index, record in enumerate(selected, start=1):
        label_name = "success" if record["label"] == 1 else "fail"
        stem = f"sample_{index:05d}_{label_name}"
        destination = output_dir / f"{stem}.npz"
        mode = link_or_copy(record["source"], destination)
        transfer_modes[mode] += 1

        metadata_source = record["source"].with_suffix(".json")
        if metadata_source.exists():
            link_or_copy(metadata_source, output_dir / f"{stem}.json")

        samples.append(
            {
                "path": destination.name,
                "label": record["label"],
                "reason": record["reason"],
                "source": str(record["source"]),
                "sha256": record["sha256"],
            }
        )

    manifest = {
        "seed": seed,
        "sources": [str(path) for path in source_dirs],
        "source_episode_count": len(records) + len(duplicates),
        "unique_episode_count": len(records),
        "duplicate_episode_count": len(duplicates),
        "available_counts": {
            "fail": len(by_label[0]),
            "success": len(by_label[1]),
        },
        "selected_counts": {
            "fail": sum(record["label"] == 0 for record in selected),
            "success": sum(record["label"] == 1 for record in selected),
        },
        "selected_finish_reasons": dict(
            Counter(record["reason"] for record in selected)
        ),
        "transfer_modes": dict(transfer_modes),
        "samples": samples,
        "duplicates": duplicates,
    }
    (output_dir / "balance_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main():
    """运行均衡数据集构建命令。

    输入：命令行中的来源目录、目标目录及随机种子。
    流程：调用 build_dataset，依次打印来源总数、去重数、
    每类样本数、结束原因与硬链接/复制数量。
    输出：无返回值；产出均衡目录及其来源清单。
    """
    args = parse_args()
    manifest = build_dataset(args.sources, args.output_dir, args.seed)
    print(f"output: {Path(args.output_dir).resolve()}")
    print(f"source episodes: {manifest['source_episode_count']}")
    print(f"unique episodes: {manifest['unique_episode_count']}")
    print(f"duplicates removed: {manifest['duplicate_episode_count']}")
    print(f"selected: {manifest['selected_counts']}")
    print(f"finish reasons: {manifest['selected_finish_reasons']}")
    print(f"transfer: {manifest['transfer_modes']}")


if __name__ == "__main__":
    main()
