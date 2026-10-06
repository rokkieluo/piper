import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import ForceCnnTcn, ForceCnnTcnConfig
from training.data import EpisodeDataset, records_from_manifest
from training.metrics import classification_metrics


def parse_args():
    """解析独立评估所需的参数。

    输入：命令行中的 checkpoint 路径、数据根目录、划分、批大小和设备。
    流程：限制可选划分及设备名称，并填充默认值。
    输出：供评估入口使用的 argparse.Namespace。
    """
    parser = argparse.ArgumentParser(description="Evaluate a phase-one checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--data-dir", default="dataset/auto")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    """加载训练运行并重算指定数据划分的分类指标。

    输入：命令行参数，以及 checkpoint 相邻的 split.json 和原始 NPZ 数据。
    流程：读取固定划分清单和权重，按 checkpoint 的模型配置及窗口策略
    构造数据集；逐批无梯度推理，汇总平均损失、混淆矩阵与类别指标。
    输出：将 JSON 格式指标打印到标准输出，不重写训练结果文件。
    """
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent
    # 使用训练运行的固定划分，避免重新随机划分改变测试集合。
    manifest = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    records = records_from_manifest(args.data_dir, manifest[args.split])

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ForceCnnTcnConfig.from_dict(checkpoint["model_config"])
    model = ForceCnnTcn(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    window_policy = checkpoint.get("window_policy", "episode_start")
    dataset = EpisodeDataset(
        records,
        window_size=checkpoint["window_size"],
        window_policy=window_policy,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    targets = []
    predictions = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for force_sequence, labels in loader:
            force_sequence = force_sequence.to(device)
            labels = labels.to(device)
            logits = model(force_sequence)
            total_loss += float(criterion(logits, labels)) * labels.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = classification_metrics(targets, predictions)
    metrics["loss"] = total_loss / len(dataset)
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
