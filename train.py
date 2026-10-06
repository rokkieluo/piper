import argparse
from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models import ForceCnnTcn, ForceCnnTcnConfig
from training.data import (
    EpisodeDataset,
    WINDOW_POLICIES,
    discover_episodes,
    manifest_from_splits,
    stratified_split,
)
from training.metrics import classification_metrics


CLASS_NAMES = ("fail", "success")


def parse_args():
    """读取并校验第一阶段训练参数。

    输入：命令行的数据目录、窗口策略、优化器参数及设备选项。
    流程：解析默认值和开关，检查窗口、轮次、批大小及耐心值为正数。
    输出：包含全部训练选项的 argparse.Namespace；错误时退出并提示。
    """
    parser = argparse.ArgumentParser(
        description="Train the phase-one Force-CNN + causal TCN baseline"
    )
    parser.add_argument("--data-dir", default="dataset/auto")
    parser.add_argument("--output-dir", default="runs/phase1")
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument(
        "--window-policy",
        choices=WINDOW_POLICIES,
        default="first_contact",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    for name in ("window_size", "epochs", "batch_size", "patience"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than 0")
    return args


def set_seed(seed):
    """固定训练涉及的随机数生成器。

    输入：整数 seed。
    流程：分别设置 Python、NumPy、PyTorch CPU 及可用 CUDA 设备种子。
    输出：无返回值；后续划分、初始化与采样使用设定的随机状态。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    """解析模型运行设备。

    输入：requested 为 auto、cpu 或 cuda。
    流程：auto 根据 CUDA 可用性选择设备；显式要求不可用 CUDA 时抛错。
    输出：对应的 torch.device。
    """
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_loader(dataset, batch_size, shuffle, workers, seed):
    """用指定采样种子构造批量数据迭代器。

    输入：数据集、批大小、是否打乱、工作进程数及种子。
    流程：初始化独立 torch.Generator，并配置 CUDA 可用时的锁页内存。
    输出：按给定批次和顺序提供 (触觉序列, 标签) 的 DataLoader。
    """
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    """执行完整一轮训练或无梯度评估。

    输入：模型、批量迭代器、损失函数、设备和可选优化器。
    流程：有优化器时前向、反传、梯度裁剪并更新权重；否则仅前向；
    按样本累加损失，收集真值与预测类别并计算分类指标。
    输出：包含平均 loss、accuracy、macro_f1 等字段的指标字典。
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets = []
    predictions = []

    # 验证阶段关闭梯度；训练阶段在更新前裁剪梯度以限制异常步长。
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for force_sequence, labels in loader:
            force_sequence = force_sequence.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(force_sequence)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += float(loss.detach()) * labels.size(0)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    metrics = classification_metrics(targets, predictions)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def class_weights(records, device):
    """按训练集频次计算各类别的损失权重。

    输入：带二分类标签的 episode 记录及目标计算设备。
    流程：统计每类样本数，按总数除以类别数和该类样本数。
    输出：位于目标设备的两元素权重张量，交给加权交叉熵使用。
    """
    counts = torch.bincount(
        torch.tensor([record.label for record in records]),
        minlength=len(CLASS_NAMES),
    ).float()
    return (counts.sum() / (len(CLASS_NAMES) * counts)).to(device)


def compact_metrics(metrics):
    """将一次训练或验证的指标转换为控制台摘要。

    输入：run_epoch 返回、包含 loss/accuracy/macro_f1 的字典。
    流程：按固定精度依次格式化三个指标。
    输出：可直接打印的单行字符串。
    """
    return (
        f"loss={metrics['loss']:.4f} "
        f"accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}"
    )


def make_run_dir(output_dir):
    """为新的训练运行分配独立目录。

    输入：训练结果的父目录路径。
    流程：使用当前时间和微秒构造目录名；如已存在则追加递增后缀。
    输出：已创建的 Path，后续保存配置、权重及指标文件。
    """
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = output_dir / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def main():
    """执行从数据发现到测试结果落盘的完整训练流程。

    输入：命令行的数据源、随机种子、窗口、超参数及绘图选项。
    流程：按 episode 分层划分并保存清单；构建数据集、模型和优化器；
    逐轮训练/验证，按验证集宏平均 F1 保存最佳权重并早停；最后加载
    最佳权重评估测试集，保存历史、结果并按选项生成图像。
    输出：无返回值；运行目录下生成 split/config/history/result 和 best.pt。
    """
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    data_dir = Path(args.data_dir).resolve()
    run_dir = make_run_dir(args.output_dir)

    # 以 episode 而非帧为单位划分数据，避免同一次抓取泄漏到不同集合。
    records = discover_episodes(data_dir, args.window_size)
    splits = stratified_split(records, seed=args.seed)
    manifest = manifest_from_splits(splits, data_dir)
    (run_dir / "split.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    run_config = vars(args).copy()
    run_config["data_dir"] = str(data_dir)
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    train_dataset = EpisodeDataset(
        splits["train"],
        window_size=args.window_size,
        window_policy=args.window_policy,
        augment=args.augment,
    )
    val_dataset = EpisodeDataset(
        splits["val"],
        window_size=args.window_size,
        window_policy=args.window_policy,
    )
    test_dataset = EpisodeDataset(
        splits["test"],
        window_size=args.window_size,
        window_policy=args.window_policy,
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.workers, args.seed
    )
    val_loader = make_loader(val_dataset, args.batch_size, False, args.workers, args.seed)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.workers, args.seed)

    model_config = ForceCnnTcnConfig()
    model = ForceCnnTcn(model_config).to(device)
    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights(splits["train"], device)
    )
    evaluation_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    split_summary = {
        name: {
            "total": len(items),
            "fail": sum(item.label == 0 for item in items),
            "success": sum(item.label == 1 for item in items),
        }
        for name, items in splits.items()
    }
    print(f"device: {device}")
    print(f"run: {run_dir}")
    print(f"window: {args.window_policy}, T={args.window_size}")
    print(f"splits: {split_summary}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    history = []
    best_macro_f1 = -1.0
    stale_epochs = 0
    checkpoint_path = run_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, train_criterion, device, optimizer
        )
        val_metrics = run_epoch(model, val_loader, evaluation_criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} train {compact_metrics(train_metrics)} "
            f"val {compact_metrics(val_metrics)}"
        )

        # 只用验证集 Macro-F1 选模型；测试集留到训练完全结束后使用。
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config.to_dict(),
                    "window_size": args.window_size,
                    "window_policy": args.window_policy,
                    "class_names": list(CLASS_NAMES),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break

    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(model, test_loader, evaluation_criterion, device)
    result = {
        "best_epoch": checkpoint["epoch"],
        "split_summary": split_summary,
        "test": test_metrics,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"test {compact_metrics(test_metrics)}")
    print(f"confusion_matrix: {test_metrics['confusion_matrix']}")
    if args.plot:
        from scripts.plot_results import generate_plots

        generate_plots([run_dir])


if __name__ == "__main__":
    main()
