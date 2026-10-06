"""绘制第一阶段训练与评估结果。"""

import argparse
import json
from pathlib import Path

import matplotlib


def parse_args():
    """读取训练结果绘图命令的参数。

    输入：一个或多个 run_* 目录、可选图片输出目录和展示开关。
    流程：通过 argparse 解析；未指定输出目录时由运行数量决定默认值。
    输出：供 main/generate_plots 使用的参数对象。
    """
    parser = argparse.ArgumentParser(
        description="Plot phase-one training curves and evaluation results"
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="one or more runs/phase1/run_* directories",
    )
    parser.add_argument(
        "--output-dir",
        help="defaults to RUN_DIR/plots for one run or runs/phase1/plots_summary",
    )
    parser.add_argument("--show", action="store_true", help="open plot windows")
    return parser.parse_args()


def load_json(path):
    """将磁盘 JSON 解析为 Python 对象。

    输入：path 指向 UTF-8 编码的 JSON 文件。
    流程：读取完整文本后交给 json.loads 解析。
    输出：对应的字典或列表；文件或 JSON 无效时透传异常。
    """
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(run_dir):
    """收集单次训练可用于绘图的数据。

    输入：一次训练运行目录 run_dir。
    流程：强制检查 history.json/result.json 存在，按需读取 config.json。
    输出：含 path、history、result、config 的字典；必需文件缺失时报错。
    """
    run_dir = Path(run_dir)
    required = ("history.json", "result.json")
    for filename in required:
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(run_dir / filename)

    config_path = run_dir / "config.json"
    return {
        "path": run_dir,
        "history": load_json(run_dir / "history.json"),
        "result": load_json(run_dir / "result.json"),
        "config": load_json(config_path) if config_path.is_file() else {},
    }


def configure_matplotlib(show):
    """准备 Matplotlib 绘图环境。

    输入：show 指示是否打开图形窗口。
    流程：非交互时先切换 Agg 后端，然后导入 pyplot 并设置
    图像分辨率、字体和坐标轴等全局绘图参数。
    输出：配置完成的 pyplot 模块对象。
    """
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    return plt


def save_figure(plt, figure, path, show):
    """统一完成一幅图的布局与保存。

    输入：pyplot、Figure、输出路径和窗口展示标志。
    流程：调整布局并写入 PNG，打印路径；按需显示后关闭 Figure。
    输出：无返回值；目标路径产生图片文件。
    """
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    print(path)
    if show:
        plt.show()
    plt.close(figure)


def plot_training_curves(plt, run, output_dir, show):
    """绘制一次训练中的 loss、accuracy 与 Macro-F1 曲线。

    输入：训练历史、最佳 epoch、输出目录与展示参数。
    流程：三幅子图分别画训练/验证指标并标出最佳轮次竖线。
    输出：无返回值；保存 training_curves.png。
    """
    history = run["history"]
    epochs = [row["epoch"] for row in history]
    best_epoch = run["result"]["best_epoch"]
    panels = (
        ("loss", "Cross-entropy loss", None),
        ("accuracy", "Accuracy", (0.0, 1.02)),
        ("macro_f1", "Macro-F1", (0.0, 1.02)),
    )

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, (metric, title, limits) in zip(axes, panels):
        for split, color in (("train", "#1565C0"), ("val", "#D1495B")):
            values = [row[split][metric] for row in history]
            axis.plot(epochs, values, label=split, color=color, linewidth=2)
        axis.axvline(
            best_epoch,
            color="#2E7D32",
            linestyle="--",
            linewidth=1.4,
            label=f"best epoch ({best_epoch})",
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.22)
        if limits:
            axis.set_ylim(*limits)
    axes[0].set_ylabel("Value")
    axes[-1].legend(loc="best")
    figure.suptitle(f"Training history: {run['path'].name}", fontsize=13)
    save_figure(plt, figure, output_dir / "training_curves.png", show)


def plot_confusion_matrix(plt, run, output_dir, show):
    """可视化测试集的绝对计数和按真实类别归一化的混淆矩阵。

    输入：测试结果中实际类别为行、预测类别为列的 2×2 矩阵。
    流程：逐行除以该真实类别总数，零样本行保持为零；并排显示
    原始计数和比例，在每个网格标注数值。
    输出：无返回值；保存 confusion_matrix.png。
    """
    matrix = run["result"]["test"]["confusion_matrix"]
    class_names = ("Fail", "Success")
    normalized = []
    # 每行对应一个真实类别，逐行归一化便于比较各类别的识别率。
    for row in matrix:
        total = sum(row)
        normalized.append([value / total if total else 0.0 for value in row])

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for axis, values, title, value_format in (
        (axes[0], matrix, "Counts", "count"),
        (axes[1], normalized, "Row-normalized", "percent"),
    ):
        image = axis.imshow(values, cmap="Blues", vmin=0, vmax=None if value_format == "count" else 1)
        axis.set_xticks(range(2), class_names)
        axis.set_yticks(range(2), class_names)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_title(title)
        for row_id in range(2):
            for column_id in range(2):
                value = values[row_id][column_id]
                text = str(value) if value_format == "count" else f"{value:.0%}"
                threshold = max(max(row) for row in values) * 0.55
                axis.text(
                    column_id,
                    row_id,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "#1F2933",
                    fontsize=12,
                    fontweight="bold",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Test confusion matrix", fontsize=13)
    save_figure(plt, figure, output_dir / "confusion_matrix.png", show)


def plot_class_metrics(plt, run, output_dir, show):
    """生成失败与成功类别的分类指标柱状图。

    输入：测试结果中的 per_class 精确率、召回率及 F1。
    流程：按两类各绘三组并列柱状条，在柱顶标注小数值。
    输出：无返回值；保存 class_metrics.png。
    """
    per_class = run["result"]["test"]["per_class"]
    metrics = ("precision", "recall", "f1")
    colors = ("#1565C0", "#D1495B", "#2E7D32")
    x_positions = (0, 1)
    width = 0.23

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for metric_id, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (metric_id - 1) * width
        values = [per_class[class_id][metric] for class_id in range(2)]
        bars = axis.bar(
            [position + offset for position in x_positions],
            values,
            width,
            label=metric.capitalize(),
            color=color,
        )
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3)
    axis.set_xticks(x_positions, ("Fail", "Success"))
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Score")
    axis.set_title("Test metrics by class")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncols=3, loc="upper center")
    save_figure(plt, figure, output_dir / "class_metrics.png", show)


def run_label(run):
    """为对比图中的单次训练生成标签。

    输入：包含 config.seed 及目录 path 的运行信息。
    流程：有种子时显示 seed 值，否则采用结果目录名称。
    输出：适合用作横轴刻度的字符串。
    """
    seed = run["config"].get("seed")
    return f"seed {seed}" if seed is not None else run["path"].name


def plot_run_comparison(plt, runs, output_dir, show):
    """比较多个训练运行的测试集表现。

    输入：至少一个 run 字典及绘图输出选项。
    流程：抽取每次运行的 accuracy 和 macro_f1，按运行标签绘制
    并列柱状条并标注分数。
    输出：无返回值；保存 run_comparison.png。
    """
    labels = [run_label(run) for run in runs]
    accuracy = [run["result"]["test"]["accuracy"] for run in runs]
    macro_f1 = [run["result"]["test"]["macro_f1"] for run in runs]
    x_positions = list(range(len(runs)))
    width = 0.34

    figure, axis = plt.subplots(figsize=(max(7, len(runs) * 1.8), 4.4))
    accuracy_bars = axis.bar(
        [position - width / 2 for position in x_positions],
        accuracy,
        width,
        label="Accuracy",
        color="#1565C0",
    )
    f1_bars = axis.bar(
        [position + width / 2 for position in x_positions],
        macro_f1,
        width,
        label="Macro-F1",
        color="#D1495B",
    )
    axis.bar_label(accuracy_bars, labels=[f"{value:.3f}" for value in accuracy], padding=3)
    axis.bar_label(f1_bars, labels=[f"{value:.3f}" for value in macro_f1], padding=3)
    axis.set_xticks(x_positions, labels)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Test score")
    axis.set_title("Test results across runs")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(ncols=2, loc="upper center")
    save_figure(plt, figure, output_dir / "run_comparison.png", show)


def main():
    """运行第一阶段结果绘图命令。

    输入：命令行提供的训练目录、输出目录和 show 开关。
    流程：解析参数后转交 generate_plots 加载并绘图。
    输出：无返回值；图片保存路径由被调用函数打印。
    """
    args = parse_args()
    generate_plots(args.run_dirs, output_dir=args.output_dir, show=args.show)


def generate_plots(run_dirs, output_dir=None, show=False):
    """批量读取训练结果并生成需要的全部图表。

    输入：run_dirs 路径列表、可选 output_dir 和 show 开关。
    流程：读取所有运行；按单次或多次选择输出目录并创建，
    为第一项生成曲线、矩阵和分类别图，多次运行额外生成对比图。
    输出：无返回值；结果以 PNG 文件保存在选定目录。
    """
    runs = [load_run(path) for path in run_dirs]
    if output_dir:
        output_dir = Path(output_dir)
    elif len(runs) == 1:
        output_dir = runs[0]["path"] / "plots"
    else:
        output_dir = Path("runs/phase1/plots_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt = configure_matplotlib(show)
    plot_training_curves(plt, runs[0], output_dir, show)
    plot_confusion_matrix(plt, runs[0], output_dir, show)
    plot_class_metrics(plt, runs[0], output_dir, show)
    if len(runs) > 1:
        plot_run_comparison(plt, runs, output_dir, show)


if __name__ == "__main__":
    main()
