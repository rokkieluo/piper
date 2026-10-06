import csv
from datetime import datetime
import json
from pathlib import Path


FIELDS = (
    "episode", "actual", "predicted", "probability_success", "correct",
    "reason", "frames", "window_start", "window_end",
)


class PredictionResults:
    """将推理结果单独保存，不混入训练数据集。"""

    def __init__(self, output_root, checkpoint, seed, predictor):
        """创建预测运行目录并记录模型及实验配置。

        输入：输出父目录、checkpoint、随机种子和预测器配置。
        流程：以含微秒的时间戳创建不覆盖旧数据的目录，保存参数，
        将空的初始统计写入 JSON 和 CSV。
        输出：无返回值；run_dir 指向本次预测结果目录。
        """
        self.run_dir = Path(output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.checkpoint = str(Path(checkpoint).resolve())
        self.seed = seed
        self.predictor = predictor
        self.episodes = []
        self.save()

    def record(self, episode):
        """持久化单个回合的预测记录。

        输入：含 episode、实际/预测标签、概率和窗口信息的字典。
        流程：复制该记录追加到列表，然后重写累计 JSON 与 CSV。
        输出：无返回值；已完成回合即使提前退出也可在磁盘查看。
        """
        self.episodes.append(dict(episode))
        self.save()

    def summary(self):
        """按已保存回合计算预测表现。

        输入：实例中所有回合的真实标签、预测标签与正确性。
        流程：过滤无预测的短片段；按真实/预测 fail、success 统计
        混淆矩阵，计算完成数、有效数、跳过数和准确率。
        输出：含上述统计的字典；没有有效片段时 accuracy 为 None。
        """
        # 少于 T 帧的回合没有预测类别，不参与准确率和混淆矩阵计算。
        valid = [row for row in self.episodes if row["predicted"] is not None]
        confusion = [
            [sum(row["actual"] == actual and row["predicted"] == predicted for row in valid)
             for predicted in ("fail", "success")]
            for actual in ("fail", "success")
        ]
        correct = sum(row["correct"] for row in valid)
        return {
            "completed": len(self.episodes),
            "evaluated": len(valid),
            "skipped": len(self.episodes) - len(valid),
            "correct": correct,
            "accuracy": correct / len(valid) if valid else None,
            "confusion_matrix": confusion,
        }

    def save(self):
        """将当前预测配置、记录和汇总写入磁盘。

        输入：checkpoint 信息、窗口配置与内存中的回合记录。
        流程：组装包含 summary 和 episodes 的 JSON，先写临时文件
        再替换正式文件；同时以 UTF-8 重写 episodes.csv 表格。
        输出：无返回值；结果位于 run_dir 下的 JSON/CSV 文件。
        """
        payload = {
            "checkpoint": self.checkpoint,
            "seed": self.seed,
            "window_policy": self.predictor.window_policy,
            "window_size": self.predictor.window_size,
            "class_names": list(self.predictor.class_names),
            "summary": self.summary(),
            "episodes": self.episodes,
        }
        # 先写临时文件再替换，防止中断时留下不完整的 JSON。
        pending = self.run_dir / "results.json.tmp"
        pending.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        pending.replace(self.run_dir / "results.json")
        with (self.run_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.episodes)

    def finish(self):
        """完成预测运行并输出最终汇总。

        输入：本次运行累计的回合记录。
        流程：再次持久化文件，打印有效数/准确率；有回合数据时
        尝试绘图，未安装 matplotlib 时只提示并保留表格数据。
        输出：无返回值；可能额外生成 prediction_results.png。
        """
        self.save()
        print(f"Prediction results: {self.run_dir / 'results.json'}")
        print(f"Episode table: {self.run_dir / 'episodes.csv'}")
        summary = self.summary()
        accuracy = "N/A" if summary["accuracy"] is None else f"{summary['accuracy']:.1%}"
        print(f"Completed: {summary['completed']}; evaluated: {summary['evaluated']}; "
              f"skipped: {summary['skipped']}; accuracy: {accuracy}")
        if not self.episodes:
            return
        try:
            self.plot()
        except ImportError:
            print("Plot unavailable: install matplotlib (pip install -r requirements-train.txt).")

    def plot(self):
        """生成用于检查预测效果的两幅统计图。

        输入：有预测的各回合成功概率、真实标签及汇总混淆矩阵。
        流程：使用无界面绘图后端；左侧以点图对照概率与真值，右侧
        绘制实际类别为行的混淆矩阵，缺少有效预测时给出空状态。
        输出：无返回值；保存 run_dir/prediction_results.png 并关闭画布。
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = [row for row in self.episodes if row["predicted"] is not None]
        matrix = self.summary()["confusion_matrix"]
        figure, (probability_axis, matrix_axis) = plt.subplots(1, 2, figsize=(12, 4.2))

        if valid:
            indices = [row["episode"] for row in valid]
            probability_axis.scatter(indices, [row["probability_success"] for row in valid],
                                     color=["#28845A" if row["correct"] else "#C44536" for row in valid],
                                     s=55, label="P(success): correct / incorrect", zorder=3)
            probability_axis.scatter(indices, [int(row["actual"] == "success") for row in valid],
                                     marker="_", color="#193D58", s=190, linewidths=2,
                                     label="Actual (0=fail, 1=success)", zorder=4)
            probability_axis.set_xticks(indices if len(indices) <= 20 else indices[::max(1, len(indices) // 20)])
            probability_axis.legend(loc="best", fontsize=8)
        else:
            probability_axis.text(0.5, 0.5, "No evaluable episodes", ha="center", va="center",
                                  transform=probability_axis.transAxes)
        probability_axis.axhline(0.5, color="#82909A", linestyle="--", linewidth=1)
        probability_axis.set_ylim(-0.08, 1.08)
        probability_axis.set_xlabel("Episode")
        probability_axis.set_ylabel("P(success) / actual label")
        probability_axis.set_title("Predictions by episode")
        probability_axis.grid(axis="y", alpha=0.2)

        matrix_axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, max(map(max, matrix))))
        matrix_axis.set_xticks((0, 1), ("Fail", "Success"))
        matrix_axis.set_yticks((0, 1), ("Fail", "Success"))
        matrix_axis.set_xlabel("Predicted")
        matrix_axis.set_ylabel("Actual")
        matrix_axis.set_title("Confusion matrix")
        for actual in range(2):
            for predicted in range(2):
                value = matrix[actual][predicted]
                matrix_axis.text(predicted, actual, str(value), ha="center", va="center",
                                 color="white" if value > max(map(max, matrix)) / 2 else "#25313A")
        figure.tight_layout()
        output = self.run_dir / "prediction_results.png"
        figure.savefig(output, dpi=180)
        plt.close(figure)
        print(f"Prediction plot: {output}")
