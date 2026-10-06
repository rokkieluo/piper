import csv
import json
from pathlib import Path
import tempfile
import unittest

from module.predict_results import PredictionResults


class FakePredictor:
    """提供结果保存测试所需的预测器配置。"""
    window_policy = "first_contact"
    window_size = 4
    class_names = ("fail", "success")


class PredictionResultsTests(unittest.TestCase):
    """验证预测结果文件和批次目录的生成。"""

    def test_results_are_saved_incrementally_and_skips_are_excluded(self):
        """验证预测结果逐回合写入及短片段统计排除规则。

        输入：三条有效预测和一条无预测记录、临时输出目录。
        流程：逐条 record 后读取 results.json 和 episodes.csv。
        输出：断言 checkpoint/配置、准确率 2/3、混淆矩阵及
        空预测 CSV 字段一致，跳过回合不进入准确率分母。
        """
        with tempfile.TemporaryDirectory() as directory:
            results = PredictionResults(directory, "best.pt", 42, FakePredictor())
            path = results.run_dir / "results.json"
            self.assertTrue(path.exists())
            self.assertEqual(json.loads(path.read_text())["summary"]["accuracy"], None)

            rows = [
                (1, "fail", "fail", 0.2, True),
                (2, "success", "fail", 0.3, False),
                (3, "success", "success", 0.8, True),
                (4, "fail", None, None, None),
            ]
            for episode, actual, predicted, probability, correct in rows:
                results.record({
                    "episode": episode, "actual": actual, "predicted": predicted,
                    "probability_success": probability, "correct": correct,
                    "reason": "test", "frames": 5 if predicted else 2,
                    "window_start": 1 if predicted else None,
                    "window_end": 5 if predicted else None,
                })

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(Path(payload["checkpoint"]).name, "best.pt")
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["window_size"], 4)
            self.assertEqual(payload["summary"], {
                "completed": 4, "evaluated": 3, "skipped": 1,
                "correct": 2, "accuracy": 2 / 3,
                "confusion_matrix": [[1, 0], [1, 1]],
            })
            self.assertEqual(len(payload["episodes"]), 4)
            with (results.run_dir / "episodes.csv").open(newline="", encoding="utf-8") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 4)
            self.assertEqual(csv_rows[1]["probability_success"], "0.3")
            self.assertEqual(csv_rows[3]["predicted"], "")

    def test_runs_have_unique_directories(self):
        """验证同一父目录可以保存多次预测。

        输入：共同的临时输出目录及同一个模型配置。
        流程：连续构造两个 PredictionResults 实例。
        输出：断言各自 run_dir 不同，防止覆盖旧预测数据。
        """
        with tempfile.TemporaryDirectory() as directory:
            first = PredictionResults(directory, "best.pt", None, FakePredictor())
            second = PredictionResults(directory, "best.pt", None, FakePredictor())
            self.assertNotEqual(first.run_dir, second.run_dir)


if __name__ == "__main__":
    unittest.main()
