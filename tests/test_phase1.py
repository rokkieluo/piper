from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from models import ForceCnnTcn
from training.data import (
    EpisodeDataset,
    discover_episodes,
    stratified_split,
)
from training.metrics import classification_metrics
from train import make_run_dir


class PhaseOneTests(unittest.TestCase):
    """验证第一阶段模型、数据窗口、划分与训练指标。"""

    def test_model_output_and_causality(self):
        """验证模型输出形状与时间卷积的因果性。

        输入：两批 20 帧触觉图，以及只改动后十帧的另一份序列。
        流程：执行完整前向，并对两组逐帧特征运行空间/时间编码器。
        输出：断言 logits 为 [2,2]，前十帧时间特征不受未来输入影响。
        """
        torch.manual_seed(0)
        model = ForceCnnTcn().eval()
        sequence = torch.rand(2, 20, 2, 30, 15)
        changed_future = sequence.clone()
        changed_future[:, 10:] = torch.rand_like(changed_future[:, 10:])

        logits = model(sequence)
        self.assertEqual(tuple(logits.shape), (2, 2))

        with torch.no_grad():
            frames = sequence.reshape(40, 2, 30, 15)
            other_frames = changed_future.reshape(40, 2, 30, 15)
            spatial = model.spatial_encoder(frames).reshape(2, 20, -1).transpose(1, 2)
            other = model.spatial_encoder(other_frames).reshape(2, 20, -1).transpose(1, 2)
            encoded = model.temporal_encoder(spatial)
            other_encoded = model.temporal_encoder(other)
        torch.testing.assert_close(encoded[:, :, :10], other_encoded[:, :, :10])

    def test_dataset_and_stratified_split(self):
        """验证数据发现、分层划分和增强后的输入格式。

        输入：临时目录中的五个成功及五个失败 NPZ 片段。
        流程：发现并分层划分样本，再从训练集读取原始及增强窗口。
        输出：断言各集合不重复、均含双类别、窗口形状和无效 taxel 正确。
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(10):
                np.savez_compressed(
                    root / f"episode_{index}.npz",
                    force_map=np.zeros((25, 2, 30, 15), dtype=np.float32),
                    success=np.asarray(index >= 5, dtype=np.bool_),
                )

            records = discover_episodes(root, window_size=20)
            splits = stratified_split(records, seed=7, train_ratio=0.6, val_ratio=0.2)
            all_paths = [record.path for items in splits.values() for record in items]
            self.assertEqual(len(all_paths), 10)
            self.assertEqual(len(set(all_paths)), 10)
            for items in splits.values():
                self.assertEqual({record.label for record in items}, {0, 1})

            sequence, label = EpisodeDataset(splits["train"], window_size=20)[0]
            self.assertEqual(tuple(sequence.shape), (20, 2, 30, 15))
            self.assertIn(label.item(), (0, 1))

            augmented, _ = EpisodeDataset(
                splits["train"], window_size=20, augment=True
            )[0]
            self.assertTrue(torch.all(augmented[:, :, :15, 12:] == 0))

    def test_metrics(self):
        """验证已知标签序列的基础分类指标。

        输入：四个真实标签和四个预测标签。
        流程：调用 classification_metrics 汇总二分类结果。
        输出：断言混淆矩阵为 [[1,1],[0,2]]、准确率为 0.75。
        """
        metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)

    def test_first_contact_window_policy(self):
        """验证 first_contact 对正常接触片段的截窗。

        输入：十二帧力图，其中状态标签从第 5 帧开始出现接触。
        流程：构造四帧窗口数据集并读取第一条样本。
        输出：断言窗口取第 5 至 8 帧，包含首次接触时刻。
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            force_map = np.zeros((12, 2, 30, 15), dtype=np.float32)
            force_map[:, :, 0, 0] = np.arange(12, dtype=np.float32)[:, None]
            state_label = np.zeros(12, dtype=np.int8)
            state_label[5:] = 1
            path = root / "contact.npz"
            np.savez_compressed(
                path,
                force_map=force_map,
                state_label=state_label,
                success=np.asarray(True, dtype=np.bool_),
            )
            records = discover_episodes(root, window_size=4)

            sequence, _ = EpisodeDataset(
                records,
                window_size=4,
                window_policy="first_contact",
            )[0]

            torch.testing.assert_close(
                sequence[:, 0, 0, 0],
                torch.tensor([5.0, 6.0, 7.0, 8.0]),
            )

    def test_no_contact_uses_episode_end_window(self):
        """验证全程无接触时使用回合末尾作为模型输入。

        输入：十二帧力图、全零状态标签及窗口大小 4。
        流程：以 first_contact 策略读取数据集。
        输出：断言返回最后四帧，即索引 8 至 11。
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            force_map = np.zeros((12, 2, 30, 15), dtype=np.float32)
            force_map[:, :, 0, 0] = np.arange(12, dtype=np.float32)[:, None]
            path = root / "no_contact.npz"
            np.savez_compressed(
                path,
                force_map=force_map,
                state_label=np.zeros(12, dtype=np.int8),
                success=np.asarray(False, dtype=np.bool_),
            )
            records = discover_episodes(root, window_size=4)

            sequence, _ = EpisodeDataset(
                records,
                window_size=4,
                window_policy="first_contact",
            )[0]

            torch.testing.assert_close(
                sequence[:, 0, 0, 0],
                torch.tensor([8.0, 9.0, 10.0, 11.0]),
            )

    def test_legacy_episode_without_state_labels_starts_at_zero(self):
        """验证旧格式片段缺少 state_label 时的兼容规则。

        输入：八帧力图与最终 success 字段，但无逐帧状态标签。
        流程：用 first_contact 策略读取四帧窗口。
        输出：断言选取开头第 0 至 3 帧。
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            force_map = np.zeros((8, 2, 30, 15), dtype=np.float32)
            force_map[:, :, 0, 0] = np.arange(8, dtype=np.float32)[:, None]
            path = root / "legacy.npz"
            np.savez_compressed(
                path,
                force_map=force_map,
                success=np.asarray(True, dtype=np.bool_),
            )
            records = discover_episodes(root, window_size=4)

            sequence, _ = EpisodeDataset(
                records,
                window_size=4,
                window_policy="first_contact",
            )[0]

            torch.testing.assert_close(
                sequence[:, 0, 0, 0],
                torch.tensor([0.0, 1.0, 2.0, 3.0]),
            )

    def test_run_directories_are_unique(self):
        """验证运行目录创建不会覆盖既有训练结果。

        输入：同一个临时父目录。
        流程：连续调用两次 make_run_dir。
        输出：断言两个路径不同且目录均已创建。
        """
        with tempfile.TemporaryDirectory() as directory:
            first = make_run_dir(directory)
            second = make_run_dir(directory)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())


if __name__ == "__main__":
    unittest.main()
