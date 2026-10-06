import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from module.auto_grasp_controller import AutoGraspConfig, AutoGraspController
from module.piper_sim import PiperSim
from module.predict_episode import PredictionEpisodeBuffer, PredictionSession, prediction_window


class FakePredictor:
    """记录输入窗口的测试用预测器。"""
    window_size = 4
    window_policy = "first_contact"
    class_names = ["fail", "success"]

    def __init__(self):
        """准备记录预测器接收到的测试窗口。

        输入：无。
        流程：创建空 windows 列表用于追踪 predict 调用。
        输出：无返回值；后续断言可读取此列表。
        """
        self.windows = []

    def predict(self, frames):
        """模拟一次固定类别的窗口预测。

        输入：按时间排序的触觉帧数组序列 frames。
        流程：从每帧首个像素提取序号并保存输入窗口。
        输出：(成功类别 1, 成功概率 0.8)，用于验证会话调用次数。
        """
        self.windows.append([float(frame[0, 0, 0]) for frame in frames])
        return 1, 0.8


def frame(value):
    """生成可追踪帧序号的双指触觉帧。

    输入：填入全部触点的数值 value。
    流程：构造 2×30×15 的 float32 常量矩阵并包装成力图字典。
    输出：符合预测缓存接口的触觉帧。
    """
    return {"force_map": np.full((2, 30, 15), value, dtype=np.float32)}


class PredictionTests(unittest.TestCase):
    """验证预测缓存、窗口选取和自动抓取流程的对接。"""

    def test_buffer_copies_frames_and_does_not_write_files(self):
        """验证预测缓存只保留内存快照。

        输入：一帧触觉图和一个空的临时目录。
        流程：记录并结束回合，然后原地改动原帧；再结束一个空回合。
        输出：断言缓存保持原值、空回合形状正确且目录没有文件。
        """
        with tempfile.TemporaryDirectory() as directory:
            buffer = PredictionEpisodeBuffer()
            buffer.start(1)
            touch = frame(1)
            buffer.record_frame(touch, stage_id=0, state_label=1)
            touch["force_map"][:] = 5
            buffer.finish(False, reason="dropped")
            self.assertEqual(buffer.last_episode[0], 1)
            self.assertEqual(buffer.last_episode[1].shape, (1, 2, 30, 15))
            self.assertEqual(buffer.last_episode[1][0, 0, 0, 0], 1)
            self.assertFalse(buffer.recording)
            self.assertEqual(list(Path(directory).iterdir()), [])

            buffer.start(2)
            buffer.finish(False, reason="empty")
            self.assertEqual(buffer.last_episode[1].shape, (0, 2, 30, 15))

    def test_window_policy_and_short_episode(self):
        """验证在线与离线窗口策略对接触时刻的处理。

        输入：十帧力图、首次接触为第 6 帧的标签以及各窗口策略。
        流程：针对未结束、已结束、无接触和不足四帧逐项计算起点。
        输出：断言在线等待完整窗口，结束后按策略从头或末尾回退。
        """
        frames = np.zeros((10, 2, 30, 15), dtype=np.float32)
        labels = np.zeros(10, dtype=np.int8)
        labels[6] = 1
        self.assertIsNone(prediction_window(frames[:9], labels[:9], 4, "first_contact"))
        self.assertEqual(prediction_window(frames, labels, 4, "first_contact"), 6)
        self.assertEqual(prediction_window(frames[:8], labels[:8], 4, "first_contact", finished=True), 4)
        self.assertIsNone(prediction_window(frames, np.zeros(10), 4, "first_contact"))
        self.assertEqual(prediction_window(frames, np.zeros(10), 4, "first_contact", finished=True), 6)
        self.assertEqual(prediction_window(frames, labels, 4, "episode_start"), 0)
        self.assertIsNone(prediction_window(frames, labels, 4, "episode_end"))
        self.assertEqual(prediction_window(frames, labels, 4, "episode_end", finished=True), 6)
        self.assertIsNone(prediction_window(frames[:3], labels[:3], 4, "first_contact", finished=True))

    def test_session_predicts_once_per_episode_and_resets(self):
        """验证跨回合会话的预测时机与汇总计数。

        输入：依次为正常接触、无接触、短片段和过晚接触的四个回合。
        流程：逐帧调用 update，再结束回合读取结果记录与模型调用窗口。
        输出：断言只在可用窗口预测、短回合跳过、结束时回退窗口并
        正确累计预测次数和真实成功数。
        """
        predictor = FakePredictor()
        session = PredictionSession(predictor)
        buffer = PredictionEpisodeBuffer()
        buffer.start(1)
        for i in range(6):
            buffer.record_frame(frame(i), stage_id=0, state_label=int(i >= 2))
            session.update(buffer)
        self.assertEqual(predictor.windows, [[2, 3, 4, 5]])
        buffer.finish(True, "success_on_B")
        self.assertIn("accuracy=1/1", session.update(buffer))
        self.assertIsNone(session.update(buffer))

        buffer.start(2)
        for i in range(6):
            buffer.record_frame(frame(i), stage_id=0, state_label=0)
            session.update(buffer)
        self.assertIsNone(session.current)
        buffer.finish(False, "no_contact")
        self.assertIn("window=2:6", session.update(buffer))
        self.assertEqual(predictor.windows, [[2, 3, 4, 5], [2, 3, 4, 5]])
        self.assertEqual((session.correct, session.total, session.successes), (1, 2, 1))

        buffer.start(3)
        buffer.record_frame(frame(0), stage_id=0, state_label=0)
        session.update(buffer)
        buffer.finish(False, "early")
        self.assertIn("skipped", session.update(buffer))
        self.assertEqual((session.skipped, session.total), (1, 2))

        buffer.start(4)
        self.assertIsNone(buffer.last_episode)
        for i in range(6):
            buffer.record_frame(frame(i), stage_id=0, state_label=int(i == 5))
            session.update(buffer)
        self.assertIsNone(session.current)
        buffer.finish(False, "late_contact")
        self.assertIn("window=2:6", session.update(buffer))
        self.assertEqual(predictor.windows[-1], [2, 3, 4, 5])

    def test_controller_can_complete_with_prediction_buffer(self):
        """验证仿真控制器可以使用内存预测缓存替代采集器。

        输入：零触觉仿真、一次抓取配置和假预测器。
        流程：步进控制器并持续更新预测会话直到结束。
        输出：断言完成一次 no_contact 回合且产生一条可评估预测。
        """
        class ZeroTouch:
            """为预测流程提供无接触触觉帧。"""

            def get_frame(self):
                """为预测仿真提供不产生接触的测试帧。

                输入：无。
                流程：构造固定形状的力图和五维受力统计零数组。
                输出：供控制器读取的触觉数据字典。
                """
                return {"force_map": np.zeros((2, 30, 15), dtype=np.float32),
                        "force_stats": np.zeros(5, dtype=np.float32)}

        sim = PiperSim()
        buffer = PredictionEpisodeBuffer()
        controller = AutoGraspController(
            sim, ZeroTouch(), buffer,
            config=AutoGraspConfig(
                num_episodes=1, gripper_close_min=0.035, gripper_close_max=0.035,
                reset_settle_time=0.01, move_timeout=0.01,
                descend_timeout=0.01, close_timeout=0.02,
            ),
            seed=42,
        )
        session = PredictionSession(FakePredictor())
        for _ in range(20):
            if not controller.should_continue():
                break
            controller.before_step()
            mujoco.mj_step(sim.m, sim.d)
            controller.after_step()
            session.update(buffer)
        self.assertEqual(controller.completed_episodes, 1)
        self.assertEqual(buffer.last_episode[4], "no_contact")
        self.assertEqual(session.total, 1)


if __name__ == "__main__":
    unittest.main()
