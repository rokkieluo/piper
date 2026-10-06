import json
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from module.auto_grasp_controller import (
    AutoGraspConfig,
    AutoGraspController,
    GraspState,
    Stage,
)
from module.data_collector import TactileEpisodeCollector
from module.piper_sim import PiperSim


class ZeroTouchProcessor:
    """提供不产生接触信号的测试用触觉输入。"""

    def get_frame(self):
        """生成没有任何触觉接触的模拟输入。

        输入：无，每次重新创建测试用力图。
        流程：为双指力图、位置图、原始力及统计量填入零值。
        输出：字段和形状与真实处理器一致的触觉帧字典。
        """
        force_map = np.zeros((2, 30, 15), dtype=np.float32)
        return {
            "force_map": force_map,
            "position_map": force_map.copy(),
            "raw_force": np.zeros((810, 3), dtype=np.float32),
            "force_stats": np.zeros(5, dtype=np.float32),
        }


class DataCollectionTests(unittest.TestCase):
    """验证采集文件结构及自动抓取的结束与标签判定。"""

    def setUp(self):
        """准备独立的自动采集测试环境。

        输入：unittest 在每个测试执行前调用此方法。
        流程：新建 PiperSim 场景及全零触觉处理器。
        输出：无返回值；仿真和处理器保存为实例成员。
        """
        self.sim = PiperSim()
        self.touch_processor = ZeroTouchProcessor()

    def test_collector_writes_frame_schema_and_metadata(self):
        """验证单帧采集的文件结构与最终标签。

        输入：临时目录、零触觉帧和带物体属性的片段元数据。
        流程：启动、记录一帧并以无接触失败结束，再读取 NPZ/JSON。
        输出：断言所有逐帧字段维度、失败标签及结束原因正确。
        """
        with tempfile.TemporaryDirectory() as directory:
            collector = TactileEpisodeCollector(
                self.sim,
                self.touch_processor,
                save_dir=directory,
            )
            collector.start(
                3,
                metadata={
                    "object": {"shape": "mjGEOM_BOX", "mass": 0.03},
                    "stage_names": {"0": "reset_settle"},
                    "state_names": {"0": "no_contact"},
                },
            )
            collector.record_frame(
                stage_id=0,
                state_label=GraspState.NO_CONTACT,
                contact_mask=[False, False],
            )
            save_path = collector.finish(False, reason="no_contact")

            self.assertIsNotNone(save_path)
            self.assertTrue(save_path.exists())
            with np.load(save_path, allow_pickle=False) as data:
                expected_frame_keys = {
                    "time",
                    "force_map",
                    "position_map",
                    "raw_force",
                    "force_stats",
                    "stage_id",
                    "state_label",
                    "contact_mask",
                    "object_on_b",
                    "gripper_object_contact",
                    "gripper_released",
                    "tcp_pos",
                    "tcp_quat",
                    "object_pos",
                    "object_quat",
                    "object_vel",
                    "gripper_ctrl",
                    "gripper_qpos",
                }
                self.assertTrue(expected_frame_keys.issubset(data.files))
                for key in expected_frame_keys:
                    self.assertEqual(data[key].shape[0], 1, key)
                self.assertFalse(bool(data["success"]))
                self.assertEqual(int(data["state_label"][0]), GraspState.NO_CONTACT)

            metadata = json.loads(save_path.with_suffix(".json").read_text())
            self.assertEqual(metadata["episode_result"], "fail")
            self.assertEqual(metadata["finish_reason"], "no_contact")
            self.assertEqual(metadata["object"]["mass"], 0.03)

    def test_no_contact_attempt_is_saved_as_failed_episode(self):
        """验证从未建立接触时仍产生完整失败样本。

        输入：零触觉处理器和缩短各阶段超时的抓取配置。
        流程：步进仿真直至单回合完成，读取采集 NPZ/JSON。
        输出：断言结果为 no_contact 且包含多帧标签与来源参数。
        """
        config = AutoGraspConfig(
            num_episodes=1,
            gripper_close_min=0.035,
            gripper_close_max=0.035,
            reset_settle_time=0.01,
            move_timeout=0.01,
            descend_timeout=0.01,
            close_timeout=0.02,
        )
        with tempfile.TemporaryDirectory() as directory:
            collector = TactileEpisodeCollector(
                self.sim,
                self.touch_processor,
                save_dir=directory,
            )
            controller = AutoGraspController(
                self.sim,
                self.touch_processor,
                collector,
                config=config,
                seed=42,
            )

            for _ in range(20):
                if not controller.should_continue():
                    break
                controller.before_step()
                mujoco.mj_step(self.sim.m, self.sim.d)
                controller.after_step()

            self.assertFalse(controller.should_continue())
            self.assertEqual(controller.completed_episodes, 1)
            self.assertEqual(controller.last_finish_reason, "no_contact")
            self.assertIsNotNone(collector.last_saved_path)

            with np.load(collector.last_saved_path, allow_pickle=False) as data:
                self.assertGreater(data["time"].shape[0], 1)
                self.assertEqual(int(data["stage_id"][0]), 0)
                self.assertIn(3, data["stage_id"])
                self.assertTrue(np.all(data["state_label"] == GraspState.NO_CONTACT))

            metadata = json.loads(collector.last_metadata_path.read_text())
            self.assertEqual(metadata["finish_reason"], "no_contact")
            self.assertEqual(metadata["attempt_index"], 1)
            self.assertIn("object", metadata)
            self.assertIn("label_thresholds", metadata)

    def test_drop_is_counted_once_then_next_attempt_is_reset(self):
        """验证掉落失败不会在等待复位期间重复写盘。

        输入：设为搬运阶段且已抬起的控制器及短暂复位等待参数。
        流程：触发掉落，继续步进直到启动第二次抓取。
        输出：断言仅有一个失败文件，完成数不重复且下一回合已记录。
        """
        config = AutoGraspConfig(
            num_episodes=3,
            drop_confirm_frames=1,
            fail_reset_wait=0.03,
        )
        with tempfile.TemporaryDirectory() as directory:
            collector = TactileEpisodeCollector(
                self.sim,
                self.touch_processor,
                save_dir=directory,
            )
            controller = AutoGraspController(
                self.sim,
                self.touch_processor,
                collector,
                config=config,
                seed=42,
            )
            controller.stage = Stage.MOVE_ABOVE_B
            controller.stage_start_time = float(self.sim.d.time)
            controller.lifted = True

            controller.before_step()
            mujoco.mj_step(self.sim.m, self.sim.d)
            controller.after_step()

            self.assertEqual(controller.completed_episodes, 1)
            self.assertEqual(controller.stage, Stage.FAIL_RESET_WAIT)
            self.assertFalse(collector.recording)
            failed_path = collector.last_saved_path

            for _ in range(4):
                controller.before_step()
                mujoco.mj_step(self.sim.m, self.sim.d)
                controller.after_step()

            self.assertEqual(controller.completed_episodes, 1)
            self.assertEqual(controller.attempt_index, 2)
            self.assertEqual(controller.stage, Stage.RESET_SETTLE)
            self.assertTrue(collector.recording)
            self.assertEqual(collector.episode_index, 2)
            self.assertEqual(collector.last_saved_path, failed_path)
            self.assertEqual(len(list(collector.save_dir.glob("*.npz"))), 1)

    def test_success_requires_placement_contact_then_release(self):
        """验证 B 区成功标签需要满足按顺序的接触和释放证据。

        输入：进入 B 区下降的控制器及模拟的物体-夹爪接触序列。
        流程：先模拟仅触碰台面、再同时触碰台面和夹爪，最后张开夹爪。
        输出：断言只在最终释放后成功，逐帧证据与 JSON 确认字段一致。
        """
        config = AutoGraspConfig(
            num_episodes=1,
            drop_confirm_frames=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            collector = TactileEpisodeCollector(
                self.sim,
                self.touch_processor,
                save_dir=directory,
            )
            controller = AutoGraspController(
                self.sim,
                self.touch_processor,
                collector,
                config=config,
                seed=42,
            )
            controller.stage = Stage.DESCEND_TO_B
            controller.stage_start_time = float(self.sim.d.time)
            controller.target_pos = controller._place_pos()
            controller.lifted = True
            controller._set_gripper(
                controller.gripper_close_target,
                set_qpos=True,
            )
            mujoco.mj_forward(self.sim.m, self.sim.d)

            with (
                patch.object(controller, "_object_on_region_b", return_value=True),
                patch.object(
                    controller,
                    "_gripper_contacts_object",
                    side_effect=[False, True, False, False],
                ),
            ):
                controller.after_step()
                self.assertEqual(controller.stage, Stage.DESCEND_TO_B)
                self.assertFalse(controller.placement_confirmed)
                self.assertEqual(controller.completed_episodes, 0)

                controller.after_step()
                self.assertEqual(controller.stage, Stage.RELEASE_ON_B)
                self.assertTrue(controller.placement_confirmed)
                self.assertEqual(controller.completed_episodes, 0)

                controller.after_step()
                self.assertEqual(controller.stage, Stage.RELEASE_ON_B)
                self.assertFalse(controller.release_confirmed)
                self.assertEqual(controller.completed_episodes, 0)

                controller._set_gripper(controller.config.open_gripper, set_qpos=True)
                mujoco.mj_forward(self.sim.m, self.sim.d)
                controller.after_step()

            self.assertEqual(controller.completed_episodes, 1)
            self.assertEqual(controller.last_finish_reason, "success_on_B")
            self.assertTrue(controller.release_confirmed)

            with np.load(collector.last_saved_path, allow_pickle=False) as data:
                np.testing.assert_array_equal(
                    data["gripper_object_contact"],
                    [False, True, False, False],
                )
                np.testing.assert_array_equal(
                    data["gripper_released"],
                    [False, False, False, True],
                )
                self.assertTrue(np.all(data["object_on_b"]))

            metadata = json.loads(collector.last_metadata_path.read_text())
            self.assertTrue(metadata["placement_confirmed"])
            self.assertTrue(metadata["release_confirmed"])

    def test_placement_target_descends_instead_of_waiting_at_fixed_height(self):
        """验证 B 区放置目标逐步下降。

        输入：B 区下降阶段及指定 placement_descent_speed。
        流程：记录初始 z 后调用一次 before_step。
        输出：断言目标高度降低量等于速度乘仿真步长。
        """
        config = AutoGraspConfig(
            num_episodes=1,
            placement_descent_speed=0.02,
        )
        with tempfile.TemporaryDirectory() as directory:
            collector = TactileEpisodeCollector(
                self.sim,
                self.touch_processor,
                save_dir=directory,
            )
            controller = AutoGraspController(
                self.sim,
                self.touch_processor,
                collector,
                config=config,
                seed=42,
            )
            controller.stage = Stage.DESCEND_TO_B
            controller.target_pos = controller._place_pos()
            initial_target_z = float(controller.target_pos[2])

            controller.before_step()

            expected_step = config.placement_descent_speed * self.sim.m.opt.timestep
            self.assertAlmostEqual(
                controller.target_pos[2],
                initial_target_z - expected_step,
            )

    def test_grasp_loss_ends_episode_without_waiting_for_timeout(self):
        """验证物体离开夹爪后及时终止搬运回合。

        输入：已抬起且正在搬运、连续无夹爪接触的控制器。
        流程：连续调用两次 after_step，避开 B 区接触并禁用掉落干扰。
        输出：断言结束原因为 grasp_lost，缓存仅有两帧。
        """
        config = AutoGraspConfig(
            num_episodes=1,
            drop_confirm_frames=100,
            grasp_loss_confirm_frames=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            collector = TactileEpisodeCollector(
                self.sim,
                self.touch_processor,
                save_dir=directory,
            )
            controller = AutoGraspController(
                self.sim,
                self.touch_processor,
                collector,
                config=config,
                seed=42,
            )
            controller.stage = Stage.MOVE_ABOVE_B
            controller.target_pos = controller._above_B_pos()
            controller.lifted = True

            with (
                patch.object(controller, "_object_on_region_b", return_value=False),
                patch.object(
                    controller,
                    "_gripper_contacts_object",
                    return_value=False,
                ),
            ):
                controller.after_step()
                self.assertEqual(controller.completed_episodes, 0)
                controller.after_step()

            self.assertEqual(controller.completed_episodes, 1)
            self.assertEqual(controller.last_finish_reason, "grasp_lost")
            self.assertEqual(collector.last_num_frames, 2)

    def test_state_labels_confirm_stability_then_detect_contact_loss(self):
        """验证稳定/滑移标签的连续帧确认规则。

        输入：已抬起物体，两侧接触力图及随后单侧接触力图。
        流程：各喂入两帧，调用 _classify_state 跟踪状态变化。
        输出：断言首次为弱接触、确认后稳定，再经两帧确认滑移。
        """
        config = AutoGraspConfig(
            stable_confirm_frames=2,
            slip_confirm_frames=2,
            side_contact_force=0.01,
            contact_start_force=0.02,
            contact_end_force=0.01,
        )
        with tempfile.TemporaryDirectory() as directory:
            collector = TactileEpisodeCollector(
                self.sim,
                self.touch_processor,
                save_dir=directory,
            )
            controller = AutoGraspController(
                self.sim,
                self.touch_processor,
                collector,
                config=config,
                seed=7,
            )
            controller.lifted = True
            controller.stage = Stage.MOVE_ABOVE_B

            bilateral = self.touch_processor.get_frame()
            bilateral["force_map"][:, 0, 0] = 0.02
            bilateral["force_stats"][0] = 0.04
            first, _ = controller._classify_state(bilateral, dropped=False)
            second, _ = controller._classify_state(bilateral, dropped=False)
            self.assertEqual(first, GraspState.WEAK_CONTACT)
            self.assertEqual(second, GraspState.STABLE_GRASP)

            unilateral = self.touch_processor.get_frame()
            unilateral["force_map"][0, 0, 0] = 0.02
            unilateral["force_stats"][0] = 0.02
            third, mask = controller._classify_state(unilateral, dropped=False)
            fourth, _ = controller._classify_state(unilateral, dropped=False)
            self.assertEqual(third, GraspState.WEAK_CONTACT)
            self.assertEqual(fourth, GraspState.SLIP_OR_DROP)
            np.testing.assert_array_equal(mask, [True, False])

    def test_initial_metadata_matches_scene_object(self):
        """验证启动回合时写入的场景元信息。

        输入：指定种子且新建的自动抓取控制器。
        流程：读取采集器在初始化首回合时保存的元数据。
        输出：断言 schema、物体属性、阶段/状态名称和时间步长一致。
        """
        with tempfile.TemporaryDirectory() as directory:
            collector = TactileEpisodeCollector(
                self.sim,
                self.touch_processor,
                save_dir=directory,
            )
            controller = AutoGraspController(
                self.sim,
                self.touch_processor,
                collector,
                config=AutoGraspConfig(num_episodes=1),
                seed=1,
            )
            metadata = collector.episode_metadata
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["attempt_index"], 1)
            self.assertEqual(metadata["seed"], 1)
            self.assertEqual(metadata["stage_names"]["0"], Stage.RESET_SETTLE)
            self.assertEqual(metadata["state_names"]["3"], "slip_or_drop")
            self.assertGreater(metadata["object"]["mass"], 0)
            self.assertEqual(len(metadata["object"]["initial_position"]), 3)
            self.assertEqual(metadata["collection_config"]["timestep"], 0.01)
            self.assertEqual(controller.stage, Stage.RESET_SETTLE)


if __name__ == "__main__":
    unittest.main()
