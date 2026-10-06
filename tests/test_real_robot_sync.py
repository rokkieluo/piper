import time
import unittest
import warnings
from types import SimpleNamespace

import mujoco
import numpy as np

from module.piper_sim import PiperSim
from module.real_robot_sync import PiperFeedbackSync


class FakeRobot:
    """模拟带时间戳的机械臂关节反馈。"""

    def __init__(self, angles):
        """建立测试机械臂的带时效关节状态。

        输入：六轴角度列表 angles。
        流程：保存角度并以当前墙钟时间初始化时间戳。
        输出：无返回值；可通过属性模拟过期或越界反馈。
        """
        self.angles = angles
        self.timestamp = time.time()

    def get_joint_angles(self):
        """模拟 SDK 的关节角度消息格式。

        输入：当前 angles 和 timestamp 属性。
        流程：将二者包装为具有 msg/timestamp 字段的对象。
        输出：供 PiperFeedbackSync 读取的反馈对象。
        """
        return SimpleNamespace(msg=self.angles, timestamp=self.timestamp)


class FakeGripper:
    """模拟带模式与时间戳的夹爪反馈。"""

    def __init__(self, width=0.04, mode="width"):
        """建立测试夹爪反馈的宽度与模式。

        输入：width 为总开度米数，mode 默认为 width。
        流程：保存给定值，并将时间戳设置为当前墙钟时间。
        输出：无返回值；属性可在用例中修改以模拟异常读数。
        """
        self.width = width
        self.mode = mode
        self.timestamp = time.time()

    def get_gripper_status(self):
        """模拟 SDK 的夹爪状态消息格式。

        输入：width、mode 和 timestamp 三项当前属性。
        流程：在外层消息的 msg 中放入带 value/mode 的状态对象。
        输出：供同步器读取的反馈对象。
        """
        return SimpleNamespace(
            msg=SimpleNamespace(value=self.width, mode=self.mode),
            timestamp=self.timestamp,
        )


class RealRobotSyncTests(unittest.TestCase):
    """验证实机反馈到仿真模型的只读映射与越界保护。"""

    def setUp(self):
        """为每个反馈映射用例准备独立的设备和仿真。

        输入：unittest 在用例前自动调用。
        流程：创建 MuJoCo 场景、默认六轴角度和夹爪状态，
        用 0.1 米实机最大开度初始化只读同步器。
        输出：无返回值；sim、robot、gripper、sync 成为测试成员。
        """
        self.sim = PiperSim()
        self.robot = FakeRobot([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0])
        self.gripper = FakeGripper()
        self.sync = PiperFeedbackSync(
            self.sim, self.robot, self.gripper, real_gripper_open_width=0.1
        )

    def test_joint_angles_and_total_gripper_width_are_mapped(self):
        """验证有效反馈写入关节位置与夹爪控制量。

        输入：六轴初始角度和夹爪总开度 0.04 米。
        流程：运行 sync，读取仿真 qpos 与夹爪执行器 ctrl。
        输出：断言六轴保持实测值、左右指正负对称且真实宽度未裁剪。
        """
        angles, width = self.sync.sync()
        np.testing.assert_allclose(angles, self.robot.angles)
        self.assertAlmostEqual(width, 0.04)
        for joint_id, angle in zip(self.sync.arm_joints, angles):
            self.assertAlmostEqual(self.sim.d.qpos[self.sim.m.jnt_qposadr[joint_id]], angle)
        left, right = (self.sim.d.qpos[self.sim.m.jnt_qposadr[j]] for j in self.sync.finger_joints)
        self.assertAlmostEqual(left, 0.014)
        self.assertAlmostEqual(right, -0.014)
        self.assertAlmostEqual(self.sim.d.ctrl[self.sync.gripper_actuator], 0.014)

    def test_stale_feedback_does_not_modify_simulation(self):
        """验证关节反馈超过最大时效时的保护。

        输入：时间戳落后当前十秒的机械臂反馈。
        流程：保存原 qpos 后尝试调用 sync。
        输出：断言抛过期异常，仿真位置与调用前完全相同。
        """
        before = self.sim.d.qpos.copy()
        self.robot.timestamp = time.time() - 10.0
        with self.assertRaisesRegex(RuntimeError, "stale joint"):
            self.sync.sync()
        np.testing.assert_array_equal(self.sim.d.qpos, before)

    def test_out_of_range_feedback_does_not_modify_simulation(self):
        """验证超出关节限位容差时拒绝整组反馈。

        输入：第一关节角设为 4 弧度的假机械臂。
        流程：记录原 qpos，调用 sync 触发限位校验。
        输出：断言抛出 joint1 错误且仿真 qpos 未被部分更新。
        """
        before = self.sim.d.qpos.copy()
        self.robot.angles[0] = 4.0
        with self.assertRaisesRegex(RuntimeError, "joint1 feedback"):
            self.sync.sync()
        np.testing.assert_array_equal(self.sim.d.qpos, before)

    def test_small_joint_limit_offset_uses_measured_angle_and_warns_once(self):
        """验证轻微越界关节角的显示策略。

        输入：第二关节实际反馈 -0.0421 弧度。
        流程：连续同步两次并捕捉越界警告。
        输出：断言实测角度未裁剪写入仿真且警告只在首次出现。
        """
        self.robot.angles[1] = -0.0421
        with self.assertWarnsRegex(RuntimeWarning, "joint2 feedback"):
            self.sync.sync()
        joint2_qpos = self.sim.m.jnt_qposadr[self.sync.arm_joints[1]]
        self.assertAlmostEqual(self.sim.d.qpos[joint2_qpos], -0.0421)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.sync.sync()
        self.assertEqual(caught, [])

    def test_real_100mm_range_maps_to_sim_70mm_range(self):
        """验证实机总开度到单侧仿真指位移的比例映射。

        输入：实机端点 0 至 0.1 米，中点 0.05 米及全开值。
        流程：分别同步中点和全开反馈并读取夹爪执行器目标。
        输出：断言仿真单侧指位移依次为 0.0175、0.035 米。
        """
        self.sync = PiperFeedbackSync(
            self.sim, self.robot, self.gripper, real_gripper_open_width=0.1
        )
        self.gripper.width = 0.05
        _, real_width = self.sync.sync()
        self.assertAlmostEqual(real_width, 0.05)
        self.assertAlmostEqual(self.sim.d.ctrl[self.sync.gripper_actuator], 0.0175)
        self.gripper.width = 0.1
        self.sync.sync()
        self.assertAlmostEqual(self.sim.d.ctrl[self.sync.gripper_actuator], 0.035)

    def test_gripper_beyond_endpoint_clamps_display_without_crashing(self):
        """验证夹爪超标定开度时的显示限幅。

        输入：比最大端点 0.1 米稍大的实机宽度 0.105 米。
        流程：同步两次并捕获警告与仿真 ctrl。
        输出：断言报告实测宽度、仿真钳至全开且仅首次警告。
        """
        self.sync = PiperFeedbackSync(
            self.sim, self.robot, self.gripper, real_gripper_open_width=0.1
        )
        self.gripper.width = 0.105
        with self.assertWarnsRegex(RuntimeWarning, "clamping simulation display only"):
            _, real_width = self.sync.sync()
        self.assertAlmostEqual(real_width, 0.105)
        self.assertAlmostEqual(self.sim.d.ctrl[self.sync.gripper_actuator], 0.035)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.sync.sync()
        self.assertEqual(caught, [])

    def test_gripper_closed_offset_is_configurable(self):
        """验证非零实机闭合端点的线性校准。

        输入：实机闭合 0.01 米、全开 0.1 米的标定。
        流程：分别同步闭合及中点宽度。
        输出：断言仿真单侧指位移映射到 0 和 0.0175 米。
        """
        self.sync = PiperFeedbackSync(
            self.sim, self.robot, self.gripper,
            real_gripper_closed_width=0.01, real_gripper_open_width=0.1,
        )
        self.gripper.width = 0.01
        self.sync.sync()
        self.assertAlmostEqual(self.sim.d.ctrl[self.sync.gripper_actuator], 0.0)
        self.gripper.width = 0.055
        self.sync.sync()
        self.assertAlmostEqual(self.sim.d.ctrl[self.sync.gripper_actuator], 0.0175)

    def test_negative_closed_feedback_can_be_calibrated(self):
        """验证负数闭合实测宽度可作为标定起点。

        输入：闭合端点 -0.005 米及同值反馈。
        流程：初始化带负端点的同步器，再同步实机反馈。
        输出：断言仿真夹爪映射为完全闭合位置 0。
        """
        self.sync = PiperFeedbackSync(
            self.sim, self.robot, self.gripper,
            real_gripper_closed_width=-0.005, real_gripper_open_width=0.1,
        )
        self.gripper.width = -0.005
        self.sync.sync()
        self.assertAlmostEqual(self.sim.d.ctrl[self.sync.gripper_actuator], 0.0)

    def test_invalid_gripper_feedback_does_not_modify_simulation(self):
        """验证夹爪 NaN 读数不会污染仿真。

        输入：非有限的实机夹爪宽度反馈。
        流程：保存原仿真 qpos 后尝试同步。
        输出：断言抛错误且所有仿真关节位置保持原样。
        """
        before = self.sim.d.qpos.copy()
        self.gripper.width = float("nan")
        with self.assertRaisesRegex(RuntimeError, "gripper width must be finite"):
            self.sync.sync()
        np.testing.assert_array_equal(self.sim.d.qpos, before)

    def test_larger_joint_limit_offset_is_rejected(self):
        """验证第二关节较大的负限位偏差会被拒绝。

        输入：第二关节 -0.2 弧度的实机反馈。
        流程：调用 sync 进行限位和容差检测。
        输出：断言抛出 joint2 越界异常。
        """
        self.robot.angles[1] = -0.2
        with self.assertRaisesRegex(RuntimeError, "joint2 feedback"):
            self.sync.sync()

    def test_angle_gripper_mode_is_rejected(self):
        """验证夹爪角度模式与宽度模式不混用。

        输入：mode 被设为 angle 的夹爪 SDK 反馈。
        流程：调用 sync 尝试读取实机夹爪状态。
        输出：断言抛模式错误，不将角度数值当作米数映射。
        """
        self.gripper.mode = "angle"
        with self.assertRaisesRegex(RuntimeError, "width mode"):
            self.sync.sync()
