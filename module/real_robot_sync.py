import time
import warnings

import mujoco
import numpy as np


class PiperFeedbackSync:
    """将 Piper 实机反馈只读映射到 MuJoCo，不发送控制命令。"""

    def __init__(self, sim, robot, gripper, real_gripper_open_width,
                 real_gripper_closed_width=0.0, max_age=0.5, joint_margin=0.1):
        """配置只读实机反馈的安全检查及夹爪行程映射。

        输入：仿真、机械臂和夹爪 SDK 对象，实测开/闭宽度（米），
        最大反馈年龄（秒）及关节限位容差（弧度）。
        流程：检查标定端点有效性，定位仿真关节和执行器并初始化警告状态。
        输出：无返回值；端点或仿真模型不合法时抛 ValueError。
        """
        if not (np.isfinite(real_gripper_closed_width)
                and np.isfinite(real_gripper_open_width)
                and real_gripper_closed_width < real_gripper_open_width):
            raise ValueError("real gripper open width must exceed closed width")
        self.sim = sim
        self.robot = robot
        self.gripper = gripper
        self.max_age = max_age
        self.joint_margin = joint_margin
        self.warned_joints = set()
        self.real_gripper_closed_width = real_gripper_closed_width
        self.real_gripper_open_width = real_gripper_open_width
        self.warned_gripper = False
        self.arm_joints = [
            mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 7)
        ]
        self.arm_actuators = [
            mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i}")
            for i in range(1, 7)
        ]
        self.finger_joints = [
            mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("joint7", "joint8")
        ]
        self.gripper_actuator = mujoco.mj_name2id(
            sim.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"
        )
        if min(*self.arm_joints, *self.arm_actuators, *self.finger_joints,
               self.gripper_actuator) < 0:
            raise ValueError("MuJoCo model is missing Piper joint or actuator names")

    def _fresh(self, feedback, name):
        """从 SDK 响应中提取时效合格的消息。

        输入：带 timestamp/msg 的反馈对象及设备名称 name。
        流程：检查是否缺失、时间戳是否位于 [0, max_age] 秒范围。
        输出：原始 msg；缺失或过期时抛 RuntimeError。
        """
        if feedback is None:
            raise RuntimeError(f"no {name} feedback from Piper")
        age = time.time() - float(feedback.timestamp)
        if not 0 <= age <= self.max_age:
            raise RuntimeError(f"stale {name} feedback ({age:.2f}s)")
        return feedback.msg

    def read(self):
        """读取并验证一组完整的实机关节和夹爪反馈。

        输入：SDK 关节角（弧度）、夹爪宽度（米）及实例中标定的端点。
        流程：拒绝过期/非有限值、错误模式和严重越界关节角；轻微
        越界只警告一次；按实测夹爪行程线性换算并钳制仿真显示值。
        输出：(六关节角数组, 单侧仿真指宽, 未裁剪的实机总夹爪宽度)；
        不修改仿真，非法反馈抛 RuntimeError。
        """
        angles = np.asarray(self._fresh(self.robot.get_joint_angles(), "joint"), dtype=float)
        status = self._fresh(self.gripper.get_gripper_status(), "gripper")
        if angles.shape != (6,) or not np.all(np.isfinite(angles)):
            raise RuntimeError("Piper joint feedback must contain six finite angles in radians")
        if status.mode != "width":
            raise RuntimeError("Piper gripper must report width mode (meters)")
        width = float(status.value)
        if not np.isfinite(width):
            raise RuntimeError("Piper gripper width must be finite")
        for joint_id, angle in zip(self.arm_joints, angles):
            low, high = self.sim.m.jnt_range[joint_id]
            if not low - self.joint_margin <= angle <= high + self.joint_margin:
                name = mujoco.mj_id2name(self.sim.m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                raise RuntimeError(
                    f"{name} feedback {angle:.4f} exceeds simulation "
                    f"[{low:.4f}, {high:.4f}] by more than {self.joint_margin:.4f} rad"
                )
            if not low <= angle <= high and joint_id not in self.warned_joints:
                name = mujoco.mj_id2name(self.sim.m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                warnings.warn(
                    f"{name} feedback {angle:.4f} is outside simulation "
                    f"[{low:.4f}, {high:.4f}]; showing measured angle without clipping",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.warned_joints.add(joint_id)
        # 按实测闭合/张开端点做线性映射，仅对显示用的夹爪宽度限幅。
        fraction = (width - self.real_gripper_closed_width) / (
            self.real_gripper_open_width - self.real_gripper_closed_width
        )
        if not 0 <= fraction <= 1 and not self.warned_gripper:
            warnings.warn(
                f"real gripper width {width:.4f} m outside calibrated "
                f"[{self.real_gripper_closed_width:.4f}, {self.real_gripper_open_width:.4f}] m; "
                "clamping simulation display only",
                RuntimeWarning,
                stacklevel=2,
            )
            self.warned_gripper = True
        low, high = self.sim.m.jnt_range[self.finger_joints[0]]
        finger = float(np.clip(fraction, 0.0, 1.0)) * high
        return angles, finger, width

    def sync(self):
        """将有效实机反馈映射到 MuJoCo 模型。

        输入：read 获取的实机六轴关节角和夹爪总宽度。
        流程：先整体验证，再写入关节位置/控制目标，将关节速度清零，
        左右指按正负方向设置，最后刷新仿真运动学。
        输出：(六关节角数组, 实机夹爪总宽度米)；不向实机发送控制命令。
        """
        # 先完整校验，再写入场景，避免一部分关节更新后因异常留下半成品状态。
        angles, finger, real_width = self.read()
        m, d = self.sim.m, self.sim.d
        for joint_id, actuator_id, angle in zip(self.arm_joints, self.arm_actuators, angles):
            d.qpos[m.jnt_qposadr[joint_id]] = angle
            d.ctrl[actuator_id] = angle
            d.qvel[m.jnt_dofadr[joint_id]] = 0.0
        for joint_id, value in zip(self.finger_joints, (finger, -finger)):
            d.qpos[m.jnt_qposadr[joint_id]] = value
            d.qvel[m.jnt_dofadr[joint_id]] = 0.0
        d.ctrl[self.gripper_actuator] = finger
        mujoco.mj_forward(m, d)
        return angles, real_width
