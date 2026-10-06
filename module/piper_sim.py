import mujoco

GRIPPER_OPEN = 0.035

class PiperSim:
    """管理 Piper 的 MuJoCo 模型和数据状态。"""

    def __init__(self, xml_path="./agilex_piper/scene_touch.xml"):
        """构建 Piper 的 MuJoCo 仿真状态。

        输入：场景 XML 文件路径，默认包含触觉传感器的场景。
        流程：加载模型与数据对象，定位夹爪执行器和左指关节，
        将夹爪置为张开状态并记录当前仿真时间。
        输出：无返回值；实例提供 m、d 及夹爪索引。
        """
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)

        self.gripper_act_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"
        )
        joint7_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, "joint7"
        )
        self.gripper_qpos_id = self.m.jnt_qposadr[joint7_id]

        self.open_gripper()
        self.last_time = self.d.time

    def open_gripper(self):
        """立即打开仿真夹爪并刷新场景运动学。

        输入：固定单指张开目标 GRIPPER_OPEN。
        流程：写入左指关节位置与夹爪执行器目标，再运行 mj_forward。
        输出：无返回值；仿真中的夹爪状态同步更新。
        """
        self.d.qpos[self.gripper_qpos_id] = GRIPPER_OPEN
        self.d.ctrl[self.gripper_act_id] = GRIPPER_OPEN
        mujoco.mj_forward(self.m, self.d)

    def handle_reset(self):
        """处理 MuJoCo 场景被外部重置后的夹爪状态。

        输入：当前 d.time 与上一次记录的仿真时间。
        流程：若时间倒退则重新打开夹爪；无论如何都刷新 last_time。
        输出：检测到重置时为 True，正常推进时为 False。
        """
        if self.d.time < self.last_time:
            self.open_gripper()
            self.last_time = self.d.time
            return True

        self.last_time = self.d.time
        return False
