import mujoco

GRIPPER_OPEN = 0.035

class PiperSim:
    def __init__(self, xml_path="./agilex_piper/scene_touch.xml"):
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
        self.d.qpos[self.gripper_qpos_id] = GRIPPER_OPEN
        self.d.ctrl[self.gripper_act_id] = GRIPPER_OPEN
        mujoco.mj_forward(self.m, self.d)

    def handle_reset(self):
        if self.d.time < self.last_time:
            self.open_gripper()
            self.last_time = self.d.time
            return True

        self.last_time = self.d.time
        return False
