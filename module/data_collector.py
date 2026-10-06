from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np


class TactileEpisodeCollector:
    """以 episode 为单位记录仿真触觉、运动状态与结果标签。"""

    def __init__(self, sim, touch_processor, save_dir="dataset/auto"):
        """建立一次采集运行及场景字段索引。

        输入：仿真 sim、触觉处理器和采集父目录 save_dir。
        流程：创建不覆盖旧数据的批次目录，定位末端、物体、夹爪
        对应的 MuJoCo 索引，并初始化空帧缓存与结果状态。
        输出：无返回值；实例准备好接收 start/record_frame 调用。
        """
        self.sim = sim
        self.m = sim.m
        self.d = sim.d
        self.touch_processor = touch_processor
        self.base_save_dir = Path(save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = self._make_run_dir()

        self.recording = False
        self.episode_index = 0
        self.start_time = 0.0
        self.last_saved_path = None
        self.last_result = None
        self.last_num_frames = 0
        self.last_metadata_path = None
        self.episode_metadata = {}

        self.tcp_site_id = self._required_id(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.object_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "object")
        self.gripper_act_id = self._required_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "gripper",
        )
        self.gripper_joint_id = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "joint7",
        )
        self.gripper_qpos_id = self.m.jnt_qposadr[self.gripper_joint_id]
        self.object_joint_id = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "object_freejoint",
        )
        self.object_qvel_id = self.m.jnt_dofadr[self.object_joint_id]
        self.clear()

    def _make_run_dir(self):
        """为本次采集运行分配独立保存目录。

        输入：采集父目录以及当前时间。
        流程：用时间戳命名目录；撞名时追加递增编号后创建。
        输出：创建完成的 Path，所有本次回合数据保存在其下。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.base_save_dir / f"run_{timestamp}"
        suffix = 1

        while run_dir.exists():
            run_dir = self.base_save_dir / f"run_{timestamp}_{suffix:02d}"
            suffix += 1

        run_dir.mkdir(parents=True)
        return run_dir

    def start(self, episode_index, metadata=None):
        """开启新的抓取片段记录。

        输入：片段序号 episode_index 和可选配置/物体元数据。
        流程：清空前一回合内存帧，标记记录中并保存起始仿真时间。
        输出：无返回值；后续 record_frame 逐帧追加到新缓存。
        """
        self.clear()
        self.recording = True
        self.episode_index = episode_index
        self.start_time = float(self.d.time)
        self.episode_metadata = dict(metadata or {})

    def record_frame(
        self,
        touch_frame=None,
        stage_id=-1,
        state_label=-1,
        contact_mask=None,
        object_on_b=False,
        gripper_object_contact=False,
        gripper_released=False,
    ):
        """向当前回合追加一帧触觉与仿真状态。

        输入：可选触觉帧、阶段/状态 ID、双侧接触掩码及放置证据。
        流程：未记录时跳过；触觉帧缺省则实时生成，接触掩码缺省
        则由力度阈值推导；追加相对时间、各触觉数组和末端/物体状态，
        对会随仿真步更新的数组复制快照。
        输出：无返回值；所有字段在 finish 时按帧序列化。
        """
        if not self.recording:
            return

        if touch_frame is None:
            touch_frame = self.touch_processor.get_frame()

        self.frames["time"].append(float(self.d.time - self.start_time))
        self.frames["force_map"].append(touch_frame["force_map"])
        self.frames["position_map"].append(touch_frame["position_map"])
        self.frames["raw_force"].append(touch_frame["raw_force"])
        self.frames["force_stats"].append(touch_frame["force_stats"])
        self.frames["stage_id"].append(int(stage_id))
        self.frames["state_label"].append(int(state_label))

        if contact_mask is None:
            contact_mask = np.any(touch_frame["force_map"] > 0.02, axis=(1, 2))
        self.frames["contact_mask"].append(np.asarray(contact_mask, dtype=np.bool_))
        self.frames["object_on_b"].append(bool(object_on_b))
        self.frames["gripper_object_contact"].append(
            bool(gripper_object_contact)
        )
        self.frames["gripper_released"].append(bool(gripper_released))

        # MuJoCo 内部数组随仿真推进会原地更新，记录时必须复制。
        self.frames["tcp_pos"].append(self.d.site_xpos[self.tcp_site_id].copy())
        tcp_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(tcp_quat, self.d.site_xmat[self.tcp_site_id])
        self.frames["tcp_quat"].append(tcp_quat)
        self.frames["object_pos"].append(self.d.xpos[self.object_body_id].copy())
        self.frames["object_quat"].append(self.d.xquat[self.object_body_id].copy())
        self.frames["object_vel"].append(
            self.d.qvel[self.object_qvel_id:self.object_qvel_id + 6].copy()
        )
        self.frames["gripper_ctrl"].append(float(self.d.ctrl[self.gripper_act_id]))
        self.frames["gripper_qpos"].append(float(self.d.qpos[self.gripper_qpos_id]))

    def finish(self, result, reason="unknown", metadata=None):
        """结束回合并将逐帧数据和元数据落盘。

        输入：最终结果 result、结束原因 reason 和可选补充 metadata。
        流程：空回合直接清空并跳过；非空回合把帧数组、二分类标签
        写入压缩 NPZ，把采集配置、帧数、持续时间和原因写入同名 JSON；
        更新最近保存信息后清空当前帧缓存。
        输出：生成的 NPZ Path；无帧可写时为 None。
        """
        if self.num_frames == 0:
            self.clear()
            return None

        result_name = "success" if result else "fail"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = (
            self.save_dir
            / f"episode_{self.episode_index:05d}_{result_name}_{timestamp}_{self.num_frames:05d}.npz"
        )

        np.savez_compressed(
            save_path,
            time=np.asarray(self.frames["time"], dtype=np.float32),
            force_map=np.asarray(self.frames["force_map"], dtype=np.float32),
            position_map=np.asarray(self.frames["position_map"], dtype=np.float32),
            raw_force=np.asarray(self.frames["raw_force"], dtype=np.float32),
            force_stats=np.asarray(self.frames["force_stats"], dtype=np.float32),
            stage_id=np.asarray(self.frames["stage_id"], dtype=np.int16),
            state_label=np.asarray(self.frames["state_label"], dtype=np.int8),
            contact_mask=np.asarray(self.frames["contact_mask"], dtype=np.bool_),
            object_on_b=np.asarray(self.frames["object_on_b"], dtype=np.bool_),
            gripper_object_contact=np.asarray(
                self.frames["gripper_object_contact"],
                dtype=np.bool_,
            ),
            gripper_released=np.asarray(
                self.frames["gripper_released"],
                dtype=np.bool_,
            ),
            tcp_pos=np.asarray(self.frames["tcp_pos"], dtype=np.float32),
            tcp_quat=np.asarray(self.frames["tcp_quat"], dtype=np.float32),
            object_pos=np.asarray(self.frames["object_pos"], dtype=np.float32),
            object_quat=np.asarray(self.frames["object_quat"], dtype=np.float32),
            object_vel=np.asarray(self.frames["object_vel"], dtype=np.float32),
            gripper_ctrl=np.asarray(self.frames["gripper_ctrl"], dtype=np.float32),
            gripper_qpos=np.asarray(self.frames["gripper_qpos"], dtype=np.float32),
            result=np.asarray(1 if result else 0, dtype=np.int64),
            success=np.asarray(bool(result), dtype=np.bool_),
        )

        metadata_path = save_path.with_suffix(".json")
        episode_metadata = dict(self.episode_metadata)
        episode_metadata.update(metadata or {})
        episode_metadata.update(
            {
                "episode_index": int(self.episode_index),
                "episode_result": result_name,
                "finish_reason": str(reason),
                "num_frames": int(self.num_frames),
                "duration": float(self.frames["time"][-1]),
                "stage_names": dict(episode_metadata.get("stage_names", {})),
                "state_names": dict(episode_metadata.get("state_names", {})),
            }
        )
        metadata_path.write_text(
            json.dumps(episode_metadata, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        self.last_saved_path = save_path
        self.last_metadata_path = metadata_path
        self.last_result = result_name
        self.last_num_frames = self.num_frames
        self.clear()
        return save_path

    def clear(self):
        """复位采集器内存中的回合状态。

        输入：无；作用于实例当前的帧字段和元数据。
        流程：为所有逐帧字段创建空列表，关闭 recording 并清空元数据。
        输出：无返回值；已保存文件和最近完成结果保留不变。
        """
        self.frames = {
            "time": [],
            "force_map": [],
            "position_map": [],
            "raw_force": [],
            "force_stats": [],
            "stage_id": [],
            "state_label": [],
            "contact_mask": [],
            "object_on_b": [],
            "gripper_object_contact": [],
            "gripper_released": [],
            "tcp_pos": [],
            "tcp_quat": [],
            "object_pos": [],
            "object_quat": [],
            "object_vel": [],
            "gripper_ctrl": [],
            "gripper_qpos": [],
        }
        self.recording = False
        self.episode_metadata = {}

    @property
    def num_frames(self):
        """读取当前未落盘回合的帧数量。

        输入：实例中 time 字段的帧列表。
        流程：直接计算列表长度，不访问磁盘。
        输出：非负整数帧数。
        """
        return len(self.frames["time"])

    def overlay_text(self):
        """构造三维 viewer 的数据采集状态提示。

        输入：记录状态、当前帧数和最近一次保存路径/结果。
        流程：分别组织固定提示与动态进度文本。
        输出：viewer.set_texts 使用的字体、位置、左右文本四元组。
        """
        status = "ON" if self.recording else "OFF"
        last = "none"
        if self.last_saved_path is not None:
            last = self.last_saved_path.name

        left = (
            "Dataset\n"
            "auto tactile episodes\n"
            "result: success/fail\n"
            "q/esc: quit"
        )
        right = (
            f"recording: {status}\n"
            f"frames: {self.num_frames}\n"
            f"last frames: {self.last_num_frames}\n"
            f"last result: {self.last_result or 'none'}\n"
            f"save dir: {self.save_dir}\n"
            f"last: {last}"
        )
        return (
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            left,
            right,
        )

    def _required_id(self, obj_type, name):
        """定位采集过程必需的 MuJoCo 对象。

        输入：对象类型 obj_type 与场景内名称 name。
        流程：调用 MuJoCo 查询模型对象 ID，并检查返回值。
        输出：有效对象 ID；缺失时抛 ValueError。
        """
        obj_id = mujoco.mj_name2id(self.m, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"{name!r} not found in MuJoCo model")
        return obj_id
