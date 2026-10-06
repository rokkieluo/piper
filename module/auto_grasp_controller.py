from dataclasses import dataclass

import mujoco
import numpy as np


class Stage:
    """定义自动抓取从复位到放置或失败等待的各阶段名称。"""
    RESET_SETTLE = "reset_settle"
    MOVE_ABOVE_A = "move_above_A"
    DESCEND_TO_A = "descend_to_A"
    CLOSE_GRIPPER = "close_gripper"
    CONTACT_HOLD = "contact_hold"
    LIFT_FROM_A = "lift_from_A"
    MOVE_ABOVE_B = "move_above_B"
    DESCEND_TO_B = "descend_to_B"
    RELEASE_ON_B = "release_on_B"
    SETTLE_ON_B = "settle_on_B"
    FAIL_RESET_WAIT = "fail_reset_wait"
    DONE = "done"


STAGE_NAMES = (
    Stage.RESET_SETTLE,
    Stage.MOVE_ABOVE_A,
    Stage.DESCEND_TO_A,
    Stage.CLOSE_GRIPPER,
    Stage.CONTACT_HOLD,
    Stage.LIFT_FROM_A,
    Stage.MOVE_ABOVE_B,
    Stage.DESCEND_TO_B,
    Stage.RELEASE_ON_B,
    Stage.SETTLE_ON_B,
    Stage.FAIL_RESET_WAIT,
    Stage.DONE,
)
STAGE_TO_ID = {name: stage_id for stage_id, name in enumerate(STAGE_NAMES)}


class GraspState:
    """定义每帧触觉状态标签的数值编码。"""
    NO_CONTACT = 0
    WEAK_CONTACT = 1
    STABLE_GRASP = 2
    SLIP_OR_DROP = 3


STATE_NAMES = {
    GraspState.NO_CONTACT: "no_contact",
    GraspState.WEAK_CONTACT: "weak_contact",
    GraspState.STABLE_GRASP: "stable_grasp",
    GraspState.SLIP_OR_DROP: "slip_or_drop",
}


@dataclass
class AutoGraspConfig:
    """集中管理轨迹、夹爪、接触判定和等待时间参数。"""
    num_episodes: int = 100
    region_size: float = 0.15
    region_a_center: tuple = (0.20, 0.09)
    region_b_center: tuple = (0.20, -0.09)
    home_qpos: tuple = (0.0, 1.57, -1.3485, 0.0, 0.0, 0.0)

    open_gripper: float = 0.035
    gripper_close_min: float = -0.001
    gripper_close_max: float = 0.016

    reset_clearance: float = 0.0005
    lift_height: float = 0.12
    grasp_z_offset: float = 0.0
    place_z_offset: float = 0.035

    ik_gain: float = 0.7
    damping: float = 0.05
    position_tolerance: float = 0.008

    reset_settle_time: float = 0.25
    move_timeout: float = 5.0
    descend_timeout: float = 4.0
    close_timeout: float = 3.0
    post_contact_hold_time: float = 0.20
    lift_timeout: float = 4.0
    transfer_timeout: float = 6.0
    release_timeout: float = 2.5
    release_open_tolerance: float = 0.002
    placement_descent_speed: float = 0.02
    placement_min_clearance: float = 0.0
    placement_limit_confirm_frames: int = 10
    settle_after_release: float = 0.5
    fail_reset_wait: float = 0.1

    contact_start_force: float = 0.02
    contact_end_force: float = 0.01
    side_contact_force: float = 0.01
    drop_force_threshold: float = 0.015
    drop_height_margin: float = 0.02
    lifted_height_threshold: float = 0.035
    drop_confirm_frames: int = 10
    grasp_loss_confirm_frames: int = 10
    stable_confirm_frames: int = 10
    slip_confirm_frames: int = 3
    stable_relative_speed: float = 0.03
    slip_relative_speed: float = 0.08


class AutoGraspController:
    """按固定抓取轨迹运行仿真，并逐帧判定抓取状态和最终结果。"""

    def __init__(self, sim, touch_processor, collector, config=None, seed=None):
        """配置固定轨迹、判定状态及首个抓取回合。

        输入：仿真 sim、触觉处理器、采集器或预测缓存、可选配置与种子。
        流程：定位末端、物体、区域、关节和执行器；设置区域大小，
        初始化接触/掉落/放置计数器，最后调用 reset_episode。
        输出：无返回值；准备好仿真及正在记录的首个 episode。
        """
        self.sim = sim
        self.m = sim.m
        self.d = sim.d
        self.touch_processor = touch_processor
        self.collector = collector
        self.config = config or AutoGraspConfig()
        self.seed = None if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.tcp_site_id = self._required_id(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.object_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "object")
        self.object_joint_id = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "object_freejoint",
        )
        self.object_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.region_a_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "region_A")
        self.region_b_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "region_B")
        link7_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "link7")
        link8_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "link8")
        self.gripper_geom_ids = frozenset(
            geom_id
            for geom_id in range(self.m.ngeom)
            if self._body_descends_from(
                int(self.m.geom_bodyid[geom_id]),
                (link7_body_id, link8_body_id),
            )
        )

        self.arm_act_ids = [
            self._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i}")
            for i in range(1, 7)
        ]
        self.arm_joint_ids = [
            self._required_id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 7)
        ]
        self.arm_qpos_ids = [self.m.jnt_qposadr[joint_id] for joint_id in self.arm_joint_ids]
        self.arm_dof_ids = [self.m.jnt_dofadr[joint_id] for joint_id in self.arm_joint_ids]

        self.gripper_act_id = self._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        self.joint7_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, "joint7")
        self.joint8_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, "joint8")
        self.gripper_qpos_id = self.m.jnt_qposadr[self.joint7_id]
        self.right_gripper_qpos_id = self.m.jnt_qposadr[self.joint8_id]
        self.object_qpos_id = self.m.jnt_qposadr[self.object_joint_id]
        self.object_qvel_id = self.m.jnt_dofadr[self.object_joint_id]

        self.region_a_center = np.asarray(self.config.region_a_center, dtype=float)
        self.region_b_center = np.asarray(self.config.region_b_center, dtype=float)
        self._configure_regions()
        self.region_top_z = self._region_top_z(self.region_a_geom_id)
        self.object_half_height = self._object_half_height()
        self.object_start_z = (
            self.region_top_z
            + self.object_half_height
            + self.config.reset_clearance
        )

        self.target_pos = np.zeros(3, dtype=float)
        self.stage = Stage.RESET_SETTLE
        self.stage_start_time = 0.0
        self.completed_episodes = 0
        self.attempt_index = 0
        self.done = False
        self.stop_reason = ""
        self.last_finish_reason = "none"
        self.gripper_close_target = self.config.gripper_close_max
        self.lifted = False
        self.drop_counter = 0
        self.grasp_loss_counter = 0
        self.placement_limit_counter = 0
        self.contact_active = False
        self.stable_counter = 0
        self.slip_counter = 0
        self.stable_confirmed = False
        self.previous_object_to_tcp = None
        self.placement_confirmed = False
        self.release_confirmed = False

        self.reset_episode()

    def before_step(self):
        """在每次仿真步进前生成机器人控制量。

        输入：当前阶段、末端目标和随机选定的夹爪闭合位置。
        流程：按阶段选择闭合或张开指令；在 B 区下降阶段逐步下调
        放置目标，再通过阻尼 IK 计算六轴执行器控制值。
        输出：无返回值；更新仿真 d.ctrl，完成状态下不做操作。
        """
        if self.done:
            return

        if self.stage in (Stage.CLOSE_GRIPPER, Stage.CONTACT_HOLD):
            self._set_gripper(self.gripper_close_target)
        elif self.stage in (Stage.LIFT_FROM_A, Stage.MOVE_ABOVE_B, Stage.DESCEND_TO_B):
            self._set_gripper(self.gripper_close_target)
        elif self.stage in (Stage.RELEASE_ON_B, Stage.SETTLE_ON_B):
            self._set_gripper(self.config.open_gripper)
        else:
            self._set_gripper(self.config.open_gripper)

        if self.stage == Stage.DESCEND_TO_B:
            self._lower_placement_target()
        self._apply_ik()

    def after_step(self):
        """在仿真步进后记录标签并决定下一阶段。

        输入：当前触觉读数、物体/夹爪碰撞和仿真时间。
        流程：更新接触和抬起状态；记录触觉、逐帧状态及放置证据；
        依次检查 B 区接触后释放、掉落、失去抓握、下降极限，最后
        根据位置或超时推进固定轨迹；等待阶段只处理阶段切换。
        输出：无返回值；可能结束当前回合并保存最终标签。
        """
        if self.done:
            return

        if self.stage in (Stage.SETTLE_ON_B, Stage.FAIL_RESET_WAIT):
            self._advance_stage()
            return

        touch_frame = self.touch_processor.get_frame()
        total_force = float(touch_frame["force_stats"][0])

        if (
            self.stage == Stage.CLOSE_GRIPPER
            and total_force >= self.config.contact_start_force
        ):
            self._enter(Stage.CONTACT_HOLD, self._grasp_pos())

        if self.object_pos[2] > (
            self.object_start_z + self.config.lifted_height_threshold
        ):
            self.lifted = True

        dropped = self._drop_detected(total_force)
        state_label, contact_mask = self._classify_state(touch_frame, dropped)
        object_on_b = self._object_on_region_b()
        gripper_object_contact = self._gripper_contacts_object()
        gripper_released = self._gripper_released(gripper_object_contact)
        self.collector.record_frame(
            touch_frame,
            stage_id=STAGE_TO_ID[self.stage],
            state_label=state_label,
            contact_mask=contact_mask,
            object_on_b=object_on_b,
            gripper_object_contact=gripper_object_contact,
            gripper_released=gripper_released,
        )

        # 成功必须分两步：先确认物体在 B 且夹爪仍接触，再张开并确认脱离。
        if (
            self.stage == Stage.DESCEND_TO_B
            and self.lifted
            and object_on_b
            and gripper_object_contact
        ):
            self.placement_confirmed = True
            self._enter(Stage.RELEASE_ON_B, self._place_pos())
            return

        if (
            self.stage == Stage.RELEASE_ON_B
            and self.placement_confirmed
            and object_on_b
            and gripper_released
        ):
            self.release_confirmed = True
            self._finish_episode(True, "success_on_B")
            return

        if dropped:
            self._finish_episode(False, "dropped")
            return

        if self._grasp_lost(gripper_object_contact):
            self._finish_episode(False, "grasp_lost")
            return

        if self._placement_limit_failed(object_on_b):
            self._finish_episode(False, "placement_limit_reached")
            return

        self._advance_stage()

    def reset_episode(self):
        """开始下一次独立的抓取尝试。

        输入：当前完成次数、配置及随机数生成器。
        流程：清空采集与判定计数器，抽取夹爪闭合目标，重置 MuJoCo，
        将机械臂、夹爪和物体归位 A 区，再进入稳定等待阶段。
        输出：无返回值；创建带参数元数据的待记录回合。
        """
        self.attempt_index += 1
        self.collector.clear()
        self.lifted = False
        self.drop_counter = 0
        self.grasp_loss_counter = 0
        self.placement_limit_counter = 0
        self.contact_active = False
        self.stable_counter = 0
        self.slip_counter = 0
        self.stable_confirmed = False
        self.previous_object_to_tcp = None
        self.placement_confirmed = False
        self.release_confirmed = False
        self.gripper_close_target = float(
            self.rng.uniform(
                self.config.gripper_close_min,
                self.config.gripper_close_max,
            )
        )

        mujoco.mj_resetData(self.m, self.d)
        self._set_arm_qpos(self.config.home_qpos)
        self._set_gripper(self.config.open_gripper, set_qpos=True)
        self._set_object_at_A()
        mujoco.mj_forward(self.m, self.d)

        self._enter(Stage.RESET_SETTLE, self._above_A_pos())
        self.collector.start(
            self.completed_episodes + 1,
            metadata=self._episode_metadata(),
        )

    @property
    def object_pos(self):
        """获取自由物体的当前世界位置。

        输入：MuJoCo 状态中的物体 body 位姿。
        流程：按初始化定位的 body ID 读取 xpos 并复制以免原地变化。
        输出：长度为 3 的位置数组副本，单位米。
        """
        return self.d.xpos[self.object_body_id].copy()

    def overlay_text(self):
        """组装 MuJoCo viewer 的自动抓取状态文本。

        输入：阶段、回合计数、夹爪闭合目标和放置确认状态。
        流程：分别生成固定操作说明和实时状态字符串。
        输出：viewer.set_texts 使用的字体、位置、左右文本四元组。
        """
        left = (
            "Auto Grasp\n"
            "A: start region\n"
            "B: success region\n"
            "record: contact -> result"
        )
        right = (
            f"stage: {self.stage}\n"
            f"episodes: {self.completed_episodes}/{self.config.num_episodes}\n"
            f"attempts: {self.attempt_index}\n"
            f"close target: {self.gripper_close_target:.4f}\n"
            f"lifted: {self.lifted}\n"
            f"placed on B: {self.placement_confirmed}\n"
            f"released: {self.release_confirmed}\n"
            f"last: {self.last_finish_reason}\n"
            f"stop: {self.stop_reason or 'running'}"
        )
        return (
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            left,
            right,
        )

    def should_continue(self):
        """判断自动抓取循环是否仍需执行。

        输入：控制器的 done 标志，达到目标回合数时会置真。
        流程：对该标志取反，不触发新的仿真步骤。
        输出：需要继续运行时为 True，否则为 False。
        """
        return not self.done

    def _advance_stage(self):
        """按阶段机规则推进抓取轨迹。

        输入：当前阶段、末端到位情况和各阶段已运行的仿真时间。
        流程：依次经历 A 区接近/闭合/抬起、搬运、B 区下降与释放；
        按到位或超时切换，无法接触、抬起、放置或释放时结束为失败；
        结束后的等待阶段再重置下一回合。
        输出：无返回值；修改阶段与目标，或触发 _finish_episode。
        """
        if self.stage == Stage.RESET_SETTLE:
            if self._stage_elapsed() >= self.config.reset_settle_time:
                self._enter(Stage.MOVE_ABOVE_A, self._above_A_pos())
            return

        if self.stage == Stage.MOVE_ABOVE_A:
            if self._target_reached() or self._stage_elapsed() >= self.config.move_timeout:
                self._enter(Stage.DESCEND_TO_A, self._grasp_pos())
            return

        if self.stage == Stage.DESCEND_TO_A:
            if self._target_reached() or self._stage_elapsed() >= self.config.descend_timeout:
                self._enter(Stage.CLOSE_GRIPPER, self._grasp_pos())
            return

        if self.stage == Stage.CLOSE_GRIPPER:
            if self._stage_elapsed() >= self.config.close_timeout:
                self._finish_episode(False, "no_contact")
            return

        if self.stage == Stage.CONTACT_HOLD:
            if self._stage_elapsed() >= self.config.post_contact_hold_time:
                self._enter(Stage.LIFT_FROM_A, self._above_A_pos())
            return

        if self.stage == Stage.LIFT_FROM_A:
            if self.lifted and (
                self._target_reached()
                or self._stage_elapsed() >= self.config.lift_timeout
            ):
                self._enter(Stage.MOVE_ABOVE_B, self._above_B_pos())
            elif self._stage_elapsed() >= self.config.lift_timeout:
                self._finish_episode(False, "lift_failed")
            return

        if self.stage == Stage.MOVE_ABOVE_B:
            if self._target_reached() or self._stage_elapsed() >= self.config.transfer_timeout:
                self._enter(Stage.DESCEND_TO_B, self._place_pos())
            return

        if self.stage == Stage.DESCEND_TO_B:
            if self._stage_elapsed() >= self.config.descend_timeout:
                self._finish_episode(False, "placement_not_confirmed")
            return

        if self.stage == Stage.RELEASE_ON_B:
            if self._stage_elapsed() >= self.config.release_timeout:
                self._finish_episode(False, "missed_B")
            return

        if self.stage == Stage.SETTLE_ON_B:
            if self._stage_elapsed() >= self.config.settle_after_release:
                self.reset_episode()
            return

        if self.stage == Stage.FAIL_RESET_WAIT:
            if self._stage_elapsed() >= self.config.fail_reset_wait:
                self.reset_episode()

    def _finish_episode(self, success, reason):
        """结算单次抓取的最终标签。

        输入：success 布尔结果和失败/成功原因字符串。
        流程：只在采集器仍记录时写入最终结果及放置证据，累加回合；
        达到目标数则结束，否则成功等待稳定、失败等待复位。
        输出：无返回值；采集器可能生成文件或更新预测缓存。
        """
        if not self.collector.recording:
            return

        self.collector.finish(
            success,
            reason=reason,
            metadata={
                "lifted": bool(self.lifted),
                "placement_confirmed": bool(self.placement_confirmed),
                "release_confirmed": bool(self.release_confirmed),
            },
        )
        self.completed_episodes += 1
        self.last_finish_reason = reason

        if self.completed_episodes >= self.config.num_episodes:
            self.done = True
            self.stage = Stage.DONE
            self.stop_reason = "target episodes reached"
            return

        if success:
            self._enter(Stage.SETTLE_ON_B, self._place_pos())
        else:
            self._enter(Stage.FAIL_RESET_WAIT, self._above_A_pos())

    def _drop_detected(self, total_force):
        """检测搬运过程中物体是否掉落。

        输入：当前双指总受力和物体位置、已抬起状态。
        流程：未抬起或进入 B 区不计掉落；否则当物体低于高度阈值
        且受力不足时累加计数，中断则清零。
        输出：连续低位且低受力达到配置帧数时为 True。
        """
        if not self.lifted:
            return False

        # B 区内正进行放置，不应将接近台面误判为抓取中掉落。
        if self._object_in_region(self.region_b_center):
            self.drop_counter = 0
            return False

        object_low = self.object_pos[2] <= (
            self.object_start_z + self.config.drop_height_margin
        )
        low_touch = total_force < self.config.drop_force_threshold

        if object_low and low_touch:
            self.drop_counter += 1
        else:
            self.drop_counter = 0

        return self.drop_counter >= self.config.drop_confirm_frames

    def _classify_state(self, touch_frame, dropped):
        """用当前及历史触觉信息分类单帧抓取状态。

        输入：包含 force_map 的触觉帧和当前是否已判定掉落。
        流程：按单侧力生成双指接触掩码；以起/止阈值滞回更新接触；
        由物体与末端相对位置差估算速度；连续满足双侧低速判稳定，
        在搬运中连续失去双侧接触或相对高速判滑移。
        输出：(GraspState 数值标签, 长度为 2 的布尔接触掩码)。
        """
        force_map = touch_frame["force_map"]
        finger_force = force_map.sum(axis=(1, 2))
        contact_mask = finger_force >= self.config.side_contact_force
        total_force = float(finger_force.sum())

        # 接触开始与结束采用不同阈值，避免临界受力引起标签抖动。
        if self.contact_active:
            self.contact_active = total_force >= self.config.contact_end_force
        else:
            self.contact_active = total_force >= self.config.contact_start_force

        object_to_tcp = self.object_pos - self.d.site_xpos[self.tcp_site_id]
        relative_speed = 0.0
        if self.previous_object_to_tcp is not None:
            relative_speed = float(
                np.linalg.norm(object_to_tcp - self.previous_object_to_tcp)
                / self.m.opt.timestep
            )
        self.previous_object_to_tcp = object_to_tcp

        bilateral = bool(contact_mask.all())
        # 稳定与滑移都需连续帧确认，避免单帧噪声改变状态标签。
        stable_candidate = (
            self.lifted
            and bilateral
            and relative_speed <= self.config.stable_relative_speed
        )
        self.stable_counter = self.stable_counter + 1 if stable_candidate else 0
        if self.stable_counter >= self.config.stable_confirm_frames:
            self.stable_confirmed = True

        transporting = self.stage in (
            Stage.LIFT_FROM_A,
            Stage.MOVE_ABOVE_B,
            Stage.DESCEND_TO_B,
        )
        slip_candidate = (
            transporting
            and self.lifted
            and self.stable_confirmed
            and (
                not bilateral
                or relative_speed >= self.config.slip_relative_speed
            )
        )
        self.slip_counter = self.slip_counter + 1 if slip_candidate else 0
        slipping = (
            self.stable_confirmed
            and self.slip_counter >= self.config.slip_confirm_frames
        )

        if dropped or slipping:
            return GraspState.SLIP_OR_DROP, contact_mask
        if self.stable_confirmed and stable_candidate:
            return GraspState.STABLE_GRASP, contact_mask
        if self.contact_active:
            return GraspState.WEAK_CONTACT, contact_mask
        return GraspState.NO_CONTACT, contact_mask

    def _episode_metadata(self):
        """汇总当前回合的采集元数据。

        输入：场景物体/区域参数、随机种子、控制配置及状态映射。
        流程：读取物体几何与质量等属性，序列化阶段名称、标签名称、
        初始位置和采集/标签阈值，供生成同名 JSON 使用。
        输出：包含 schema_version 等字段的元数据字典。
        """
        geom_type = mujoco.mjtGeom(int(self.m.geom_type[self.object_geom_id])).name
        return {
            "schema_version": 2,
            "attempt_index": int(self.attempt_index),
            "seed": self.seed,
            "stage_names": {
                str(stage_id): name for stage_id, name in enumerate(STAGE_NAMES)
            },
            "state_names": {
                str(state_id): name for state_id, name in STATE_NAMES.items()
            },
            "object": {
                "shape": geom_type,
                "size": self.m.geom_size[self.object_geom_id].tolist(),
                "mass": float(self.m.body_mass[self.object_body_id]),
                "friction": self.m.geom_friction[self.object_geom_id].tolist(),
                "initial_position": self.object_pos.tolist(),
            },
            "gripper_close_target": float(self.gripper_close_target),
            "collection_config": {
                "timestep": float(self.m.opt.timestep),
                "region_size": float(self.config.region_size),
                "region_a_center": self.region_a_center.tolist(),
                "region_b_center": self.region_b_center.tolist(),
                "lift_height": float(self.config.lift_height),
                "place_z_offset": float(self.config.place_z_offset),
                "release_open_tolerance": float(
                    self.config.release_open_tolerance
                ),
                "placement_descent_speed": float(
                    self.config.placement_descent_speed
                ),
                "placement_min_clearance": float(
                    self.config.placement_min_clearance
                ),
            },
            "label_thresholds": {
                "contact_start_force": self.config.contact_start_force,
                "contact_end_force": self.config.contact_end_force,
                "side_contact_force": self.config.side_contact_force,
                "stable_relative_speed": self.config.stable_relative_speed,
                "slip_relative_speed": self.config.slip_relative_speed,
                "stable_confirm_frames": self.config.stable_confirm_frames,
                "slip_confirm_frames": self.config.slip_confirm_frames,
                "drop_confirm_frames": self.config.drop_confirm_frames,
                "grasp_loss_confirm_frames": (
                    self.config.grasp_loss_confirm_frames
                ),
                "placement_limit_confirm_frames": (
                    self.config.placement_limit_confirm_frames
                ),
            },
        }

    def _object_on_region_b(self):
        """核验物体已实际放置在 B 区。

        输入：物体当前位置、B 区中心与 MuJoCo 碰撞记录。
        流程：先判断物体水平投影在 B 区范围内，再验证物体与 B 台面碰撞。
        输出：两项条件同时满足时为 True，否则为 False。
        """
        return self._object_in_region(self.region_b_center) and self._object_contacts_region_b()

    def _object_in_region(self, center):
        """判断物体是否位于指定区域的水平范围。

        输入：区域中心 center 的 x/y 坐标及配置的正方形边长。
        流程：计算物体世界位置到中心的两个轴向偏差，并与半边长比较。
        输出：水平投影落入区域时为 True；不检查高度或实际接触。
        """
        pos = self.object_pos
        half = self.config.region_size * 0.5
        return (
            abs(pos[0] - center[0]) <= half
            and abs(pos[1] - center[1]) <= half
        )

    def _object_contacts_region_b(self):
        """读取物体与 B 区台面的实际接触证据。

        输入：当前帧 MuJoCo contact 列表及物体/B 区几何体 ID。
        流程：逐对检查碰撞几何体，不依赖碰撞列表中的先后顺序。
        输出：找到该对接触时为 True，否则为 False。
        """
        for i in range(self.d.ncon):
            contact = self.d.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            if (
                geom1 == self.object_geom_id and geom2 == self.region_b_geom_id
            ) or (
                geom2 == self.object_geom_id and geom1 == self.region_b_geom_id
            ):
                return True
        return False

    def _gripper_contacts_object(self):
        """核验夹爪是否仍与被抓取物体接触。

        输入：当前碰撞列表、物体几何体 ID 及夹爪后代几何体集合。
        流程：筛选包含物体的碰撞对，判断另一侧是否属于夹爪。
        输出：存在物体-夹爪接触时为 True，否则为 False。
        """
        for i in range(self.d.ncon):
            contact = self.d.contact[i]
            if contact.geom1 == self.object_geom_id:
                other_geom = int(contact.geom2)
            elif contact.geom2 == self.object_geom_id:
                other_geom = int(contact.geom1)
            else:
                continue

            if other_geom in self.gripper_geom_ids:
                return True
        return False

    def _gripper_released(self, gripper_object_contact):
        """确认放置后的夹爪释放动作是否完成。

        输入：是否仍与物体接触、当前单指关节位置与张开容差。
        流程：比较夹爪位置和全开位置，且要求物体-夹爪接触消失。
        输出：两者均满足时为 True，不单凭控制指令判断释放。
        """
        sufficiently_open = self.d.qpos[self.gripper_qpos_id] >= (
            self.config.open_gripper - self.config.release_open_tolerance
        )
        return bool(sufficiently_open and not gripper_object_contact)

    def _grasp_lost(self, gripper_object_contact):
        """判定搬运中持续失去对物体的夹持。

        输入：当前物体-夹爪接触状态和阶段、抬起状态。
        流程：仅在抬起/搬运/B 区下降时累计无接触帧，恢复接触即清零。
        输出：无接触持续达到 grasp_loss_confirm_frames 时为 True。
        """
        monitoring = self.stage in (
            Stage.LIFT_FROM_A,
            Stage.MOVE_ABOVE_B,
            Stage.DESCEND_TO_B,
        )
        if monitoring and self.lifted and not gripper_object_contact:
            self.grasp_loss_counter += 1
        else:
            self.grasp_loss_counter = 0
        return self.grasp_loss_counter >= self.config.grasp_loss_confirm_frames

    def _lower_placement_target(self):
        """将 B 区放置目标逐帧向台面靠近。

        输入：当前目标高度、下降速度、仿真步长和最低安全高度。
        流程：按速度乘步长递减 z，并用目标下限防止继续下降。
        输出：无返回值；原地更新 target_pos 的高度。
        """
        lower_limit = self.object_start_z + self.config.placement_min_clearance
        descent_step = self.config.placement_descent_speed * self.m.opt.timestep
        self.target_pos[2] = max(lower_limit, self.target_pos[2] - descent_step)

    def _placement_limit_failed(self, object_on_b):
        """确认已下降至极限但物体未放到 B 区。

        输入：是否触碰 B 区、当前阶段、目标高度和末端到位情况。
        流程：只在 B 区下降且未接触时检测；目标到下限并到位则
        累加帧数，其他情况清零，避免瞬时到位误判。
        输出：持续满足配置帧数时为 True，触发放置失败。
        """
        if self.stage != Stage.DESCEND_TO_B or object_on_b:
            self.placement_limit_counter = 0
            return False

        lower_limit = self.object_start_z + self.config.placement_min_clearance
        at_limit = self.target_pos[2] <= lower_limit + 1e-9
        if at_limit and self._target_reached():
            self.placement_limit_counter += 1
        else:
            self.placement_limit_counter = 0
        return (
            self.placement_limit_counter
            >= self.config.placement_limit_confirm_frames
        )

    def _body_descends_from(self, body_id, ancestor_ids):
        """沿 MuJoCo body 树确认夹爪几何体的归属。

        输入：待检查 body ID 和夹爪父级 body ID 集合。
        流程：从当前 body 沿 parentid 向根遍历，任一祖先匹配即停止。
        输出：属于指定夹爪子树时为 True，否则为 False。
        """
        while body_id > 0:
            if body_id in ancestor_ids:
                return True
            body_id = int(self.m.body_parentid[body_id])
        return False

    def _above_A_pos(self):
        """计算 A 区上方的末端目标位置。

        输入：A 区中心、物体初始高度及配置的 lift_height。
        流程：采用中心 x/y，初始高度加抬起距离作为 z。
        输出：长度为 3 的米制位置数组。
        """
        return np.array(
            [
                self.region_a_center[0],
                self.region_a_center[1],
                self.object_start_z + self.config.lift_height,
            ],
            dtype=float,
        )

    def _above_B_pos(self):
        """计算 B 区上方的末端搬运目标。

        输入：B 区中心、物体初始高度与 lift_height。
        流程：采用 B 区 x/y，保持物体抬起后的目标高度。
        输出：长度为 3 的米制位置数组。
        """
        return np.array(
            [
                self.region_b_center[0],
                self.region_b_center[1],
                self.object_start_z + self.config.lift_height,
            ],
            dtype=float,
        )

    def _grasp_pos(self):
        """计算 A 区夹持物体时的末端目标。

        输入：A 区中心、物体初始高度和 grasp_z_offset。
        流程：中心坐标用于 x/y，按配置偏移修正夹持高度。
        输出：长度为 3 的米制位置数组。
        """
        return np.array(
            [
                self.region_a_center[0],
                self.region_a_center[1],
                self.object_start_z + self.config.grasp_z_offset,
            ],
            dtype=float,
        )

    def _place_pos(self):
        """计算 B 区的初始放置目标。

        输入：B 区中心、物体初始高度和 place_z_offset。
        流程：固定 x/y 为 B 区中心，初始 z 为物体高度加偏移。
        输出：长度为 3 的米制位置数组，随后下降阶段会逐步降低 z。
        """
        return np.array(
            [
                self.region_b_center[0],
                self.region_b_center[1],
                self.object_start_z + self.config.place_z_offset,
            ],
            dtype=float,
        )

    def _enter(self, stage, target_pos):
        """进入抓取轨迹的新阶段。

        输入：阶段名称 stage 和三维目标点 target_pos。
        流程：记录阶段、当前仿真时间及转换为浮点数组的目标位置。
        输出：无返回值；后续到位判断和阶段超时从此时开始计算。
        """
        self.stage = stage
        self.stage_start_time = float(self.d.time)
        self.target_pos = np.asarray(target_pos, dtype=float)

    def _stage_elapsed(self):
        """计算当前阶段已经持续的时间。

        输入：MuJoCo 当前时间及进入阶段时保存的起点时间。
        流程：两者相减并转换为 Python 浮点数。
        输出：该阶段的已运行时间，单位秒。
        """
        return float(self.d.time - self.stage_start_time)

    def _target_reached(self):
        """判断末端是否已经接近三维目标点。

        输入：当前 tcp 世界位置、target_pos 和 position_tolerance。
        流程：求位置差的欧氏范数并与阈值比较。
        输出：误差不超过配置容差时为 True。
        """
        error = self.target_pos - self.d.site_xpos[self.tcp_site_id]
        return float(np.linalg.norm(error)) <= self.config.position_tolerance

    def _apply_ik(self):
        """将末端三维位置误差转换为六轴关节控制目标。

        输入：当前末端/目标位置、MuJoCo 位置雅可比和 IK 增益/阻尼。
        流程：求阻尼最小二乘关节增量，加到当前关节位置，再按执行器
        控制范围限幅；只处理位置，不求姿态雅可比控制。
        输出：无返回值；更新六轴的 d.ctrl。
        """
        current_pos = self.d.site_xpos[self.tcp_site_id].copy()
        error = self.target_pos - current_pos

        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.tcp_site_id)

        # 阻尼抑制奇异姿态附近的关节速度突增，输出还要经过执行器限幅。
        arm_jac = jacp[:, self.arm_dof_ids]
        damping_matrix = (self.config.damping ** 2) * np.eye(3)
        dq = arm_jac.T @ np.linalg.solve(
            arm_jac @ arm_jac.T + damping_matrix,
            error,
        )
        dq *= self.config.ik_gain

        for i, (act_id, qpos_id) in enumerate(zip(self.arm_act_ids, self.arm_qpos_ids)):
            target = self.d.qpos[qpos_id] + dq[i]
            self.d.ctrl[act_id] = self._clamp_ctrl(act_id, target)

    def _set_arm_qpos(self, values):
        """把机械臂复位到指定关节配置。

        输入：与六个机械臂关节一一对应的目标角度 values。
        流程：同时设置关节位置与受限的执行器目标，再清零速度。
        输出：无返回值；直接修改仿真 d.qpos、d.ctrl、d.qvel。
        """
        for act_id, qpos_id, value in zip(self.arm_act_ids, self.arm_qpos_ids, values):
            self.d.qpos[qpos_id] = value
            self.d.ctrl[act_id] = self._clamp_ctrl(act_id, value)

        self.d.qvel[:] = 0.0

    def _set_gripper(self, value, set_qpos=False):
        """更新夹爪单侧目标位置。

        输入：目标指位移 value 和是否立即修改关节位置 set_qpos。
        流程：按执行器范围限制目标；常规步骤只写控制量，复位时
        同时将两指 qpos 设置为正负对称位置。
        输出：无返回值；更新夹爪执行器及可选关节状态。
        """
        value = self._clamp_ctrl(self.gripper_act_id, value)
        self.d.ctrl[self.gripper_act_id] = value
        if set_qpos:
            self.d.qpos[self.gripper_qpos_id] = value
            self.d.qpos[self.right_gripper_qpos_id] = -value

    def _set_object_at_A(self):
        """重置被抓取物体在 A 区的初始状态。

        输入：A 区中心和预先计算的物体初始高度。
        流程：设置自由关节平移与单位四元数姿态，清零六维速度。
        输出：无返回值；更新仿真中的物体 qpos/qvel。
        """
        self.d.qpos[self.object_qpos_id:self.object_qpos_id + 3] = [
            self.region_a_center[0],
            self.region_a_center[1],
            self.object_start_z,
        ]
        self.d.qpos[self.object_qpos_id + 3:self.object_qpos_id + 7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qvel[self.object_qvel_id:self.object_qvel_id + 6] = 0.0

    def _object_half_height(self):
        """根据物体几何计算其竖向半尺寸。

        输入：场景中物体几何类型和尺寸参数。
        流程：球体取半径，胶囊取半径加半长，其他类型取 z 尺寸。
        输出：浮点高度，单位米，用于将物体放在台面上。
        """
        geom_type = self.m.geom_type[self.object_geom_id]
        size = self.m.geom_size[self.object_geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(size[0])
        if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return float(size[0] + size[1])
        return float(size[2])

    def _region_top_z(self, geom_id):
        """读取指定区域台面的顶面高度。

        输入：区域几何体 ID。
        流程：取几何体 z 位置加其 z 方向半尺寸。
        输出：浮点顶面高度，单位米。
        """
        return float(self.m.geom_pos[geom_id][2] + self.m.geom_size[geom_id][2])

    def _configure_regions(self):
        """按配置布置 A、B 正方形台面。

        输入：配置的区域边长及 A/B 中心平面坐标。
        流程：分别修改两个几何体的位置 x/y 与半边长尺寸。
        输出：无返回值；直接更新 MuJoCo 模型中的 geom_pos/geom_size。
        """
        half_size = self.config.region_size * 0.5
        for geom_id, center in (
            (self.region_a_geom_id, self.region_a_center),
            (self.region_b_geom_id, self.region_b_center),
        ):
            self.m.geom_pos[geom_id][0] = center[0]
            self.m.geom_pos[geom_id][1] = center[1]
            self.m.geom_size[geom_id][0] = half_size
            self.m.geom_size[geom_id][1] = half_size

    def _clamp_ctrl(self, act_id, value):
        """约束关节或夹爪执行器的目标值。

        输入：执行器 ID 和期望目标 value。
        流程：受限执行器按 ctrlrange 钳制，不受限则原样保留。
        输出：可写入 d.ctrl 的数值。
        """
        if not self.m.actuator_ctrllimited[act_id]:
            return value

        low, high = self.m.actuator_ctrlrange[act_id]
        return min(max(value, low), high)

    def _required_id(self, obj_type, name):
        """强制定位运行所需的 MuJoCo 场景对象。

        输入：对象类型 obj_type 和场景名称 name。
        流程：调用 MuJoCo 名称到 ID 查询，并检查是否找到。
        输出：对象整数 ID；缺失时抛 ValueError 以中止初始化。
        """
        obj_id = mujoco.mj_name2id(self.m, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"{name!r} not found in MuJoCo model")
        return obj_id
