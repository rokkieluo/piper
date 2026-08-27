from dataclasses import dataclass

import mujoco
import numpy as np


class Stage:
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


@dataclass
class AutoGraspConfig:
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
    settle_after_release: float = 0.5
    fail_reset_wait: float = 0.1

    contact_start_force: float = 0.02
    drop_force_threshold: float = 0.015
    drop_height_margin: float = 0.02
    lifted_height_threshold: float = 0.035
    drop_confirm_frames: int = 10
    max_skipped_attempts: int = 50


class AutoGraspController:
    def __init__(self, sim, touch_processor, collector, config=None, seed=None):
        self.sim = sim
        self.m = sim.m
        self.d = sim.d
        self.touch_processor = touch_processor
        self.collector = collector
        self.config = config or AutoGraspConfig()
        self.rng = np.random.default_rng(seed)

        self.tcp_site_id = self._required_id(mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.object_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "object")
        self.object_joint_id = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "object_freejoint",
        )
        self.object_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.region_a_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "region_A")
        self.region_b_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "region_B")

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
        self.skipped_attempts = 0
        self.done = False
        self.stop_reason = ""
        self.last_finish_reason = "none"
        self.gripper_close_target = self.config.gripper_close_max
        self.lifted = False
        self.drop_counter = 0

        self.reset_episode()

    def before_step(self):
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

        self._apply_ik()

    def after_step(self):
        if self.done:
            return

        touch_frame = self.touch_processor.get_frame()
        total_force = float(touch_frame["force_stats"][0])

        if (
            self.stage == Stage.CLOSE_GRIPPER
            and not self.collector.recording
            and total_force >= self.config.contact_start_force
        ):
            self.collector.start(self.completed_episodes + 1)
            self._enter(Stage.CONTACT_HOLD, self._grasp_pos())

        if self.collector.recording:
            self.collector.record_frame(touch_frame)

        if self.collector.recording and self.object_pos[2] > (
            self.object_start_z + self.config.lifted_height_threshold
        ):
            self.lifted = True

        if self.stage == Stage.RELEASE_ON_B and self._object_on_region_b():
            self._finish_episode(True, "success_on_B")
            return

        if self._drop_detected(total_force):
            self._finish_episode(False, "dropped")
            return

        self._advance_stage()

    def reset_episode(self):
        self.attempt_index += 1
        self.collector.clear()
        self.lifted = False
        self.drop_counter = 0
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

    @property
    def object_pos(self):
        return self.d.xpos[self.object_body_id].copy()

    def overlay_text(self):
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
            f"skipped: {self.skipped_attempts}\n"
            f"close target: {self.gripper_close_target:.4f}\n"
            f"lifted: {self.lifted}\n"
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
        return not self.done

    def _advance_stage(self):
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
                self._skip_attempt("no_contact")
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
            if self._target_reached() or self._stage_elapsed() >= self.config.descend_timeout:
                self._enter(Stage.RELEASE_ON_B, self._place_pos())
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
        self.collector.finish(success)
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

    def _skip_attempt(self, reason):
        self.skipped_attempts += 1
        self.last_finish_reason = reason
        if self.skipped_attempts >= self.config.max_skipped_attempts:
            self.done = True
            self.stage = Stage.DONE
            self.stop_reason = "too many skipped attempts"
            return
        self.reset_episode()

    def _drop_detected(self, total_force):
        if not self.collector.recording or not self.lifted:
            return False

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

    def _object_on_region_b(self):
        return self._object_in_region(self.region_b_center) and self._object_contacts_region_b()

    def _object_in_region(self, center):
        pos = self.object_pos
        half = self.config.region_size * 0.5
        return (
            abs(pos[0] - center[0]) <= half
            and abs(pos[1] - center[1]) <= half
        )

    def _object_contacts_region_b(self):
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

    def _above_A_pos(self):
        return np.array(
            [
                self.region_a_center[0],
                self.region_a_center[1],
                self.object_start_z + self.config.lift_height,
            ],
            dtype=float,
        )

    def _above_B_pos(self):
        return np.array(
            [
                self.region_b_center[0],
                self.region_b_center[1],
                self.object_start_z + self.config.lift_height,
            ],
            dtype=float,
        )

    def _grasp_pos(self):
        return np.array(
            [
                self.region_a_center[0],
                self.region_a_center[1],
                self.object_start_z + self.config.grasp_z_offset,
            ],
            dtype=float,
        )

    def _place_pos(self):
        return np.array(
            [
                self.region_b_center[0],
                self.region_b_center[1],
                self.object_start_z + self.config.place_z_offset,
            ],
            dtype=float,
        )

    def _enter(self, stage, target_pos):
        self.stage = stage
        self.stage_start_time = float(self.d.time)
        self.target_pos = np.asarray(target_pos, dtype=float)

    def _stage_elapsed(self):
        return float(self.d.time - self.stage_start_time)

    def _target_reached(self):
        error = self.target_pos - self.d.site_xpos[self.tcp_site_id]
        return float(np.linalg.norm(error)) <= self.config.position_tolerance

    def _apply_ik(self):
        current_pos = self.d.site_xpos[self.tcp_site_id].copy()
        error = self.target_pos - current_pos

        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.tcp_site_id)

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
        for act_id, qpos_id, value in zip(self.arm_act_ids, self.arm_qpos_ids, values):
            self.d.qpos[qpos_id] = value
            self.d.ctrl[act_id] = self._clamp_ctrl(act_id, value)

        self.d.qvel[:] = 0.0

    def _set_gripper(self, value, set_qpos=False):
        value = self._clamp_ctrl(self.gripper_act_id, value)
        self.d.ctrl[self.gripper_act_id] = value
        if set_qpos:
            self.d.qpos[self.gripper_qpos_id] = value
            self.d.qpos[self.right_gripper_qpos_id] = -value

    def _set_object_at_A(self):
        self.d.qpos[self.object_qpos_id:self.object_qpos_id + 3] = [
            self.region_a_center[0],
            self.region_a_center[1],
            self.object_start_z,
        ]
        self.d.qpos[self.object_qpos_id + 3:self.object_qpos_id + 7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qvel[self.object_qvel_id:self.object_qvel_id + 6] = 0.0

    def _object_half_height(self):
        geom_type = self.m.geom_type[self.object_geom_id]
        size = self.m.geom_size[self.object_geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(size[0])
        if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return float(size[0] + size[1])
        return float(size[2])

    def _region_top_z(self, geom_id):
        return float(self.m.geom_pos[geom_id][2] + self.m.geom_size[geom_id][2])

    def _configure_regions(self):
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
        if not self.m.actuator_ctrllimited[act_id]:
            return value

        low, high = self.m.actuator_ctrlrange[act_id]
        return min(max(value, low), high)

    def _required_id(self, obj_type, name):
        obj_id = mujoco.mj_name2id(self.m, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"{name!r} not found in MuJoCo model")
        return obj_id
