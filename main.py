import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from module.auto_grasp_controller import AutoGraspConfig, AutoGraspController
from module.data_collector import TactileEpisodeCollector
from module.piper_sim import PiperSim
from module.predict_episode import EpisodePredictor, PredictionEpisodeBuffer, PredictionSession
from module.predict_results import PredictionResults
from module.real_robot_sync import PiperFeedbackSync
from module.touch_display import TouchDisplay
from module.touch_processor import TouchProcessor


def parse_args():
    """解析并验证程序的命令行参数。

    输入：命令行中的运行模式、抓取阈值、实机 CAN 参数和显示选项。
    流程：先由 argparse 解析参数，再检查模式依赖、夹爪标定端点、
    时间/力度阈值及确认帧数等约束；非法参数由解析器报错退出。
    输出：经过检查的 argparse.Namespace，供各运行模式共享。
    """
    parser = argparse.ArgumentParser(description="Automatic Piper tactile data collection")
    parser.add_argument(
        "--mode", choices=("auto", "predict", "robot-follow"), default="auto",
        help="auto: collect simulation data; predict: test a checkpoint in simulation; robot-follow: read-only real robot visualization",
    )
    parser.add_argument("--checkpoint", help="best.pt checkpoint (required for predict)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="inference device for predict (default: auto)")
    parser.add_argument("--predict-output-dir", default="runs/predict",
                        help="parent directory for predict result runs (default: runs/predict)")
    parser.add_argument("--can-channel", default="can0")
    parser.add_argument(
        "--piper-firmware", choices=("default", "v183", "v188", "v189"),
        default="default", help="pyAgxArm driver profile matching the robot firmware",
    )
    parser.add_argument("--feedback-max-age", type=float, default=0.5)
    parser.add_argument("--real-gripper-closed-width", type=float, default=0.0,
                        help="measured real gripper width at the closed endpoint (meters)")
    parser.add_argument("--real-gripper-open-width", type=float, default=None,
                        help="measured real gripper width at the open endpoint (meters); defaults to SDK configuration")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--save-dir", default="dataset/auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--region-size", type=float, default=0.15)
    parser.add_argument("--a-x", type=float, default=0.20)
    parser.add_argument("--a-y", type=float, default=0.09)
    parser.add_argument("--b-x", type=float, default=0.20)
    parser.add_argument("--b-y", type=float, default=-0.09)
    parser.add_argument("--gripper-close-min", type=float, default=-0.001)
    parser.add_argument("--gripper-close-max", type=float, default=0.016)
    parser.add_argument("--contact-start-force", type=float, default=0.02)
    parser.add_argument("--contact-end-force", type=float, default=0.01)
    parser.add_argument("--side-contact-force", type=float, default=0.01)
    parser.add_argument("--drop-force-threshold", type=float, default=0.015)
    parser.add_argument("--drop-confirm-frames", type=int, default=10)
    parser.add_argument("--stable-confirm-frames", type=int, default=10)
    parser.add_argument("--slip-confirm-frames", type=int, default=3)
    parser.add_argument("--stable-relative-speed", type=float, default=0.03)
    parser.add_argument("--slip-relative-speed", type=float, default=0.08)
    parser.add_argument("--release-open-tolerance", type=float, default=0.002)
    parser.add_argument("--placement-descent-speed", type=float, default=0.02)
    parser.add_argument("--placement-min-clearance", type=float, default=0.0)
    parser.add_argument("--placement-limit-confirm-frames", type=int, default=10)
    parser.add_argument("--grasp-loss-confirm-frames", type=int, default=10)
    parser.add_argument("--lift-height", type=float, default=0.12)
    parser.add_argument("--place-z-offset", type=float, default=0.035)
    parser.add_argument("--viewer", action="store_true", help="show the MuJoCo robot viewer")

    args = parser.parse_args()
    if args.mode == "predict" and not args.checkpoint:
        parser.error("--checkpoint is required for predict mode")
    if args.episodes <= 0:
        parser.error("--episodes must be greater than 0")
    if args.feedback_max_age <= 0:
        parser.error("--feedback-max-age must be greater than 0")
    if not np.isfinite(args.real_gripper_closed_width):
        parser.error("--real-gripper-closed-width must be finite")
    if args.real_gripper_open_width is not None and (
        not np.isfinite(args.real_gripper_open_width)
        or args.real_gripper_open_width <= args.real_gripper_closed_width
    ):
        parser.error("--real-gripper-open-width must exceed --real-gripper-closed-width")
    if args.region_size <= 0:
        parser.error("--region-size must be greater than 0")
    if args.gripper_close_min > args.gripper_close_max:
        parser.error("--gripper-close-min must be <= --gripper-close-max")
    if args.contact_start_force < 0:
        parser.error("--contact-start-force must be >= 0")
    if not 0 <= args.contact_end_force <= args.contact_start_force:
        parser.error(
            "--contact-end-force must be between 0 and --contact-start-force"
        )
    if args.side_contact_force < 0:
        parser.error("--side-contact-force must be >= 0")
    if args.drop_force_threshold < 0:
        parser.error("--drop-force-threshold must be >= 0")
    for name in (
        "drop_confirm_frames",
        "stable_confirm_frames",
        "slip_confirm_frames",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than 0")
    if args.stable_relative_speed < 0 or args.slip_relative_speed < 0:
        parser.error("relative speed thresholds must be >= 0")
    if args.stable_relative_speed > args.slip_relative_speed:
        parser.error(
            "--stable-relative-speed must be <= --slip-relative-speed"
        )
    if not 0 <= args.release_open_tolerance < AutoGraspConfig.open_gripper:
        parser.error(
            "--release-open-tolerance must be >= 0 and smaller than 0.035"
        )
    if args.placement_descent_speed <= 0:
        parser.error("--placement-descent-speed must be greater than 0")
    if args.placement_min_clearance < 0:
        parser.error("--placement-min-clearance must be >= 0")
    if args.placement_limit_confirm_frames <= 0:
        parser.error("--placement-limit-confirm-frames must be greater than 0")
    if args.grasp_loss_confirm_frames <= 0:
        parser.error("--grasp-loss-confirm-frames must be greater than 0")
    return args


def build_config(args):
    """从命令行参数生成自动抓取配置。

    输入：parse_args 返回的参数对象，包含区域、夹爪、接触与轨迹设置。
    流程：逐项将命令行字段映射到控制器配置，不修改原参数。
    输出：AutoGraspConfig，供自动采集和仿真预测复用。
    """
    return AutoGraspConfig(
        num_episodes=args.episodes,
        region_size=args.region_size,
        region_a_center=(args.a_x, args.a_y),
        region_b_center=(args.b_x, args.b_y),
        gripper_close_min=args.gripper_close_min,
        gripper_close_max=args.gripper_close_max,
        contact_start_force=args.contact_start_force,
        contact_end_force=args.contact_end_force,
        side_contact_force=args.side_contact_force,
        drop_force_threshold=args.drop_force_threshold,
        drop_confirm_frames=args.drop_confirm_frames,
        stable_confirm_frames=args.stable_confirm_frames,
        slip_confirm_frames=args.slip_confirm_frames,
        stable_relative_speed=args.stable_relative_speed,
        slip_relative_speed=args.slip_relative_speed,
        release_open_tolerance=args.release_open_tolerance,
        placement_descent_speed=args.placement_descent_speed,
        placement_min_clearance=args.placement_min_clearance,
        placement_limit_confirm_frames=args.placement_limit_confirm_frames,
        grasp_loss_confirm_frames=args.grasp_loss_confirm_frames,
        lift_height=args.lift_height,
        place_z_offset=args.place_z_offset,
    )


def main():
    """启动所选模式并管理显示与结果写入的生命周期。

    输入：来自命令行的模式、模型路径、采集参数和 viewer 开关。
    流程：创建 MuJoCo 场景；实机模式连接反馈，其余模式共用轨迹控制器，
    分别接入磁盘采集器或内存预测缓存；按需启动 viewer 和运行循环。
    输出：无返回值；退出时关闭触觉窗口，并收尾保存预测统计。
    """
    args = parse_args()
    sim = PiperSim()
    if args.mode == "robot-follow":
        run_robot_follow(sim, args)
        return

    # 预测与采集复用同一运动控制器，仅替换帧记录方式。
    predictor = EpisodePredictor(args.checkpoint, args.device) if args.mode == "predict" else None
    touch_processor = TouchProcessor(sim.m, sim.d)
    touch_display = TouchDisplay(sim.m, sim.d)
    reporter = None
    try:
        collector = (
            PredictionEpisodeBuffer() if predictor is not None
            else TactileEpisodeCollector(sim, touch_processor, save_dir=args.save_dir)
        )
        controller = AutoGraspController(
            sim,
            touch_processor,
            collector,
            config=build_config(args),
            seed=args.seed,
        )
        session = PredictionSession(predictor) if predictor is not None else None
        if session is not None:
            reporter = PredictionResults(args.predict_output_dir, args.checkpoint, args.seed, predictor)
        if args.viewer:
            with mujoco.viewer.launch_passive(sim.m, sim.d) as viewer:
                if session is not None:
                    run_predict_loop(sim, touch_display, controller, collector, session, reporter, viewer)
                else:
                    run_loop(sim, touch_display, controller, collector, viewer)
        else:
            if session is not None:
                run_predict_loop(sim, touch_display, controller, collector, session, reporter)
            else:
                run_loop(sim, touch_display, controller, collector)
    finally:
        # 用户关闭窗口或中断时仍保存已完成 episode 的预测结果。
        touch_display.close()
        if reporter is not None:
            reporter.finish()


def run_robot_follow(sim, args):
    """建立 Piper 实机到仿真的只读反馈连接。

    输入：仿真对象 sim 和 CAN、固件版本、夹爪标定等命令行参数。
    流程：创建 SDK 设备并连接，读取或采用指定的夹爪开度端点，
    建立映射并等待首帧有效反馈，随后运行可选 viewer 的同步循环。
    输出：无返回值；连接不可用时抛错，退出时始终断开实机连接。
    """
    try:
        from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config
    except ImportError as exc:
        raise RuntimeError("robot-follow requires pyAgxArm in the active Python environment") from exc

    config = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=args.piper_firmware,
        interface="socketcan",
        channel=args.can_channel,
    )
    robot = AgxArmFactory.create_arm(config)
    try:
        robot.connect()
        gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        real_open_width = args.real_gripper_open_width
        if real_open_width is None:
            pendant = gripper.get_gripper_teaching_pendant_param()
            real_open_width = None if pendant is None else pendant.msg.max_range_config
        if real_open_width is None or not np.isfinite(real_open_width) or real_open_width <= args.real_gripper_closed_width:
            raise RuntimeError(
                "cannot determine real gripper range; pass --real-gripper-open-width "
                "with the measured fully-open feedback width in meters"
            )
        feedback = PiperFeedbackSync(
            sim, robot, gripper, max_age=args.feedback_max_age,
            real_gripper_closed_width=args.real_gripper_closed_width,
            real_gripper_open_width=real_open_width,
        )
        wait_for_robot_feedback(feedback)
        print(
            f"Piper feedback ready (read-only); real gripper endpoints: "
            f"{args.real_gripper_closed_width:.4f}..{real_open_width:.4f} m. "
            "No motion commands or dataset labels are sent."
        )
        if args.viewer:
            with mujoco.viewer.launch_passive(sim.m, sim.d) as viewer:
                run_robot_follow_loop(feedback, viewer)
        else:
            run_robot_follow_loop(feedback)
    finally:
        robot.disconnect()


def wait_for_robot_feedback(feedback):
    """等待实机发送可用于初始化的首帧反馈。

    输入：提供 sync 方法的反馈同步器。
    流程：在五秒内重复读取；仅对缺失或过期反馈短暂重试，
    对格式、模式和限位错误立即停止。
    输出：成功时无返回值；超时或非法反馈时抛 RuntimeError。
    """
    deadline = time.monotonic() + 5.0
    while True:
        try:
            feedback.sync()
            return
        except RuntimeError as exc:
            if not str(exc).startswith(("no ", "stale ")) or time.monotonic() >= deadline:
                raise RuntimeError(f"Piper feedback unavailable at startup: {exc}") from exc
            time.sleep(0.02)


def run_robot_follow_loop(feedback, viewer=None):
    """持续显示实机反馈对应的仿真姿态。

    输入：反馈同步器和可选 MuJoCo viewer。
    流程：每个仿真周期读取并同步关节、刷新 viewer，约每秒打印
    实机角度及实机/仿真夹爪宽度；反馈中断时停止循环。
    输出：无返回值；只更新仿真状态，不向实机发送运动指令。
    """
    next_log = 0.0
    period = feedback.sim.m.opt.timestep
    while viewer is None or viewer.is_running():
        started = time.monotonic()
        try:
            angles, width = feedback.sync()
        except RuntimeError as exc:
            raise RuntimeError(f"Piper feedback stopped; visualization halted: {exc}") from exc
        if viewer is not None:
            viewer.sync()
        if started >= next_log:
            print(
                f"joints(rad): {np.array2string(angles, precision=3)}  "
                f"real gripper(m): {width:.4f}  "
                f"sim gripper(m): {2 * feedback.sim.d.ctrl[feedback.gripper_actuator]:.4f}"
            )
            next_log = started + 1.0
        remaining = period - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)


def run_loop(sim, touch_display, controller, collector, viewer=None):
    """推进自动抓取仿真并采集逐帧数据。

    输入：仿真、触觉显示器、轨迹控制器、磁盘采集器及可选 viewer。
    流程：每步先设置控制量，再推进 MuJoCo 并记录标签；回合结束
    输出保存路径，更新触觉和三维画面，处理退出键并按步长限速。
    输出：无返回值；数据文件由采集器在回合结束时写入。
    """
    last_completed = controller.completed_episodes

    while controller.should_continue():
        if viewer is not None and not viewer.is_running():
            break

        step_start = time.time()

        controller.before_step()
        mujoco.mj_step(sim.m, sim.d)
        controller.after_step()

        if controller.completed_episodes != last_completed:
            if collector.last_saved_path is not None:
                print(
                    f"episode {controller.completed_episodes}: "
                    f"{collector.last_result}, "
                    f"reason={controller.last_finish_reason}, "
                    f"{collector.last_num_frames} frames, "
                    f"{collector.last_saved_path}"
                )
            last_completed = controller.completed_episodes

        key = touch_display.show()
        if key == ord("q") or key == 27:
            break

        if viewer is not None:
            viewer.set_texts([
                controller.overlay_text(),
                collector.overlay_text(),
            ])
            viewer.sync()

        time_until_next_step = sim.m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def run_predict_loop(sim, touch_display, controller, collector, session, reporter, viewer=None):
    """沿自动采集轨迹测试模型并记录逐回合预测。

    输入：仿真、显示器、控制器、内存缓存、预测会话、结果保存器和 viewer。
    流程：每步仿真后将触觉帧送入预测会话；得到回合结果即打印并
    持久化，随后处理触觉窗口、viewer 和仿真节拍。
    输出：无返回值；预测不控制运动，也不生成采集数据集。
    """
    print("Simulation prediction only; the model does not control motion or save datasets.")
    while controller.should_continue():
        if viewer is not None and not viewer.is_running():
            break

        step_start = time.time()
        controller.before_step()
        mujoco.mj_step(sim.m, sim.d)
        controller.after_step()

        result = session.update(collector)
        if result is not None:
            # episode 结束即写入结果，提前退出也不会丢失已完成的预测。
            reporter.record(session.last_record)
            print(result)

        key = touch_display.show()
        if key == ord("q") or key == 27:
            break

        if viewer is not None:
            viewer.set_texts([controller.overlay_text(), session.overlay_text()])
            viewer.sync()

        time_until_next_step = sim.m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
