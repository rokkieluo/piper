import argparse
import time

import mujoco
import mujoco.viewer

from module.auto_grasp_controller import AutoGraspConfig, AutoGraspController
from module.data_collector import TactileEpisodeCollector
from module.piper_sim import PiperSim
from module.touch_display import TouchDisplay
from module.touch_processor import TouchProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Automatic Piper tactile data collection")
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
    parser.add_argument("--drop-force-threshold", type=float, default=0.015)
    parser.add_argument("--drop-confirm-frames", type=int, default=10)
    parser.add_argument("--lift-height", type=float, default=0.12)
    parser.add_argument("--place-z-offset", type=float, default=0.035)
    parser.add_argument("--viewer", action="store_true", help="show the MuJoCo robot viewer")

    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be greater than 0")
    if args.region_size <= 0:
        parser.error("--region-size must be greater than 0")
    if args.gripper_close_min > args.gripper_close_max:
        parser.error("--gripper-close-min must be <= --gripper-close-max")
    if args.drop_confirm_frames <= 0:
        parser.error("--drop-confirm-frames must be greater than 0")
    return args


def build_config(args):
    return AutoGraspConfig(
        num_episodes=args.episodes,
        region_size=args.region_size,
        region_a_center=(args.a_x, args.a_y),
        region_b_center=(args.b_x, args.b_y),
        gripper_close_min=args.gripper_close_min,
        gripper_close_max=args.gripper_close_max,
        contact_start_force=args.contact_start_force,
        drop_force_threshold=args.drop_force_threshold,
        drop_confirm_frames=args.drop_confirm_frames,
        lift_height=args.lift_height,
        place_z_offset=args.place_z_offset,
    )


def main():
    args = parse_args()
    sim = PiperSim()
    touch_processor = TouchProcessor(sim.m, sim.d)
    touch_display = TouchDisplay(sim.m, sim.d)
    collector = TactileEpisodeCollector(sim, touch_processor, save_dir=args.save_dir)
    controller = AutoGraspController(
        sim,
        touch_processor,
        collector,
        config=build_config(args),
        seed=args.seed,
    )

    try:
        if args.viewer:
            with mujoco.viewer.launch_passive(sim.m, sim.d) as viewer:
                run_loop(sim, touch_display, controller, collector, viewer)
        else:
            run_loop(sim, touch_display, controller, collector)
    finally:
        touch_display.close()


def run_loop(sim, touch_display, controller, collector, viewer=None):
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


if __name__ == "__main__":
    main()
