from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np


class TactileEpisodeCollector:
    def __init__(self, sim, touch_processor, save_dir="dataset/auto"):
        self.sim = sim
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
        self.clear()

    def _make_run_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.base_save_dir / f"run_{timestamp}"
        suffix = 1

        while run_dir.exists():
            run_dir = self.base_save_dir / f"run_{timestamp}_{suffix:02d}"
            suffix += 1

        run_dir.mkdir(parents=True)
        return run_dir

    def start(self, episode_index):
        self.clear()
        self.recording = True
        self.episode_index = episode_index
        self.start_time = float(self.d.time)

    def record_frame(self, touch_frame=None):
        if not self.recording:
            return

        if touch_frame is None:
            touch_frame = self.touch_processor.get_frame()

        self.frames["time"].append(float(self.d.time - self.start_time))
        self.frames["force_map"].append(touch_frame["force_map"])
        self.frames["position_map"].append(touch_frame["position_map"])
        self.frames["raw_force"].append(touch_frame["raw_force"])
        self.frames["force_stats"].append(touch_frame["force_stats"])

    def finish(self, result):
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
            result=np.asarray(1 if result else 0, dtype=np.int64),
            success=np.asarray(bool(result), dtype=np.bool_),
        )

        self.last_saved_path = save_path
        self.last_result = result_name
        self.last_num_frames = self.num_frames
        self.clear()
        return save_path

    def clear(self):
        self.frames = {
            "time": [],
            "force_map": [],
            "position_map": [],
            "raw_force": [],
            "force_stats": [],
        }
        self.recording = False

    @property
    def num_frames(self):
        return len(self.frames["time"])

    def overlay_text(self):
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
