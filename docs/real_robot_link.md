# Piper real robot feedback link

`main.py` defaults to `--mode auto`: the existing MuJoCo-only data collection
loop is unchanged. `--mode robot-follow` is a read-only link from the real Piper
to MuJoCo, using the installed `pyAgxArm` package.

```bash
conda activate base
# Configure and activate the CAN interface following the pyAgxArm CAN guide first.
python main.py --mode robot-follow --can-channel can0 --piper-firmware default --viewer
```

Set `--piper-firmware` to the actual driver profile (`default`, `v183`, `v188`,
or `v189`). The link reads `get_joint_angles()` in radians and AGX gripper
`get_gripper_status()` in width mode (meters). At startup, the read-only link
queries the real gripper's configured maximum travel via
`get_gripper_teaching_pendant_param()`. For example, a 100 mm real range is
scaled to the MuJoCo model's 70 mm total opening; 50 mm real maps to 35 mm
in simulation. The simulator uses half the mapped total opening for each finger.
If the SDK cannot report the maximum, pass `--real-gripper-open-width 0.10`
with a **verified** fully-open feedback width in meters. If the real gripper's
measured fully-closed feedback is not zero, pass e.g.
`--real-gripper-closed-width 0.005` (negative offsets are also accepted).
These options affect visualization only;
they do not configure the actual gripper. The log prints both real and simulated
widths. Widths beyond the calibrated endpoints are displayed at the closest
simulated endpoint with a one-time warning, without changing the original
logged real width.

Missing, expired (default 0.5 s),
nonfinite, or joint feedback exceeding a simulation limit by more than 0.1 rad
stops the visualization. Smaller joint-limit differences are shown using the
measured angle (without clipping) and warned about once per joint. This margin
is for read-only visualization, **not** a physical motion safety limit. On exit, the CAN
connection is closed. This mode never enables the arm, sends motion commands,
or starts the automatic episode collector.

The robot joints and gripper pose are measured feedback. The object, contact
forces, and tactile display in MuJoCo are **simulation only**: no real object
tracking or real tactile sensor is connected, and no real success/failure
labels are produced. Verify the robot variant, model joint limits, gripper
type, scene alignment, and safe workspace before considering bidirectional
control. A physical emergency-stop remains necessary for any future
robot-commanding mode.

SDK reference: https://github.com/agilexrobotics/pyAgxArm
