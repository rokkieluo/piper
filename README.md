# Piper 项目运行指南

[![Coded with Codex](https://vibecoded.fyi/badges/flat/agents/codex.svg)](https://vibecoded.fyi/)

## 0. 环境与运行入口

| 入口 | 用途 | 输入 | 主要输出 |
| --- | --- | --- | --- |
| `main.py --mode auto` | 在仿真中自动抓取并采集 | 场景、抓取配置、随机种子 | NPZ 片段及同名 JSON 元数据 |
| `train.py` | 训练抓取成功/失败二分类模型 | NPZ 数据集 | 权重、划分清单、训练指标和图表 |
| `evaluate.py` | 评估已有数据集的固定划分 | 权重、训练时的 `split.json`、数据根目录 | 终端 JSON 指标 |
| `main.py --mode predict` | 在新仿真回合中测试模型 | 权重及抓取配置 | 逐回合预测、CSV、JSON 和图表 |
| `main.py --mode robot-follow` | 将实机姿态同步到仿真显示 | CAN 实机关节/夹爪反馈 | 三维姿态和终端反馈日志 |


`auto` 和 `predict` 即使不加 `--viewer`，仍会打开 OpenCV 触觉窗口，因此需要图形桌面；当前没有纯无窗口采集/预测选项。`--viewer` 额外打开 MuJoCo 三维窗口。采集和预测时按触觉窗口中的 `q` 或 `Esc` 退出，或关闭三维窗口；实机模式可关闭三维窗口或在终端按 `Ctrl+C`。

## 1. 采集数据集

### 1.1 先进行小批次验证

在装有仿真依赖的环境中运行：

```bash
conda activate base
python main.py --mode auto \
  --episodes 10 \
  --save-dir dataset/auto \
  --seed 42 \
  --viewer
```

每次运行会新建 `dataset/auto/run_*`。从终端记录本次实际目录，然后检查成功/失败数量、结束原因与逐帧状态：

```bash
# 将此路径替换为刚刚采集生成的 run_* 目录。
COLLECT_RUN=dataset/auto/run_YYYYMMDD_HHMMSS
python scripts/showdata.py "$COLLECT_RUN" --summary
python scripts/showdata.py "$COLLECT_RUN" --map force_map --delay 30 --scale 12
```

浏览器中：空格暂停/播放，`n/p` 切换下一/上一片段，`d/a` 逐帧前后移动，`]/[` 前后跳十帧，`f` 切换力图/位置图，`r` 回到当前片段开头，`q/Esc` 退出。默认按每张单指图的峰值调整色阶，适合观察弱接触；加 `--fixed-scale` 使用固定 `[0,1]` 色阶，便于跨帧比较力度。

确认物体复位、搬运下降、放置释放和标签变化正确后，再进行较大批次采集：

```bash
python main.py --mode auto \
  --episodes 500 \
  --save-dir dataset/auto \
  --seed 42 \
  --gripper-close-min -0.001 \
  --gripper-close-max 0.016 \
  --placement-descent-speed 0.02 \
  --grasp-loss-confirm-frames 10 \
  --viewer
```

可换 `--seed 7` 或 `--seed 2026` 采集另一批随机夹爪闭合目标。种子控制闭合目标的随机抽样；当前流程没有随机更换物体形状、质量、摩擦或初始平面位置。

### 1.2 抓取流程与标签

每个 episode 表示一次完整尝试：复位物体和机器人 → A 区上方 → 下降 → 闭合 → 接触保持 → 抬起 → 搬运至 B 上方 → 下降接触 → 张开释放。记录从复位后开始，到成功或失败时结束。失败后等待并复位下一次尝试，避免重复保存同一失败状态。

成功判定需要按顺序满足：先抬起物体；在 B 区下降时，物体位于 B 的水平范围且接触 B 台面，同时夹爪仍接触物体；随后夹爪接近全开并脱离物体，且物体仍接触 B 台面。当前成功在这次释放确认时标记，之后的稳定等待不用于重新判定该回合结果。

| 标签字段 | 编码或含义 | 当前用途 |
| --- | --- | --- |
| `success` / `result` | 成功 `True/1`，失败 `False/0` | 当前模型的 episode 级监督标签 |
| `state_label` | `0` 无接触，`1` 弱接触，`2` 稳定抓握，`3` 滑移或掉落 | 逐帧状态；训练使用它定位首次接触 |
| `stage_id` | 抓取阶段编号，名称映射保存在 JSON | 追踪动作流程 |
| `contact_mask` | 每帧左右指接触布尔值 | 检查双侧接触 |
| `object_on_b` | 物体位于并接触 B 区 | 放置证据 |
| `gripper_object_contact` | 夹爪是否接触物体 | 抓握及释放证据 |
| `gripper_released` | 接近全开且已脱离物体 | 释放证据 |

| `finish_reason` | 结束原因 |
| --- | --- |
| `success_on_B` | 确认放置并释放成功 |
| `no_contact` | 闭合阶段超时仍无足够触觉接触 |
| `lift_failed` | 有接触但未在期限内抬起 |
| `dropped` | 抬起后的物体在 B 外低位、低受力持续达到确认帧数 |
| `grasp_lost` | 抬起/搬运/放置中持续失去物体-夹爪接触 |
| `placement_limit_reached` | 末端已到下降下限，仍无法确认物体接触 B |
| `placement_not_confirmed` | B 区下降超时仍未确认放置 |
| `missed_B` | 释放阶段超时仍未满足成功条件 |

### 1.3 采集及预测共用参数

下面默认值来自 `main.py`，同样适用于 `--mode predict`。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--mode` | `auto` | `auto` 采集、`predict` 仿真测试、`robot-follow` 实机反馈显示 |
| `--episodes` | `100` | 完成的成功与失败回合总数，必须大于零 |
| `--save-dir` | `dataset/auto` | 采集父目录；只在 auto 中生成数据集 |
| `--seed` | 不指定 | 夹爪闭合目标抽样种子；明确指定以便复现 |
| `--viewer` | 关闭 | 是否打开 MuJoCo 三维窗口 |
| `--region-size` | `0.15` | A/B 正方形区域边长，单位米 |
| `--a-x` / `--a-y` | `0.20` / `0.09` | A 区平面中心，单位米 |
| `--b-x` / `--b-y` | `0.20` / `-0.09` | B 区平面中心，单位米 |
| `--gripper-close-min` / `--gripper-close-max` | `-0.001` / `0.016` | 每回合闭合位置目标的均匀抽样范围，单位米，左指单侧位移 |
| `--lift-height` | `0.12` | 相对物体初始中心高度的末端抬起高度，单位米 |
| `--place-z-offset` | `0.035` | B 区初始放置目标相对物体初始中心高度的偏移，单位米 |
| `--placement-descent-speed` | `0.02` | B 区目标点持续下降速度，单位米/秒 |
| `--placement-min-clearance` | `0.0` | 目标下降下限相对物体初始中心高度的余量，单位米 |
| `--release-open-tolerance` | `0.002` | 判断夹爪接近全开的单侧位置容差，单位米 |

夹爪参数设置的是位置目标，实际受力还受物体尺寸、摩擦和执行器模型影响；当前没有通过命令行直接指定牛顿单位夹持力的参数。

| 标签/失败判定参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--contact-start-force` | `0.02` | 开始接触所需双指总力度阈值 |
| `--contact-end-force` | `0.01` | 已接触后维持接触的总力度阈值，形成滞回 |
| `--side-contact-force` | `0.01` | 判定单指接触的该指力度和阈值 |
| `--drop-force-threshold` | `0.015` | 判定掉落时的低总力度阈值 |
| `--drop-confirm-frames` | `10` | 掉落连续确认帧数 |
| `--grasp-loss-confirm-frames` | `10` | 搬运中失去夹爪接触的连续确认帧数 |
| `--placement-limit-confirm-frames` | `10` | 到下降下限仍未放置的连续确认帧数 |
| `--stable-confirm-frames` | `10` | 稳定抓握连续确认帧数 |
| `--slip-confirm-frames` | `3` | 滑移连续确认帧数 |
| `--stable-relative-speed` | `0.03` | 物体相对末端速度的稳定阈值，单位米/秒 |
| `--slip-relative-speed` | `0.08` | 物体相对末端速度的滑移阈值，单位米/秒 |

上述力度阈值基于归一化 `force_map` 的求和，不是直接的牛顿数。单个触点三轴力模长以 `TouchProcessor.max_force=3.0` 裁剪和归一化。当前仿真步长是 `0.01` 秒，十帧相当于约 `0.1` 秒仿真时间。

约束：接触结束阈值应介于零和开始阈值之间；稳定速度阈值不大于滑移阈值；闭合最小目标不大于最大目标；确认帧数为正整数；释放容差满足 `0 <= tolerance < 0.035`。

全开位置、IK 参数和阶段超时目前在 `module/auto_grasp_controller.py` 的 `AutoGraspConfig` 中设置，没有对应命令行参数：

| 配置字段 | 默认值 | 含义 |
| --- | --- | --- |
| `open_gripper` | `0.035` 米 | 单侧指全开目标 |
| `ik_gain` / `damping` / `position_tolerance` | `0.7` / `0.05` / `0.008` 米 | IK 增益、阻尼和末端到位容差 |
| `reset_settle_time` / `post_contact_hold_time` | `0.25` / `0.20` 秒 | 复位稳定及接触保持等待 |
| `move_timeout` / `descend_timeout` | `5.0` / `4.0` 秒 | 接近和下降阶段上限 |
| `close_timeout` / `lift_timeout` | `3.0` / `4.0` 秒 | 闭合和抬起阶段上限 |
| `transfer_timeout` / `release_timeout` | `6.0` / `2.5` 秒 | 搬运和释放阶段上限 |
| `fail_reset_wait` / `settle_after_release` | `0.1` / `0.5` 秒 | 失败复位及成功后等待 |

### 1.4 文件结构与均衡整理

```text
dataset/auto/run_*/
  episode_00001_success_*.npz
  episode_00001_success_*.json
  episode_00002_fail_*.npz
  episode_00002_fail_*.json
```

NPZ 保存触觉、状态标签、末端/物体位姿、速度和最终结果。JSON 保存结束原因、标签映射、物体尺寸/质量/摩擦、种子、参数及放置/释放确认信息。空片段不写文件，正常无接触失败片段会保存。

合并多个批次时可用均衡脚本，按文件 SHA-256 去重，以较少类别的数量为基准，从另一类别抽取同样数量；抽样优先保留少见结束原因。以下命令使用已有的两批数据，输出到一个新目录：

```bash
python scripts/build_balanced_dataset.py \
  dataset/auto/run_20260915_154736 \
  dataset/auto/run_20260915_170600 \
  --output-dir dataset/balanced/combined_new_1to1 \
  --seed 42

python scripts/showdata.py dataset/balanced/combined_new_1to1 --summary
```

`sources` 是一个或多个批次目录，脚本仅扫描各目录直接包含的 NPZ；`--output-dir` 必须是尚不存在的目录，`--seed` 默认 `42`。输出中包含 `balance_manifest.json`，记录来源、去重与抽样统计。优先使用硬链接，无法硬链接时复制；使用硬链接的源文件和目标文件共享内容，后续不要原地改写片段文件。

已有的均衡目录是 `dataset/balanced/run_20260915_combined_1to1`，后面的训练示例使用它。

## 2. 训练数据集

### 2.1 确定训练输入与划分

当前网络为逐帧 CNN + 因果 TCN，输入窗口形状 `[T, 2, 30, 15]`，输出失败/成功二分类。每个 episode 取一个固定长度窗口，最终标签作监督；逐帧状态标签用于截窗，尚未作为多类别训练目标。

训练前先用 `scripts/showdata.py --summary` 核对类别。训练加载器会跳过少于 `T` 帧的片段，要求每个类别至少有三个有效片段，才能为训练、验证、测试各留样本。默认 `T=40`，当前 `0.01` 秒步长下窗口覆盖约 `0.4` 秒。

数据按类别分别随机打乱，以 episode 为单位划分为约 `70%/15%/15%` 的 train/val/test。比例会因取整和少量样本保护有所偏差；具体路径保存在 `split.json`。当前不按采集批次或物体身份分组划分，因此测试结果反映当前数据分布内的表现。

| 窗口策略 | 截取方式 |
| --- | --- |
| `first_contact`（默认） | 首个 `state_label > 0` 的帧起取 T 帧；接触太晚则回退末尾，无接触也取末尾；旧数据无状态标签时从头开始 |
| `episode_start` | 取片段开头 T 帧 |
| `episode_end` | 取片段末尾 T 帧 |

### 2.2 训练命令

```bash
conda activate base
python train.py \
  --data-dir dataset/balanced/run_20260915_combined_1to1 \
  --output-dir runs/phase1 \
  --window-size 40 \
  --window-policy first_contact \
  --epochs 100 \
  --patience 15 \
  --batch-size 16 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --seed 42 \
  --device auto
```

`100` 是最大训练轮数。连续 `15` 轮验证集 Macro-F1 没有严格提升即早停，保存的是验证集 Macro-F1 最佳权重。训练使用类别加权交叉熵、AdamW、梯度裁剪和余弦学习率；验证和测试采用未加权交叉熵。

使用三个种子检查结果波动：

```bash
for TRAIN_SEED in 42 7 2026; do
  python train.py \
    --data-dir dataset/balanced/run_20260915_combined_1to1 \
    --window-size 40 \
    --window-policy first_contact \
    --epochs 100 --patience 15 \
    --seed "$TRAIN_SEED"
done
```

每次命令都会生成新目录。种子同时影响数据划分、模型初始化、批次顺序和增强，所以三种子结果包含这些因素共同带来的波动。将 `--window-size 40` 改为 `60` 可训练更长窗口；比较窗口长度时需同时留意可用片段是否因长度过滤而变化。

### 2.3 训练参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--data-dir` | `dataset/auto` | 递归读取此目录中的 NPZ；建议明确指定选定批次或均衡目录 |
| `--output-dir` | `runs/phase1` | 训练结果父目录 |
| `--window-size` | `40` | 输入帧数 T |
| `--window-policy` | `first_contact` | 窗口策略，见上表 |
| `--epochs` | `100` | 最大训练轮数 |
| `--patience` | `15` | 验证集宏平均 F1 不提升的早停耐心值 |
| `--batch-size` | `16` | 每批 episode 数；显存不足时减小 |
| `--learning-rate` | `3e-4` | AdamW 初始学习率 |
| `--weight-decay` | `1e-4` | 权重衰减 |
| `--seed` | `42` | 划分、初始化和采样随机种子 |
| `--workers` | `0` | 数据加载进程数，零表示主进程加载 |
| `--device` | `auto` | 有可用 CUDA 时用 GPU，否则 CPU；也可显式选 `cpu/cuda` |
| `--augment` / `--no-augment` | 开启 | 仅训练集力度缩放、加噪声及触点丢弃 |
| `--plot` / `--no-plot` | 开启 | 训练结束后是否自动输出图表 |

窗口大小、轮数、批大小和 patience 必须为正整数。显式指定 `--device cuda` 但 CUDA 不可用时程序报错，不会自动回退 CPU。

### 2.4 输出与图表

```text
runs/phase1/run_*/
  best.pt              最佳权重、模型配置、窗口策略、类别名称和最佳 epoch
  config.json          命令行训练配置
  split.json           固定 train/val/test 相对路径清单
  history.json         逐轮训练与验证指标
  result.json          最佳 epoch、划分统计及测试集指标
  plots/
    training_curves.png Loss、Accuracy、Macro-F1 与最佳 epoch
    confusion_matrix.png 原始计数与按真实类别归一化的矩阵
    class_metrics.png   失败/成功的 Precision、Recall、F1
```

已有运行可以重新绘图：

```bash
python scripts/plot_results.py runs/phase1/run_20260915_192936_239054
```

比较三次运行时，把下列三个路径替换为终端显示的实际运行目录：

```bash
python scripts/plot_results.py \
  runs/phase1/RUN_SEED_42 \
  runs/phase1/RUN_SEED_7 \
  runs/phase1/RUN_SEED_2026 \
  --output-dir runs/phase1/plots_summary
```

脚本以第一项运行生成单次曲线/指标图，并额外生成所有运行的 `run_comparison.png`。默认只保存文件，加 `--show` 同时打开图表窗口。重点同时查看 Macro-F1 和失败/成功各自的召回率，避免仅以 Accuracy 判断不均衡数据上的表现。

## 3. 预测与数据集评估

### 3.1 对已有 NPZ 数据集评估

已有数据的评估使用 `evaluate.py`，复用权重旁边的 `split.json`，不会重新随机划分。数据根目录必须与训练时的相对路径结构匹配。下面模型当时使用的正是已有均衡目录：

```bash
conda activate cuda
python evaluate.py runs/phase1/run_20260915_192936_239054/best.pt \
  --data-dir dataset/balanced/run_20260915_combined_1to1 \
  --split test \
  --batch-size 32 \
  --device auto
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| 位置参数 `checkpoint` | 必填 | `best.pt` 路径，相邻目录需要包含 `split.json` |
| `--data-dir` | `dataset/auto` | 训练数据根目录，必须能按清单相对路径找到文件 |
| `--split` | `test` | 可选 `train`、`val`、`test`；最终性能使用 test |
| `--batch-size` | `32` | 每批评估样本数量 |
| `--device` | `auto` | `auto/cpu/cuda` |

T 和窗口策略从 checkpoint 自动读取。终端输出 JSON，包含 loss、accuracy、macro_f1、per_class 和 confusion_matrix；当前不会输出逐条 NPZ 的预测表，也不能直接把任意新目录替换为训练数据根目录来自动评估所有文件。

### 3.2 在仿真中运行新抓取回合

使用你此前测试通过的命令：

```bash
conda activate cuda
python main.py --mode predict \
  --checkpoint runs/phase1/run_20260915_192936_239054/best.pt \
  --episodes 10 \
  --seed 42 \
  --device auto \
  --predict-output-dir runs/predict \
  --viewer
```

要测试新训练模型，将 `--checkpoint` 替换为新运行的 `best.pt`。`predict` 复用 `auto` 的控制器、轨迹、夹爪目标抽样与成功/失败判定；使用相同种子和抓取参数可以对照相同抽样序列。若采集时调整过区域或夹爪参数，预测时应显式传入同样的参数，它们不会从训练权重中恢复。

模型读取触觉序列进行分类，不参与轨迹或力度控制，也不会向 `dataset/` 保存新的采集文件。`robot-follow` 当前未接入这个模型预测流程。

| 预测专用参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--checkpoint` | 必填 | 训练产生的 `best.pt` |
| `--device` | `auto` | 推理设备，可选 `cpu/cuda` |
| `--predict-output-dir` | `runs/predict` | 预测记录父目录，每次新建 run_* |

回合数、种子、区域、夹爪和各判定阈值使用第 1.3 节的共用参数。预测的 T 和截窗策略来自 checkpoint，`main.py` 没有 `--window-size` 或 `--window-policy` 选项。

### 3.3 预测时机与结果读取

`first_contact` 在首次接触后凑齐 T 帧时进行在线预测；回合结束时按训练截窗规则确定最终窗口，必要时重算。无接触回合在结束后取末尾 T 帧；`episode_end` 总是在结束后预测。少于 T 帧的回合标记 `skipped`，不参与准确率或混淆矩阵。

终端逐回合打印实际结果、结束原因、预测类别、`P(success)`、窗口范围和累计准确率；viewer 右侧显示在线预测及统计。例如：

```text
episode 1: actual=success (reason=success_on_B), predicted=success,
P(success)=0.830, window=320:360, accuracy=1/1 (100.0%)
```

这里的数值仅演示输出格式，窗口为零起点、右端不包含的帧区间。

```text
runs/predict/run_*/
  results.json          模型信息、逐回合结果、准确率和混淆矩阵
  episodes.csv          episode、真值、预测、成功概率、原因和窗口位置
  prediction_results.png 各回合概率/真值对照与混淆矩阵
```

JSON/CSV 在每个回合结束后更新，提前退出仍保留已完成回合。图表在退出时生成；缺少 Matplotlib 时保留表格并提示安装。成功概率是 softmax 分数，当前没有概率校准。

## 4. 仿真与实机联动

### 4.1 联动范围

当前 `robot-follow` 的数据流为实机 SDK 反馈 → 校验与夹爪标定映射 → MuJoCo 姿态 → viewer。程序读取六轴角度（弧度）和夹爪宽度（米），只更新仿真，不使能实机或发送运动指令，也不启动自动采集和结果标签判定。

移动实机时，仿真机械臂与夹爪跟随反馈姿态。仿真物体、接触和触觉不代表实机测量；当前没有实机物体跟踪和真实触觉采集。SDK 项目地址：[pyAgxArm](https://github.com/agilexrobotics/pyAgxArm)。

### 4.2 启动 CAN 并检查反馈通道

实机上电、USB-CAN 已连接且系统存在 `can0` 后执行：

```bash
ip -details link show can0
sudo ip link set can0 up type can bitrate 1000000
ip -details link show can0
```

检查输出含 `UP`、`LOWER_UP`，CAN 状态为 `ERROR-ACTIVE`，速率为 `1000000`。`ERROR-ACTIVE` 是 CAN 正常可通信的状态名称。接口已启动但需要修改速率时，先停用再配置：

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000
```

若系统中没有 `can0`，先处理 USB-CAN 识别与驱动；仅将接口置为 UP 不代表实机一定正在发送有效反馈。

### 4.3 使用实测夹爪端点运行

目前测得的当前夹爪反馈端点为完全闭合 `-0.0107` 米、完全打开 `0.0872` 米。使用已安装 SDK 的 `base` 环境：

```bash
conda activate base
cd /home/luo/project/piper
python main.py --mode robot-follow \
  --can-channel can0 \
  --piper-firmware default \
  --real-gripper-closed-width -0.0107 \
  --real-gripper-open-width 0.0872 \
  --feedback-max-age 0.5 \
  --viewer
```

启动后等待首个有效反馈，最长约五秒；连接正常会打印只读提示和夹爪标定范围。终端约每秒打印：

```text
joints(rad): [...]  real gripper(m): ...  sim gripper(m): ...
```

检查完全闭合时仿真总开度约为 `0.0000` 米，完全打开时约为 `0.0700` 米。这是当前映射的结果，替代未标定时实机尚未到端点而仿真提前全开/全闭的情况。

### 4.4 夹爪行程映射

实机反馈宽度 `w` 先计算相对行程比例，再换算单指位移：

```text
fraction = (w - real_closed) / (real_open - real_closed)
fraction = clip(fraction, 0, 1)
左指位置 = fraction × 0.035 米
右指位置 = -左指位置
仿真总开度 = 2 × 左指位置
```

| 当前位置 | 实机反馈总宽度 | 仿真显示总开度 |
| --- | --- | --- |
| 标定闭合端点 | `-0.0107` 米 | `0.0000` 米 |
| 标定行程中点 | `0.03825` 米 | `0.0350` 米 |
| 标定全开端点 | `0.0872` 米 | `0.0700` 米 |

负的实测闭合端点可以作为偏移校准。超出这两个端点时，仿真显示被限制在最近端点，并只警告一次；日志中的实机宽度仍保留原值。标定参数只影响仿真显示，不修改实机夹爪配置。更换夹爪或标定变化后，应重新测量端点。

### 4.5 当前仿真关节限位与反馈校验

`PiperSim` 默认加载 `agilex_piper/scene_touch.xml`，它包含 `piper_touch.xml`。以下是该模型的限位，不代表已经核实的实机机械限位；六轴反馈不做角度缩放或零点偏移，直接使用 SDK 的弧度值。

| 关节 | 模型范围 | 单位 | 反馈同步行为 |
| --- | --- | --- | --- |
| `joint1` | `[-2.618, 2.618]` | 弧度 | 实测角度写入仿真 |
| `joint2` | `[0, 3.14]` | 弧度 | 实测角度写入仿真 |
| `joint3` | `[-2.697, 0]` | 弧度 | 实测角度写入仿真 |
| `joint4` | `[-1.832, 1.832]` | 弧度 | 实测角度写入仿真 |
| `joint5` | `[-1.22, 1.22]` | 弧度 | 实测角度写入仿真 |
| `joint6` | `[-3.14, 3.14]` | 弧度 | 实测角度写入仿真 |
| `joint7` 左指 | `[-0.001, 0.035]` | 米 | 联动显示使用 `[0, 0.035]` 的线性映射 |
| `joint8` 右指 | `[-0.035, 0]` | 米 | 联动显示使用负的左指位移 |

六轴角度越过模型限位不超过 `0.1` 弧度时，仍显示实测角度，每个关节仅警告一次。例如 `joint2=-0.0421` 弧度在允许容差内。超出容差则停止显示并报错；`joint2` 的反馈校验接受区间是 `[-0.1, 3.24]` 弧度。这个容差是只读显示的检查设置，不是实机运动许可范围。

`joint_margin=0.1` 在 `module/real_robot_sync.py` 的构造函数中设置，没有 `--joint-margin` 命令行参数。若需校对模型限位，修改位置在 `agilex_piper/piper_touch.xml`，应依据实际设备的规格和标定值确认，不能将显示容差当作实机限位配置。

### 4.6 实机模式参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--mode robot-follow` | 默认模式是 auto | 显式选择只读实机联动 |
| `--can-channel` | `can0` | 已启动的 SocketCAN 接口名 |
| `--piper-firmware` | `default` | SDK 驱动档案：`default/v183/v188/v189`，按实际固件选择 |
| `--feedback-max-age` | `0.5` 秒 | 反馈最大允许年龄，必须大于零 |
| `--real-gripper-closed-width` | `0.0` 米 | 实测完全闭合反馈值，可为负数 |
| `--real-gripper-open-width` | 未指定 | 未指定时向 SDK 查询最大行程配置；建议使用本机实测 `0.0872` |
| `--viewer` | 关闭 | 打开三维姿态窗口；不加时只更新场景并打印日志 |

实测全开值必须为有限数且大于闭合值。缺少、过期、非有限反馈、非宽度模式夹爪消息或六轴严重越界都会停止联动；退出时断开 SDK 连接。

### 4.7 常见问题

| 现象 | 检查或处理 |
| --- | --- |
| `[WARN] Device is DOWN` | 检查 `ip -details link show can0`，确认以 1 Mbit/s 启动接口 |
| `no joint feedback` | 检查实机上电、CAN 连线、接口名称、速率及固件驱动档案；UP 状态本身不能保证反馈到达 |
| `stale joint/gripper feedback` | 检查反馈是否中断、时钟及 USB-CAN 状态；按实际刷新周期设置最大年龄 |
| `joint2 feedback ... outside ...` 警告 | 偏差在容差内时只是一次提醒；仍显示实际角度 |
| `exceeds simulation ... by more than ...` | 核对型号、实际角度和模型限位；该反馈被拒绝，仿真停止更新 |
| 仿真夹爪提前全开或全闭 | 使用实测闭合/打开端点，确认单位为米且反馈为 width 模式 |
| Wayland 窗口位置警告 | 单独的窗口位置警告不等于 CAN 错误；若伴随 OpenGL 显示失败，可在 Xorg/X11 桌面会话中运行并检查显卡驱动 |

## 5. 查阅命令帮助

本文参数以当前代码为准，后续新增选项可通过以下命令查看：

```bash
python main.py --help
python train.py --help
python evaluate.py --help
python scripts/showdata.py --help
python scripts/build_balanced_dataset.py --help
python scripts/plot_results.py --help
```
