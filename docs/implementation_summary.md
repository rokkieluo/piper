# 项目实现汇总

本文档记录当前已经完成的数据采集、标签、数据整理和第一阶段训练改动。

## 1. 相关文件

```text
module/auto_grasp_controller.py   自动抓取状态机、成功/失败判定、逐帧标签
module/data_collector.py           NPZ/JSON 数据保存
main.py                           采集命令行参数和主循环
scripts/showdata.py               数据摘要和逐帧可视化
training/data.py                  训练窗口截取和数据集加载
train.py                          第一阶段成功/失败二分类训练
evaluate.py                       checkpoint 独立评估
scripts/build_balanced_dataset.py 去重并构建 1:1 均衡数据集
tests/                            采集、标签和训练回归测试
```

## 2. 数据采集流程

每次尝试从 MuJoCo reset 后开始记录，经过以下阶段：

```text
reset_settle -> move_above_A -> descend_to_A -> close_gripper
-> contact_hold -> lift_from_A -> move_above_B -> descend_to_B
-> release_on_B -> settle_on_B
```

失败时进入 `fail_reset_wait`，等待后重新 reset。失败 episode 也会保存，不再丢弃。

每个 episode 生成一对同名文件：

```text
episode_XXXXX_success_*.npz
episode_XXXXX_success_*.json
```

NPZ 文件保存逐帧数组，JSON 保存解释信息和采集参数。

## 3. Episode 级标签

NPZ 中的 `success`/`result` 是当前第一阶段训练使用的 episode 级二分类标签：

```text
0 / fail       失败
1 / success    成功
```

JSON 中的 `finish_reason` 用于区分失败原因：

```text
success_on_B              满足完整放置和释放条件
no_contact                关闭夹爪超时仍未检测到接触
lift_failed               有接触但未成功抬起
dropped                   抬起后物体掉落到低处
grasp_lost                抬起/转移/放置期间持续失去夹爪-物体接触
placement_not_confirmed   到达放置阶段但未确认物体、B、夹爪同时满足条件
placement_limit_reached   到达安全最低放置高度仍未接触 B
missed_B                  已进入释放阶段，但释放超时仍未完成放置
```

### 严格成功条件

成功不能只根据物体位置判断，必须满足以下顺序：

1. 物体已经被抬起。
2. 物体与 B 区发生接触。
3. 同一时刻夹爪仍与物体接触。
4. 确认以上条件后进入 `release_on_B`。
5. 夹爪张开到目标开度，并且夹爪与物体脱离。
6. 物体仍然接触 B 区，才记录 `success_on_B`。

夹爪-物体接触通过 MuJoCo 碰撞对判断，而不是只依赖触觉力阈值。

## 4. 逐帧状态标签

NPZ 中的 `state_label` 为四分类状态：

```text
0 no_contact      未接触
1 weak_contact    弱接触、单侧接触或尚未形成稳定夹持
2 stable_grasp    双侧接触、物体已抬起且满足稳定确认帧数
3 slip_or_drop    滑移或掉落
```

标签只使用当前帧和历史帧计算，不读取未来信息。当前使用：

- 接触开始/结束迟滞，减少力信号抖动；
- 左右手指分别统计接触力；
- 双侧接触判断；
- 物体相对 TCP 的有限差分速度；
- `stable_confirm_frames` 稳定确认窗口；
- `slip_confirm_frames` 滑移确认窗口。

相关阈值会写入每个 episode 的 JSON `label_thresholds`。

## 5. NPZ 逐帧字段

所有字段第一维长度均为帧数 `N`：

```text
time
force_map                  [N, 2, 30, 15]
position_map               [N, 2, 30, 15]
raw_force
force_stats
stage_id
state_label
contact_mask               左右手指接触布尔值
object_on_b                 当前帧物体是否接触 B
gripper_object_contact     当前帧夹爪是否接触物体
gripper_released            当前帧是否已张开且脱离物体
tcp_pos / tcp_quat
object_pos / object_quat / object_vel
gripper_ctrl / gripper_qpos
success / result
```

JSON 还保存：

```text
episode_index / attempt_index
finish_reason
stage_names / state_names
object.shape / size / mass / friction / initial_position
gripper_close_target
seed
collection_config
label_thresholds
placement_confirmed / release_confirmed
```

## 6. 采集参数

新增参数可通过 `main.py --help` 查看，主要包括：

```text
--contact-start-force
--contact-end-force
--side-contact-force
--stable-confirm-frames
--slip-confirm-frames
--stable-relative-speed
--slip-relative-speed
--release-open-tolerance
--placement-descent-speed
--placement-min-clearance
--placement-limit-confirm-frames
--grasp-loss-confirm-frames
```

放置阶段不再固定停在一个高度，而是持续下降；夹持丢失连续达到确认帧数时快速失败，
避免无效等待数秒。

## 7. 训练数据窗口

采集器从 reset 开始记录，因此训练不能固定读取 NPZ 的前 `T` 帧。当前
`training/data.py` 支持三种窗口策略：

```text
first_contact    从首次 state_label > 0 的帧开始；无接触失败取末尾窗口
episode_start    从 episode 第 0 帧开始
episode_end      从 episode 末尾开始
```

第一阶段默认使用：

```text
T = 40
window_policy = first_contact
```

窗口策略会保存到训练 checkpoint，`evaluate.py` 会自动复用。

当前第一阶段仍然是 episode 级二分类；`state_label` 已保存，但尚未用于四分类网络训练。

## 8. 均衡数据集

使用 `scripts/build_balanced_dataset.py` 构建均衡集：

```bash
python scripts/build_balanced_dataset.py \
  dataset/auto/run_20260915_154736 \
  dataset/auto/run_20260915_170600 \
  --output-dir dataset/balanced/run_20260915_combined_1to1 \
  --seed 42
```

构建时按 NPZ 内容 SHA-256 去重。两组源数据实际逐文件完全重复，因此：

```text
源文件：1000
去重后：500
最终失败：122
最终成功：122
```

最终数据集：

```text
dataset/balanced/run_20260915_combined_1to1
```

选择记录在 `balance_manifest.json`，原始两个 run 未修改。

## 9. 当前均衡集训练结果

均衡集三个随机种子的测试结果：

```text
seed 42:   Accuracy 89.47%, Macro-F1 89.44%
seed 7:    Accuracy 92.11%, Macro-F1 92.10%
seed 2026: Accuracy 86.84%, Macro-F1 86.76%
```

三种子平均：

```text
Accuracy  89.47% +/- 2.63%
Macro-F1  89.43% +/- 2.67%
```

三个测试集合并后的混淆矩阵为：

```text
                预测失败    预测成功
真实失败            51          6
真实成功             6         51
```

三种子对比图：

```text
runs/phase1/plots_summary/balanced_20260915/run_comparison.png
```

## 10. 推荐命令

采集并查看：

```bash
python main.py --episodes 20 --save-dir dataset/auto --seed 42 --viewer
python scripts/showdata.py dataset/auto --summary
python scripts/showdata.py dataset/auto
```

训练：

```bash
python train.py \
  --data-dir dataset/balanced/run_20260915_combined_1to1 \
  --window-size 40 \
  --window-policy first_contact \
  --epochs 100 \
  --patience 15 \
  --seed 42
```

评估：

```bash
python evaluate.py runs/phase1/run_YYYYMMDD_HHMMSS/best.pt \
  --data-dir dataset/balanced/run_20260915_combined_1to1 \
  --split test
```

## 11. 验证状态和注意事项

当前自动化测试共 15 项，覆盖：

- NPZ 字段和 JSON 元数据；
- `no_contact` 失败保存；
- 掉落只计数一次并正确 reset；
- 严格放置后释放成功条件；
- 渐进下降和快速夹持丢失；
- 四类状态标签转换；
- 训练窗口策略和旧数据兼容；
- 模型输出、因果性和数据分层。

运行测试：

```bash
python -m unittest discover -s tests -v
```

后续正式实验仍需增加非重复成功轨迹，并按物体参数或物体身份做泛化划分。当前训练结果
属于第一阶段二分类基线，不等同于最终的四类逐帧状态识别结果。
