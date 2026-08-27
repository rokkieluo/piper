# 基于位置-力强度交叉融合的双分支触觉网络方案

本文档整理当前 Piper + MuJoCo 触觉仿真项目的研究方案，目标是形成一条可以支撑小论文的完整技术路线：

> 利用仿真采集的触觉阵列数据，将触觉信号解耦为接触位置分布与接触力强度两个分支，通过接触感知交叉融合网络判断物体抓取状态。

建议论文题目方向：

```text
基于接触感知位置-力强度交叉融合的机械臂抓取状态识别方法
```

英文可写作：

```text
Contact-Aware Position-Force Cross Fusion Network for Tactile Grasp State Recognition
```

简称：

```text
CPF-FusionNet
```

---

## 1. 研究目标

### 1.1 任务定义

输入一段连续触觉数据，判断当前机械臂夹爪对物体的抓取状态。

推荐第一阶段做 4 分类：

```text
0 no_contact      未接触
1 weak_contact    弱接触 / 单侧接触 / 未形成稳定夹持
2 stable_grasp    稳定抓取
3 slip_or_drop     滑移或掉落
```

如果初期数据采集和标签设计还不稳定，可以先降成 3 分类：

```text
0 no_contact
1 unstable_contact
2 stable_grasp
```

后续扩展：

```text
0 no_contact
1 left_contact
2 right_contact
3 bilateral_unstable
4 stable_grasp
5 slip
6 drop
```

### 1.2 核心研究问题

普通做法通常直接把触觉图当成一张灰度图，送入 CNN 或 CNN+LSTM：

```text
force_map -> CNN -> temporal model -> classifier
```

这种方法的问题是：它把“力在哪里”和“力有多大”混在一起，融合过程缺少物理解释。

本方案希望回答：

```text
接触位置分布和接触力强度是否可以解耦建模？
二者如何融合才能更好判断稳定抓取、弱接触和滑移？
```

---

## 2. 相关研究启发

近几年触觉学习的趋势可以概括为四点：

1. 触觉不再只是手工特征，而是学习通用触觉表征。
2. 触觉图像或触觉阵列逐渐被 token 化，用 Transformer 或 attention 建模。
3. 多模态融合从简单 concat 转向 cross-attention、force-guided attention、阶段自适应融合。
4. 抓取稳定性、滑移检测等任务越来越强调时序触觉。

可参考工作：

- Sparsh: Self-supervised touch representations for vision-based tactile sensing, CoRL 2024。它强调通过自监督预训练学习通用触觉表征，并在 TacBench 上评估多个触觉任务。链接：https://arxiv.org/abs/2410.24090
- UniT: Data Efficient Tactile Representation with Generalization to Unseen Objects, 2024/2025。它使用紧凑潜空间做触觉表征，并强调对未见物体的泛化。链接：https://arxiv.org/abs/2408.06481
- Transferable Tactile Transformers for Representation Learning Across Diverse Sensors and Tasks, 2024。它说明 tactile transformer 可以跨传感器和任务学习触觉表征。链接：https://arxiv.org/abs/2406.13640
- Visuo-Tactile Transformers for Manipulation, 2022。它使用 self-attention 和 cross-modal attention 融合视觉与触觉，说明 cross-attention 对触觉融合有价值。链接：https://arxiv.org/abs/2210.00121
- Adaptive Visuo-Tactile Fusion with Predictive Force Attention, 2025。它使用 force-guided attention 做自适应融合，给本方案的“力引导位置分支”提供了很好的方法启发。链接：https://arxiv.org/html/2505.13982v2
- Grasp Stability Assessment Through Attention-Guided Cross-Modality Fusion and Transfer Learning, 2023。它指出简单特征级融合不足，并用 attention/cross-attention 做抓取稳定性评估。链接：https://arxiv.org/abs/2308.00980

本方案与这些工作的关系：

```text
不是做视觉-触觉融合，而是做触觉内部的双因素融合。
不是简单双分支 concat，而是做位置-力强度的物理含义解耦与交叉融合。
不是只判断单帧接触，而是用短时序列识别动态抓取状态。
```

---

## 3. 当前仿真触觉数据格式

当前项目关键文件：

```text
agilex_piper/piper_touch.xml      触觉传感器和机械臂模型
agilex_piper/scene_touch.xml      场景和物体
module/touch_processor.py         触觉数据处理
module/touch_display.py           触觉数据显示
module/auto_grasp_controller.py   自动抓取控制与成功/失败检测
module/data_collector.py          episode 级触觉数据保存
module/piper_sim.py               MuJoCo 模型封装
main.py                           自动采集主循环
```

### 3.1 MuJoCo 原始传感器数据

MuJoCo 中每个 force sensor 输出 3 维力：

```text
[Fx, Fy, Fz]
```

在 Python 中位置为：

```python
d.sensordata[adr:adr + 3]
```

当前模型中：

```text
sensor 数量: 810
每个 sensor 维度: 3
sensordata 总长度: 810 * 3 = 2430
```

四块触觉区域：

```text
link7_touch_A: 15 x 15 = 225 点
link7_touch_B: 15 x 12 = 180 点
link8_touch_A: 15 x 15 = 225 点
link8_touch_B: 15 x 12 = 180 点
```

其中 link7 和 link8 对应两侧夹爪。

### 3.2 当前代码中的力大小图

当前 `touch_display.py` 中使用力的模长：

```python
force_norm = sqrt(Fx^2 + Fy^2 + Fz^2)
```

并归一化到 0 到 1：

```python
force_value = clip(force_norm, 0, max_force) / max_force
```

默认：

```python
max_force = 3.0
```

所以每个触觉点最终是一个标量：

```text
0.0 表示无接触
1.0 表示达到或超过 max_force
```

### 3.3 推荐训练数据格式

不要保存 OpenCV resize 之后的图像。训练时应该保存原始触觉阵列。

建议把每一帧触觉整理成：

```text
finger7 = concat(link7_A, pad(link7_B)) -> 30 x 15
finger8 = concat(pad(link8_B), link8_A) -> 30 x 15
force_map = stack(finger7, finger8)     -> 2 x 30 x 15
```

其中 `link7_B` 和 `link8_B` 原始是 `15 x 12`，需要补零到 `15 x 15`。

每个时间窗口：

```text
force_seq: [T, 2, 30, 15]
```

推荐：

```text
T = 10, 20, 30 分别实验
```

如果仿真 timestep 是 0.01s，并且每帧都采样：

```text
T=20 表示 0.2s 触觉历史
```

---

## 4. 位置分支与力强度分支

### 4.1 力强度分支 M

力强度图直接使用归一化后的力模长：

```text
M_t in R^{2 x 30 x 15}
```

其中每个元素：

```text
M_t[c, h, w] = clip(||F||, 0, Fmax) / Fmax
```

它表达：

```text
哪个触觉点受到多大的力
```

### 4.2 位置分布分支 P

位置分支不直接使用二值图，而是使用归一化接触分布：

```text
P_t = M_t / (sum(M_t) + eps)
```

可以按整帧归一化：

```text
P_t = M_t / (sum over all fingers and pixels + eps)
```

也可以按每个手指归一化：

```text
P_t[c] = M_t[c] / (sum over pixels in finger c + eps)
```

推荐第一版使用每个手指单独归一化，因为左右手指接触强度可能不同，单独归一化更能保留每个手指内部的接触位置分布。

它表达：

```text
力主要分布在哪里
```

它弱化总力大小，强调空间结构。

### 4.3 可加入的物理统计量

从 `M_t` 和 `P_t` 中可以提取辅助物理特征：

```text
total_force      总接触力
max_force        最大触觉点力
contact_area     大于阈值的接触点数量
center_x/y       接触中心
entropy          接触分布熵
left_right_diff  左右手指总力差
force_slope      总力变化率
```

这些统计量可以作为融合模块的 gate 输入，而不是直接替代深度网络。

---

## 5. 网络总体结构

推荐网络名：

```text
CPF-FusionNet
```

整体结构：

```text
force_seq M:    [B, T, 2, 30, 15]
position_seq P: [B, T, 2, 30, 15]

        M ---- Force Encoder ---- M tokens ----\
                                                 Contact-Aware Cross Fusion ---- Temporal Encoder ---- Classifier
        P -- Position Encoder -- P tokens ----/
```

### 5.1 输入

每个样本：

```text
M_seq: [T, 2, 30, 15]
P_seq: [T, 2, 30, 15]
label: int
```

训练 batch：

```text
M_seq: [B, T, 2, 30, 15]
P_seq: [B, T, 2, 30, 15]
y:     [B]
```

### 5.2 双分支编码器

两个分支结构可以相同，但参数不共享。

推荐轻量 CNN：

```text
Conv2d(2, 16, 3, padding=1) + BN + ReLU
Conv2d(16, 32, 3, padding=1) + BN + ReLU
MaxPool2d(2)
Conv2d(32, 64, 3, padding=1) + BN + ReLU
AdaptiveAvgPool2d((5, 5))
```

单帧输出：

```text
[64, 5, 5]
```

再展平为 token：

```text
25 tokens, 每个 token 64 维
```

对于序列：

```text
M_tokens: [B, T, 25, D]
P_tokens: [B, T, 25, D]
```

### 5.3 创新融合模块：Contact-Aware Cross Fusion

普通融合：

```text
concat(M_feature, P_feature)
```

本文方法：

```text
1. 力统计量生成接触可靠性 gate
2. force tokens 与 position tokens 做 cross-attention
3. 加入左右手指平衡约束或接触一致性辅助损失
```

#### 5.3.1 接触可靠性 Gate

先从力强度图提取统计量：

```text
s_t = [total_force, max_force, contact_area, entropy, left_right_diff, force_slope]
```

通过 MLP 得到 gate：

```text
g_t = sigmoid(MLP(s_t))
```

用 gate 调制位置特征：

```text
P'_tokens = P_tokens * g_t
```

含义：

```text
接触力过弱时，位置图不稳定，降低位置分支权重。
接触力稳定时，位置图更可靠，提高位置分支权重。
```

这是本方案的第一个创新点。

#### 5.3.2 位置-力 Cross Attention

做双向交叉注意力：

```text
F_p2m = CrossAttention(Q=P'_tokens, K=M_tokens, V=M_tokens)
F_m2p = CrossAttention(Q=M_tokens, K=P'_tokens, V=P'_tokens)
```

再融合：

```text
F_fuse = LayerNorm(M_tokens + P'_tokens + F_p2m + F_m2p)
```

物理解释：

```text
位置分支查询力分支：某个接触区域的力强度是否合理？
力分支查询位置分支：某个力峰值是否出现在合理接触位置？
```

这是本方案的第二个创新点。

#### 5.3.3 接触一致性辅助损失

可以加入一个轻量辅助损失，提高融合可解释性。

例如左右手指平衡：

```text
L_balance = |sum(M_left) - sum(M_right)|
```

但不要对所有状态都强制平衡。只在 `stable_grasp` 类别上使用：

```text
if label == stable_grasp:
    L_aux = L_balance
else:
    L_aux = 0
```

总损失：

```text
L = L_cls + lambda_aux * L_aux
```

建议：

```text
lambda_aux = 0.01 或 0.05
```

这是可选创新点，初版可以先实现分类损失，后续作为增强实验。

### 5.4 时序编码器

抓取状态需要时序信息，尤其是滑移和掉落。

推荐优先级：

```text
第一版: TCN
第二版: GRU
第三版: 小型 Transformer
```

建议第一版使用 TCN：

```text
Temporal Conv1d, kernel_size=3, dilation=1
Temporal Conv1d, kernel_size=3, dilation=2
Temporal Conv1d, kernel_size=3, dilation=4
GlobalAvgPool over time
```

原因：

```text
数据量较小时 TCN 比 Transformer 更稳
训练速度快
容易做消融
能捕捉短时滑移趋势
```

融合后每帧得到：

```text
z_t: [B, D]
```

时序输入：

```text
Z: [B, T, D]
```

输出：

```text
h: [B, D]
```

分类头：

```text
MLP(D -> 128 -> num_classes)
```

---

## 6. 数据集采集完整流程

### 6.1 Episode 随机化

每次仿真采集一个 episode，随机以下参数：

```text
物体位置: x, y, z
物体尺寸: box size 或不同 shape
物体质量: mass
摩擦系数: friction
夹爪初始开合: gripper initial
夹爪闭合速度: close_speed
机械臂接近方向: tcp approach pose
是否抬升: lift motion
外部扰动: 可选
```

第一阶段可以只用 box：

```text
box 位置随机
box 尺寸随机
box 质量随机
box 摩擦随机
```

第二阶段增加：

```text
cylinder
sphere
不同长宽比 cuboid
```

### 6.2 单个 Episode 阶段

推荐 episode 分成 5 个阶段：

```text
1. reset
2. approach: 末端移动到物体附近
3. close: 夹爪闭合直到接触
4. lift: 抬升物体
5. hold: 保持一段时间，观察是否稳定/滑移/掉落
```

每一帧记录：

```text
time
force_map: [2, 30, 15]
position_map: [2, 30, 15]
raw_force_vector: 可选, [810, 3]
tcp_pos
tcp_quat
gripper_ctrl
gripper_qpos
object_pos
object_quat
object_vel
label_frame
episode_id
```

### 6.3 自动标签规则

#### no_contact

```text
total_force < force_threshold
```

建议：

```text
force_threshold = 0.02 或 0.05
```

需要根据仿真噪声调整。

#### weak_contact

满足：

```text
total_force >= force_threshold
```

但以下任意一项成立：

```text
只有单侧手指接触
左右力差过大
物体未被成功抬起
接触面积过小
```

#### stable_grasp

满足：

```text
物体被抬起到目标高度
保持 N 帧不掉落
物体相对 TCP 速度较小
左右手指均有接触
```

可写为：

```text
object_z > initial_z + lift_height_threshold
relative_speed < speed_threshold
left_force > left_threshold
right_force > right_threshold
```

#### slip_or_drop

滑移：

```text
有接触力
物体相对夹爪位置持续变化
接触中心在触觉图上持续移动
```

掉落：

```text
lift 后 object_z 下降
或 object_z 回到地面附近
```

### 6.4 窗口切片

不要把整个 episode 直接作为一个样本。建议使用滑动窗口：

```text
window_size = T
stride = 2 或 5
```

例如：

```text
episode 长度 300 帧
T = 20
stride = 5
得到约 57 个训练样本
```

窗口标签：

```text
使用窗口最后一帧标签
```

或：

```text
使用窗口内多数标签
```

抓取状态识别更推荐最后一帧标签，因为它符合在线判断：

```text
根据过去 T 帧，判断当前状态
```

### 6.5 数据保存格式

推荐使用 `.npz`，简单直接：

```text
dataset/
  train/
    episode_000001.npz
    episode_000002.npz
  val/
  test/
```

每个 `.npz`：

```python
{
    "force_map": np.ndarray,      # [N, 2, 30, 15]
    "position_map": np.ndarray,   # [N, 2, 30, 15]
    "raw_force": np.ndarray,      # optional, [N, 810, 3]
    "tcp_pos": np.ndarray,        # [N, 3]
    "object_pos": np.ndarray,     # [N, 3]
    "gripper_ctrl": np.ndarray,   # [N]
    "label": np.ndarray,          # [N]
    "meta": dict                  # object mass, size, friction, etc.
}
```

训练时再由 Dataset 类切窗口：

```python
M_seq = force_map[i:i+T]
P_seq = position_map[i:i+T]
y = label[i+T-1]
```

---

## 7. 建议代码模块

建议新增：

```text
module/touch_processor.py       触觉数据转 force_map / position_map
module/data_collector.py        自动采集 episode
module/labeler.py               自动标签规则
module/trajectory.py            approach/close/lift/hold 控制流程
models/cpf_fusion_net.py        网络结构
train.py                        训练入口
eval.py                         评估入口
configs/tactile_grasp.yaml      参数配置
```

### 7.1 touch_processor.py

职责：

```text
从 d.sensordata 读取原始 force vector
生成 force_map [2, 30, 15]
生成 position_map [2, 30, 15]
生成统计量 force_stats
```

核心函数：

```python
class TouchProcessor:
    def get_raw_force(self) -> np.ndarray:
        # [810, 3]

    def get_force_map(self) -> np.ndarray:
        # [2, 30, 15]

    def get_position_map(self) -> np.ndarray:
        # [2, 30, 15]

    def get_force_stats(self) -> dict:
        # total_force, max_force, contact_area, entropy, left_right_diff
```

### 7.2 data_collector.py

职责：

```text
自动运行仿真 episode
随机化物体参数
执行抓取动作
保存数据
```

### 7.3 labeler.py

职责：

```text
根据接触力、物体高度、相对速度、左右接触情况生成标签
```

### 7.4 cpf_fusion_net.py

职责：

```text
实现双分支 CNN
实现 force-guided gate
实现 position-force cross attention
实现 temporal encoder
输出分类结果
```

---

## 8. 实验设计

### 8.1 主实验

比较方法：

```text
Baseline 1: Force-CNN
Baseline 2: Position-CNN
Baseline 3: Force+Position Concat
Baseline 4: Late Fusion
Ours: CPF-FusionNet
```

输入相同：

```text
T 帧触觉序列
```

评价指标：

```text
Accuracy
Macro F1
Per-class F1
Confusion Matrix
Stable grasp recall
Slip/drop recall
```

其中小论文中最重要的是：

```text
Macro F1
stable_grasp recall
slip_or_drop recall
```

因为抓取状态分类往往类别不均衡，accuracy 可能虚高。

### 8.2 消融实验

必须做：

```text
1. w/o position branch
2. w/o force branch
3. concat instead of cross fusion
4. w/o force-guided gate
5. w/o temporal encoder
```

可选：

```text
6. T=1, 10, 20, 30 的时间窗口对比
7. CNN vs GRU vs TCN vs Transformer temporal encoder
8. 是否加入 balance auxiliary loss
```

### 8.3 泛化实验

训练集：

```text
固定几种 box 尺寸、质量、摩擦
```

测试集：

```text
未见过的尺寸
未见过的质量
未见过的摩擦
未见过的位置
```

如果时间够，增加：

```text
未见过的形状
```

论文里可以写：

```text
The proposed fusion module improves robustness under object and contact variations.
```

### 8.4 可视化

建议展示：

```text
force_map 热图
position_map 热图
cross-attention map
不同抓取状态下的时序力曲线
混淆矩阵
```

可视化例子：

```text
stable_grasp: 左右手指接触面积稳定，总力缓慢变化
weak_contact: 单侧接触或接触面积小
slip: 接触中心沿某方向持续移动，总力波动
drop: lift 后总力突然下降，object_z 下降
```

---

## 9. 推荐实现顺序

### Step 1: 整理触觉处理模块

目标：

```text
从当前 touch_display.py 中抽出训练用数据处理逻辑
```

输出：

```text
force_map: [2, 30, 15]
position_map: [2, 30, 15]
force_stats: dict
```

检查：

```text
随机 step 仿真，打印 shape 和数值范围
```

### Step 2: 采集单条 episode

目标：

```text
手动或半自动完成一次接近、闭合、抬升、保持
```

输出：

```text
episode_000001.npz
```

检查：

```text
force_map.shape == [N, 2, 30, 15]
label.shape == [N]
```

### Step 3: 自动标签

目标：

```text
根据规则自动生成 no_contact / weak_contact / stable_grasp / slip_or_drop
```

检查：

```text
可视化 3 到 5 个 episode 的标签随时间变化
```

### Step 4: 批量采集数据

第一批：

```text
200 episodes
```

第二批：

```text
1000 episodes
```

划分：

```text
train: 70%
val: 15%
test: 15%
```

注意：

```text
按 episode 划分，不要按窗口随机划分
```

否则同一个 episode 的相邻窗口会同时出现在 train 和 test，导致数据泄漏。

### Step 5: 训练 baseline

先训练最简单版本：

```text
Force-CNN + TCN
```

确认：

```text
loss 能下降
val accuracy 明显高于随机
```

### Step 6: 训练双分支 concat

实现：

```text
Force encoder + Position encoder + concat + TCN
```

目的：

```text
作为你的融合方法 baseline
```

### Step 7: 实现 CPF-FusionNet

加入：

```text
force-guided gate
position-force cross attention
```

比较：

```text
CPF-FusionNet vs concat
```

### Step 8: 消融实验

至少做：

```text
Force only
Position only
Concat
Cross attention only
Gate + Cross attention
```

### Step 9: 写论文

论文结构：

```text
Abstract
1 Introduction
2 Related Work
3 Method
   3.1 Tactile Representation
   3.2 Position-Force Decoupled Branches
   3.3 Contact-Aware Cross Fusion
   3.4 Temporal Grasp State Recognition
4 Dataset Generation
5 Experiments
6 Conclusion
```

---

## 10. 论文创新点整理

### 创新点 1: 位置-力强度解耦触觉表示

将触觉信号从单一力图拆成：

```text
force intensity map M
contact position distribution P
```

区别于普通触觉图分类方法。

### 创新点 2: 接触感知 Gate

利用总力、接触面积、左右力差、熵等物理统计量动态调整位置分支可靠性。

论文表达：

```text
The reliability of contact-position features is conditioned on force statistics.
```

### 创新点 3: 位置-力交叉注意力融合

用 cross-attention 建模：

```text
where the contact happens
how strong the contact is
whether the force-position correspondence is physically consistent
```

比简单 concat 更有解释性。

### 创新点 4: 短时序抓取状态识别

利用连续触觉序列识别滑移和掉落，而不是只做单帧分类。

---

## 11. 风险与应对

### 风险 1: 仿真触觉数据过于理想

应对：

```text
加入噪声
随机摩擦
随机质量
随机接触位置
随机物体尺寸
```

### 风险 2: 类别不平衡

应对：

```text
控制采集策略，主动制造失败和滑移样本
使用 class weight 或 focal loss
报告 macro F1
```

### 风险 3: 标签规则不稳定

应对：

```text
先用 3 类标签
可视化检查标签
人工抽查 20 条 episode
```

### 风险 4: 网络创新不明显

应对：

```text
必须做 concat baseline
必须做 w/o gate 消融
必须可视化 attention map
```

### 风险 5: Transformer 数据量不够

应对：

```text
主模型使用 CNN + Cross Attention + TCN
Transformer 只作为可选对比
```

---

## 12. 最小可行版本

如果时间有限，按下面最小闭环完成：

```text
1. 只做 box 物体
2. 采集 300 episode
3. 做 3 分类: no_contact / unstable / stable
4. 输入 T=20 帧
5. 实现 force branch + position branch
6. 对比 concat 和 CPF-Fusion
7. 做 4 个消融实验
```

最小论文结果表：

```text
Method                 Accuracy    Macro F1    Stable Recall
Force only
Position only
Concat fusion
Cross attention
CPF-FusionNet
```

最小论文图：

```text
1. 系统结构图
2. 触觉数据格式图
3. CPF-FusionNet 结构图
4. 不同状态触觉热图
5. 混淆矩阵
6. 消融实验柱状图
```

---

## 13. 推荐下一步

下一步优先写代码：

```text
module/touch_processor.py
```

这个模块是后续采集数据和训练网络的基础。

它应该把当前 `touch_display.py` 中的显示逻辑和数据处理逻辑拆开：

```text
touch_display.py    只负责显示
touch_processor.py  负责产生训练数据
```

完成后再写：

```text
module/data_collector.py
```

最后再写模型和训练代码。

推荐阶段目标：

```text
第 1 周: 触觉数据处理 + 单 episode 保存
第 2 周: 自动采集 + 自动标签
第 3 周: baseline 网络
第 4 周: CPF-FusionNet + 消融
第 5 周: 可视化 + 论文初稿
```
