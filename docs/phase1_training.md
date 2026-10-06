# 第一阶段：抓取结果二分类基线

## 任务定义

每个 episode 只产生一个训练样本：

```text
输入：首次接触后前 20 帧 force_map，形状 [20, 2, 30, 15]
输出：episode 最终结果，0=fail，1=success
```

新版采集器从 reset 开始录制。默认的 `first_contact` 窗口策略通过逐帧
`state_label` 找到首次有效接触，并从该帧开始截取固定长度窗口；完全没有接触的
失败 episode 使用末尾窗口。没有 `state_label` 的旧数据仍从第 0 帧读取，因为旧版
采集器本身就是从首次接触开始录制。这个基线不按滑动窗口重复赋予最终标签，避免
模型从动作阶段和序列长度中推断结果。

数据先按类别打乱，再以 episode 为单位分为 train/val/test。训练运行目录中的
`split.json` 固定记录划分，评估时复用相同清单。

## 模型

```text
force_map -> framewise CNN -> causal TCN -> last timestep -> 2-class logits
```

CNN 在每帧上提取双指空间特征。TCN 包含膨胀率为 1、2、4 的三个因果残差块，
不会读取预测时刻之后的触觉数据。该模型是验证数据和训练闭环的基线，不是最终的
CPF-FusionNet。数据加载器对补零产生的无效 taxel 应用固定掩码，数据增强不会在
这些位置制造虚假接触。

## 环境与训练

当前机器可直接使用已有 CUDA 环境：

```bash
conda activate cuda
python train.py --data-dir dataset/auto/run_20260915_154736 \
  --window-policy first_contact
```

快速检查可以减少训练轮数：

```bash
python train.py --data-dir dataset/auto --epochs 5 --patience 3
```

可用窗口策略为 `first_contact`、`episode_start` 和 `episode_end`。训练配置与
checkpoint 都会保存所选策略，独立评估会自动复用。默认配置为 batch size 16、
AdamW、学习率 `3e-4`、类别加权交叉熵、梯度裁剪、
余弦学习率和基于验证集 Macro-F1 的早停。训练结束后会自动绘图；不需要绘图时
使用 `--no-plot`。输出保存在 `runs/phase1/run_*`：

```text
best.pt       验证集 Macro-F1 最优权重
config.json   本次训练参数
split.json    固定的数据划分
history.json  每轮训练与验证指标
result.json   最优权重在测试集上的最终结果
plots/        训练曲线、混淆矩阵和分类别指标
```

## 独立评估

```bash
python evaluate.py runs/phase1/run_YYYYMMDD_HHMMSS/best.pt \
  --data-dir dataset/auto --split test
```

## 在仿真中测试模型

在安装了 PyTorch 和项目训练依赖的环境中，指定要测试的训练结果：

```bash
python main.py --mode predict \
  --checkpoint runs/phase1/run_YYYYMMDD_HHMMSS/best.pt \
  --episodes 10 --seed 42 --viewer
```

`predict` 与 `auto` 使用同一个自动抓取控制器及参数，夹爪目标、轨迹和仿真
成功/失败判定保持一致；模型仅观察触觉帧，不参与运动控制，也不会保存新的数据集。
可使用相同的 `--seed` 和抓取参数对照 `auto` 的仿真轨迹。默认自动选择 CUDA/CPU，
可用 `--device cpu` 强制使用 CPU。模型输入长度和截取策略读取 checkpoint，
无需另外指定 `T`。`first_contact` 在首次接触后凑齐 T 帧时显示预测；
若 episode 过早结束，则用训练时的回退窗口在结束时预测；无接触 episode
在结束时使用最后 T 帧。`episode_end` 仅在结束后预测。

viewer 右上角显示当前预测和累计正确数，终端显示逐个 episode 的真实结果、
预测类别、成功概率与窗口位置。少于 T 帧的 episode 会跳过，不计入准确率。
成功概率是模型输出的 softmax 值，未经过概率校准；这些结果衡量的是仿真
中的分类能力，并不能直接代表实机效果。

每次运行会自动新建 `runs/predict/run_*` 目录；可以用 `--predict-output-dir`
修改父目录。`results.json` 包含 checkpoint、窗口策略、逐次预测以及准确率和
混淆矩阵；`episodes.csv` 便于用表格查看。两者在每次 episode 结束后更新，
即使提前退出也保留已经完成的结果。结束时生成 `prediction_results.png`，
展示各次成功概率、仿真真实标签和混淆矩阵。绘图需要 Matplotlib；若未安装，
JSON 和 CSV 仍正常保存，终端会显示安装提示。已有旧运行未记录逐次预测，
需要重新运行上述命令才能生成这些结果。

## 结果绘图

绘制单次运行的训练曲线、混淆矩阵和分类别指标：

```bash
python scripts/plot_results.py runs/phase1/run_YYYYMMDD_HHMMSS
```

比较多个随机种子，并把图保存到指定目录：

```bash
python scripts/plot_results.py RUN_SEED_42 RUN_SEED_7 RUN_SEED_2026 \
  --output-dir runs/phase1/plots_summary
```

默认只保存 PNG 文件。在有图形桌面的环境中加入 `--show` 可以同时打开窗口。

## 验证

```bash
python -m unittest discover -s tests -v
```

现有 100 个 episode 只用于打通流程。测试指标不能作为最终论文结果；正式实验需要
扩充数据、逐帧标签，并按物体身份做泛化划分。
