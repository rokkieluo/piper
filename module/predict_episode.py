import mujoco
import numpy as np


class PredictionEpisodeBuffer:
    """兼容采集器接口的内存缓存，不向数据集写入触觉帧。"""

    def __init__(self):
        """初始化只用于预测的内存片段缓存。

        输入：无，缓存不依赖数据目录。
        流程：设置上次结果为空，再建立当前帧/标签列表。
        输出：无返回值；不创建任何数据集文件。
        """
        self.last_result = None
        self.last_finish_reason = None
        self.last_episode = None
        self.clear()

    def clear(self):
        """清除尚未完成的回合帧数据。

        输入：当前实例的 force_maps 和 state_labels 列表。
        流程：将记录标志置假，重置两个逐帧列表。
        输出：无返回值；已结束回合的 last_episode 保留。
        """
        self.recording = False
        self.force_maps = []
        self.state_labels = []

    def start(self, episode_index, metadata=None):
        """启动一个新的在线预测回合。

        输入：片段序号和为兼容采集器接口提供的可选 metadata。
        流程：清空当前缓存及上一回合完整序列，更新片段序号并开启记录。
        输出：无返回值；metadata 不写盘也不参与预测。
        """
        self.clear()
        self.last_episode = None
        self.recording = True
        self.episode_index = episode_index

    def record_frame(self, touch_frame, stage_id, state_label, **kwargs):
        """将一帧触觉输入复制到在线预测缓存。

        输入：含 force_map 的触觉帧和状态标签；其余阶段参数保持接口兼容。
        流程：仅在 recording 状态复制 float32 力图并追加整数标签。
        输出：无返回值；缓存与后续仿真帧互不共享数组内存。
        """
        if self.recording:
            self.force_maps.append(np.array(touch_frame["force_map"], dtype=np.float32, copy=True))
            self.state_labels.append(int(state_label))

    def finish(self, result, reason="unknown", metadata=None):
        """封存一个回合的触觉序列及真实结果。

        输入：成功标志 result、结束原因和可选 metadata。
        流程：将逐帧力图堆叠为 [N, 2, 30, 15]，保存状态标签、
        回合编号及真值；空序列保留零帧形状，再清空进行中的缓存。
        输出：始终返回 None；未在记录时不修改 last_episode，也不写文件。
        """
        if not self.recording:
            return None
        self.last_episode = (
            self.episode_index,
            np.stack(self.force_maps) if self.force_maps else np.empty((0, 2, 30, 15), dtype=np.float32),
            np.asarray(self.state_labels, dtype=np.int8),
            bool(result),
            reason,
        )
        self.last_result = "success" if result else "fail"
        self.last_finish_reason = reason
        self.clear()
        return None


class EpisodePredictor:
    """读取训练 checkpoint 并执行固定窗口的触觉二分类。"""

    def __init__(self, checkpoint_path, device="auto"):
        """从训练权重建立可用于在线测试的预测器。

        输入：best.pt 路径及 auto/cpu/cuda 设备选择。
        流程：延迟导入 PyTorch，选择设备并加载权重；验证窗口大小、
        策略和 fail/success 类别顺序，恢复模型并生成有效 taxel 掩码。
        输出：无返回值；实例保存模型及窗口配置，缺依赖或配置错误时抛错。
        """
        try:
            import torch
            from models import ForceCnnTcn, ForceCnnTcnConfig
            from training.data import WINDOW_POLICIES, tactile_validity_mask
        except ImportError as exc:
            raise RuntimeError("predict mode requires PyTorch; install requirements-train.txt") from exc

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.torch = torch
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.window_size = int(checkpoint["window_size"])
        self.window_policy = checkpoint.get("window_policy", "episode_start")
        self.class_names = checkpoint.get("class_names", ["fail", "success"])
        if self.window_size <= 0 or self.window_policy not in WINDOW_POLICIES:
            raise ValueError("invalid checkpoint window settings")
        if list(self.class_names) != ["fail", "success"]:
            raise ValueError("predict mode expects fail/success checkpoint classes")
        self.model = ForceCnnTcn(ForceCnnTcnConfig.from_dict(checkpoint["model_config"]))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()
        self.mask = tactile_validity_mask()

    def predict(self, force_maps):
        """对单个固定长度片段执行模型推理。

        输入：恰好 window_size 帧的双指力图 [T, 2, 30, 15]。
        流程：转 float32、补批次维、屏蔽无效 taxel，在无梯度模式
        下获得 softmax 两类概率并比较成功与失败分数。
        输出：(类别 ID 0/1, 成功概率浮点值)；帧数不符时抛 ValueError。
        """
        if len(force_maps) != self.window_size:
            raise ValueError(f"expected {self.window_size} force frames")
        batch = self.torch.as_tensor(np.asarray(force_maps, dtype=np.float32)).unsqueeze(0)
        batch = (batch * self.mask).to(self.device)
        with self.torch.inference_mode():
            scores = self.model(batch).softmax(dim=1)[0].cpu().tolist()
        return int(scores[1] >= scores[0]), float(scores[1])


def prediction_window(force_maps, state_labels, size, policy, finished=False):
    """选取在线预测或回合结束后的触觉窗口起点。

    输入：力图帧、逐帧状态标签、窗口长度、训练策略及 finished 标志。
    流程：在线只使用已到达且满足策略的完整窗口；末尾策略和无接触
    回合需等结束；结束时首次接触过晚则回退到最后一个完整窗口。
    输出：起始帧整数；窗口不足或尚不能预测时返回 None，策略非法抛错。
    """
    count = len(force_maps)
    if count < size:
        return None
    if policy not in ("first_contact", "episode_start", "episode_end"):
        raise ValueError(f"unknown window policy: {policy}")
    # 在线推理只能使用已到达的帧；episode_end 和无接触样本必须等结果揭晓。
    if not finished:
        if policy == "episode_end":
            return None
        if policy == "episode_start":
            return 0
        contact = np.flatnonzero(np.asarray(state_labels) > 0)
        if not contact.size or count < int(contact[0]) + size:
            return None
        return int(contact[0])
    if policy == "episode_start":
        return 0
    if policy == "episode_end":
        return count - size
    contact = np.flatnonzero(np.asarray(state_labels) > 0)
    return min(int(contact[0]), count - size) if contact.size else count - size


class PredictionSession:
    """跟踪跨回合的在线预测与仿真真值对比。"""

    def __init__(self, predictor):
        """建立跨回合预测统计对象。

        输入：已恢复权重与窗口配置的 predictor。
        流程：记录预测器，清空当前/上一回合索引及结果，并将正确、
        评估、跳过、真实成功等计数置零。
        输出：无返回值；等待 update 接收逐帧缓存。
        """
        self.predictor = predictor
        self.episode_index = None
        self.completed_index = None
        self.current = None
        self.last_result = "none"
        self.last_label = "none"
        self.last_record = None
        self.total = 0
        self.correct = 0
        self.skipped = 0
        self.successes = 0

    def update(self, collector):
        """更新当前回合预测并在结束时与真值对照。

        输入：PredictionEpisodeBuffer 的进行中帧或最近结束的片段。
        流程：新回合清除旧预测；窗口齐备时尝试在线推理；回合结束
        后按训练策略确定最终窗口，必要时重算；不足 T 帧则跳过，
        否则记录类别、成功概率、正确性和累计准确率。
        输出：仅在新回合完成时返回可打印的结果字符串，并更新
        last_record 供结果保存器使用；其他时候返回 None。
        """
        if collector.recording and collector.episode_index != self.episode_index:
            self.episode_index = collector.episode_index
            self.current = None

        if collector.recording and self.current is None:
            start = prediction_window(
                collector.force_maps, collector.state_labels,
                self.predictor.window_size, self.predictor.window_policy,
            )
            if start is not None:
                self.current = (
                    start,
                    self.predictor.predict(collector.force_maps[start:start + self.predictor.window_size]),
                )

        episode = collector.last_episode
        if episode is None or episode[0] == self.completed_index:
            return None
        index, force_maps, state_labels, actual, reason = episode
        self.completed_index = index
        self.successes += int(actual)
        start = prediction_window(
            force_maps, state_labels, self.predictor.window_size,
            self.predictor.window_policy, finished=True,
        )
        if start is None:
            self.skipped += 1
            self.last_label = f"#{index}: skipped (<T)"
            self.last_record = {
                "episode": index, "actual": "success" if actual else "fail",
                "predicted": None, "probability_success": None,
                "correct": None, "reason": reason, "frames": len(force_maps),
                "window_start": None, "window_end": None,
            }
            self.last_result = f"episode {index}: skipped (only {len(force_maps)} frames, T={self.predictor.window_size})"
            return self.last_result

        # 首次接触过晚时，训练窗口会回退至末尾，需要在结束时重算预测。
        if self.current is None or self.current[0] != start:
            self.current = (start, self.predictor.predict(force_maps[start:start + self.predictor.window_size]))
        predicted, success_probability = self.current[1]
        self.total += 1
        self.correct += int(predicted == actual)
        self.last_label = (
            f"#{index}: {'success' if actual else 'fail'} / "
            f"{self.predictor.class_names[predicted]}"
        )
        self.last_record = {
            "episode": index, "actual": "success" if actual else "fail",
            "predicted": self.predictor.class_names[predicted],
            "probability_success": success_probability,
            "correct": bool(predicted == actual), "reason": reason,
            "frames": len(force_maps), "window_start": start,
            "window_end": start + self.predictor.window_size,
        }
        self.last_result = (
            f"episode {index}: actual={'success' if actual else 'fail'} (reason={reason}), "
            f"predicted={self.predictor.class_names[predicted]}, "
            f"P(success)={success_probability:.3f}, window={start}:{start + self.predictor.window_size}, "
            f"accuracy={self.correct}/{self.total} ({self.correct / self.total:.1%})"
        )
        return self.last_result

    def overlay_text(self):
        """生成仿真 viewer 的预测状态提示。

        输入：模型窗口策略、当前在线预测和累计评估计数。
        流程：将缺少窗口、已有成功概率与上次真值对照格式化为文本。
        输出：viewer.set_texts 接收的字体、位置及左右文本四元组。
        """
        prediction = self.current
        label = "waiting for window" if prediction is None else (
            f"{self.predictor.class_names[prediction[1][0]]} "
            f"(P(success)={prediction[1][1]:.2f})"
        )
        return (
            mujoco.mjtFontScale.mjFONTSCALE_100,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "Prediction",
            f"policy: {self.predictor.window_policy}, T={self.predictor.window_size}\n"
            f"current: {label}\n"
            f"accuracy: {self.correct}/{self.total}\n"
            f"skipped (<T): {self.skipped}\n"
            f"actual successes: {self.successes}\n"
            f"last (actual/pred): {self.last_label}",
        )
