import mujoco
import numpy as np


class TouchProcessor:
    """将 MuJoCo 触觉传感器原始读数转换为触觉图与统计量。"""

    def __init__(self, m, d, max_force=3.0, eps=1e-8):
        """初始化触觉传感器索引与力度缩放参数。

        输入：MuJoCo 模型 m、状态 d、最大力度 max_force 和数值稳定项 eps。
        流程：解析模型内传感器名称，定位双指四组触点的数据地址。
        输出：无返回值；保存索引供后续逐帧读取。
        """
        self.m = m
        self.d = d
        self.max_force = max_force
        self.eps = eps
        self.names = [
            name.decode("utf-8")
            for name in m.names.split(b"\x00")
            if name
        ]

        self.link7_touch_A = self.get_touch_data_adr("link7_touch_A_point")
        self.link7_touch_B = self.get_touch_data_adr("link7_touch_B_point")
        self.link8_touch_A = self.get_touch_data_adr("link8_touch_A_point")
        self.link8_touch_B = self.get_touch_data_adr("link8_touch_B_point")
        self.all_touch_adrs = (
            self.link7_touch_A
            + self.link7_touch_B
            + self.link8_touch_A
            + self.link8_touch_B
        )

    def get_touch_data_adr(self, name):
        """定位某一组触觉传感器在 sensordata 中的位置。

        输入：待查找的传感器名称片段 name。
        流程：遍历模型名称，匹配后查询传感器 ID 和数据起始地址。
        输出：按模型名称顺序排列的地址列表。
        """
        touch_data_adr = []

        for sensor_name in self.names:
            if name in sensor_name:
                sensor_id = mujoco.mj_name2id(
                    self.m,
                    mujoco.mjtObj.mjOBJ_SENSOR,
                    sensor_name,
                )
                touch_data_adr.append(self.m.sensor_adr[sensor_id])

        return touch_data_adr

    def get_raw_force(self):
        """提取当前仿真帧中全部传感器的原始三轴力。

        输入：初始化时保存的触点地址及当前 d.sensordata。
        流程：逐触点复制连续的三个分量，不进行裁剪或归一化。
        输出：形状 [触点数, 3] 的 float32 力数组。
        """
        raw_force = np.zeros((len(self.all_touch_adrs), 3), dtype=np.float32)

        for i, adr in enumerate(self.all_touch_adrs):
            raw_force[i] = self.d.sensordata[adr:adr + 3]

        return raw_force

    def get_force_values(self, touch_data_adr):
        """计算一组触点的归一化受力。

        输入：该组触点在 sensordata 中的地址列表。
        流程：逐点求三轴力模长，裁剪到 [0, max_force] 后除以 max_force。
        输出：长度为触点数、范围 [0, 1] 的 float32 数组。
        """
        force_values = np.zeros(len(touch_data_adr), dtype=np.float32)

        for i, adr in enumerate(touch_data_adr):
            force_norm = mujoco.mju_norm3(self.d.sensordata[adr:adr + 3])
            force_values[i] = mujoco.mju_clip(force_norm, 0.0, self.max_force)

        return force_values / self.max_force

    def get_force_map(self):
        """将四组触点拼成空间对齐的双指触觉图。

        输入：当前仿真帧的两指 A/B 指段读数。
        流程：各指段恢复二维网格、补齐较窄指段，按两指的安装方向
        翻转或拼接，使相应位置在力图中对齐。
        输出：float32 数组 [2, 30, 15]，第 0/1 维分别对应两根手指。
        """
        link7_A = self.get_force_values(self.link7_touch_A).reshape((15, 15))
        link7_B = self.pad_to_15x15(
            self.get_force_values(self.link7_touch_B).reshape((15, 12))
        )
        link8_A = self.get_force_values(self.link8_touch_A).reshape((15, 15))
        link8_B = self.pad_to_15x15(
            self.get_force_values(self.link8_touch_B).reshape((15, 12))
        )

        # 两指的安装朝向不同；统一翻转后同位置的 taxel 才能对应。
        finger7 = np.flipud(np.vstack([link7_A, link7_B]))
        finger8 = np.vstack([link8_B, link8_A])

        return np.stack([finger7, finger8], axis=0).astype(np.float32)

    def get_position_map(self, force_map=None):
        """计算双指各自的受力空间分布。

        输入：可选 force_map [2, 30, 15]；省略则读取当前仿真帧。
        流程：每根手指的像素值分别除以该指总力；接近零时保持全零。
        输出：与力图形状相同的 float32 位置分布图。
        """
        if force_map is None:
            force_map = self.get_force_map()

        position_map = np.zeros_like(force_map, dtype=np.float32)

        for finger_id in range(force_map.shape[0]):
            finger_force = force_map[finger_id]
            force_sum = finger_force.sum()
            if force_sum > self.eps:
                position_map[finger_id] = finger_force / force_sum

        return position_map

    def get_force_stats(self, force_map=None):
        """将一帧触觉图汇总为五个力度特征。

        输入：可选双指 force_map；省略则现场读取。
        流程：分别求双指受力和、最大值、超过 0.02 的触点数、
        整体归一化分布的熵及左右受力差。
        输出：float32 数组 [总力, 峰值, 接触面积, 熵, 左右力差]。
        """
        if force_map is None:
            force_map = self.get_force_map()

        left_force = float(force_map[0].sum())
        right_force = float(force_map[1].sum())
        total_force = left_force + right_force
        contact_mask = force_map > 0.02
        contact_area = float(contact_mask.sum())
        max_force = float(force_map.max())
        distribution = force_map.reshape(-1)
        distribution = distribution / (distribution.sum() + self.eps)
        entropy = float(-(distribution * np.log(distribution + self.eps)).sum())

        return np.array(
            [
                total_force,
                max_force,
                contact_area,
                entropy,
                left_force - right_force,
            ],
            dtype=np.float32,
        )

    def get_frame(self):
        """汇总当前仿真帧所需的完整触觉信息。

        输入：实例内的 MuJoCo 模型、状态和已定位的触点地址。
        流程：先生成力图，再以同一力图计算位置分布和统计量，
        同时提取未归一化的三轴传感器读数。
        输出：含 force_map、position_map、raw_force、force_stats 的字典。
        """
        force_map = self.get_force_map()

        return {
            "force_map": force_map,
            "position_map": self.get_position_map(force_map),
            "raw_force": self.get_raw_force(),
            "force_stats": self.get_force_stats(force_map),
        }

    def pad_to_15x15(self, array):
        """补齐窄指段的触觉网格。

        输入：形状 [15, 12] 的二维触觉数组。
        流程：在右侧追加三列零值，不更改真实触点数值。
        输出：形状 [15, 15] 的数组。
        """
        return np.pad(array, ((0, 0), (0, 3)), mode="constant")
