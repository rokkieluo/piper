import cv2
import mujoco
import numpy as np


class TouchDisplay:
    """将双指触觉力渲染为 OpenCV 窗口并读取退出按键。"""

    def __init__(self, m, d, max_force=3.0):
        """初始化触觉窗口使用的传感器与显示比例。

        输入：MuJoCo 模型 m、状态 d 和力度显示上限 max_force。
        流程：读取模型名称并定位双指的四个触点组。
        输出：无返回值；保存地址以供后续实时显示。
        """
        self.m = m
        self.d = d
        self.max_force = max_force

        self.names = [
            name.decode("utf-8")
            for name in m.names.split(b"\x00")
            if name
        ]

        self.link7_touch_A = self.get_touch_data_adr("link7_touch_A_point")
        self.link7_touch_B = self.get_touch_data_adr("link7_touch_B_point")
        self.link8_touch_A = self.get_touch_data_adr("link8_touch_A_point")
        self.link8_touch_B = self.get_touch_data_adr("link8_touch_B_point")

    def get_touch_data_adr(self, name):
        """查找匹配一组指段的触觉数据地址。

        输入：传感器名称片段 name。
        流程：从模型名称中匹配传感器，再读取各传感器数据起点。
        输出：与模型顺序一致的地址列表。
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

    def get_touch_data(self, touch_data_adr):
        """读取指定触点组的归一化力值。

        输入：各触点在 MuJoCo sensordata 中的起始地址。
        流程：逐点求三维力模长，并以 max_force 裁剪与归一化。
        输出：对应触点数的一维 [0, 1] 浮点数组。
        """
        touch_data = np.zeros(len(touch_data_adr))

        for i, adr in enumerate(touch_data_adr):
            data = mujoco.mju_norm3(self.d.sensordata[adr:adr + 3])
            touch_data[i] = mujoco.mju_clip(data, 0.0, self.max_force) / self.max_force

        return touch_data

    def show(self):
        """显示当前帧的两根手指触觉图并接收键盘事件。

        输入：当前仿真帧的四组触点数据。
        流程：生成各指段灰度图、按真实朝向拼接/翻转，在两个窗口显示。
        输出：非阻塞读取到的按键码低八位，供主循环判断退出指令。
        """
        link7_touch_A = self.make_image(self.link7_touch_A, 15, 15, 150, 150)
        link7_touch_B = self.make_image(self.link7_touch_B, 15, 12, 150, 120)
        link8_touch_A = self.make_image(self.link8_touch_A, 15, 15, 150, 150)
        link8_touch_B = self.make_image(self.link8_touch_B, 15, 12, 150, 120)

        link7_touch = cv2.vconcat([link7_touch_A, link7_touch_B])
        link7_touch = cv2.flip(link7_touch, 0)

        link8_touch = cv2.vconcat([link8_touch_B, link8_touch_A])

        cv2.imshow("link7_touch", link7_touch)
        cv2.imshow("link8_touch", link8_touch)
        return cv2.waitKey(1) & 0xFF

    def make_image(self, adr, rows, cols, width, height):
        """把一组传感器数据转换成可显示的指段图像。

        输入：传感器地址、原始网格行列数和目标图像宽高。
        流程：归一化力值乘 255，恢复二维矩阵后缩放。
        输出：uint8 灰度图，尺寸为 [height, width]。
        """
        data = (self.get_touch_data(adr) * 255).reshape((rows, cols)).astype(np.uint8)
        return cv2.resize(data, (width, height))

    def close(self):
        """释放触觉可视化窗口。

        输入：无，操作当前进程已创建的 OpenCV 窗口。
        流程：调用 OpenCV 的窗口销毁接口。
        输出：无返回值；窗口资源被释放。
        """
        cv2.destroyAllWindows()
