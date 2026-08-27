import cv2
import mujoco
import numpy as np


class TouchDisplay:
    def __init__(self, m, d, max_force=3.0):
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
        touch_data = np.zeros(len(touch_data_adr))

        for i, adr in enumerate(touch_data_adr):
            data = mujoco.mju_norm3(self.d.sensordata[adr:adr + 3])
            touch_data[i] = mujoco.mju_clip(data, 0.0, self.max_force) / self.max_force

        return touch_data

    def show(self):
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
        data = (self.get_touch_data(adr) * 255).reshape((rows, cols)).astype(np.uint8)
        return cv2.resize(data, (width, height))

    def close(self):
        cv2.destroyAllWindows()
