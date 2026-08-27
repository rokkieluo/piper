import mujoco
import numpy as np


class TouchProcessor:
    def __init__(self, m, d, max_force=3.0, eps=1e-8):
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
        raw_force = np.zeros((len(self.all_touch_adrs), 3), dtype=np.float32)

        for i, adr in enumerate(self.all_touch_adrs):
            raw_force[i] = self.d.sensordata[adr:adr + 3]

        return raw_force

    def get_force_values(self, touch_data_adr):
        force_values = np.zeros(len(touch_data_adr), dtype=np.float32)

        for i, adr in enumerate(touch_data_adr):
            force_norm = mujoco.mju_norm3(self.d.sensordata[adr:adr + 3])
            force_values[i] = mujoco.mju_clip(force_norm, 0.0, self.max_force)

        return force_values / self.max_force

    def get_force_map(self):
        link7_A = self.get_force_values(self.link7_touch_A).reshape((15, 15))
        link7_B = self.pad_to_15x15(
            self.get_force_values(self.link7_touch_B).reshape((15, 12))
        )
        link8_A = self.get_force_values(self.link8_touch_A).reshape((15, 15))
        link8_B = self.pad_to_15x15(
            self.get_force_values(self.link8_touch_B).reshape((15, 12))
        )

        finger7 = np.flipud(np.vstack([link7_A, link7_B]))
        finger8 = np.vstack([link8_B, link8_A])

        return np.stack([finger7, finger8], axis=0).astype(np.float32)

    def get_position_map(self, force_map=None):
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
        force_map = self.get_force_map()

        return {
            "force_map": force_map,
            "position_map": self.get_position_map(force_map),
            "raw_force": self.get_raw_force(),
            "force_stats": self.get_force_stats(force_map),
        }

    def pad_to_15x15(self, array):
        return np.pad(array, ((0, 0), (0, 3)), mode="constant")
