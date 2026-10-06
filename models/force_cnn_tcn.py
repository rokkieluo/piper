from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ForceCnnTcnConfig:
    """定义逐帧 CNN 与因果 TCN 的通道数和分类参数。"""
    input_channels: int = 2
    spatial_dim: int = 64
    tcn_channels: tuple[int, ...] = (64, 64, 64)
    kernel_size: int = 3
    dropout: float = 0.1
    num_classes: int = 2

    def to_dict(self):
        """将模型配置转换为 checkpoint 可保存的映射。

        输入：当前配置实例的通道数、核大小和 dropout 等字段。
        流程：展开 dataclass，并把 tcn_channels 元组转为列表。
        输出：可用于模型配置持久化的字典。
        """
        config = asdict(self)
        config["tcn_channels"] = list(self.tcn_channels)
        return config

    @classmethod
    def from_dict(cls, values):
        """从保存的配置字典恢复模型参数。

        输入：checkpoint 中包含 tcn_channels 列表的 values。
        流程：先复制输入，再将通道列表转回不可变元组。
        输出：ForceCnnTcnConfig 实例，不修改原字典。
        """
        values = dict(values)
        values["tcn_channels"] = tuple(values["tcn_channels"])
        return cls(**values)


class SpatialEncoder(nn.Module):
    """从单帧双指触觉图中提取固定长度的空间特征。"""

    def __init__(self, input_channels, output_dim):
        """创建单帧触觉空间编码器。

        输入：双指图的输入通道数和目标特征维度。
        流程：堆叠卷积、归一化、激活与池化，再映射到指定维数。
        输出：无返回值；模块参数保存在实例中等待 forward 调用。
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, output_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, output_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 2)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_dim * 4 * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, frames):
        """将合并批次与时间维的触觉图编码为特征。

        输入：frames，形状 [B*T, C, H, W]。
        流程：卷积提取空间信息，池化并线性投影。
        输出：形状 [B*T, output_dim] 的特征张量。
        """
        return self.projection(self.features(frames))


class CausalConv1d(nn.Conv1d):
    """只使用当前及过去时刻的膨胀一维卷积。"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        """初始化只看过去的时间卷积层。

        输入：输入/输出通道、卷积核长度与膨胀系数。
        流程：计算 (kernel_size-1)*dilation 的填充长度并配置 Conv1d。
        输出：无返回值；填充长度供 forward 裁剪未来方向的数据。
        """
        self.left_padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.left_padding,
            dilation=dilation,
        )

    def forward(self, sequence):
        """按因果约束处理一维时间序列。

        输入：sequence，形状 [B, 输入通道, T]。
        流程：执行带填充的卷积，并从输出末尾裁掉填充长度。
        输出：形状 [B, 输出通道, T] 的特征，每步仅依赖当前及历史帧。
        """
        output = super().forward(sequence)
        if self.left_padding:
            output = output[..., :-self.left_padding]
        return output


class ChannelLayerNorm(nn.Module):
    """逐时间步对通道维做归一化。"""

    def __init__(self, channels):
        """为指定特征通道建立逐时间步归一化层。

        输入：每帧的特征通道数 channels。
        流程：创建匹配通道维的 LayerNorm 并保存在实例中。
        输出：无返回值；归一化操作在 forward 时执行。
        """
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, sequence):
        """对每个时间步的特征通道做 LayerNorm。

        输入：形状 [B, C, T] 的序列。
        流程：转置为 [B, T, C]、归一化最后一维，再转回原布局。
        输出：与输入形状相同的归一化序列。
        """
        return self.norm(sequence.transpose(1, 2)).transpose(1, 2)


class TemporalResidualBlock(nn.Module):
    """包含两层因果卷积与跳跃连接的时间残差块。"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        """创建包含两层因果卷积的时间残差块。

        输入：输入/输出通道数、核长度、膨胀系数与 dropout 概率。
        流程：建立卷积/归一化/激活/随机失活支路，并在通道变化时
        使用 1×1 卷积投影残差，否则保留恒等映射。
        输出：无返回值；构造出的层供 forward 组合使用。
        """
        super().__init__()
        self.layers = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            ChannelLayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            ChannelLayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.GELU()

    def forward(self, sequence):
        """计算时间卷积和残差支路的融合特征。

        输入：形状 [B, 输入通道, T] 的序列。
        流程：并行计算两层因果卷积和残差投影，相加后通过 GELU。
        输出：形状 [B, 输出通道, T] 的时序特征。
        """
        return self.activation(self.layers(sequence) + self.residual(sequence))


class ForceCnnTcn(nn.Module):
    """使用逐帧触觉 CNN 和因果 TCN 预测抓取最终结果。"""

    def __init__(self, config=None):
        """按配置构建完整的触觉分类网络。

        输入：可选 ForceCnnTcnConfig；省略时使用默认配置。
        流程：创建逐帧 CNN；按通道列表堆叠膨胀率 1、2、4… 的 TCN；
        最后配置线性分类层。
        输出：无返回值；网络模块及配置保存在实例中。
        """
        super().__init__()
        self.config = config or ForceCnnTcnConfig()
        self.spatial_encoder = SpatialEncoder(
            self.config.input_channels,
            self.config.spatial_dim,
        )

        blocks = []
        in_channels = self.config.spatial_dim
        for block_id, out_channels in enumerate(self.config.tcn_channels):
            blocks.append(
                TemporalResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.config.kernel_size,
                    dilation=2 ** block_id,
                    dropout=self.config.dropout,
                )
            )
            in_channels = out_channels
        self.temporal_encoder = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, self.config.num_classes),
        )

    def forward(self, force_sequence):
        """将触觉时序映射到抓取成功与失败的类别分数。

        输入：force_sequence，[B, T, C, H, W]，C 须等于配置通道数。
        流程：校验维度；逐帧共用 CNN 提取空间特征，再沿时间经过
        因果 TCN，取最后一个时间步的特征送入分类层。
        输出：[B, num_classes] 的未归一化 logits；形状错误时抛 ValueError。
        """
        if force_sequence.ndim != 5:
            raise ValueError("force_sequence must have shape [B, T, 2, 30, 15]")

        batch_size, time_steps, channels, height, width = force_sequence.shape
        if channels != self.config.input_channels:
            raise ValueError(
                f"expected {self.config.input_channels} input channels, got {channels}"
            )

        # CNN 共用同一组权重处理每帧，再把时间维恢复供 TCN 建模。
        frames = force_sequence.reshape(
            batch_size * time_steps,
            channels,
            height,
            width,
        )
        spatial = self.spatial_encoder(frames)
        spatial = spatial.reshape(batch_size, time_steps, -1).transpose(1, 2)
        temporal = self.temporal_encoder(spatial)
        return self.classifier(temporal[:, :, -1])
