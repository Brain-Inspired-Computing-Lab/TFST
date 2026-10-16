"""Time-fidelity embedding module."""

import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode


class TF_Module(nn.Module):
    """Projects a multivariate signal into the spiking embedding space.

    The projected spikes are broadcast against the raw input across heads, so
    ``embed_dims`` must equal ``num_heads * in_channels``.
    """

    def __init__(
        self,
        num_heads=8,
        in_channels=2,
        embed_dims=256,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.C = in_channels

        self.proj_dense = nn.Linear(in_channels, embed_dims)
        self.proj_bn = nn.BatchNorm1d(embed_dims)
        self.proj_lif = LIFNode(
            step_mode="m", tau=2.0, detach_reset=True, backend="cupy", v_reset=None
        )

    def forward(self, x, hook=None):
        # x: (T, B, in_channels)
        T, B, C = x.shape
        x_feat = x

        x = self.proj_dense(x.flatten(0, 1))
        x = self.proj_bn(x)
        x = x.reshape(T, B, -1).contiguous()
        x = self.proj_lif(x)

        # Fuse the spiking embedding with the raw signal, per head.
        x = x.view(T, B, self.num_heads, C) + x_feat.unsqueeze(2)
        x = x.reshape(T, B, -1).contiguous()

        return x, C, hook
