"""TFST: a Time-Fidelity Spiking Transformer for biosignal classification.

The network embeds a multivariate signal with :class:`TF_Module`, applies a
stack of spiking encoder blocks, and classifies the time-averaged output.
"""

import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from module import Enc_Block, TF_Module


class TFST(nn.Module):
    """Time-Fidelity Spiking Transformer.

    Args:
        in_channels: Number of input signal channels.
        num_classes: Number of output classes.
        T: Number of spiking time steps.
        depths: Number of encoder blocks.
        num_heads: Attention heads per block.
        embed_dims: Embedding width.
        mlp_ratios: Hidden expansion ratio of the MLP block.
        attention_type: Either ``"MLSA"`` or ``"MRHA"``.
    """

    def __init__(
        self,
        pretrained=False,
        pretrained_cfg=None,
        in_channels=22,
        num_classes=4,
        T=1001,
        depths=1,
        num_heads=8,
        embed_dims=176,
        mlp_ratios=4,
        attention_type="MLSA",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        self.attention_type = attention_type

        self.patch_embed = TF_Module(
            num_heads=num_heads,
            in_channels=in_channels,
            embed_dims=embed_dims,
        )
        self.block = nn.ModuleList(
            [
                Enc_Block(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    attention_type=attention_type,
                    layer=j,
                )
                for j in range(self.depths)
            ]
        )

        # Classification head
        self.head_lif = LIFNode(
            step_mode="m", tau=2.0, detach_reset=True, backend="cupy", v_reset=None
        )
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x, hook=None):
        x, _, hook = self.patch_embed(x, hook=hook)
        for blk in self.block:
            x, _, hook = blk(x, hook=hook)
        return x, hook

    def forward(self, x, hook=None):
        # x: (T, B, in_channels)
        x, hook = self.forward_features(x, hook=hook)
        T, B, _ = x.shape

        x = self.head_lif(x)
        x = self.head(x.flatten(0, 1))
        x = x.reshape(T, B, -1)

        return x.mean(0)  # mean over time -> (B, num_classes)


@register_model
def TFS(**kwargs):
    model = TFST(**kwargs)
    model.default_cfg = _cfg()
    return model
