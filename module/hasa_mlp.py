"""Spiking self-attention blocks.

Two attention mechanisms share one block implementation, selected by
``attention_type``:

* ``MLSA`` - Masked Logical Spiking Attention. Pure boolean AND routing, no
  multiplications.
* ``MRHA`` - Mixed Routing/Hamming Attention. Half the heads use the boolean
  route, half use the Hamming route.
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode
from timm.models.layers import DropPath


class MLP_Block(nn.Module):
    """Two-layer spiking feed-forward block with residual connections."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features

        self.fc1_dense = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = LIFNode(
            step_mode="m", tau=2.0, detach_reset=True, backend="cupy", v_reset=None
        )

        self.fc2_dense = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = LIFNode(
            step_mode="m", tau=2.0, detach_reset=True, backend="cupy", v_reset=None
        )

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x, hook=None):
        # x: (T, B, C)
        T, B, C = x.shape
        identity = x

        x = self.fc1_lif(x)
        x = self.fc1_dense(x.flatten(0, 1))
        x = self.fc1_bn(x)
        x = x.reshape(T, B, self.c_hidden).contiguous()

        if self.res:
            x = identity + x
            identity = x

        x = self.fc2_lif(x)
        x = self.fc2_dense(x.flatten(0, 1))
        x = self.fc2_bn(x)
        x = x.reshape(T, B, C).contiguous()

        return x + identity, hook


class MLSA_Block(nn.Module):
    """Spiking self-attention over the time dimension.

    Dispatches to the MLSA or MRHA route; see the module docstring. Q, K and V
    are produced identically in both cases.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        attention_type="MLSA",
        layer=0,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} should be divided by num_heads {num_heads}."
        )

        self.dim = dim
        self.num_heads = num_heads
        self.attention_type = attention_type.upper()

        self.q_dense = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(
            step_mode="m", tau=2.0, detach_reset=True, backend="cupy", v_reset=None
        )

        self.k_dense = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNode(
            step_mode="m", tau=2.0, detach_reset=True, backend="cupy", v_reset=None
        )

        self.v_dense = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = LIFNode(
            step_mode="m", tau=2.0, detach_reset=True, backend="cupy", v_reset=None
        )

        self.proj_dense = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)

        self.shortcut_lif = LIFNode(
            step_mode="m", tau=2.0, detach_reset=True, backend="cupy", v_reset=None
        )
        self.layer = layer

    def _spiking_projection(self, x_flat, dense, bn, lif, T, B, C):
        """Dense -> BN -> LIF -> split into heads. Returns (T, B, H, C // H)."""
        y = bn(dense(x_flat))
        y = lif(y.reshape(T, B, C).contiguous())
        return y.view(T, B, self.num_heads, C // self.num_heads)

    def forward(self, x, hook=None):
        # x: (T, B, C)
        T, B, C = x.shape
        identity = x

        x_for_qkv = self.shortcut_lif(x).flatten(0, 1)
        q = self._spiking_projection(
            x_for_qkv, self.q_dense, self.q_bn, self.q_lif, T, B, C
        )
        k = self._spiking_projection(
            x_for_qkv, self.k_dense, self.k_bn, self.k_lif, T, B, C
        )
        v = self._spiking_projection(
            x_for_qkv, self.v_dense, self.v_bn, self.v_lif, T, B, C
        )

        if self.attention_type == "MLSA":
            x = self._mlsa_attention(q, k, v, T, B, C)
        elif self.attention_type == "MRHA":
            x = self._mrha_attention(q, k, v, T, B, C)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

        x = self.proj_bn(self.proj_dense(x.flatten(0, 1)))
        x = x.reshape(T, B, C).contiguous()

        return x + identity, v, hook

    def _mlsa_attention(self, q, k, v, T, B, C):
        """Boolean AND routing; multiplication-free."""
        kv = (k.bool() & v.bool()).sum(dim=-1, keepdim=True).bool()
        x = (q.bool() & kv).to(torch.float32)
        return x.reshape(T, B, C).contiguous()

    def _mrha_attention(self, q, k, v, T, B, C):
        """Boolean route on the first half of the heads, Hamming on the second."""
        q_d, q_h = torch.chunk(q, 2, dim=2)
        k_d, k_h = torch.chunk(k, 2, dim=2)
        v_d, v_h = torch.chunk(v, 2, dim=2)

        # Boolean route
        kv_d = (k_d.bool() & v_d.bool()).float().sum(dim=-1, keepdim=True).bool()
        x_d = (q_d.bool() & kv_d).float()

        # Hamming route: (T, B, H, D) -> (B, H, T, D) for the matmuls
        q_h = q_h.permute(1, 2, 0, 3)
        k_h = k_h.permute(1, 2, 0, 3)
        v_h = v_h.permute(1, 2, 0, 3)
        x_h = q_h @ ((2 * k_h - 1).transpose(-2, -1) @ v_h)
        x_h = x_h.permute(2, 0, 1, 3).contiguous()

        x = torch.cat([x_d, x_h], dim=2)
        return x.reshape(T, B, C).contiguous()


class Enc_Block(nn.Module):
    """One encoder block: spiking attention followed by a spiking MLP."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        attention_type="MLSA",
        layer=0,
    ):
        super().__init__()
        self.attn = MLSA_Block(
            dim,
            num_heads=num_heads,
            attention_type=attention_type,
            layer=layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = MLP_Block(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            layer=layer,
        )

    def forward(self, x, hook=None):
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x_attn, hook=hook)
        return x, attn, hook
