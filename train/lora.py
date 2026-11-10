"""Adapted from compute_network/yolo/train/yolo_world_lora.py; see README_CN.md."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod, sqrt
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_patterns: tuple[str, ...] = ("attn.gl", "attn.proj_conv.conv", "model.22.cv3")
    train_bias: str = "none"
    freeze_clip: bool = True
    freeze_batchnorm: bool = True

    def __post_init__(self):
        if self.rank <= 0 or self.alpha <= 0 or not 0 <= self.dropout < 1:
            raise ValueError('Invalid LoRA rank/alpha/dropout')
        if not self.target_patterns or any(not p.strip() for p in self.target_patterns):
            raise ValueError('Nonempty LoRA target patterns required')
        if self.train_bias != 'none' or not self.freeze_clip or not self.freeze_batchnorm:
            raise ValueError('This workflow requires frozen bias, CLIP and BatchNorm')

    @classmethod
    def from_targets(
        cls,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        targets: Iterable[str],
        train_bias: str,
        freeze_clip: bool,
    ) -> LoRAConfig:
        target_patterns = tuple(pattern.strip() for pattern in targets if pattern.strip())
        if not target_patterns:
            raise ValueError("At least one LoRA target pattern is required.")
        if rank <= 0:
            raise ValueError("LoRA rank must be a positive integer.")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0.0, 1.0).")
        if train_bias not in {"none", "all"}:
            raise ValueError("LoRA train_bias must be 'none' or 'all'.")
        return cls(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_patterns=target_patterns,
            train_bias=train_bias,
            freeze_clip=freeze_clip,
        )


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = min(rank, base.in_features, base.out_features)
        self.scale = alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        self.reset_parameters()
        self._freeze_base()

    def _freeze_base(self) -> None:
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scale

    def merged_module(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        merged.load_state_dict(self.base.state_dict())
        delta = (self.lora_B @ self.lora_A).to(dtype=merged.weight.dtype)
        merged.weight.data.add_(delta * self.scale)
        return merged


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.fan_in = (base.in_channels // base.groups) * prod(base.kernel_size)
        self.rank = min(rank, self.fan_in, base.out_channels)
        self.scale = alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.fan_in))
        self.lora_B = nn.Parameter(torch.zeros(base.out_channels, self.rank))
        self.reset_parameters()
        self._freeze_base()

    def _freeze_base(self) -> None:
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=sqrt(5))
        nn.init.zeros_(self.lora_B)

    def _delta_weight(self) -> torch.Tensor:
        delta = self.lora_B @ self.lora_A
        return delta.view(
            self.base.out_channels,
            self.base.in_channels // self.base.groups,
            *self.base.kernel_size,
        ).to(dtype=self.base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = self._delta_weight() * self.scale
        lora_out = F.conv2d(
            self.dropout(x),
            delta,
            bias=None,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=self.base.groups,
        )
        return base_out + lora_out

    def merged_module(self) -> nn.Conv2d:
        merged = nn.Conv2d(
            self.base.in_channels,
            self.base.out_channels,
            kernel_size=self.base.kernel_size,
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=self.base.groups,
            bias=self.base.bias is not None,
            padding_mode=self.base.padding_mode,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        merged.load_state_dict(self.base.state_dict())
        merged.weight.data.add_(self._delta_weight() * self.scale)
        return merged


def _matches_target(name: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in name for pattern in patterns)


def _replace_child_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    if isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = new_module
    else:
        setattr(parent, child_name, new_module)


def apply_lora_adapters(model: nn.Module, config: LoRAConfig) -> list[str]:
    matched: list[str] = []
    for full_name, module in list(model.named_modules()):
        if not _matches_target(full_name, config.target_patterns):
            continue
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            continue
        if isinstance(module, nn.Linear):
            parent_name, _, child_name = full_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            _replace_child_module(parent, child_name, LoRALinear(module, config.rank, config.alpha, config.dropout))
            matched.append(full_name)
        elif isinstance(module, nn.Conv2d):
            parent_name, _, child_name = full_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            _replace_child_module(parent, child_name, LoRAConv2d(module, config.rank, config.alpha, config.dropout))
            matched.append(full_name)

    if not matched:
        raise RuntimeError(
            "No modules matched the requested LoRA target patterns: "
            + ", ".join(config.target_patterns)
        )
    return matched


def configure_lora_trainable(model: nn.Module, train_bias: str) -> tuple[int, int]:
    total_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        total_params += parameter.numel()
        is_lora_param = name.endswith("lora_A") or name.endswith("lora_B")
        is_bias_param = train_bias == "all" and name.endswith(".bias")
        parameter.requires_grad = is_lora_param or is_bias_param
        if parameter.requires_grad:
            trainable_params += parameter.numel()
    return trainable_params, total_params


def merge_and_strip_lora(model: nn.Module) -> nn.Module:
    for child_name, child in list(model.named_children()):
        if isinstance(child, LoRALinear):
            _replace_child_module(model, child_name, child.merged_module())
        elif isinstance(child, LoRAConv2d):
            _replace_child_module(model, child_name, child.merged_module())
        else:
            merge_and_strip_lora(child)
    return model


