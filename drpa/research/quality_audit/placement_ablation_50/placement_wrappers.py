"""Project-side wrappers for the frozen 50% placement ablation.

The official VoxTell source is not modified. Both wrappers preserve the
canonical forward path and enable only the placement under test plus the
shared rank-4 cross-attention LoRA.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from voxtell_peft_wrapper import VoxTellPEFTWrapper, _unique_named_parameters
from drpa8_wrapper import LowRankWeightUpdate


class ProjectionPlacementWrapper(VoxTellPEFTWrapper):
    """B1 plus rank-8 residual adapters on project_to_decoder_channels."""

    def __init__(self, model_dir: str, embedding_bank: str,
                 device: torch.device | None = None) -> None:
        super().__init__(model_dir, embedding_bank, rank=4, alpha=8.0,
                         dropout=0.05, device=device)
        self.projection_rank = 8
        self.projection_targets = []
        for pidx, projection in enumerate(self.model.project_to_decoder_channels):
            linear_count = 0
            for child_name, child in projection.named_modules():
                if not isinstance(child, nn.Linear):
                    continue
                adapter = LowRankWeightUpdate(tuple(child.weight.shape), 8, 8.0).to(child.weight.device)
                register_parametrization(child, "weight", adapter)
                self.projection_targets.append(f"project_to_decoder_channels.{pidx}.{child_name}.weight")
                linear_count += 1
            if linear_count != 2:
                raise RuntimeError(f"expected 2 projection linears at {pidx}, got {linear_count}")
        for param in self.model.parameters():
            param.requires_grad = False
        for layer in self.model.transformer_decoder.layers:
            for target in (layer.multihead_attn.parametrizations.in_proj_weight[0],
                           layer.multihead_attn.out_proj.parametrizations.weight[0]):
                for param in target.parameters():
                    param.requires_grad = True
        for projection in self.model.project_to_decoder_channels:
            for module in projection.modules():
                if isinstance(module, LowRankWeightUpdate):
                    for param in module.parameters():
                        param.requires_grad = True
        if device is not None:
            self.to(device)

    def set_training_mode(self) -> None:
        self.model.eval()
        for projection in self.model.project_to_decoder_channels:
            projection.train()
        for layer in self.model.transformer_decoder.layers:
            layer.multihead_attn.parametrizations.in_proj_weight[0].train()
            layer.multihead_attn.out_proj.parametrizations.weight[0].train()

    def trainable_parameter_groups(self) -> Iterable[Tuple[str, nn.Parameter, str]]:
        for name, param in _unique_named_parameters(self.model):
            if not param.requires_grad:
                continue
            if "project_to_decoder_channels" in name and ".parametrizations." in name:
                group = "projection_adapter"
            elif "multihead_attn" in name and ".parametrizations." in name:
                group = "cross_attention_lora"
            else:
                raise RuntimeError(f"unexpected P1 parameter: {name}")
            yield name, param, group


class DecoderPlacementWrapper(VoxTellPEFTWrapper):
    """B1 plus decoder.stages trainable; projection weights remain frozen."""

    def __init__(self, model_dir: str, embedding_bank: str,
                 device: torch.device | None = None) -> None:
        super().__init__(model_dir, embedding_bank, rank=4, alpha=8.0,
                         dropout=0.05, device=device)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.decoder.stages.parameters():
            param.requires_grad = True
        for layer in self.model.transformer_decoder.layers:
            for target in (layer.multihead_attn.parametrizations.in_proj_weight[0],
                           layer.multihead_attn.out_proj.parametrizations.weight[0]):
                for param in target.parameters():
                    param.requires_grad = True
        if device is not None:
            self.to(device)

    def set_training_mode(self) -> None:
        self.model.eval()
        self.model.decoder.stages.train()
        for layer in self.model.transformer_decoder.layers:
            layer.multihead_attn.parametrizations.in_proj_weight[0].train()
            layer.multihead_attn.out_proj.parametrizations.weight[0].train()

    def trainable_parameter_groups(self) -> Iterable[Tuple[str, nn.Parameter, str]]:
        for name, param in _unique_named_parameters(self.model):
            if not param.requires_grad:
                continue
            if name.startswith("decoder.stages."):
                group = "decoder_stages"
            elif "multihead_attn" in name and ".parametrizations." in name:
                group = "cross_attention_lora"
            else:
                raise RuntimeError(f"unexpected P2 parameter: {name}")
            yield name, param, group


def build_placement_wrapper(kind: str, model_dir: str, embedding_bank: str,
                            device: torch.device) -> nn.Module:
    if kind == "b1_projection":
        w = ProjectionPlacementWrapper(model_dir, embedding_bank, device=device)
    elif kind == "b1_decoder":
        w = DecoderPlacementWrapper(model_dir, embedding_bank, device=device)
    else:
        raise ValueError(kind)
    w.set_training_mode()
    return w
