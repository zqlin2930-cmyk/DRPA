"""B3-Full wrapper for the full-data paired comparison.

This is a project-side adapter. It does not modify official VoxTell sources.
It keeps the proven B3 graph-boundary implementation while enabling only
decoder.stages and the original projection weights, plus cross-attention LoRA.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

from voxtell_decoder_capacity_wrapper import VoxTellDecoderCapacityWrapper
from voxtell_peft_wrapper import _unique_named_parameters


class B3FullDataWrapper(VoxTellDecoderCapacityWrapper):
    """Full projection + decoder-stage FT with cross-attention LoRA."""

    decoder_owned_children = ("stages",)

    def set_training_mode(self) -> None:
        self.model.eval()
        self.model.decoder.stages.train()
        for projection in self.model.project_to_decoder_channels:
            projection.train()
        for layer in self.model.transformer_decoder.layers:
            layer.multihead_attn.parametrizations.in_proj_weight[0].train()
            layer.multihead_attn.out_proj.parametrizations.weight[0].train()

    def trainable_parameter_groups(self) -> Iterable[Tuple[str, torch.nn.Parameter, str]]:
        for name, param in _unique_named_parameters(self.model):
            if not param.requires_grad:
                continue
            if ".parametrizations." in name and "multihead_attn" in name:
                group = "cross_attention_lora"
            elif name.startswith("decoder.stages."):
                group = "decoder_stages"
            elif name.startswith("project_to_decoder_channels."):
                group = "projection_adapter"
            else:
                raise RuntimeError(f"unexpected B3-Full trainable parameter: {name}")
            yield name, param, group

    def save_checkpoint(self, path: str, metadata: Dict) -> None:
        payload = {
            "format": "voxtell_mtl_b3_full_data_v1",
            "base_model_dir": self.model_dir,
            "cross_attention_rank": self.rank,
            "cross_attention_alpha": self.alpha,
            "cross_attention_dropout": self.dropout,
            "trainable_parameter_groups": {
                group: sum(p.numel() for _, p, g in self.trainable_parameter_groups() if g == group)
                for group in {g for _, _, g in self.trainable_parameter_groups()}
            },
            "trainable_state_dict": {
                name: param.detach().cpu().clone()
                for name, param, _ in self.trainable_parameter_groups()
            },
            "metadata": metadata,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
