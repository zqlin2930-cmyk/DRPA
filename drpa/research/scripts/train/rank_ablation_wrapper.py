"""Generalized DRPA wrapper used only for the frozen rank-4/rank-16 ablation.

This leaves the canonical DRPA-8 wrapper untouched.  It has exactly the same
trainable placement as DRPA-8 (cross-attention LoRA, all decoder stages, and
zero-initialized residual projection adapters); only ``projection_rank`` is
variable.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from voxtell_peft_wrapper import VoxTellPEFTWrapper, _unique_named_parameters
from drpa8_wrapper import DRPA8Wrapper, LowRankWeightUpdate


class RankAblationDRPAWrapper(DRPA8Wrapper):
    """Canonical DRPA placement with an explicitly supplied projection rank."""

    def __init__(self, model_dir: str, embedding_bank: str, *, projection_rank: int,
                 projection_alpha: float = 8.0, cross_attention_rank: int = 4,
                 cross_attention_alpha: float = 8.0,
                 cross_attention_dropout: float = 0.05,
                 device: torch.device | None = None) -> None:
        # DRPA8Wrapper intentionally rejects values other than eight.  Rebuild
        # its construction contract directly rather than weakening that class.
        VoxTellPEFTWrapper.__init__(
            self, model_dir, embedding_bank, cross_attention_rank,
            cross_attention_alpha, cross_attention_dropout, device,
        )
        if projection_rank not in (4, 8, 16):
            raise ValueError("rank ablation permits only pre-registered ranks 4, 8, 16")
        self.projection_rank = int(projection_rank)
        self.projection_alpha = float(projection_alpha)
        self.projection_targets = []
        self._attach_ranked_projection_adapters()
        self._set_trainable_contract()

    def _attach_ranked_projection_adapters(self) -> None:
        for pidx, projection in enumerate(self.model.project_to_decoder_channels):
            count = 0
            for child_name, child in projection.named_modules():
                if not isinstance(child, nn.Linear):
                    continue
                adapter = LowRankWeightUpdate(
                    tuple(child.weight.shape), self.projection_rank,
                    self.projection_alpha,
                ).to(child.weight.device)
                register_parametrization(child, "weight", adapter)
                self.projection_targets.append(
                    f"project_to_decoder_channels.{pidx}.{child_name}.weight"
                )
                count += 1
            if count != 2:
                raise RuntimeError(f"projection index {pidx}: expected 2 Linear layers, got {count}")
        if len(self.projection_targets) != 2 * len(self.model.project_to_decoder_channels):
            raise RuntimeError("not every projection received both residual adapters")

    def trainable_parameter_groups(self) -> Iterable[Tuple[str, nn.Parameter, str]]:
        for name, parameter in _unique_named_parameters(self.model):
            if not parameter.requires_grad:
                continue
            if ".parametrizations." in name and "project_to_decoder_channels" in name:
                group = "projection_adapter"
            elif ".parametrizations." in name and "multihead_attn" in name:
                group = "cross_attention_lora"
            elif name.startswith("decoder.stages."):
                group = "decoder_stages"
            else:
                raise RuntimeError(f"unexpected rank-ablation trainable parameter: {name}")
            yield name, parameter, group
