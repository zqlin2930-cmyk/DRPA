#!/usr/bin/env python3
"""Project-side DRPA-8 wrapper; the official VoxTell source is untouched."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn.utils.parametrize import register_parametrization
from torch.utils.checkpoint import checkpoint

from voxtell_peft_wrapper import VoxTellPEFTWrapper, _unique_named_parameters


class LowRankWeightUpdate(nn.Module):
    """Base weight plus a zero-initialized rank-r update."""

    def __init__(self, shape: Tuple[int, ...], rank: int, alpha: float = 8.0) -> None:
        super().__init__()
        if len(shape) < 2:
            raise ValueError(f"DRPA-8 requires a matrix-like weight, got {shape}")
        self.original_shape = tuple(int(x) for x in shape)
        self.rows = int(shape[0])
        self.cols = int(torch.tensor(shape[1:]).prod().item())
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.cols))
        self.lora_B = nn.Parameter(torch.zeros(self.rows, self.rank))
        # Initialization is independent of B3. B=0 makes the initial network
        # exactly the pretrained network up to floating-point parametrization.
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def delta_matrix(self) -> Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, base_weight: Tensor) -> Tensor:
        delta = self.delta_matrix().reshape(self.original_shape)
        return base_weight + delta.to(dtype=base_weight.dtype)


class DRPA8Wrapper(VoxTellPEFTWrapper):
    """Fresh VoxTell + cross-attention LoRA + all-scale rank-8 projections.

    Only decoder.stages, projection adapters, and existing cross-attention
    LoRA are trainable. Original projection weights are parametrization bases
    and remain frozen.
    """

    def __init__(self, model_dir: str, embedding_bank: str,
                 projection_rank: int = 8, projection_alpha: float = 8.0,
                 cross_attention_rank: int = 4, cross_attention_alpha: float = 8.0,
                 cross_attention_dropout: float = 0.05,
                 device: torch.device | None = None) -> None:
        if projection_rank != 8:
            raise ValueError("DRPA-8 fixes projection_rank=8")
        super().__init__(model_dir, embedding_bank, cross_attention_rank,
                         cross_attention_alpha, cross_attention_dropout, device)
        self.projection_rank = int(projection_rank)
        self.projection_alpha = float(projection_alpha)
        self.projection_targets: List[str] = []
        self._attach_projection_adapters()
        self._set_trainable_contract()

    def _attach_projection_adapters(self) -> None:
        projections = self.model.project_to_decoder_channels
        for pidx, projection in enumerate(projections):
            linear_count = 0
            for child_name, child in projection.named_modules():
                if not isinstance(child, nn.Linear):
                    continue
                adapter = LowRankWeightUpdate(tuple(child.weight.shape), self.projection_rank,
                                              self.projection_alpha).to(child.weight.device)
                register_parametrization(child, "weight", adapter)
                target = f"project_to_decoder_channels.{pidx}.{child_name}.weight"
                self.projection_targets.append(target)
                linear_count += 1
            if linear_count != 2:
                raise RuntimeError(
                    f"Expected two effective Linear projection weights at index {pidx}, "
                    f"found {linear_count}"
                )
        if len(self.projection_targets) != 2 * len(projections):
            raise RuntimeError("Not every projection received both rank-8 adapters")

    def _set_trainable_contract(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        for stage in self.model.decoder.stages.parameters():
            stage.requires_grad = True
        for projection in self.model.project_to_decoder_channels:
            for module in projection.modules():
                if isinstance(module, LowRankWeightUpdate):
                    for param in module.parameters():
                        param.requires_grad = True
        for layer in self.model.transformer_decoder.layers:
            for target in (layer.multihead_attn.parametrizations.in_proj_weight[0],
                           layer.multihead_attn.out_proj.parametrizations.weight[0]):
                for param in target.parameters():
                    param.requires_grad = True

    def set_training_mode(self) -> None:
        self.model.eval()
        self.model.decoder.stages.train()
        for projection in self.model.project_to_decoder_channels:
            projection.train()
        for layer in self.model.transformer_decoder.layers:
            layer.multihead_attn.parametrizations.in_proj_weight[0].train()
            layer.multihead_attn.out_proj.parametrizations.weight[0].train()

    def forward(self, img: Tensor, text_embedding: Tensor) -> Tensor:
        m = self.model
        with torch.no_grad():
            skips = m.encoder(img)
            selected_feature = skips[m.selected_decoder_layer]
            bottleneck_embed = rearrange(selected_feature, 'b c d h w -> b h w d c')
            bottleneck_embed = m.project_bottleneck_embed(bottleneck_embed)
            bottleneck_embed = rearrange(bottleneck_embed, 'b h w d c -> (h w d) b c')
            text_embed = text_embedding.squeeze(2)
            text_embed = repeat(text_embed, 'b n dim -> n b dim')
            text_embed = m.project_text_embed(text_embed)
        mask_embedding, _ = m.transformer_decoder(
            tgt=text_embed, memory=bottleneck_embed, pos=m.pos_embed,
            memory_key_padding_mask=None,
        )
        mask_embedding = repeat(mask_embedding, 'n b dim -> b n dim')
        mask_embeddings = [projection(mask_embedding)
                           for projection in m.project_to_decoder_channels]
        outs = []
        num_prompts = text_embedding.shape[1]
        for prompt_idx in range(num_prompts):
            prompt_embeds = [x[:, prompt_idx:prompt_idx + 1]
                             for x in mask_embeddings]
            if torch.is_grad_enabled():
                def decode_from_flat(*args):
                    n_skips = len(skips)
                    dec_skips = list(args[:n_skips])
                    dec_embeds = list(args[n_skips:])
                    return m.decoder(dec_skips, dec_embeds)[0]
                decoded = checkpoint(
                    decode_from_flat, *(list(skips) + prompt_embeds),
                    use_reentrant=False, preserve_rng_state=True,
                )
                outs.append([decoded])
            else:
                outs.append(m.decoder(skips, prompt_embeds))
        outs = [torch.cat(scale_outs, dim=1) for scale_outs in zip(*outs)]
        result = outs[0] if not m.deep_supervision else outs
        if torch.is_grad_enabled() and (not result.requires_grad or result.grad_fn is None):
            raise RuntimeError("DRPA-8 graph detached at final logits")
        return result

    def trainable_parameter_groups(self) -> Iterable[Tuple[str, nn.Parameter, str]]:
        for name, param in _unique_named_parameters(self.model):
            if not param.requires_grad:
                continue
            if ".parametrizations." in name:
                if "project_to_decoder_channels" in name:
                    group = "projection_adapter"
                elif "multihead_attn" in name:
                    group = "cross_attention_lora"
                else:
                    group = "unexpected"
            elif name.startswith("decoder.stages."):
                group = "decoder_stages"
            else:
                group = "unexpected"
            yield name, param, group

    def trainable_parameter_count_by_group(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for _, param, group in self.trainable_parameter_groups():
            out[group] = out.get(group, 0) + param.numel()
        return out

    def projection_update_summary(self) -> List[Dict[str, float | str | int]]:
        rows = []
        for pidx, projection in enumerate(self.model.project_to_decoder_channels):
            for child_name, child in projection.named_modules():
                if not isinstance(child, nn.Linear):
                    continue
                adapter = child.parametrizations.weight[0]
                delta = adapter.delta_matrix().detach().float()
                base = child.parametrizations.weight.original.detach().float().reshape(delta.shape)
                rows.append({
                    "projection": f"project_to_decoder_channels.{pidx}.{child_name}",
                    "rank": adapter.rank,
                    "parameter_count": int(adapter.lora_A.numel() + adapter.lora_B.numel()),
                    "delta_l2": float(delta.norm().cpu()),
                    "base_l2": float(base.norm().cpu()),
                    "relative_update": float((delta.norm() / base.norm().clamp_min(1e-12)).cpu()),
                    "A_norm": float(adapter.lora_A.detach().float().norm().cpu()),
                    "B_norm": float(adapter.lora_B.detach().float().norm().cpu()),
                })
        return rows

    def adapter_state_dict(self) -> Dict[str, Tensor]:
        return {name: param.detach().cpu().clone()
                for name, param, _ in self.trainable_parameter_groups()}

    def save_checkpoint(self, path: str, metadata: Dict) -> None:
        payload = {
            "format": "voxtell_mtl_drpa8_v1",
            "base_model_dir": self.model_dir,
            "projection_rank": self.projection_rank,
            "projection_alpha": self.projection_alpha,
            "cross_attention_rank": self.rank,
            "cross_attention_alpha": self.alpha,
            "cross_attention_dropout": self.dropout,
            "adapter_state_dict": self.adapter_state_dict(),
            "metadata": metadata,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> Dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "voxtell_mtl_drpa8_v1":
            raise ValueError(f"Unsupported DRPA-8 checkpoint: {payload.get('format')}")
        expected = dict((n, p) for n, p, _ in self.trainable_parameter_groups())
        incoming = payload["adapter_state_dict"]
        if set(expected) != set(incoming):
            raise RuntimeError("DRPA-8 trainable parameter keys do not match")
        with torch.no_grad():
            for name, param in expected.items():
                param.copy_(incoming[name].to(param.device, dtype=param.dtype))
        return payload
