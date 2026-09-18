"""Parameter-efficient wrapper for the official VoxTellModel.

The official source tree is not modified. The wrapper loads the published full
text-conditioned model, freezes every original parameter, and parametrizes only
cross-attention QKV (PyTorch's combined in_proj_weight) and output projections
with low-rank adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrize import register_parametrization

from voxtell.inference.predictor import VoxTellPredictor


class LowRankWeight(nn.Module):
    """W + scale * B @ dropout(A), used as a weight parametrization."""

    def __init__(self, out_features: int, in_features: int, rank: int,
                 alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = float(dropout)
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, base_weight: Tensor) -> Tensor:
        a = F.dropout(self.lora_A, p=self.dropout, training=self.training)
        return base_weight + self.scaling * (self.lora_B @ a)


def _unique_named_parameters(module: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    seen = set()
    for name, param in module.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        yield name, param


class VoxTellPEFTWrapper(nn.Module):
    """Keep the official VoxTellModel forward and add only cross-attention LoRA."""

    def __init__(self, model_dir: str, embedding_bank: str,
                 rank: int = 4, alpha: float = 8.0, dropout: float = 0.05,
                 device: torch.device | None = None) -> None:
        super().__init__()
        self.model_dir = str(model_dir)
        self.embedding_bank = str(embedding_bank)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)

        predictor = VoxTellPredictor(
            model_dir=self.model_dir,
            device=torch.device("cpu"),
            embedding_bank=self.embedding_bank,
            use_precomputed_embeddings=True,
        )
        self.predictor = predictor
        self.model = predictor.network
        self.base_total_params = sum(p.numel() for _, p in _unique_named_parameters(self.model))
        self._freeze_all_original_parameters()
        self.lora_targets: List[str] = []
        self._attach_cross_attention_lora()
        self._freeze_all_then_enable_lora()
        if device is not None:
            self.to(device)

    def _freeze_all_original_parameters(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def _attach_cross_attention_lora(self) -> None:
        layers = self.model.transformer_decoder.layers
        for layer_idx, layer in enumerate(layers):
            # VoxTell's TransformerDecoderLayer calls multihead_attn for
            # text-query -> image-memory cross-attention. PyTorch combines
            # q/k/v into in_proj_weight and stores output in out_proj.weight.
            cross_attn = layer.multihead_attn
            if cross_attn.in_proj_weight is None:
                raise RuntimeError("Expected combined MultiheadAttention in_proj_weight")
            register_parametrization(
                cross_attn,
                "in_proj_weight",
                LowRankWeight(
                    out_features=cross_attn.in_proj_weight.shape[0],
                    in_features=cross_attn.in_proj_weight.shape[1],
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                ),
            )
            register_parametrization(
                cross_attn.out_proj,
                "weight",
                LowRankWeight(
                    out_features=cross_attn.out_proj.weight.shape[0],
                    in_features=cross_attn.out_proj.weight.shape[1],
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                ),
            )
            self.lora_targets.extend([
                f"transformer_decoder.layers.{layer_idx}.multihead_attn.in_proj_weight",
                f"transformer_decoder.layers.{layer_idx}.multihead_attn.out_proj.weight",
            ])

    def _freeze_all_then_enable_lora(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        for layer in self.model.transformer_decoder.layers:
            for target in (layer.multihead_attn.parametrizations.in_proj_weight[0],
                           layer.multihead_attn.out_proj.parametrizations.weight[0]):
                for param in target.parameters():
                    param.requires_grad = True

    def forward(self, img: Tensor, text_embedding: Tensor) -> Tensor:
        """Call the unchanged official VoxTellModel(img, text_embedding) path."""
        return self.model(img, text_embedding)

    def trainable_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        return ((name, param) for name, param in _unique_named_parameters(self.model)
                if param.requires_grad)

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [param for _, param in self.trainable_named_parameters()]

    def adapter_parameter_count(self) -> int:
        return sum(param.numel() for param in self.trainable_parameters())

    def effective_total_parameter_count(self) -> int:
        return self.base_total_params + self.adapter_parameter_count()

    def adapter_state_dict(self) -> Dict[str, Tensor]:
        return {name: param.detach().cpu().clone()
                for name, param in self.trainable_named_parameters()}

    def save_adapter_checkpoint(self, path: str, metadata: Dict) -> None:
        payload = {
            "format": "voxtell_mtl_peft_lora_v1",
            "base_model_dir": self.model_dir,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "adapter_state_dict": self.adapter_state_dict(),
            "metadata": metadata,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_adapter_checkpoint(self, path: str) -> Dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "voxtell_mtl_peft_lora_v1":
            raise ValueError(f"Unsupported PEFT checkpoint format: {payload.get('format')}")
        expected = dict(self.trainable_named_parameters())
        incoming = payload["adapter_state_dict"]
        if set(expected) != set(incoming):
            raise RuntimeError("PEFT adapter keys do not match the wrapped model")
        with torch.no_grad():
            for name, param in expected.items():
                param.copy_(incoming[name].to(param.device, dtype=param.dtype))
        return payload


def module_group(name: str) -> str:
    if ".parametrizations." in name:
        if "multihead_attn" in name and "in_proj_weight" in name:
            return "transformer_decoder.cross_attention.in_proj_weight_lora"
        if "multihead_attn" in name and "out_proj" in name:
            return "transformer_decoder.cross_attention.out_proj_weight_lora"
        return "peft_adapter"
    if name.startswith("encoder."):
        return "image_encoder"
    if name.startswith("project_text_embed"):
        return "project_text_embed"
    if name.startswith("project_bottleneck_embed"):
        return "project_bottleneck_embed"
    if name.startswith("project_to_decoder_channels"):
        return "project_to_decoder_channels"
    if name.startswith("transformer_decoder"):
        return "transformer_decoder_base"
    if name.startswith("decoder"):
        return "decoder_base"
    return name.split(".")[0]
