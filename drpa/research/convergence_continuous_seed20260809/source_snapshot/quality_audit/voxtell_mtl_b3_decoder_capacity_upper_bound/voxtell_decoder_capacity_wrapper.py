"""B3 wrapper: cross-attention LoRA plus decoder-owned capacity unfreezing.

The official VoxTellDecoder stores model.encoder as an alias. This wrapper
therefore enables only decoder-owned stages/transpconvs/seg_layers and all
project_to_decoder_channels weights; decoder.encoder remains frozen.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import torch
from torch import nn
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint
from voxtell_peft_wrapper import VoxTellPEFTWrapper, _unique_named_parameters

class VoxTellDecoderCapacityWrapper(VoxTellPEFTWrapper):
    """Keep image encoder and prompt decoder frozen; train decoder capacity."""
    decoder_owned_children = ("stages", "transpconvs", "seg_layers")

    def __init__(self, model_dir: str, embedding_bank: str,
                 rank: int = 4, alpha: float = 8.0, dropout: float = 0.05,
                 device: torch.device | None = None) -> None:
        super().__init__(model_dir, embedding_bank, rank, alpha, dropout, device)
        self._enable_decoder_capacity()

    def _enable_decoder_capacity(self) -> None:
        for child_name in self.decoder_owned_children:
            child = getattr(self.model.decoder, child_name)
            for param in child.parameters():
                param.requires_grad = True
        for projection in self.model.project_to_decoder_channels:
            for param in projection.parameters():
                param.requires_grad = True

    def forward(self, img: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """Run the unchanged computation with frozen encoder activations detached.

        Detaching the frozen image encoder and frozen input projections is an
        exact memory optimization: no trainable parameter is downstream of
        those operations, while transformer LoRA, all mask projections and
        decoder-owned parameters retain their normal autograd path.
        """
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
            tgt=text_embed,
            memory=bottleneck_embed,
            pos=m.pos_embed,
            memory_key_padding_mask=None,
        )
        mask_embedding = repeat(mask_embedding, 'n b dim -> b n dim')
        mask_embeddings = [projection(mask_embedding) for projection in m.project_to_decoder_channels]
        outs = []
        num_prompts = text_embedding.shape[1]
        for prompt_idx in range(num_prompts):
            prompt_embeds = [x[:, prompt_idx:prompt_idx + 1] for x in mask_embeddings]
            if torch.is_grad_enabled():
                def decode_from_flat(*args):
                    n_skips = len(skips)
                    dec_skips = list(args[:n_skips])
                    dec_embeds = list(args[n_skips:])
                    return m.decoder(dec_skips, dec_embeds)[0]
                decoded = checkpoint(
                    decode_from_flat,
                    *(list(skips) + prompt_embeds),
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                outs.append([decoded])
            else:
                outs.append(m.decoder(skips, prompt_embeds))
        outs = [torch.cat(scale_outs, dim=1) for scale_outs in zip(*outs)]
        result = outs[0] if not m.deep_supervision else outs
        if torch.is_grad_enabled():
            if not result.requires_grad or result.grad_fn is None:
                raise RuntimeError(
                    "B3 wrapper graph is detached: final logits must require gradients "
                    "and have a grad_fn during training"
                )
        return result

    def decoder_capacity_parameter_count(self) -> int:
        return sum(
            p.numel()
            for child_name in self.decoder_owned_children
            for p in getattr(self.model.decoder, child_name).parameters()
        ) + sum(p.numel() for p in self.model.project_to_decoder_channels.parameters())

    def decoder_capacity_target_summary(self) -> Dict[str, List[str]]:
        return {
            "decoder_owned": [f"decoder.{x}" for x in self.decoder_owned_children],
            "all_project_to_decoder_channels": ["project_to_decoder_channels"],
            "cross_attention_lora": [x for x in self.lora_targets],
            "excluded_shared_alias": ["decoder.encoder (alias of model.encoder)"],
        }

    def decoder_encoder_alias_is_frozen(self) -> bool:
        enc = self.model.encoder
        dec_enc = self.model.decoder.encoder
        if enc is not dec_enc:
            raise RuntimeError("VoxTellDecoder.encoder is not the expected model.encoder alias")
        return all(not p.requires_grad for p in enc.parameters())

    def trainable_parameter_groups(self) -> Iterable[Tuple[str, nn.Parameter, str]]:
        for name, param in _unique_named_parameters(self.model):
            if not param.requires_grad:
                continue
            if ".parametrizations." in name:
                group = "cross_attention_lora"
            elif name.startswith("decoder."):
                group = "decoder_owned"
            elif name.startswith("project_to_decoder_channels."):
                group = "project_to_decoder_channels"
            else:
                group = "unexpected"
            yield name, param, group
