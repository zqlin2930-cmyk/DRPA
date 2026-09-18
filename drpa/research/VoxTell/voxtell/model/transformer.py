"""
Text prompt decoder implementation for VoxTell.

Code modified from DETR transformer:
https://github.com/facebookresearch/detr
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""

import copy
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerDecoder(nn.Module):
    """
    Transformer decoder consisting of multiple decoder layers.
    
    This decoder processes target sequences with attention to memory (encoder output),
    optionally returning intermediate layer outputs. It receives the text prompt embeddings
    as queries and attends to image features from the encoder. It outputs refined text-image
    fused features for segmentation mask prediction.
    
    Args:
        decoder_layer: A single transformer decoder layer to be cloned.
        num_layers: Number of decoder layers to stack.
        norm: Optional normalization layer applied to the final output.
        return_intermediate: If True, returns outputs from all layers stacked together.
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        return_intermediate: bool = False
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Forward pass through all decoder layers.
        
        Args:
            tgt: Target sequence tensor of shape [T, B, C].
            memory: Memory (encoder output) tensor of shape [T, B, C].
            tgt_mask: Attention mask for target self-attention.
            memory_mask: Attention mask for cross-attention to memory.
            tgt_key_padding_mask: Padding mask for target keys.
            memory_key_padding_mask: Padding mask for memory keys.
            pos: Positional embeddings for memory.
            query_pos: Positional embeddings for queries.
            
        Returns:
            If return_intermediate is True, returns stacked intermediate outputs.
            Otherwise, returns tuple of (final_output, attention_weights_list).
        """
        output = tgt
        T, B, C = memory.shape
        intermediate = []
        atten_layers = []
        
        for n, layer in enumerate(self.layers):
            residual = True
            output, ws = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                residual=residual
            )
            atten_layers.append(ws)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output, atten_layers



class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with self-attention, cross-attention, and FFN.
    
    This layer implements:
    1. Self-attention on the target sequence
    2. Cross-attention between target and memory (image encoder output)
    3. Position-wise feed-forward network
    
    Args:
        d_model: Dimension of the model (embedding dimension).
        nhead: Number of attention heads.
        dim_feedforward: Dimension of the feedforward network.
        dropout: Dropout probability.
        activation: Activation function name ('relu', 'gelu', or 'glu').
        normalize_before: If True, applies layer norm before attention/FFN (pre-norm).
                         If False, applies after (post-norm).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """
        Add positional embeddings to tensor if provided.
        
        Args:
            tensor: Input tensor.
            pos: Optional positional embeddings to add.
            
        Returns:
            Tensor with positional embeddings added, or original tensor if pos is None.
        """
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        residual: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """
        Post-norm forward pass: attention/FFN first, then normalization.
        
        Args:
            tgt: Target sequence.
            memory: Memory (encoder output).
            tgt_mask: Self-attention mask for target.
            memory_mask: Cross-attention mask.
            tgt_key_padding_mask: Padding mask for target keys.
            memory_key_padding_mask: Padding mask for memory keys.
            pos: Positional embeddings for memory.
            query_pos: Positional embeddings for queries.
            residual: Whether to use residual connections.
            
        Returns:
            Tuple of (output_tensor, attention_weights).
        """
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, ws = self.self_attn(
            q, k, value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = self.norm1(tgt)
        tgt2, ws = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )

        # Cross-attention with residual connection
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, ws

    def forward_pre(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Pre-norm forward pass: normalization first, then attention/FFN.
        
        Args:
            tgt: Target sequence.
            memory: Memory (encoder output).
            tgt_mask: Self-attention mask for target.
            memory_mask: Cross-attention mask.
            tgt_key_padding_mask: Padding mask for target keys.
            memory_key_padding_mask: Padding mask for memory keys.
            pos: Positional embeddings for memory.
            query_pos: Positional embeddings for queries.
            
        Returns:
            Tuple of (output_tensor, attention_weights).
        """
        tgt2 = self.norm2(tgt)
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        
        # Feed-forward network with pre-norm
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, attn_weights
    

    def forward_pre_selfattention(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Alternative forward pass with cross-attention before self-attention.
        
        This variant applies operations in the order:
        1. Cross-attention (without normalization)
        2. Self-attention (with pre-norm)
        3. Feed-forward network (with pre-norm)
        
        Args:
            tgt: Target sequence.
            memory: Memory (encoder output).
            tgt_mask: Self-attention mask for target.
            memory_mask: Cross-attention mask.
            tgt_key_padding_mask: Padding mask for target keys.
            memory_key_padding_mask: Padding mask for memory keys.
            pos: Positional embeddings for memory.
            query_pos: Positional embeddings for queries.
            
        Returns:
            Tuple of (output_tensor, attention_weights).
        """
        # Cross-attention without pre-normalization
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + tgt2
        
        # Self-attention with pre-norm
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2, ws = self.self_attn(q, k, value=tgt2)
        tgt = tgt + tgt2
        
        # Feed-forward network with pre-norm
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt, attn_weights


    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        residual: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the decoder layer.
        
        Dispatches to either pre-norm or post-norm variant based on configuration.
        
        Args:
            tgt: Target sequence.
            memory: Memory (encoder output).
            tgt_mask: Self-attention mask for target.
            memory_mask: Cross-attention mask.
            tgt_key_padding_mask: Padding mask for target keys.
            memory_key_padding_mask: Padding mask for memory keys.
            pos: Positional embeddings for memory.
            query_pos: Positional embeddings for queries.
            residual: Whether to use residual connections (used in post-norm variant).
            
        Returns:
            Tuple of (output_tensor, attention_weights).
        """
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, residual
        )


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """
    Create N identical copies of a module.
    
    Args:
        module: The module to clone.
        N: Number of clones to create.
        
    Returns:
        ModuleList containing N deep copies of the input module.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """
    Return an activation function given a string identifier.
    
    Args:
        activation: Name of the activation function ('relu', 'gelu', or 'glu').
        
    Returns:
        The corresponding activation function.
        
    Raises:
        RuntimeError: If the activation name is not recognized.
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")