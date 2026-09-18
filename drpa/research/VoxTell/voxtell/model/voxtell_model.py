import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from typing import List, Type, Union, Tuple
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import InitWeights_He, init_last_bn_before_add_to_0
from einops import rearrange, repeat
from positional_encodings.torch_encodings import PositionalEncoding3D

from voxtell.model.transformer import TransformerDecoder, TransformerDecoderLayer


class VoxTellModel(nn.Module):
    """
    VoxTell segmentation model with text-prompted decoder.
    
    This model combines a ResidualEncoder backbone with a transformer-based decoder
    that uses text embeddings to generate segmentation masks. It supports multi-stage
    decoding with optional deep supervision.
    
    Attributes:
        encoder: ResidualEncoder backbone for feature extraction.
        decoder: VoxTellDecoder for multi-scale feature decoding.
        transformer_decoder: Transformer for fusing text and image features.
        deep_supervision: Whether to return multi-scale predictions.
    """
    
    # Class constants for transformer architecture (text prompt decoder)
    TRANSFORMER_NUM_HEADS = 8
    TRANSFORMER_NUM_LAYERS = 6
    
    # Decoder configuration for different stages
    DECODER_CONFIGS = {
        0: {"channels": 32, "shape": (192, 192, 192)},
        1: {"channels": 64, "shape": (96, 96, 96)},
        2: {"channels": 128, "shape": (48, 48, 48)},
        3: {"channels": 256, "shape": (24, 24, 24)},
        4: {"channels": 320, "shape": (12, 12, 12)},
        5: {"channels": 320, "shape": (6, 6, 6)},
    }
    
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
        # Text-prompted segmentation parameters
        num_maskformer_stages: int = 5,
        query_dim: int = 864,
        decoder_layer: int = 4,
        text_embedding_dim: int = 1024,
        num_heads: int = 1,
        project_to_decoder_hidden_dim: int = 432
    ) -> None:
        """
        Initialize the VoxTell model.
        
        Args:
            input_channels: Number of input channels.
            n_stages: Number of encoder stages.
            features_per_stage: Number of features at each stage.
            conv_op: Convolution operation type.
            kernel_sizes: Kernel sizes for convolutions.
            strides: Strides for downsampling.
            n_blocks_per_stage: Number of residual blocks per stage.
            n_conv_per_stage_decoder: Number of convolutions per stage.
            conv_bias: Whether to use bias in convolutions.
            norm_op: Normalization operation.
            norm_op_kwargs: Normalization operation keyword arguments.
            dropout_op: Dropout operation.
            dropout_op_kwargs: Dropout operation keyword arguments.
            nonlin: Non-linearity operation.
            nonlin_kwargs: Non-linearity keyword arguments.
            deep_supervision: Whether to use deep supervision.
            block: Residual block type (BasicBlockD or BottleneckD).
            bottleneck_channels: Channels in bottleneck layers.
            stem_channels: Channels in stem layer.
            num_maskformer_stages: Number of stages to fuse text-image embeddings in decoder.
            query_dim: Dimension of query embeddings.
            decoder_layer: Which decoder layer to use as memory for text prompt decoder (0-5).
            text_embedding_dim: Dimension of text embeddings.
            num_heads: Number of channels added per U-Net stage for mask embedding fusion.
            project_to_decoder_hidden_dim: Hidden dimension for projection to decoder.
        """
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        
        assert len(n_blocks_per_stage) == n_stages, (
            f"n_blocks_per_stage must have as many entries as we have resolution stages. "
            f"Expected: {n_stages}, got: {len(n_blocks_per_stage)} ({n_blocks_per_stage})"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            f"n_conv_per_stage_decoder must have one less entry than resolution stages. "
            f"Expected: {n_stages - 1}, got: {len(n_conv_per_stage_decoder)} ({n_conv_per_stage_decoder})"
        )
        if num_maskformer_stages != 5:
            assert not deep_supervision, (
                "Deep supervision is not supported for num_maskformer_stages != 5."
            )
        self.deep_supervision = deep_supervision
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.project_to_decoder_hidden_dim = project_to_decoder_hidden_dim
        self.text_embedding_dim = text_embedding_dim
        
        # Initialize encoder backbone
        self.encoder = ResidualEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
            return_skips=True, disable_default_stem=False, stem_channels=stem_channels
        )
        
        # Initialize decoder
        self.decoder = VoxTellDecoder(
            self.encoder, 1, n_conv_per_stage_decoder, deep_supervision,
            num_maskformer_stages, num_heads=self.num_heads
        )
        
        # Select decoder layer configuration
        self.selected_decoder_layer = decoder_layer
        if decoder_layer not in self.DECODER_CONFIGS:
            raise ValueError(
                f"decoder_layer must be in {list(self.DECODER_CONFIGS.keys())}, got {decoder_layer}"
            )
        selected_config = self.DECODER_CONFIGS[decoder_layer]
        
        h, w, d = selected_config["shape"]

        # Project bottleneck embeddings to query dimension
        self.project_bottleneck_embed = nn.Sequential(
            nn.Linear(selected_config["channels"], query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )

        # Project text embeddings to query dimension
        text_hidden_dim = 2048
        self.project_text_embed = nn.Sequential(
            nn.Linear(self.text_embedding_dim, text_hidden_dim),
            nn.GELU(),
            nn.Linear(text_hidden_dim, query_dim),
        )

        # Project decoder output to image channels for each mask-former stage
        self.project_to_decoder_channels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(query_dim, self.project_to_decoder_hidden_dim),
                nn.GELU(),
                nn.Linear(
                    self.project_to_decoder_hidden_dim,
                    decoder_config["channels"] * self.num_heads if stage_idx != 0 else decoder_config["channels"]
                )
            )
            for stage_idx, decoder_config in enumerate(
                list(self.DECODER_CONFIGS.values())[:num_maskformer_stages]
            )
        ])
        
        # Initialize 3D positional encoding
        # Shape: (H*W*D, batch_size, query_dim)
        pos_embed = PositionalEncoding3D(query_dim)(torch.zeros(1, h, w, d, query_dim))
        pos_embed = rearrange(pos_embed, 'b h w d c -> (h w d) b c')
        self.register_buffer('pos_embed', pos_embed)

        # Initialize transformer decoder for fusing text and image features (prompt decoder)
        transformer_layer = TransformerDecoderLayer(
            d_model=query_dim,
            nhead=self.TRANSFORMER_NUM_HEADS,
            normalize_before=True
        )
        decoder_norm = nn.LayerNorm(query_dim)
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=transformer_layer,
            num_layers=self.TRANSFORMER_NUM_LAYERS,
            norm=decoder_norm
        )

    def forward(
        self,
        img: torch.Tensor,
        text_embedding: torch.Tensor = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through VoxTell model.
        
        Args:
            img: Input image tensor of shape (B, C, D, H, W).
            text_embedding: Pre-computed text embeddings of shape (B, N, D) where
                N is number of prompts and D is embedding dimension.
            
        Returns:
            If deep_supervision is False, returns single prediction tensor of shape (B, N, D, H, W).
            If deep_supervision is True, returns list of prediction tensors at different scales.
        """
        # Extract multi-scale features from encoder
        skips = self.encoder(img)

        # Select encoder features 
        selected_feature = skips[self.selected_decoder_layer]

        # Reshape and project features to query dimension
        # Shape: (B, C, D, H, W) -> (B, H, W, D, C) -> (B, H, W, D, query_dim)
        bottleneck_embed = rearrange(selected_feature, 'b c d h w -> b h w d c')
        bottleneck_embed = self.project_bottleneck_embed(bottleneck_embed)
        # Shape: (B, H, W, D, query_dim) -> (H*W*D, B, query_dim) for transformer
        bottleneck_embed = rearrange(bottleneck_embed, 'b h w d c -> (h w d) b c')

        # Remove singleton dimension from text embeddings and project
        # Shape: (B, N, 1, D) -> (B, N, D)
        text_embedding = text_embedding.squeeze(2)
        # Shape: (B, N, D) -> (N, B, D) as required by transformer decoder
        text_embed = repeat(text_embedding, 'b n dim -> n b dim')
        text_embed = self.project_text_embed(text_embed)

        # Fuse text and image features through transformer decoder
        # Output shape: (N, B, query_dim)
        mask_embedding, _ = self.transformer_decoder(
            tgt=text_embed,
            memory=bottleneck_embed,
            pos=self.pos_embed,
            memory_key_padding_mask=None
        )
        # Shape: (N, B, query_dim) -> (B, N, query_dim)
        mask_embedding = repeat(mask_embedding, 'n b dim -> b n dim')

        # Project mask embeddings to decoder channel dimensions for each stage
        mask_embeddings = [
            projection(mask_embedding)
            for projection in self.project_to_decoder_channels
        ]

        # Generate segmentation outputs for each text prompt
        outs = []
        num_prompts = text_embedding.shape[1]
        for prompt_idx in range(num_prompts):
            # Extract embeddings for this prompt across all stages
            prompt_embeds = [m[:, prompt_idx:prompt_idx + 1] for m in mask_embeddings]
            outs.append(self.decoder(skips, prompt_embeds))
        
        # Concatenate outputs across prompts for each scale
        outs = [torch.cat(scale_outs, dim=1) for scale_outs in zip(*outs)]

        if not self.deep_supervision:
            outs = outs[0]

        return outs

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class VoxTellDecoder(nn.Module):
    """
    Decoder for VoxTell with mask-embedding fusion.
    
    This decoder upsamples features from the encoder and fuses them with
    mask embeddings from text prompts at multiple scales. It supports
    deep supervision for multi-scale training.
    
    The decoder processes features from bottleneck to highest resolution,
    incorporating mask embeddings through einsum operations at each stage.
    """
    
    def __init__(
        self,
        encoder: Union[PlainConvEncoder, ResidualEncoder],
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        num_maskformer_stages: int = 5,
        nonlin_first: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        conv_bias: bool = None,
        num_heads: int = 1
    ) -> None:
        """
        Initialize VoxTell decoder.
        
        The decoder upsamples features from encoder stages and fuses them with
        mask embeddings. Each stage consists of:
        1) Transpose convolution to upsample lower resolution features
        2) Concatenation with skip connections from encoder
        3) Convolutional blocks to merge features
        4) Fusion with mask embeddings via einsum
        5) Optional segmentation output for deep supervision
        
        Args:
            encoder: Encoder module (PlainConvEncoder or ResidualEncoder).
            num_classes: Number of output classes (typically 1 for binary segmentation).
            n_conv_per_stage: Number of convolution blocks per decoder stage.
            deep_supervision: Whether to output predictions at multiple scales.
            num_maskformer_stages: Number of stages to fuse mask embeddings.
            nonlin_first: Whether to apply non-linearity before convolution.
            norm_op: Normalization operation (inherited from encoder if None).
            norm_op_kwargs: Normalization keyword arguments.
            dropout_op: Dropout operation (inherited from encoder if None).
            dropout_op_kwargs: Dropout keyword arguments.
            nonlin: Non-linearity operation (inherited from encoder if None).
            nonlin_kwargs: Non-linearity keyword arguments.
            conv_bias: Whether to use bias in convolutions (inherited from encoder if None).
            num_heads: Number of attention heads for mask embedding fusion.
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.num_heads = num_heads
        
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        
        assert len(n_conv_per_stage) == n_stages_encoder - 1, (
            f"n_conv_per_stage must have one less entry than encoder stages. "
            f"Expected: {n_stages_encoder - 1}, got: {len(n_conv_per_stage)}"
        )

        # Inherit hyperparameters from encoder if not specified
        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        # Build decoder stages from bottleneck to highest resolution
        stages = []
        transpconvs = []
        seg_layers = []
        for stage_idx in range(1, n_stages_encoder):
            # Determine input channels: add num_heads for stages with mask embedding fusion
            if stage_idx <= n_stages_encoder - num_maskformer_stages:
                input_features_below = encoder.output_channels[-stage_idx]
            else:
                input_features_below = encoder.output_channels[-stage_idx] + num_heads
            
            input_features_skip = encoder.output_channels[-(stage_idx + 1)]
            stride_for_transpconv = encoder.strides[-stage_idx]
            
            # Transpose convolution for upsampling
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip,
                stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            
            # Convolutional blocks for feature merging
            # Input features: 2x input_features_skip (concatenated skip + upsampled features)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[stage_idx - 1], encoder.conv_op,
                2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(stage_idx + 1)], 1,
                conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, nonlin_first
            ))

            # Segmentation output layer (always built for parameter loading compatibility)
            # This allows models trained with deep_supervision=True to be loaded
            # for inference with deep_supervision=False
            seg_layers.append(encoder.conv_op(
                input_features_skip + num_heads, num_classes,
                1, 1, 0, bias=True
            ))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(
        self,
        skips: List[torch.Tensor],
        mask_embeddings: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Forward pass through decoder with mask embedding fusion.
        
        Processes features from bottleneck to highest resolution, fusing
        mask embeddings at multiple stages via einsum operations.
        
        Args:
            skips: List of encoder skip connections in computation order.
                Last entry should be bottleneck features.
            mask_embeddings: List of mask embeddings for each decoder stage,
                in order from lowest to highest resolution.
                
        Returns:
            List of segmentation predictions. If deep_supervision=False,
            returns single-element list with highest resolution prediction.
            If deep_supervision=True, returns predictions at all scales
            from highest to lowest resolution.
        """
        lres_input = skips[-1]
        seg_outputs = []
        
        # Reverse mask embeddings to match decoder stage order (bottleneck first)
        mask_embeddings = mask_embeddings[::-1]
        
        for stage_idx in range(len(self.stages)):
            # Upsample and concatenate with skip connection
            x = self.transpconvs[stage_idx](lres_input)
            x = torch.cat((x, skips[-(stage_idx + 2)]), dim=1)
            x = self.stages[stage_idx](x)
            
            # Apply mask embedding fusion for relevant stages
            if stage_idx == (len(self.stages) - 1):
                # Final stage: generate segmentation via einsum
                # x: (B, C, H, W, D), mask_embeddings[-1]: (B, N, C)
                # Output: (B, N, H, W, D)
                seg_pred = torch.einsum('b c h w d, b n c -> b n h w d', x, mask_embeddings[-1])
                seg_outputs.append(seg_pred)
            elif stage_idx >= len(self.stages) - len(mask_embeddings):
                # Intermediate stages with mask embedding fusion
                mask_embedding = mask_embeddings.pop(0)
                batch_size, _, channels = mask_embedding.shape
                
                # Reshape for multi-head fusion and compute attention-weighted features
                # Shape: (B, num_heads, C // num_heads)
                mask_embedding_reshaped = mask_embedding.view(batch_size, self.num_heads, -1)
                fusion_features = torch.einsum(
                    'b c h w d, b n c -> b n h w d',
                    x, mask_embedding_reshaped
                )
                
                # Concatenate fused features with spatial features
                x = torch.cat((x, fusion_features), dim=1)
                seg_outputs.append(self.seg_layers[stage_idx](x))
            
            lres_input = x

        # Reverse outputs to have highest resolution first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            return seg_outputs[:1]
        else:
            return seg_outputs