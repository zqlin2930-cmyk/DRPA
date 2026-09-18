# VoxTell-FullFT Parameter Structure Audit

Status: FULLFT_PARAMETER_DEFINITION_FROZEN

## Scope

This is a structural, read-only audit. No training, optimizer.step, checkpoint write, GT or manifest change occurred. Qwen3-Embedding-4B is external to VoxTellModel, remains frozen, and is excluded from every segmentation-network count.

Canonical audit configuration: VoxTell v1.1, normalize_before=True, deep_supervision=False, batch 1, 192-cubed image, eight prompts. FullFT means every unique VoxTell segmentation-network parameter has requires_grad=True.

## Three distinct parameter definitions

| field | definition | VoxTell-FullFT |
|---|---|---:|
| FULLFT_CONFIGURED_TRAINABLE | all unique requires_grad=True parameters | 440,029,541 |
| FULLFT_FORWARD_REACHABLE | configured parameters executed in canonical forward | 339,292,452 |
| FULLFT_LOSS_GRADIENT_ACTIVE | configured parameters on the final segmentation-loss path | 339,291,552 |

The configured count was independently reproduced from named_parameters(remove_duplicate=True), id(parameter) deduplication, and AdamW optimizer registration. The historical 540,558,277 is a non-canonical state_dict count from incomplete alias handling; it must not be used in paper tables.

## Actual forward-path evidence

MRI -> encoder -> project_bottleneck_embed -> project_text_embed -> transformer_decoder -> project_to_decoder_channels -> prompt-wise decoder -> final einsum logits.

Hooks on the instantiated canonical model found:

- encoder, both input projections, all five decoder-channel projections, all six cross-attention modules, FFN, norm2/norm3, decoder stages, transpconvs and seg_layers 0-3 execute;
- transformer self_attn and norm1 execute zero times under normalize_before=True / forward_pre;
- seg_layers 0-3 execute but their intermediate deep-supervision outputs are discarded under deep_supervision=False;
- seg_layers.4 is registered but not executed; the final mask is the highest-resolution einsum output.

Forward-reachable therefore excludes self_attn (100,712,448), norm1 (24,576), and seg_layers.4 (65). Final-loss activity also excludes seg_layers.0-3 (900).

The exact 192-cubed FP32 backward gate, including the activation/gradient-checkpointing attempt, OOMed before completion. Hence loss-gradient activity is exact execution-plus-final-output structural accounting, not an empirically completed full-resolution gradient-norm measurement.

## Alias reconciliation

The checkpoint state_dict has 724,690,373 elements in 1,095 entries, including 281,121,888 duplicate alias elements and a 3,538,944-element pos_embed buffer. Aliases include decoder.encoder and internal conv/all_modules entries. They are not extra optimizer parameters. Unique registered model parameters equal 440,029,541.

## Same-contract comparator accounting

| model | configured | forward-reachable | loss-gradient-active | trainable groups |
|---|---:|---:|---:|---|
| B1 | 294,912 | 294,912 | 294,912 | rank-4 cross-attention LoRA |
| DRPA-8 | 10,969,696 | 10,969,696 | 10,969,696 | B1 LoRA + rank-8 projection adapters + decoder.stages |
| B3-Canonical | 81,930,624 | 81,930,624 | 81,930,624 | B1 LoRA + full decoder-channel projections + decoder.stages |
| VoxTell-FullFT | 440,029,541 | 339,292,452 | 339,291,552 | all unique segmentation-network parameters |

B1, DRPA-8 and B3-Canonical trainable groups are all final-logit/loss active in their project-side wrappers. B3 intentionally leaves decoder.transpconvs and decoder.seg_layers frozen, so it is not FullFT.

The three wrapper contracts were instantiated read-only in the canonical voxtell environment. Unique requires_grad counts matched the ledger: B1 294,912; DRPA-8 10,969,696; B3-Canonical 81,930,624. No forward, backward, optimizer step, or state write occurred.

This is formal B3-Canonical. It differs from the historical Data-Capacity screen B3, which additionally enabled decoder-owned modules and reported 83,873,893 trainable parameters; do not merge the two in paper tables.

## Paper rule

Primary parameter-efficiency tables must use the unique configured count and label it explicitly:

- B1: 294,912
- DRPA-8: 10,969,696
- B3-Canonical: 81,930,624
- VoxTell-FullFT: 440,029,541

