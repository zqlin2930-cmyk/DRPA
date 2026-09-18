# VoxTell-FullFT Baseline Definition Specification

Status: `FULLFT_BASELINE_READY` at the definition level; empirical full-resolution gradient smoke test is still a required pre-training gate because the remote host currently reports no available GPU.

## 1. Purpose and naming

This document defines a true full fine-tuning reference for the official VoxTell v1.1 segmentation network. It is deliberately distinct from the project’s B3 arms:

```text
B3-Canonical  = LoRA + selected projection/decoder scopes
VoxTell-FullFT = all unique VoxTell segmentation-network parameters nominally unfrozen
```

The external `Qwen3-Embedding-4B` text encoder remains frozen and is not part of the segmentation checkpoint or optimizer parameter count.

No training was launched by this audit.

## 2. Authoritative construction and dataflow

The exact network construction is in:

- `VoxTell/voxtell/inference/predictor.py:140-175`: reads `plans.json`, instantiates `VoxTellModel`, and loads `fold_0/checkpoint_final.pth`.
- `VoxTell/voxtell/model/voxtell_model.py:131-203`: creates encoder, decoder, bottleneck/text projections, five mask projections, positional buffer, and six-layer Transformer decoder.
- `VoxTell/voxtell/model/voxtell_model.py:221-272`: `encoder -> selected bottleneck -> text projection -> Transformer prompt decoder -> five mask projections -> prompt-wise decoder`.
- `VoxTell/voxtell/model/voxtell_model.py:428-472`: decoder transpose-convolution, skip concatenation, convolutional stages, intermediate mask fusion, final `einsum` readout.
- `VoxTell/voxtell/model/transformer.py:251-265,355-359`: the configured `normalize_before=True` dispatches to `forward_pre`.

The FullFT forward is unchanged. Only `requires_grad`/optimizer scope changes.

## 3. Nominal FullFT scope

The official plans specify six encoder stages `[32,64,128,256,320,320]`, five decoder stages, query dimension `2048`, text embedding dimension `2560`, `num_heads=32`, and five mask-former stages. FullFT nominally unfreezes the following unique parameters:

| Module | Parameters | FullFT role |
|---|---:|---|
| `encoder` | 180,593,152 | MRI feature extraction |
| `project_bottleneck_embed` | 4,853,760 | image-to-query projection |
| `project_text_embed` | 9,441,280 | text-to-query projection |
| `transformer_decoder` | 251,858,944 | prompt/image fusion and query refinement |
| `project_to_decoder_channels` | 71,403,552 | mask-embedding projections |
| `decoder.stages` | 20,464,320 | skip/upsampling feature merge convolutions |
| `decoder.transpconvs` | 1,942,304 | decoder upsampling |
| `decoder.seg_layers` | 965 | intermediate deep-supervision heads retained for checkpoint compatibility |
| **Unique nominal total** | **540,558,277** | **VoxTell-FullFT** |

The checkpoint state dict contains an alias copy under `decoder.encoder` and a `pos_embed` buffer. They are excluded from the unique parameter total; counting raw state-dict entries without deduplication would overcount.

## 4. Gradient-active audit

The source-level current final-loss path has an important distinction from nominal FullFT:

1. `TransformerDecoderLayer` is created with `normalize_before=True` (`voxtell_model.py:192-196`). Its `forward_pre` (`transformer.py:251-265`) calls `norm2 -> multihead_attn -> norm3 -> linear1/linear2`; it does not call `self_attn` or `norm1`.
2. Therefore Transformer `self_attn` and `norm1` contain `100,737,024` parameters that are nominally trainable under FullFT but are not reached by the current final forward path.
3. `decoder.seg_layers` are executed for intermediate outputs (`voxtell_model.py:447-462`), but after output reversal `deep_supervision=False` returns only `seg_outputs[:1]` (`469-472`). Their `965` parameters therefore do not receive final-loss gradients.

Consequently:

```text
nominal requires_grad FullFT                 540,558,277
source-level final-loss gradient-active     439,820,288
```

The empirical gate before any authorized training is one fixed-batch FP32 forward/backward with no `optimizer.step`, recording finite/nonzero gradients and the exact active set. Because `nvidia-smi` currently reports no devices, that empirical smoke test was not executed in this audit. No gradient-active count is claimed as GPU-measured evidence; `439,820,288` is the source-derived expectation.

## 5. Relation to existing B3/DRPA counts

| Arm | Trainable count used in its existing contract | Relation |
|---|---:|---|
| B1 | 294,912 | cross-attention LoRA only |
| DRPA-8 | 10,969,696 | LoRA + rank-8 projection adapters + decoder stages |
| B3-Canonical | 81,930,624 | LoRA + full projections + decoder stages |
| B3 Data–Capacity historical scope | 83,873,893 | additionally includes decoder transpconvs/seg layers |
| **VoxTell-FullFT nominal** | **540,558,277** | all unique network parameters |

Relative nominal trainable multipliers:

- FullFT / B3-Canonical = `6.5978x`;
- FullFT / DRPA-8 = `49.2774x`;
- FullFT / B1 = `1,832.9477x`.

Using only source-derived final-loss-active parameters, FullFT active/B3-Canonical = `5.3682x`. This is a diagnostic ratio, not a replacement for the nominal FullFT definition.

## 6. Resource estimates

All figures below are estimates from parameter counts, not measured FullFT runtime.

### Memory

With FP32 weights and AdamW:

| Item | Calculation | Logical size |
|---|---:|---:|
| unique model weights | `540,558,277 × 4` | 2.162 GB / 2.014 GiB |
| gradients | same | 2.162 GB / 2.014 GiB |
| AdamW moments | `540,558,277 × 2 × 4` | 4.324 GB / 4.027 GiB |
| persistent lower bound | weights + gradients + moments | **8.647 GB / 8.055 GiB** |

The formal B3 FP32 run measured `17,224 MB` peak GPU memory. Adding the extra optimizer/gradient states for previously frozen parameters gives a same-activation lower-bound estimate of approximately `22.7 GB`. Actual FullFT peak is expected to be higher because B3’s wrapper uses `no_grad()` around the encoder and checkpointed decoder logic, whereas FullFT must retain encoder activations for backward. Exact peak must be measured on the target GPU; this audit does not claim a measured FullFT peak.

Qwen is frozen and external. If precomputed embeddings are used, it need not be resident on the training GPU. Loading Qwen online would add a separate frozen-model memory cost and is excluded from the FullFT segmentation estimate.

### Checkpoint

- Unique FullFT model parameters at FP32: about `2.162 GB` logical payload.
- A raw official network state dict also contains the shared `decoder.encoder` alias and `pos_embed`; its logical serialized tensor payload is `2.899 GB` before archive/compression behavior.
- FullFT model plus two FP32 AdamW moments is about `6.487 GB` logical using unique weights, or about `7.223 GB` if a saved state dict repeats the alias/buffer payload. Actual `torch.save` size depends on serialization and compression; it must be measured after an authorized pilot checkpoint.

## 7. Pre-registered LR stability pilot

This is a protocol design only; it was not run.

- Fixed dataset: the existing 10% manifest (`99` visits / the manifest’s fixed PTID subset), with the same preprocessing, eight prompt order, seed, optimizer family, weight decay, batch contract, and validation manifest as the B3/DRPA screen.
- Fixed short steps: the run must declare its finite pilot step count in the config before launch; no final performance-based retuning is allowed.
- Candidate global LR values: `5e-6`, `1e-5`, `2e-5`; no additional values.
- Selection criterion: finite loss, finite gradients and parameters, no OOM, no unexplained foreground explosion, and bounded update norms. Validation Dice is not used to select LR.
- Recommended first candidate: `1e-5`, because it is the canonical full-spatial adaptation scale; the neighboring two values are stability references only.
- If more than one candidate is stable, retain `1e-5` as the predeclared primary rather than choosing by validation score.

## 8. 10% versus 100% expected compute

Under the existing fixed `6000 optimizer-step` contract:

| Training fraction | Visits | Optimizer steps | Exposure-equivalent epochs | Prompt-decoder passes* |
|---|---:|---:|---:|---:|
| 10% | 99 | 6,000 | 60.61 | 48,000 |
| 100% | 971 | 6,000 | 6.18 | 48,000 |

`*` Eight prompt-wise decoder passes per optimizer step; the image encoder is called once per visit forward in the canonical path.

Thus raw optimizer-step compute is nominally 1:1 between 10% and 100% under fixed steps. The 10% arm repeatedly cycles through a much smaller cohort; 100% covers substantially more unique visits. A matched-epoch 100% run would require about `9.81x` the optimizer steps of the 10% run and is a different experiment.

For wall-clock planning, the formal B3 run’s measured median step time was approximately `2.31 s`, implying about `3.85 h` for 6000 steps at that B3 configuration. FullFT will be slower because encoder backward activations are retained; no defensible FullFT sec/step was measured here.

## 9. Launch gates and status

Before any FullFT training is authorized, the runner must:

1. instantiate the official network from `plans.json` and `checkpoint_final.pth`;
2. print the complete nominal trainable module list and deduplicated count;
3. run one fixed-batch FP32 forward/backward with no `optimizer.step`;
4. record finite/nonzero gradient status, dead parameters, and peak memory;
5. save no checkpoint during the smoke test;
6. use the pre-registered LR pilot without looking at final validation performance.

Current result: `FULLFT_FP32_CHECKPOINTING_OOM`. The FP32 runtime gate and the activation/gradient-checkpointing gate both failed before backward completion. No training started; no LR pilot or formal FullFT run is claimed.
