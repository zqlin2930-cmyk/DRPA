# Experiment/source map

All paths in this table are relative to `drpa/research/` (or to a prepared workspace after relocation). Configuration files collected alongside historical code describe those historical runs; the continuous-run `effective_protocol_v3.json` written by a new queue overrides the old fixed-length stopping scope.

| Component | Source | Role / required inputs |
|---|---|---|
| B1 cross-attention LoRA | `quality_audit/voxtell_mtl_peft/voxtell_peft_wrapper.py` | Frozen VoxTell model and prompt bank; 294,912 trainable parameters |
| DRPA-8 | `quality_audit/voxtell_mtl_drpa8_pilot/drpa8_wrapper.py` | Cross-attention LoRA, rank8 projection updates, decoder stages |
| PD-FT / B3-Canonical | `quality_audit/voxtell_full_data_b3_vs_drpa_efficiency/b3_full_data_wrapper.py` | Canonical 81,930,624-parameter baseline |
| FullFT | `scripts/fullft/fullft_runtime_gate.py`, `scripts/fullft/train_fullft_10pct_formal.py` | Full segmentation-network fine tuning; frozen text embeddings |
| Canonical preprocessing | `quality_audit/voxtell_mtl_b1_bilateral_crop/b1_preprocessing.py` | Native-reader/reference-assisted192³ crop; image and label paths |
| Dataset and embeddings | `quality_audit/voxtell_mtl_peft/mtl_grounding_dataset.py`, `text_embedding_cache.py` in the same directory | MRI/label records, offline prompt embeddings |
| Original metrics | `quality_audit/voxtell_mtl_drpa8_pilot/evaluate_drpa8.py` | Raw/LCC evaluation; original geometry and empty-mask conventions |
| Optimized metrics | `convergence_continuous_seed20260809/validation_v2/fast_validation.py` | Same required raw metrics; shared HD95/Surface Dice distances,4 CPU threads; optional LCC/component/max-FP diagnostics omitted |
| Original6000-step DRPA | `drpa8_full_data_train.py` | Historical sparse validation/precision settings; not interchangeable with later FP32 convergence |
| Canonical FP32 baseline | `quality_audit/voxtell_full_data_b3_vs_drpa_efficiency/b3_fp32_formal_runner.py` | Canonical PD-FT training |
| Data-capacity experiments | `quality_audit/canonical_data_capacity_completion/canonical_b3_data_capacity_runner.py` | Author's frozen nested PTID subsets and source artifacts |
| Placement ablation | `quality_audit/placement_ablation_50/placement_wrappers.py`, `placement_runner.py`; `scripts/train/placement_multiseed_50_runner.py` | Original50% subset, component choices, seeds |
| Rank ablation | `scripts/train/rank_ablation_wrapper.py`, `rank_ablation_50_runner.py` | Rank4/8/16 variants and corresponding manifests |
| Multi-seed aggregation | `quality_audit/placement_multiseed_50/final_statistics/finalize_statistics.py`, `scripts/analysis/rank_ablation_finalize.py` | Existing run outputs; no fabricated completion records |
| Frozen pretrained baseline | `scripts/evaluate/run_voxtell_frozen_canonical_baseline.py` | Frozen initialization, private canonical validation manifests |
| External frozen evaluation | `scripts/evaluate/run_oasis_external_frozen_test.py` | OASIS data, original label mapping, trained model assets and frozen external manifest |
| External B1 few-shot | `scripts/oasis_b1_fewshot_20260912/` | Private5-support/15-query split, parent outputs and source guards |
| External FullFT few-shot | `scripts/oasis_fullft_fewshot_20260912/` | Same frozen external protocol plus original FullFT checkpoint |
| Resource benchmarking | `scripts/analysis/pro6000_pipeline_benchmark.py`, `benchmark_x3.py`, `quality_audit/same_gpu_resource_benchmark.py` | Hardware-specific measurements; source weights and benchmark inputs |
| Continuous FP32 training | `convergence_continuous_seed20260809/validation_v2/experiment_v2.py` | Fresh0→9000+ trajectory; use `python -m drpa convergence` for a new three-model queue |
| Stopping and reporting | `convergence_continuous_seed20260809/plateau_v3/policy.py`, `controller.py`, `report_v3.py` | Uniform validation-only practical-plateau decisions; different final endpoints supported |

## Historical-only operational code

`takeover.py`, `watch_boundary.py`, `deploy_v3.py`, `supervise_v2.py`, `supervise_v3.py`, recovery launchers, files with `.before_`/`.pre_` in their names, and convergence `source_snapshot/` are provenance/operations source from an already-running author experiment. They may refer to fixed process identities, old gates, old outputs or source hashes. They are not the entrypoint for a fresh public run. The new distribution queue in `drpa/convergence.py` launches from initialization and uses the inherited controller without adopting old processes.

The older B3 decoder-capacity wrapper is retained because historical evaluation imports it. Its exploratory configuration must not be relabelled as canonical PD-FT.

## Coverage

The source snapshot contains every `.py`, `.sh`, `.yaml`, `.yml` and `.toml` file found in the selected author workspace, excluding Git metadata and caches, plus selected method configs and upstream attribution. Exact coverage is recorded in `drpa/source_inventory.json`. No claim is made that an unavailable historical artifact or code on an uninspected machine was recovered. There are no unresolved **static import names** in the collected source after accounting for standard-library, bundled modules and documented external packages; dynamic state/data prerequisites remain experiment specific.
