# Private assets and reproducibility boundary

## Canonical continuous queue

Use `examples/assets.example.json` as a local mapping; real manifests and paths must remain private.

| Prepared-workspace location | Input |
|---|---|
| `VoxTell_weights/voxtell_v1.1/` | Official compatible model directory containing `plans.json` and `fold_0/checkpoint_final.pth` |
| `VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz` | Compatible offline prompt bank, with the original8 prompts available |
| `quality_audit/voxtell_mtl_drpa8_full_data/full_data_train_cases.csv` |971 visits /337 PTIDs |
| `quality_audit/voxtell_mtl_drpa8_full_data/full_data_val_cases.csv` |247 visits /85 PTIDs, disjoint from training |
| `quality_audit/drpa_data_capacity_scaling/manifests/train_100pct.csv` |Same training rows/order as above, used by FullFT |
| `quality_audit/drpa_data_capacity_scaling/manifests/val_100pct_frozen.csv` |Same validation rows/order as above, used by FullFT |

Required manifest columns include `case_id`, `ptid`, `image_path`, `label_path`. Paths must point to authorized existing volumes. The preprocessing convention derives participant identity from the part of `case_id` preceding the first underscore; preserve the original IDs privately and never substitute independently shuffled aliases between the two manifests. The synthetic example is a schema illustration only.

The optional project cache `quality_audit/voxtell_mtl_peft/text_embedding_cache.npz` is not required if the official bank already contains all prompts. The inherited preflight requires all eight prompt embeddings to be present in the official bank and equal across model branches. It does not silently compute a different bank. Checkpoint and embedding licenses/access conditions are separate from the code license; obtain them through the upstream project or your authorized storage.

The author's observed initialization SHA256 was `f45e61c34c56af7b71711a6de54e8414dbc4cb52441003894f3370ed68d8feaa`. The fresh preflight records the supplied initialization hash; for exact author reproduction compare it with this value. The full original frozen cohort cannot be reconstructed from the public synthetic example or sample counts alone.

## Preprocessing and labels

The distributed crop specification is retained exactly:192³ input, training-derived normalized center, original native-reader/restoration behavior. Eight prompts cover bilateral hippocampus, entorhinal cortex, parahippocampal gyrus and amygdala. The original code includes reference-assisted preprocessing; this is material to the interpretation of the experiment. Do not change geometry, label mapping, prompt order, threshold or voxel spacing and continue to claim the same protocol.

## Other experiment families

The original data-capacity,50% ablation, external frozen/few-shot and historical recovery scripts use additional private manifests, source-model checkpoints, hash receipts and parent-result completion gates. These assets are intentionally not published. Preparation allows additional non-code file mappings, but does not disable source/result guards or synthesize successful prior runs. Path relocation changes source hashes; a new independently specified experiment needs its own preflight/receipts rather than reusing author runtime receipts as if the code were unchanged.

## Source transformations

`drpa/source_inventory.json` records source and release hashes per file. Project-side absolute server locations were replaced by explicit workspace/data/interpreter tokens; the preparation command resolves them in a new directory. Historical direct entrypoints have execution guards. Limited benchmark-case identifiers and private compute hostnames in configuration were removed. Cgroup memory detection in the convergence preflight handles both numeric limits and unrestricted hosts. Model tensors, losses, optimizers, prompts, metric formulas and stopping thresholds were not rewritten by these packaging changes.

The VoxTell source files are byte-identical to the collected originals and retain their original attribution. Runtime outputs, patient-level logs, binary tensors, dataset identifiers and credentials are outside the release.
