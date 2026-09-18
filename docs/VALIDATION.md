# Release validation

The machine-readable release check result is in `RELEASE_AUDIT.json` once packaging is complete. Tests are run on isolated copies, not in the author's active experiment directory.

The verification scope includes:

- Every collected source/config file matches its release inventory; upstream collected VoxTell files retain their original hashes.
- All Python files parse and all JSON files decode.
- Isolated workspace relocation, private-asset links, overwrite/path-traversal protection and source-modification detection.
- Direct historical execution is blocked; missing research assets cannot launch training.
- Static import-name closure against bundled modules, the standard library and listed external dependencies.
- CPU adapter zero-update initialization and nonzero trainable gradient, without loading weights or initializing CUDA.
-26 synthetic cases comparing original and optimized HD95/Surface Dice exactly, including empty/full masks and anisotropic spacing.
- The existing10 practical-plateau tests, including threshold boundaries, deterioration/conflict, a real dummy-process stop and reporting models with9000/10500/12000 endpoints.
- Package/wheel source inclusion and final ZIP extraction/integrity.
- Pattern checks for patient identifiers, common credentials/private keys, private compute hosts, unexpected large files and forbidden runtime/data artifacts.

A pattern scan is not a universal proof that arbitrary secrets are absent. The selected inputs exclude runtime logs, manifests, tensors, model weights, raw Git history and session instructions by construction.

No end-to-end relocated GPU training or patient-level inference is launched for this packaging task. The original FullFT job continues separately. Scientific reproduction still requires the actual authorized assets, environment and full execution of the protocol.
