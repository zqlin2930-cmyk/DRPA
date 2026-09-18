#!/usr/bin/env python3
"""Paper-line canonical evaluation entrypoint.

This thin entrypoint delegates to the audited historical evaluator. It never
trains or modifies checkpoints. The default root is backward compatible and
can be overridden with --base-dir or MTL_MODEL_ROOT.
"""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__"))
EVALUATOR_REL = Path("quality_audit/voxtell_mtl_drpa8_pilot/evaluate_drpa8.py")

def contract(root):
    return {
        "evaluator_source": str(root / EVALUATOR_REL),
        "threshold": "sigmoid(logit) >= 0.5",
        "primary_mask": "raw mask (lcc=0); LCC is diagnostic only",
        "hd95": "bidirectional surface distances, reader-space spacing, physical millimetres",
        "surface_dice": "bidirectional surface-distance fraction within physical 2.0 mm",
        "empty_rules": {
            "dice": "1.0 only when both masks are empty",
            "hd95": "NaN if either mask is empty",
            "surface_dice": "0.0 if either mask is empty",
            "max_fp_distance": "0.0 if no FP; NaN if FP exists and GT is empty",
        },
        "fp_volume": "sum(pred & ~gt) * abs(det(affine[:3,:3])) / 1000, mL",
        "fn_volume": "sum(gt & ~pred) * abs(det(affine[:3,:3])) / 1000, mL",
        "components": "scipy.ndimage.label default 3-D cross connectivity (6-connected)",
        "aggregation": "visit/ROI macro metrics; PTID-clustered paired comparisons",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(DEFAULT_ROOT))
    ap.add_argument("--print-contract", action="store_true")
    ap.add_argument("--validation-manifest")
    ap.add_argument("--checkpoint")
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    root = Path(args.base_dir)
    if args.print_contract or not args.validation_manifest:
        print(json.dumps(contract(root), indent=2))
        return
    raise SystemExit("This audit does not execute inference. Use the printed contract in an authorized evaluation run.")

if __name__ == "__main__":
    main()
