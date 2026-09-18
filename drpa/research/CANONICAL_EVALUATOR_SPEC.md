# CANONICAL_EVALUATOR_SPEC

Status: frozen for future paper-line evaluation. Historical output files are unchanged. The selected implementation is quality_audit/voxtell_mtl_drpa8_pilot/evaluate_drpa8.py, which is called by the Data-Capacity runner.

## Mask and geometry

- Apply sigmoid and binarize with probability >= 0.5.
- Use the raw mask as primary (lcc=0). Largest-connected-component output is diagnostic only.
- Use reader-space affine voxel sizes for physical spacing.
- HD95 is reported in millimetres.
- Surface Dice uses a physical 2.0 mm tolerance.

## Metrics

Dice is 2 times intersection divided by the sum of prediction and reference voxels. If both masks are empty, Dice is 1.0.

HD95 is the 95th percentile of bidirectional surface distances computed with distance_transform_edt and the physical spacing. If either mask is empty, HD95 is NaN; aggregation must not silently convert it to zero.

Surface Dice is the fraction of both-direction surface points within 2.0 physical millimetres. If either mask is empty, the historical evaluator returns 0.0.

FP volume is the predicted-positive/reference-negative voxel count times the affine voxel volume divided by 1000, in mL. FN volume is the corresponding reference-positive/predicted-negative quantity. Historical CSVs do not consistently contain FN; future canonical rows must add it without overwriting history.

Connected components use scipy.ndimage.label default 3-D cross connectivity, conventionally 6-connected.

## Aggregation

The historical summary computes macro means over visit-by-ROI rows for overall and named-ROI scopes. Paired analyses must aggregate within PTID first and use PTID-clustered bootstrap CIs and win/tie proportions. Voxel-level significance testing is prohibited.

## Version rule

A future evaluator change requires a versioned side-by-side numerical audit. No historical report may be overwritten to make metrics look comparable.
