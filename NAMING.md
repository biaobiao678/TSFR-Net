# Code naming

The public repository uses names aligned with the TSFR-Net paper:

- `multiscale_refinement.py` / `MultiScaleRefinementPath`: one path of the dual-path multi-scale refinement stage.
- `feature_aggregation.py` / `CrossScaleFeatureAggregation`: base cross-scale aggregation before task routing.
- `tsfr_net.py` / `TSFRNet`: the full TSFR-Net model.
- `DetailToSemanticBranch`: Det2Sem branch.
- `RCMAFusion`: RCMA feature branch.
- `TaskSpecificHeads`: task-specific prediction heads.
- `MassResidualCorrection` and `CarbResidualCorrection`: targeted residual correction branches.

Legacy experiment variant strings are retained for reproducibility. Old checkpoint parameter keys are automatically remapped when loaded.
