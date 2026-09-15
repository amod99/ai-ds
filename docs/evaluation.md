# Evaluation protocol

All successful experiments use the same shuffled, seeded cross-validation split. Binary
classification defaults to ROC-AUC with F1 and accuracy as secondary metrics; multiclass
classification uses weighted one-vs-rest ROC-AUC. Regression defaults to RMSE with MAE and
R² as secondary metrics.

Scores are calculated only on validation folds. The engine compares ROC-AUC/F1/accuracy/R²
by maximizing them and compares RMSE/MAE by minimizing them. The saved final pipeline is
the validation-selected winner refit once on the complete rows with observed targets.

All imputers, one-hot/frequency/hash encoders, and target-mean encoders live inside the
sklearn pipeline supplied to cross-validation. In particular, target encoding is fitted on
each training fold and applies those learned statistics to its validation fold; validation
targets never participate in their own category statistics.

The run controller stops at the earliest of experiment limit, runtime limit, convergence
patience, failure threshold, or planner stop. These checks are recorded in the report and
SQLite trajectory.
