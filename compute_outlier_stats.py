"""
compute_outlier_stats.py

Computes global and 5-subcluster Mahalanobis distance statistics from training
features (`data/manifest_fitur.csv`) for multi-layer novelty detection in
`src/pipeline.py` without requiring a separate third-class dataset.
"""

import joblib
import pandas as pd

from src.crowdsource import DATASET_CLASSES, FEATURE_COLS, compute_outlier_stats


def main() -> None:
    df = pd.read_csv("data/manifest_fitur.csv")
    scaler = joblib.load("models/scaler.joblib")

    X_scaled = scaler.transform(df[FEATURE_COLS].values)
    y = df["label"].values

    class_stats = compute_outlier_stats(X_scaled, y)

    for label in DATASET_CLASSES:
        stats = class_stats[label]
        print(
            f"Class '{label}': threshold={stats['distance_threshold']:.2f}, "
            f"subclusters={len(stats['subclusters'])}, samples={stats['n_samples']}"
        )

    joblib.dump(class_stats, "models/outlier_stats.joblib")
    print("\nDone. Outlier statistics saved to models/outlier_stats.joblib")


if __name__ == "__main__":
    main()
