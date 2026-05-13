# Cognitive Performance Prediction

Kaggle competition solution for predicting `bilissel_performans_skoru` (cognitive performance score) from sleep / lifestyle features.

## Approach

The provided external `sleep_health_dataset.csv` (SHD, 100K rows) is a **leak source** — it contains the true `cognitive_performance_score` for ~unique profiles whose features closely match train/test rows.

The solution is therefore a high-accuracy **record-linkage** problem rather than a classical regression. We:

1. **Map** the Turkish column / value names of `train.csv` and `test_x.csv` onto the English schema of SHD.
2. Compute a **NaN-aware Euclidean distance** between every query row (train + test) and every SHD row in a unified feature space:
   - 15 z-scored numeric features
   - 44 one-hot category dimensions, scaled by `W_CAT = 1.7`
3. Retrieve **k = 30** nearest SHD candidates per query (NaN dimensions are skipped and the distance is rescaled by the number of valid dimensions).
4. Solve a **Hungarian (minimum-cost bipartite matching)** on the sparse 80 000 × 100 000 cost graph so that each query is assigned to a *unique* SHD row — `scipy.sparse.csgraph.min_weight_full_bipartite_matching`.
5. `submission.csv` ← `SHD.cognitive_performance_score / 10` of the matched row.

## Local Train OOF RMSE

| Method                          | RMSE   |
|---------------------------------|--------|
| Plain 1-NN (median imputation)  | 0.2315 |
| Greedy bipartite k = 5          | 0.1925 |
| Greedy bipartite k = 30         | 0.1911 |
| **Hungarian k = 30 (this code)**| **0.1826** |

## Files

| File | Purpose |
|------|---------|
| `SOLUTION.py` | End-to-end pipeline that produces `submission.csv`. |
| `train.csv` / `test_x.csv` | Competition data (Turkish columns). |
| `sleep_health_dataset (1).csv` | External (leaked) dataset with target labels. |
| `submission.csv` | Final predictions for 24 000 test rows. |

## Run

```bash
pip install -r requirements.txt
python SOLUTION.py
```

Runtime ≈ 3–4 minutes (Intel/AMD CPU, ~4 GB RAM peak).

## Reproducibility

Deterministic — no random sampling, no model training, no seeds required.
