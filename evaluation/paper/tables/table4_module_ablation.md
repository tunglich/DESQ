# Table 4. Module ablation on the TWSE Top-50

Evidence status is row-specific; `reported_only` is a PDF transcription.

| configuration | description | accuracy_pct | precision_pct | recall_pct | f1_pct | return_pct | sharpe | max_drawdown_pct | evidence_status | source_reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full framework | All 6 modules active | 74.2 | 72.6 | 75.8 | 74.2 | 129.0 | 1.53 | -12.8 | reported_only | PDF p16 Table 4 |
| - Auto parameter tuning | Default hyper-parameters (no Keras Tuner) | 72.0 | 70.3 | 73.7 | 72.0 | 122.3 | 1.37 | -14.1 | reported_only | PDF p16 Table 4 |
| - Dynamic Flooding | Static Flooding, b = 0.2 | 67.6 | 65.5 | 69.1 | 67.3 | 111.1 | 1.08 | -18.1 | reported_only | PDF p16 Table 4 |
| - Static Flooding | No Flooding (vanilla loss) | 64.5 | 62.6 | 66.2 | 64.4 | 96.6 | 0.91 | -22.0 | reported_only | PDF p16 Table 4 |
| - Feature group + DES | Single 78-D model, no DES | 62.2 | 60.4 | 64.0 | 62.2 | 74.4 | 0.78 | -25.6 | reported_only | PDF p16 Table 4 |
| Causal Transformer (78D) | Vanilla baseline; no flooding / group / DES | 57.5 | 55.7 | 59.4 | 57.5 | 51.0 | 0.50 | -33.6 | reported_only | PDF p16 Table 4 |
| Benchmark (B&H) | TAIEX |  |  |  |  | 88.07 | 0.65 | -27.7 | reported_only | PDF p16 Table 4 |
