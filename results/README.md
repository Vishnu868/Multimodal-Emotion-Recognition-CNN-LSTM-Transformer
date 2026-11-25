# Results

Real-data results on AFFEC (and the synthetic sanity check), copied from `outputs/` after training.

| Folder | Experiment |
|---|---|
| `main_stimulus4_subject/` | **Main result**: EEG + Physio + Eye, 4 emotions, subject-independent 5-fold CV |
| `main_stimulus4_random/` | Same model, subject-dependent random split (5-fold) |
| `ablation_affec_stimulus4_subject/` | Every modality subset, subject-independent 3-fold (`results.md` = table) |
| `stimulus6_subject/`, `felt_va4_subject/` | 6 emotions and felt valence × arousal quadrants |
| `synthetic_random_all/` | Synthetic-data pipeline validation |
| `models/` | Trained checkpoints (`state_dict`, `classes`, `config`) |

Each folder holds `metrics.json` (per-fold and mean ± std accuracy, macro-F1, balanced accuracy, confusion matrix), `classification_report.txt` and `results_fold*.png` (training curves + confusion matrix).

Run a trained model:

```bash
python src/predict.py --checkpoint results/models/fusion_eeg_physio_eye_stimulus4_subject_fold0.pt --data data/processed/affec_T50.npz
```
