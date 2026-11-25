# Voice spoofing detector

Frozen XLS-R 300M with LoRA adapters, weighted hidden states, master-token attention pooling and
OC-Softmax with SupCon and evidential auxiliary losses. Trained on ASVspoof 2019 LA (train) only and
evaluated without fine-tuning on ASVspoof 2019 LA, ASVspoof 2021 LA, In-the-Wild and ASVspoof 5.

## Files

| File | Contents |
|---|---|
| `01_config.py` | Configuration: imports, seed, dataset and model paths, hyperparameters, checkpoint paths. |
| `02_augmentation.py` | Augmentations: RawBoost, telephone channel simulation, reverberation and media codecs. |
| `03_data_loading.py` | Data loading: protocols, audio decoding, 4 s cropping, training dataset. |
| `04_model_and_losses.py` | Detector: XLS-R 300M + LoRA, layer weighting, master-token pooling, OC-Softmax, SupCon, evidential head. |
| `05_metrics.py` | Metrics: EER and threshold, per-attack EER, dev-set evaluation. |
| `06_training.py` | Training loop: AdamW, OneCycleLR, FP16, best/final checkpoints, resume. |
| `07_evaluation.py` | Evaluation: test sets (ASVspoof 2019 LA, 2021 LA, In-the-Wild, ASVspoof 5), checkpoint selection. |
| `08_decision.py` | Decision: threshold, calibration, rejection of uncertain decisions, single-file scoring. |
| `run.py` | Entry point: training, evaluation, single-file decision. |

## Usage

1. `pip install -r requirements.txt` (FFmpeg is also required).
2. Set dataset paths and the `facebook/wav2vec2-xls-r-300m` model path in `01_config.py`.
3. Run:

```
python run.py train                     # ~8 h on 2 x NVIDIA T4
python run.py evaluate
python run.py decide path/to/file.flac
```

## Results (EER, %)

| ASVspoof 2019 LA | ASVspoof 2021 LA | In-the-Wild | ASVspoof 5 |
|---|---|---|---|
| 1.20 | 1.75 | 10.71 | 15.73 |
