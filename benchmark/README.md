# Benchmark OpenFake 500

Thu muc nay dung de benchmark 3 model phat hien anh AI tren 500 anh OpenFake can bang.

## Du lieu

Nguon dataset:

```text
https://huggingface.co/datasets/ComplexDataLab/OpenFake
```

Protocol hien tai:

- Chon 500 anh tu manifest co san.
- 250 anh real.
- 250 anh fake.
- Mau duoc chon rai deu theo thu tu manifest de giu tinh dai dien va chay nhanh.
- Khong dung calibration split.

## Model

1. `Hybrid XRayon Physical`
   - File: `model/hybrid_xrayon_physical.onnx`
   - Input: `3 x 224 x 224`

2. `XRayon RGB Only`
   - File: `model/xrayon_rgb_only.onnx`
   - Input: `3 x 256 x 256`

3. `UniversalFakeDetect`
   - Repo: `benchmark/external/UniversalFakeDetect`
   - Paper/repo: `https://github.com/WisconsinAIVision/UniversalFakeDetect`

Voi 2 model ONNX, `fake_score` lay tu output class AI, mac dinh index `1`.

## Threshold

Moi model duoc danh gia tai 3 nguong co dinh:

```text
0.5
0.65
0.8
```

Quy tac:

```text
fake_score >= threshold => fake
fake_score < threshold  => real
```

## Chay benchmark

Kiem tra du lieu:

```powershell
python benchmark\prepare_data.py
```

Chay 3 model va ghi raw score:

```powershell
python benchmark\run_models.py --force --batch-size 8
```

Neu chi muon chay 2 model ONNX:

```powershell
python benchmark\run_models.py --models xrayon --force
```

Tinh metrics o 3 threshold:

```powershell
python benchmark\evaluate_results.py
```

## Ket qua

File chinh:

```text
benchmark/results/openfake_1k/metrics.csv
benchmark/results/openfake_1k/report.json
```

Raw score:

```text
benchmark/results/openfake_1k/predictions/raw_hybrid_xrayon_physical.csv
benchmark/results/openfake_1k/predictions/raw_xrayon_rgb_only.csv
benchmark/results/openfake_1k/predictions/raw_universalfakedetect.csv
```

Prediction theo threshold:

```text
benchmark/results/openfake_1k/predictions/fixed_threshold_predictions_hybrid_xrayon_physical.csv
benchmark/results/openfake_1k/predictions/fixed_threshold_predictions_xrayon_rgb_only.csv
benchmark/results/openfake_1k/predictions/fixed_threshold_predictions_universalfakedetect.csv
```

Notebook `benchmark/view_results.ipynb` chi doc `metrics.csv` va `report.json` de xem bang, bieu do, confusion matrix.
