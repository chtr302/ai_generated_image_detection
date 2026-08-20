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

1. `AI Detection`
   - File: `model/AI Detection.onnx`
   - Input: `3 x 256 x 256`
   - Output: logits 2 lop, lay fake_score bang softmax class AI

2. `AI Detection + Physic`
   - File: `model/AI Detection + Physic.onnx`
   - Input: `3 x 256 x 256`
   - Output: `prob_ai`

3. `UniversalFakeDetect`
   - Repo: `benchmark/external/UniversalFakeDetect`
   - Paper/repo: `https://github.com/WisconsinAIVision/UniversalFakeDetect`

Voi model ONNX 2 lop, class AI mac dinh index `1`.

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

Lan dau clone repo, tai data OpenFake va clone UniversalFakeDetect:

```powershell
python benchmark\setup_benchmark.py
```

Hai thu muc sau duoc tao local va khong can push len git:

```text
benchmark/data
benchmark/external
```

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
python benchmark\run_models.py --models onnx --force
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
benchmark/results/openfake_1k/predictions/raw_ai_detection.csv
benchmark/results/openfake_1k/predictions/raw_ai_detection_physic.csv
benchmark/results/openfake_1k/predictions/raw_universalfakedetect.csv
```

Prediction theo threshold:

```text
benchmark/results/openfake_1k/predictions/fixed_threshold_predictions_ai_detection.csv
benchmark/results/openfake_1k/predictions/fixed_threshold_predictions_ai_detection_physic.csv
benchmark/results/openfake_1k/predictions/fixed_threshold_predictions_universalfakedetect.csv
```

Notebook `benchmark/view_results.ipynb` chi doc `metrics.csv` va `report.json` de xem bang, bieu do, confusion matrix.
