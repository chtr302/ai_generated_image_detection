# AI Core Pipeline

Tai lieu nay mo ta nhanh `ai_core` sau khi cap nhat theo data loader moi.

## Hugging Face Defactify dataset

Mac dinh train script co the dung truc tiep dataset:

```text
Rajarshi-Roy-research/Defactify_Image_Dataset
```

Loader doc cac cot Defactify nhu sau:

```text
Image    -> anh PIL
Label_A  -> binary label: 0 real, 1 AI-generated
Label_B  -> source model label de luu metadata
Caption  -> caption/prompt de luu metadata
```

Code dung `datasets.load_dataset(..., streaming=True)`, nen khong tai toan bo dataset 7.51 GB ve may truoc khi train. Can cai dependency:

```bash
python -m pip install datasets pillow torchvision
```

Train nhanh tu Hugging Face streaming:

```bash
python -m src.model.train --hf-dataset Rajarshi-Roy-research/Defactify_Image_Dataset --output-dir outputs/ai_detector --image-size 384 --batch-size 8 --epochs 20 --grad-accum 2 --amp fp16 --nec 10
```

Debug tren it batch truoc khi train that:

```bash
python -m src.model.train --hf-dataset Rajarshi-Roy-research/Defactify_Image_Dataset --max-train-steps 10 --image-size 128 --batch-size 2 --epochs 1 --amp none
```

Neu muon cache/tai split theo cach map-style cu, them `--hf-no-streaming`.

## Local data layout

Data loader van ho tro quet label va split tu ten folder local. Co the dung dataset da chia split:

```text
data/
  train/
    real/
    ai/
  val/
    real/
    ai/
  test/
    real/
    ai/
```

Hoac dataset chua chia split:

```text
data/
  real/
  ai/
```

Neu chua co split, `DataLoaderConfig` tu chia train/val/test theo ti le mac dinh 0.8/0.1/0.1.

## Train single GPU

Mac dinh script dung Hugging Face streaming:

```powershell
.\scripts\train_single_gpu.ps1 -BatchSize 8 -Epochs 20 -Nec 10
```

Dung folder local:

```powershell
.\scripts\train_single_gpu.ps1 -DataRoot data -BatchSize 8 -Epochs 20 -Nec 10
```

Lenh Python local tuong duong:

```bash
python -m src.model.train --data-root data --output-dir outputs/ai_detector --image-size 384 --batch-size 8 --epochs 20 --grad-accum 2 --amp fp16 --nec 10
```

## Train DDP Dual T4

Linux server, dung Hugging Face streaming:

```bash
BATCH_SIZE=8 GRAD_ACCUM=2 EPOCHS=20 NEC=10 bash scripts/train_dual_t4_ddp.sh
```

Linux server, dung folder local:

```bash
DATA_ROOT=data BATCH_SIZE=8 GRAD_ACCUM=2 EPOCHS=20 NEC=10 bash scripts/train_dual_t4_ddp.sh
```

PowerShell:

```powershell
.\scripts\train_dual_t4_ddp.ps1 -BatchSize 8 -GradAccum 2 -Epochs 20 -Nec 10
```

Effective batch size tren Dual T4:

```text
2 GPU * batch_size 8 * grad_accum 2 = 32 images/update
```

## Colab

Notebook/script Colab can cai `datasets` va co the train truc tiep bang streaming:

```bash
bash scripts/train_colab.sh
```

De test nhe truoc:

```bash
MAX_TRAIN_STEPS=10 IMAGE_SIZE=128 BATCH_SIZE=2 EPOCHS=1 AMP=none bash scripts/train_colab.sh
```

## Inference

```bash
python -m src.model.inference --image path/to/image.jpg --checkpoint outputs/ai_detector/best.pt --nec 10
```

## Unit tests

Chay tat ca test bang unittest:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s src/tests -p "test_*.py"
```

Cac nhom test hien co:

```text
src/tests/test_data_utils.py       # label, split, local/HF dataloader
src/tests/test_model_parts.py      # preprocessing, branch, fusion, detector
src/tests/test_forward_shapes.py   # contract output cua detector
src/tests/test_train_runtime.py    # train loop/evaluate/loaders runtime
src/tests/test_train_scripts.py    # train scripts local/HF args
```

## Verification

Da chay thanh cong:

```text
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s src/tests -p "test_*.py"
Ran 17 tests - OK
```

Smoke test voi dataset Hugging Face that chua chay tren may nay vi moi truong local chua cai package `datasets`.
