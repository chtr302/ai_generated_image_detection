# AI Core Pipeline

Tai lieu nay mo ta nhanh `ai_core` sau khi cap nhat theo data loader moi.

## Data layout

Data loader moi quet label va split tu ten folder. Co the dung dataset da chia split:

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

```powershell
.\scripts\train_single_gpu.ps1 -DataRoot data -BatchSize 8 -Epochs 20 -Nec 10
```

Lenh Python tuong duong:

```bash
python -m src.model.train --data-root data --output-dir outputs/ai_detector --image-size 384 --batch-size 8 --epochs 20 --grad-accum 2 --amp fp16 --nec 10
```

## Train DDP Dual T4

Linux server:

```bash
DATA_ROOT=data BATCH_SIZE=8 GRAD_ACCUM=2 EPOCHS=20 NEC=10 bash scripts/train_dual_t4_ddp.sh
```

PowerShell:

```powershell
.\scripts\train_dual_t4_ddp.ps1 -DataRoot data -BatchSize 8 -GradAccum 2 -Epochs 20 -Nec 10
```

Effective batch size tren Dual T4:

```text
2 GPU * batch_size 8 * grad_accum 2 = 32 images/update
```

## Inference

```bash
python -m src.model.inference --image path/to/image.jpg --checkpoint outputs/ai_detector/best.pt --nec 10
```

## Unit tests

Chay tat ca test bang unittest:

```bash
python -m unittest discover -s src/tests -v
```

Cac nhom test hien co:

```text
src/tests/test_data_utils.py       # label, split, dataloader
src/tests/test_model_parts.py      # preprocessing, branch, fusion, detector
src/tests/test_forward_shapes.py   # contract output cua detector
src/tests/test_train_runtime.py    # train loop/evaluate/loaders runtime
src/tests/test_train_scripts.py    # script train dung --data-root
```

## Smoke train da kiem tra

Da chay thanh cong train CLI voi dataset tam, 1 epoch, image size 128:

```text
python -m src.model.train --data-root <temp>/data --output-dir <temp>/out --image-size 128 --batch-size 2 --epochs 1 --num-workers 0 --amp none --nec 5
returncode 0
```
