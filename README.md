# AI Generated Image Detection

Web app Flask chay local de phat hien anh that hay anh do AI tao ra. App hien tai tich hop truc tiep 2 model ONNX trong thu muc `model/`:

- `hybrid_xrayon_physical.onnx`
- `xrayon_rgb_only.onnx`

Web app hien thi output cua tung model va chon ket qua theo model co confidence lon nhat.

Model duoc chay trong worker process rieng. Khi bat model, app chi mo khoa phan tich; model chua load vao RAM. Khi co request phan tich dau tien, worker moi duoc tao va load 2 ONNX model. Khi tat model, worker process bi dung de RAM model duoc tra lai cho he dieu hanh.

## Cau truc chinh

```text
src/
|-- web/
|   |-- app.py
|   |-- services/
|   |   |-- inference.py
|   |-- templates/
|   |   |-- index.html
|   |-- static/
|       |-- css/styles.css
|       |-- js/app.js
|-- model/
|-- data/
|-- forensics/
|-- xai/
benchmark/
model/
|-- hybrid_xrayon_physical.onnx
|-- xrayon_rgb_only.onnx
```

## Chay web app

```powershell
python -m src.web.app
```

Mo trinh duyet:

```text
http://127.0.0.1:5000
```

## API web

- `GET /`: giao dien upload anh.
- `GET /health`: kiem tra service.
- `GET /api/model-state`: trang thai bat/tat model.
- `GET /api/model-detail`: thong tin 2 model ONNX.
- `POST /api/load-url`: tai preview anh tu URL.
- `POST /api/analyze`: phan tich mot anh upload hoac URL.

App khong con dung mock model, anh mau, sample route, hoac batch demo trong web app.

## Bien moi truong tuy chon

- `AIGID_MODEL_DIR`: thu muc chua 2 file ONNX, mac dinh la `model/`.
- `AIGID_AI_CLASS_INDEX`: index lop AI trong output model, mac dinh `1`.
- `AIGID_MODEL_ENABLED`: dat `0` de tat chuc nang phan tich khi khoi dong.
- `AIGID_MODEL_CONTROL_KEY`: key bao ve API bat/tat model.
- `AIGID_WORKER_TIMEOUT_SECONDS`: thoi gian toi da cho mot lan phan tich, mac dinh `120`.

## Benchmark

Thu muc `benchmark/` duoc giu lai rieng de tiep tuc sua va danh gia sau.
