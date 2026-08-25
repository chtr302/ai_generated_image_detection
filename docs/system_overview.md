# Tong quan he thong AI Generated Image Detection

## 1. Muc dich

He thong `AI Generated Image Detection` la web app Flask dung de kiem tra anh dau vao la `Anh AI` hay `Anh that`. Nguoi dung co the upload anh tu may hoac nhap URL anh. He thong validate dau vao, hien preview, chay 2 model ONNX trong worker process rieng va tra ve ket qua de nguoi dung de hieu.

| Hang muc | Mo ta |
| --- | --- |
| Dau vao | File `jpg/jpeg/png/webp` hoac URL anh `http/https` |
| Gioi han anh | 10 MB |
| Ket qua | Nhan du doan, diem AI, confidence, thoi gian xu ly, evidence tung model |
| Model | `AI Detection.onnx`, `AI Detection + Physic.onnx` |
| Engine | `onnxruntime` CPU |

## 2. Thanh phan chinh

| Thanh phan | File/thu muc | Vai tro |
| --- | --- | --- |
| Frontend | `src/web/templates/index.html`, `src/web/static/css/styles.css`, `src/web/static/js/app.js` | Giao dien, validate nhanh, preview, goi API, render ket qua |
| Backend Flask | `src/web/app.py` | Route API, validate request, dieu phoi inference |
| Inference service | `src/web/services/inference.py` | Tai anh URL, preprocess, chay model, tong hop ket qua |
| Worker process | Tao trong `inference.py` | Giu ONNX session rieng de giai phong RAM khi tat model |
| Model ONNX | `model/` | Chua 2 model phan loai anh |

## 3. Yeu cau chuc nang

| ID | Yeu cau | Ket qua mong doi |
| --- | --- | --- |
| FR-01 | Kiem tra trang thai model | UI biet co the phan tich hay khong truoc khi gui anh |
| FR-02 | Nhan anh local | Nguoi dung chon file hop le va xem preview |
| FR-03 | Nhan URL anh | He thong validate URL, tai anh va hien preview neu hop le |
| FR-04 | Phan tich anh | Backend preprocess anh va gui job sang worker |
| FR-05 | Tra ket qua de hieu | UI hien nhan, diem AI, confidence, thoi gian, evidence |
| FR-06 | Xem chi tiet model | Nguoi dung xem output tung model trong dialog |
| FR-07 | Bat/tat model | Operator co the tat model va giai phong RAM |

## 4. Luong xu ly tong quat

| Flow | Quy trinh |
| --- | --- |
| Chuan bi | Mo web -> Kiem tra model-state -> Upload/URL -> Validate -> Preview |
| Phan tich | Bam phan tich -> Backend validate -> Worker preprocess -> 2 ONNX -> Chon ket qua -> Render UI |
| Chi tiet/quan ly | Bam chi tiet -> Model detail popup; hoac tat model -> Stop worker -> Khoa UI |

## 5. API chinh

| API | Chuc nang |
| --- | --- |
| `GET /health` | Kiem tra server song |
| `GET /api/model-state` | Lay trang thai model |
| `POST /api/model-state` | Bat/tat model |
| `GET /api/model-detail` | Lay chi tiet model va ket qua |
| `POST /api/load-url` | Kiem tra URL va tao preview |
| `POST /api/analyze` | Phan tich anh |

## 6. Response ket qua mau

```json
{
  "job_id": "job_xxx",
  "input_name": "image.png",
  "input_kind": "upload",
  "final_label": "AI-generated",
  "final_score": 0.93,
  "selected_confidence": 0.94,
  "selected_model": {
    "key": "ai_detection",
    "name": "AI Detection",
    "vote": "AI-generated",
    "confidence": 0.94,
    "prob_real": 0.06,
    "prob_ai": 0.93,
    "raw_output": [0.06, 0.94]
  },
  "decision_status": "selected_highest_confidence",
  "model_scores": [
    {
      "key": "ai_detection",
      "model": "AI Detection",
      "prob_real": 0.06,
      "prob_ai": 0.93,
      "confidence": 0.94,
      "vote": "AI-generated"
    },
    {
      "key": "ai_detection_physic",
      "model": "AI Detection + Physic",
      "prob_real": 0.22,
      "prob_ai": 0.78,
      "confidence": 0.78,
      "vote": "AI-generated"
    }
  ],
  "processing_time_ms": 456,
  "engine": "onnxruntime"
}
```

## 7. Bien moi truong

| Bien moi truong | Mac dinh | Y nghia |
| --- | --- | --- |
| `AIGID_HOST` | `0.0.0.0` | Host server |
| `AIGID_PORT` | `5000` | Port server |
| `AIGID_MODEL_ENABLED` | `1` | Bat/tat model khi khoi dong |
| `AIGID_MODEL_DIR` | `model/` | Thu muc chua ONNX |
| `AIGID_AI_CLASS_INDEX` | `1` | Index class AI |
| `AIGID_WORKER_TIMEOUT_SECONDS` | `120` | Timeout inference |
| `AIGID_MODEL_CONTROL_KEY` | rong | Key bao ve API bat/tat model |

## 8. Gioi han hien tai

| Gioi han | Ghi chu |
| --- | --- |
| Ensemble | Chon model co confidence cao nhat, chua average/voting weight |
| URL | Preview va analyze deu tai anh tu remote; anh remote thay doi thi ket qua co the khac |
| Production | App dang toi uu cho local/LAN, chua cau hinh production server |
| Provider | ONNX chay CPU provider mac dinh |
| File size | Logic anh gioi han 10 MB, Flask request gioi han 64 MB |
