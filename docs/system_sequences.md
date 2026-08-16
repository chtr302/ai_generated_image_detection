# Sequence diagrams he thong AI Generated Image Detection

## 1. Cach chia sequence

Tai lieu dung 1 sequence tong de nguoi doc nam flow end-to-end, sau do chia thanh 3 sequence chinh:

| Sequence | Muc dich |
| --- | --- |
| Sequence tong | Mo ta toan bo flow tu mo app den hien ket qua |
| Sequence 1 | Khoi dong va chuan bi anh dau vao |
| Sequence 2 | Phan tich anh va tra ket qua |
| Sequence 3 | Xem chi tiet va quan ly model |

## 2. Sequence tong - Toan bo luong he thong

```mermaid
sequenceDiagram
    autonumber
    actor User as Nguoi dung
    participant UI as Frontend
    participant API as Flask API
    participant INF as Inference service
    participant WK as Worker process
    participant M as Detection models

    User->>UI: Mo web app
    UI->>API: GET / va GET /api/model-state
    API-->>UI: Giao dien va trang thai model
    User->>UI: Upload file hoac nhap URL
    UI->>UI: Validate nhanh va tao preview
    opt Neu nhap URL
        UI->>API: POST /api/load-url
        API->>INF: Tai va kiem tra anh tu URL
        INF-->>API: Anh hop le hoac loi
        API-->>UI: Preview URL
    end
    User->>UI: Bam Phan tich
    UI->>API: POST /api/analyze
    API->>INF: Gui anh hop le de phan tich
    INF->>WK: Gui job inference
    WK->>M: Chay 2 model ONNX
    M-->>WK: Score va evidence
    WK-->>INF: Ket qua tung model
    INF-->>API: Ket qua tong hop
    API-->>UI: Nhan, diem AI, confidence, thoi gian
    UI-->>User: Hien ket qua va cho xem chi tiet
```

## 3. Sequence 1 - Khoi dong va chuan bi anh dau vao

```mermaid
sequenceDiagram
    autonumber
    actor User as Nguoi dung
    participant UI as Frontend
    participant API as Flask API
    participant INF as Inference service

    User->>UI: Mo web app
    UI->>API: GET /
    API-->>UI: Tra giao dien HTML/CSS/JS
    UI->>API: GET /api/model-state
    API-->>UI: Trang thai model ON/OFF

    alt Upload file anh
        User->>UI: Chon hoac keo tha file
        UI->>UI: Validate dinh dang va kich thuoc
        UI->>UI: Tao preview bang FileReader
        UI-->>User: Hien Anh preview
    else Nhap URL anh
        User->>UI: Nhap URL
        UI->>UI: Debounce 650 ms va validate http/https
        UI->>API: POST /api/load-url
        API->>INF: validate_url + fetch_url_image
        INF-->>API: Bytes anh hoac loi
        API-->>UI: preview_data_url
        UI-->>User: Hien Anh preview
    end
```

## 4. Sequence 2 - Phan tich anh va tra ket qua

```mermaid
sequenceDiagram
    autonumber
    actor User as Nguoi dung
    participant UI as Frontend
    participant API as Flask API
    participant INF as Inference service
    participant WK as Worker process
    participant M1 as Hybrid ONNX
    participant M2 as RGB ONNX

    User->>UI: Bam Phan tich
    UI->>UI: Khoa input, URL, nut, theme, chi tiet
    UI->>API: POST /api/analyze
    API->>API: Kiem tra model ON va validate dau vao

    opt Neu dau vao la URL
        API->>INF: fetch_url_image(url)
        INF-->>API: Bytes anh
    end

    API->>INF: analyze_upload(input_ref, data)
    INF->>WK: Gui job qua Pipe
    WK->>WK: Decode anh -> RGB -> resize/crop/normalize
    WK->>M1: Chay Hybrid XRayon Physical
    M1-->>WK: score, confidence, vote
    WK->>M2: Chay XRayon RGB Only
    M2-->>WK: score, confidence, vote
    WK->>WK: Chon model co confidence cao nhat
    WK-->>INF: Ket qua tong hop
    INF-->>API: Ket qua phan tich
    API-->>UI: JSON ket qua
    UI-->>User: Hien nhan, diem AI, thoi gian va anh da phan tich
```

## 5. Sequence 3 - Xem chi tiet va quan ly model

```mermaid
sequenceDiagram
    autonumber
    actor User as Nguoi dung / Operator
    participant UI as Frontend
    participant API as Flask API
    participant INF as Inference service
    participant WK as Worker process

    alt Xem chi tiet ket qua
        User->>UI: Bam Chi tiet ket qua
        UI->>API: GET /api/model-detail
        API->>INF: model_detail_payload
        INF-->>API: Trang thai model, worker, file ONNX
        API-->>UI: JSON chi tiet
        UI-->>User: Popup ket qua anh va output tung model
    else Tat model
        User->>API: POST /api/model-state enabled=false
        API->>API: Kiem tra control key neu co
        API->>INF: clear_model_cache_and_worker
        INF->>WK: Gui lenh stop
        WK-->>INF: Dong ONNX session va thoat
        API-->>UI: Model OFF
        UI-->>User: Khoa upload, URL va phan tich
    end
```
