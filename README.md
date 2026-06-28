# Explainable AI-Generated Image Recognition System
He thong nhan dien va giai thich hinh anh do AI tao ra su dung mo hinh lai Spatial-Frequency CNN-ViT ket hop voi co che giai thich Concept Bottleneck Models (CBM).

---

## 1. Project Directory Structure

```text
ai_generated_image_recognition_system/
├── data/                        # Quan ly du lieu va DataLoader
│   ├── download_scripts/        # Scripts tai dataset (GenImage, CIFAKE)
│   ├── dataloader.py            # PyTorch Dataset
│   └── augmentations.py         # Kyt thuat tang cuong anh (JPEG, Blur)
├── forensics/                   # Bo trich xuat dau vet vat ly (FFT, DWT, Demosaicing)
│   ├── fft_extractor.py
│   ├── dwt_extractor.py
│   ├── demosaicing_extractor.py
│   └── chromatic_aberration.py
├── models/                      # Cau truc va huan luyen mo hinh
│   ├── architectures/           # Hybrid CNN-ViT backbone
│   ├── train.py                 # Script huan luyen
│   ├── evaluate.py              # Đanh gia roc, f1
│   └── loss.py                  # Joint Loss Function
├── xai/                         # Bo giai thich quyet dinh
│   ├── concept_bottleneck.py    # Concept Bottleneck Layer
│   ├── gradcam.py               # Grad-CAM Visualizations
│   └── attribution.py           # Integrated Gradients
├── web/                         # Giao dien chay offline cục bộ
│   ├── app.py                   # Streamlit Frontend
│   └── api/                     # FastAPI Backend
├── tests/                       # Unit tests
├── docs/                        # Tai lieu nghien cuu
└── requirements.txt
```

---

## 2. Team Roles & Pipeline Ownership

*   **Forensics & XAI Lead:** Phụ trách xây dựng các module trích xuất đặc trưng vật lý quang học (`forensics/`) và tích hợp cơ chế giải thích (`xai/` - Concept Bottleneck, Grad-CAM, Integrated Gradients).
*   **ML Engineer:** Phụ trách chuẩn bị dữ liệu (`data/`), thiết kế backbone hybrid và thực hiện huấn luyện, tối ưu hóa mô hình (`models/`).
*   **System Engineer:** Phụ trách xây dựng giao diện ứng dụng web local (`web/`), tích hợp mô hình qua API và thực hiện kiểm thử hệ thống.

---

## 3. Project Roadmap (7 Weeks)

*   **Tuan 1:** Thống nhất sơ đồ luồng dữ liệu và thiết lập môi trường phát triển chung.
*   **Tuan 2:** Hoàn thiện module FFT, DWT và xây dựng DataLoader chuẩn hóa.
*   **Tuan 3:** Hoàn thiện module Demosaicing, Chromatic Aberration và xây dựng cấu trúc mô hình hybrid CNN-ViT.
*   **Tuan 4:** Bàn giao các Forensics Layers để nhúng vào mô hình. Huấn luyện baseline model.
*   **Tuan 5:** Huấn luyện nâng cao với Joint Loss. Xây dựng lớp Concept Bottleneck và tích hợp Grad-CAM.
*   **Tuan 6:** Kiểm thử mô hình trên dữ liệu unseen (Flux, Midjourney v6). Hoàn thiện Web App cục bộ.
*   **Tuan 7:** Đóng gói mã nguồn, viết báo cáo tổng kết và chuẩn bị slide thuyết trình.