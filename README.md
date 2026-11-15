# Corn Kernel Mold Detection System (SVM + GLCM/HSV)

Web-based computer vision application for detecting mold contamination on corn kernels using **Otsu + CLAHE Watershed segmentation**, **K-Means fungal lesion isolation**, **GLCM texture + HSV color feature extraction**, **3-layer novelty detection**, and an **RBF-Kernel Support Vector Machine (SVM)** classifier.

Built with **FastAPI**, **OpenCV**, **scikit-learn**, **scikit-image**, and modular **ES6 Vanilla JavaScript**.

---

## Key Features

- **Image Upload & Live Webcam Detection**
  - Drag-and-drop photo upload for single close-ups or multi-kernel piles across dark, navy-blue, and white backgrounds.
  - Real-time webcam detection (~2.5 FPS lightweight live preview) with automatic full-pipeline analysis on camera stop.
- **5-Stage Interactive Pipeline Inspector**
  - Inspect every stage sequentially (*Step-by-Step View*) or expanded (*Show All Stages*):
    1. Input & ROI Standardization (`224×224` centered crop)
    2. Preprocessing (Grayscale, `5×5` Gaussian blur, and intensity histogram)
    3. Segmentation (Polarity-aware Otsu + CLAHE Watershed & `K=3` HSV K-Means lesion mask)
    4. Feature Extraction (4 GLCM + 3 background-compensated HSV features & 7D Z-score chart)
    5. Novelty Gate & RBF-SVM Hyperplane Evaluation
  - Clickable kernel cards, mobile pagers, full feature matrix table, and CSV feature export.
- **Multi-Layer Novelty Detection (`bukan_jagung`)**
  - Rejects non-corn objects before SVM classification using physical color/warmth gates, morphological constraints, and 5-subcluster Mahalanobis distance in 7D scaled space.
  - Applies a **Majority Non-Corn Rule** and batch quality grading thresholds:
    - `0%` contamination -> *Excellent Quality*
    - `<= 15%` contamination -> *Standard Quality (Light Sorting Required)*
    - `> 15%` contamination -> *Poor Quality (Mold Contaminated / Unfit)*
- **Crowdsourcing & Continuous Training Dashboard**
  - Automatically queues valid user detections to `data/staging/` (deduplicated via pixel hashes) with optional user correctness feedback.
  - Password-protected admin panel for curating queued images (`approve` / `reject`), running incremental SVM re-training (`GridSearchCV` + Mahalanobis stats), and rolling back to previous model backups.

---

## Detection Pipeline

```text
Input Image
   │
   ▼  [1. ROI Standardization]    auto_resize_and_crop -> 224×224 BGR
   ▼  [2. Preprocessing]          Grayscale + 5×5 Gaussian Blur + 128×128 patch
   ▼  [3. Segmentation]           Warmth-guided Otsu + CLAHE Watershed -> Kernel ROIs
   │                              HSV K-Means (K=3) -> Fungal lesion mask & % mold area
   ▼  [4. Feature Extraction]     GLCM (Contrast, Correlation, Energy, Homogeneity)
   │                              HSV (Mean Hue, Saturation, Value)
   ▼  [5. Novelty Gate & SVM]     Layer 1-3 check_is_corn -> StandardScaler -> RBF-SVM
   │
   ▼
Batch Quality Summary + Annotated Visuals + JSON/CSV Results
```

---

## Project Structure

```text
root/
├── app.py                        # FastAPI server, API routes, & base64 response formatting
├── compute_outlier_stats.py      # Standalone script to rebuild Mahalanobis outlier stats
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for environment variables (ADMIN_PASSWORD)
├── src/
│   ├── features.py               # Preprocessing & GLCM/HSV feature extraction
│   ├── segmentation.py           # ROI crop, Otsu+Watershed segmentation, K-Means mold isolation
│   ├── novelty.py                # 3-layer novelty detection & decision calibration
│   ├── pipeline.py               # CornDiseasePipeline facade (used by app & notebooks)
│   └── crowdsource/
│       ├── __init__.py           # Public crowdsourcing package exports
│       ├── auth.py               # Token-based admin authentication
│       ├── storage.py            # Staging queue, hash deduplication, & registry management
│       └── retrain.py            # Incremental SVM re-training, outlier stats, & rollback
├── templates/
│   └── index.html                # Main single-page UI & admin dashboard modal/panel
├── static/
│   ├── style.css                 # Application styling & responsive layout
│   └── js/
│       ├── app.js                # Entry point, upload handling, & summary rendering
│       ├── webcam.js             # Real-time webcam capture & live frame detection
│       ├── inspector.js          # 5-stage pipeline inspector, charts, & CSV export
│       └── admin.js              # Staging feedback notice & admin curation/retrain UI
├── models/
│   ├── svm_model_v1.joblib       # Trained RBF-SVM classifier
│   ├── scaler.joblib             # Fitted StandardScaler (7 features)
│   └── outlier_stats.joblib      # Class & 5-subcluster Mahalanobis thresholds
├── notebooks/                    # Research & training notebooks (01 to 07)
└── scripts/
    └── run_all_notebooks.py      # Batch runner for notebooks 01–07
```

---

## Quick Start

### 1. Clone & Create Virtual Environment

```bash
git clone <your-repo-url>
cd jagung-svm
python -m venv venv
```

Activate the virtual environment:

- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (Optional)

Copy `.env.example` to `.env` and set `ADMIN_PASSWORD` if you want to enable the admin curation & re-training panel:

```env
ADMIN_PASSWORD=your_secure_password
```

> If `ADMIN_PASSWORD` is not set, kernel detection via upload and webcam still works normally, while admin login endpoints return `503`.

### 4. Run the Web Application

```bash
python app.py
```

Or run Uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Open **`http://localhost:8080`** in your browser.

> **Note on Webcam Access:** Modern browsers allow camera access (`navigator.mediaDevices.getUserMedia`) on `http://localhost` / `127.0.0.1` or `https://` origins. Accessing the app from another device over plain `http://<LAN-IP>` requires HTTPS or browser origin flags.

---

## API Overview

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Render web interface |
| `POST` | `/api/detect/upload` | Run full 5-stage detection on uploaded image file |
| `POST` | `/api/detect/frame` | Lightweight live webcam frame detection (skips K-Means) |
| `POST` | `/api/detect/frame/full` | Full analysis on the final captured webcam frame |
| `POST` | `/api/staging/feedback` | Submit optional user feedback on a queued detection |
| `POST` | `/api/admin/login` | Authenticate admin and issue session token |
| `GET` | `/api/admin/stats` | Dataset counts, staging queue stats, model meta, & history |
| `GET` | `/api/admin/staging` | List queued staging items (`sehat`, `terkontaminasi`, `campuran`) |
| `POST` | `/api/admin/staging/approve` | Approve & move image into `data/raw/<label>/` |
| `POST` | `/api/admin/staging/reject` | Delete image from staging queue |
| `POST` | `/api/admin/retrain` | Trigger background incremental SVM re-training |
| `GET` | `/api/admin/retrain/status` | Poll background re-training progress & evaluation metrics |
| `POST` | `/api/admin/rollback` | Restore previous backed-up model version & manifests |

---

## Research Notebooks & Utility Scripts

The `notebooks/` directory contains the full experimental workflow:

1. `01_eksplorasi_dataset.ipynb` — Dataset inspection & standardization requirements
2. `02_segmentasi_otsu_kmeans.ipynb` — Otsu + Watershed & K-Means segmentation
3. `03_preprocessing.ipynb` — ROI cropping, grayscale conversion, & Gaussian smoothing
4. `04_ekstraksi_fitur.ipynb` — GLCM & background-compensated HSV feature extraction
5. `05_feature_scaling.ipynb` — Z-score standardization (`StandardScaler`)
6. `06_svm_training.ipynb` — RBF-SVM hyperparameter tuning & training
7. `07_evaluasi.ipynb` — Model evaluation & end-to-end novelty-gate testing

Utility scripts:
- **Re-execute all notebooks:** `python scripts/run_all_notebooks.py`
- **Recompute novelty detection statistics:** `python compute_outlier_stats.py`

> Raw/processed datasets (`data/`), model backup history (`models/history/`), virtual environments (`venv/`), and `.env` are excluded from Git via `.gitignore`.
