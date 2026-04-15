# NeuroScan AI — Alzheimer's MRI Classifier

> Predict-only Flask web app + full DL experiment scripts for Alzheimer's dementia classification.

## Dataset
Extracted from `Data__2_.zip`:
```
Data/
├── Mild Dementia/       (~5,000 images)
├── Moderate Dementia/   (~489 images)
└── Non Demented/        (~24,000 images)
```

## Project Structure
```
neuroscan_final/
│
├── flask_app/
│   ├── app.py                  ← Flask web app (predict-only)
│   ├── templates/
│   │   ├── index.html          ← Upload page
│   │   └── result.html         ← Results + Grad-CAM
│   └── static/uploads/         ← Uploaded images
│
├── models/
│   └── cnn_model.h5            ← Saved after training
│
├── train_scripts/
│   ├── train_model.py                   ← Train CNN ← RUN THIS FIRST
│   └── DL_All_Experiments_Alzheimers.py ← All 8 DL experiments
│
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Extract dataset
Place extracted `Data/` folder at project root:
```
Data/  Mild Dementia/  Moderate Dementia/  Non Demented/
```

### 3. Train
```bash
python train_scripts/train_model.py
```
Saves `models/cnn_model.h5`

### 4. Run web app
```bash
cd flask_app && python app.py
```
Visit http://127.0.0.1:5000

## Run All 8 Experiments
```bash
python train_scripts/DL_All_Experiments_Alzheimers.py
```

> ⚠️ For educational use only. Not a medical diagnostic tool.
