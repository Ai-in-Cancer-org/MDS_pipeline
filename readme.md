# MDS Pipeline

A PyTorch + MLflow pipeline for training, evaluating, and logging image classification models.  
Supports binary and multi-class datasets with cross-validation, ROC logging, and imbalance handling.

---

---

## ⚙️ Features
- ✅ Training with ResNet, DenseNet, EfficientNet, MobileNet, ConvNeXt, RegNet
- ✅ Binary / multi-class classification
- ✅ Cross-validation (`--cv`)
- ✅ MLflow experiment logging
- ✅ ROC curve plotting and CSV export
- ✅ Early stopping
- ✅ Class imbalance handling (oversampling)

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/classification-pipeline.git
cd classification-pipeline
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data

### 4. Run Training
```bash
python main.py \
    --data-paths train.csv test.csv \
    --imgs-path /path/to/images \
    --experiment-name MyExperiment \
    --run_name resnet50_test \
    --model resnet50 \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001 \
    --log_model --log_roc \
    --random-crop-scale 0.7 1.0 \
    --color-jitter 0.3 0.3 0.3 0.2 
```

### 5. Test only
```bash
python main.py \
    --data-paths train.csv test.csv \
    --imgs-path /path/to/images \
    --model resnet50 \
    --test-only
```

### 6. MLFlow tracking
```bash
mlflow ui --backend-store-uri ./mlruns
```







