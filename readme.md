# Classification Pipeline

A PyTorch + MLflow pipeline for training, evaluating, and logging image classification models.  
Supports binary and multi-class datasets with cross-validation, ROC logging, and imbalance handling.

---

## 📂 Project Structure

classification-pipeline/
│
├── main.py # Entry point
├── train.py # Training loop & CV
├── data.py # Dataset & dataloaders
├── models.py # Model building & head reshaping
├── inference.py # Prediction, ROC, occlusion maps
├── utils_mlflow.py # MLflow helpers
├── metrics.py # Early stopping & metrics
├── parse.py # Argparse CLI
├── requirements.txt # Dependencies
└── README.md # Documentation