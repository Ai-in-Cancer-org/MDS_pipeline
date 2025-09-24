import os, tempfile
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
import mlflow
from utils_mlflow import retry

def classification_model_predict(model, img, input_size, device):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
        transforms.Resize(tuple(input_size)),
        transforms.ToTensor(),
        normalize])
    model.eval()
    img_tensor = img_transforms(img).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor.to(device))
    return prediction

def calc_log_ROC(model, data_loader, device, classes, cv_str, run_name):
    model.eval()
    result = []
    targets = []
    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            m = torch.nn.Softmax(dim=1)
            output = m(output)
            result.append(output.cpu())
            targets.append(target.cpu())
    result = torch.cat(result)
    targets = torch.cat(targets)
    y = targets.numpy()
    scores = result[:, -1].numpy()
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=len(classes)-1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(loc="lower right")

    output_directory = f"output_roc/{run_name}"
    os.makedirs(output_directory, exist_ok=True)
    savepath = os.path.join(output_directory,f'{cv_str}_ROC.png')
    plt.savefig(savepath, dpi=1000)

    roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    csv_path = os.path.join(output_directory, f'{cv_str}_ROC.csv')
    roc_data.to_csv(csv_path, index=False)

    retry(lambda: mlflow.log_artifact(savepath, artifact_path=f"{cv_str}_ROC_Curve"), 5)
    retry(lambda: mlflow.log_artifact(csv_path, artifact_path=f"{cv_str}_ROC_Data"), 5)

    print("ROC saved:", savepath)
