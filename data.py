import os, time, torch
import pandas as pd
from PIL import Image
from torchsampler import ImbalancedDatasetSampler
import torchvision.transforms as transforms
import mlflow
import tempfile
from utils_mlflow import retry

class CSVDataset(object):
    def __init__(self, root, path_to_csv, transforms, dataset_type='binary', class_names=[]):
        self.root = root
        self.transforms = transforms
        df = pd.read_csv(path_to_csv)
        self.imgs = list(df.iloc[:, 0])
        self.labels = list(df[df.columns[1]])

        if dataset_type == 'binary':
            self.classes = class_names if class_names else ['non_'+df.columns[1], df.columns[1]]
        elif dataset_type == 'multi_class':
            self.classes = class_names if class_names else df.iloc[:, 1].unique()
        else:
            raise ValueError("Invalid dataset_type")

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path)
        target = self.labels[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

class ConvertToRGB:
    def __call__(self, img):
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return img

def log_data_split(args, indices, splitpoint, cv_str=""):
    df = pd.read_csv(args.data_path)
    df_train = df.iloc[indices[:splitpoint],:]
    df_test = df.iloc[indices[splitpoint:],:]
    with tempfile.TemporaryDirectory() as tmpdirname:
        df_train.to_csv(tmpdirname+"/train_set.csv", index=False)
        df_test.to_csv(tmpdirname+"/test_set.csv", index=False)
        mlflow.log_artifacts(tmpdirname, artifact_path=f"{cv_str}data")
        mlflow.log_artifact(args.data_path, artifact_path=f"{cv_str}data")

def output_dir(args, cv):
    return args.output_dir+'/'+args.run_name+'/'+str(cv)

def data_paths(args, cv):
    return [p.format(cv=cv) for p in args.data_paths]

def cv_str(cv):
    return f"cv{cv}_"

def load_data(args, cv):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset_train = CSVDataset(
        args.imgs_path,
        data_paths(args, cv)[0],
        transforms.Compose([
            ConvertToRGB(),
            transforms.RandomResizedCrop(tuple(args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        args.dataset_type,
        args.class_names)

    dataset_test = CSVDataset(
        args.imgs_path,
        data_paths(args, cv)[1],
        transforms.Compose([
            ConvertToRGB(),
            transforms.Resize(tuple(args.input_size)),
            transforms.ToTensor(),
            normalize]),
        args.dataset_type,
        args.class_names)

    class_names = dataset_train.classes

    splitpoint = int(len(dataset_train)*(1-args.test_split))
    indices = torch.randperm(len(dataset_train)).tolist()

    dataset_train = torch.utils.data.Subset(dataset_train, indices[:splitpoint])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[splitpoint:])

    if args.data_path:
        log_data_split(args, indices, splitpoint)
    else:
        retry(lambda: mlflow.log_artifact(data_paths(args, cv)[0], artifact_path=f"{cv_str(cv)}data"), 5)
        retry(lambda: mlflow.log_artifact(data_paths(args, cv)[1], artifact_path=f"{cv_str(cv)}data"), 5)

    train_sampler = ImbalancedDatasetSampler(dataset_train, labels=[y for _,y in dataset_train]) if args.balance_samples else torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler, class_names
