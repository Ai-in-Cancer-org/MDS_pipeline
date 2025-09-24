import os, time, copy, datetime, shutil, gc
import torch
import torch.nn as nn
import mlflow
import optuna
from pathlib import Path

from data import load_data, output_dir, cv_str
from metrics import EarlyStopping
from inference import calc_log_ROC
from utils_mlflow import retry
import utils as utils

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('labelBalance', utils.SmoothedValue(window_size=1, fmt='{global_avg}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    print(f'#### {datetime.datetime.now().strftime("%d.%m.%y %H:%M")} - Start Epoch {epoch}')
    header = f'Epoch: [{epoch}] Batch:'
    from torchmetrics.classification import F1Score
    f1 = F1Score(num_classes=2, task='binary').to(device)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        target = target.type(torch.LongTensor)
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc2 = utils.accuracy(output, target, topk=(1, 2))
        max_output, _ = output.max(dim=1)
        max_output_binary = (output[:, 1] == max_output).float()

        f1_score = f1(max_output_binary, target)
        batch_size = image.shape[0]

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['f1_score'].update(f1_score.item(), n=batch_size)
        metric_logger.meters['labelBalance'].update(torch.sum(target), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    print(f'#### {datetime.datetime.now().strftime("%d.%m.%y %H:%M")} - End Epoch {epoch}')
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg / 100


def inner_main(args, cv, trial=None, resume=False):
    if utils.is_main_process():
        if args.output_dir:
            if Path(output_dir(args, cv)).exists():
                shutil.rmtree(output_dir(args, cv))
            utils.mkdir(output_dir(args, cv))

    dataset, dataset_test, train_sampler, test_sampler, class_names = load_data(args, cv)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    import torchvision
    from models import reshape_classification_head
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)
    model = reshape_classification_head(model, args, class_names)

    device_arg = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    device = torch.device(device_arg)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    best_model_performance = 0.0
    best_model = copy.deepcopy(model_without_ddp)
    best_model_epoch = 0

    early_stopping = EarlyStopping(patience=10, verbose=True)
    print(f"Total epochs {args.epochs}")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()

        from evaluate import evaluate
        loss, acc, f1_score, recall_per_class, precision_per_class, f1_per_class = evaluate(
            model, criterion, data_loader_test, device=device, class_names=class_names, cv_str=cv_str(cv))

        f1_macro = sum([v for v in f1_per_class.values()]) / len(f1_per_class.values())

        if utils.is_main_process():
            retry(lambda: mlflow.log_metric(f'{cv_str(cv)}Accuracy_Train', train_acc, step=epoch), 5)
            retry(lambda: mlflow.log_metric(f'{cv_str(cv)}Accuracy_Val', acc, step=epoch), 5)
            retry(lambda: mlflow.log_metric(f'{cv_str(cv)}Loss_Train', train_loss, step=epoch), 5)
            retry(lambda: mlflow.log_metric(f'{cv_str(cv)}Loss_Val', loss, step=epoch), 5)
            retry(lambda: mlflow.log_metric(f'{cv_str(cv)}F1_Micro_Val', f1_score, step=epoch), 5)
            retry(lambda: mlflow.log_metric(f'{cv_str(cv)}F1_Macro_Val', f1_macro, step=epoch), 5)
            retry(lambda: mlflow.log_metrics(recall_per_class, step=epoch), 5)
            retry(lambda: mlflow.log_metrics(precision_per_class, step=epoch), 5)

            if trial is not None and not resume:
                trial.report(f1_score, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if f1_score > best_model_performance:
                best_model = copy.deepcopy(model_without_ddp)
                best_model_performance = f1_score
                best_model_epoch = epoch
                if args.output_dir:
                    checkpoint = {
                        'model': best_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'last_epoch': best_model_epoch,
                        'args': args}
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(output_dir(args, cv), 'model.pth'))
                    with open(os.path.join(output_dir(args, cv), 'best_model_epoch.log'), "a") as epoch_log:
                        epoch_log.write(f"Logged model at epoch {best_model_epoch}\n")

    if utils.is_main_process():
        retry(lambda: mlflow.log_metric(f'{cv_str(cv)}F1_Best', best_model_performance, best_model_epoch), 5)
        if args.log_roc:
            calc_log_ROC(best_model, data_loader_test, device=device, classes=class_names, cv_str=cv_str(cv), run_name=args.run_name)
        if args.log_model:
            retry(lambda: mlflow.log_artifact(os.path.join(output_dir(args, cv), 'model.pth'),
                                              artifact_path=f"{cv_str(cv)}pytorch-model"), 5)

    del model, best_model, optimizer, criterion, lr_scheduler
    torch.cuda.empty_cache()
    gc.collect()

    return best_model_performance
