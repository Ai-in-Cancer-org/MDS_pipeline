import argparse

class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-path', help='csv with dataset')
    group.add_argument('--data-paths', nargs=2, help='train/test csvs')
    parser.add_argument('--imgs-path', default='/', help='Root folder with images')
    parser.add_argument('--run_name', default='default')
    parser.add_argument('--experiment-name')
    parser.add_argument('--input-size', type=int, nargs=2, default=[150,150])
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('-b','--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('-j','--workers', default=16, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay','--wd', default=1e-4, type=float, dest='weight_decay')
    parser.add_argument('--lr-step-size', default=10, type=int)
    parser.add_argument('--lr-gamma', default=0.1, type=float)
    parser.add_argument('--print-freq', default=10, type=int)
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--run-uuid', dest="run_uuid", default=None)
    parser.add_argument('--log_model', action='store_true')
    parser.add_argument('--log_roc', action='store_true')
    parser.add_argument('--balance_samples', action='store_true')
    parser.add_argument('--balance_val_set', action='store_true')
    parser.add_argument('--dataset-type', default="binary")
    parser.add_argument('--class_names', nargs="+", default=[])
    parser.add_argument('--cv','--cross-validation', dest="nr_cv", default=1, type=int)
    parser.add_argument('--cv-start', dest="cv_start", default=0, type=int)
    parser.add_argument("--user_attr", dest="user_attr", action=StoreDictKeyPair, default={})
    parser.add_argument("--system_attr", dest="system_attr", action=StoreDictKeyPair, default={})
    parser.add_argument('--random-crop-scale', type=float, nargs=2, default=[0.8, 1.0], help='scale range for RandomResizedCrop (min max)')
    parser.add_argument('--color-jitter', type=float, nargs=4, default=[0.2, 0.2, 0.2, 0.1], help='Color jitter params (brightness contrast saturation hue)')
    parser.add_argument('--affine-degrees', type=float, default=15, help='Max rotation for RandomAffine')
    parser.add_argument('--affine-translate', type=float, nargs=2, default=[0.1, 0.1], help='Translation fraction (h, w)')
    parser.add_argument('--affine-scale', type=float, nargs=2, default=[0.9, 1.1], help='Scaling range')
    parser.add_argument('--affine-shear', type=float, default=10, help='Shear angle')
    
    
    args = parser.parse_args()
    return args
