import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN network end to end.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--dataset', type=str, default='imaterialist',
                        help='Training dataset. Now support imaterialist and coco.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=str, default='',
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./mask_rcnn_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=str, default='',
                        help='Learning rate, default is 0.00125 for coco single gpu training.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='',
                        help='epochs at which learning rate decays. default is 17,23 for coco.')
    parser.add_argument('--lr-warmup', type=str, default='',
                        help='warmup iterations to adjust learning rate, default is 8000 for coco.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=str, default='',
                        help='Weight decay, default is 1e-4 for coco')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Print helpful debugging info once set.')
    # FPN options
    parser.add_argument('--use-fpn', action='store_true',
                        help='Whether to use feature pyramid network.')

    # Performance options
    parser.add_argument('--disable-hybridization', action='store_true',
                        help='Whether to disable hybridize the entire model. '
                             'Memory usage and speed will decrese.')
    parser.add_argument('--static-alloc', action='store_true',
                        help='Whether to use static memory allocation. Memory usage will increase.')

    args = parser.parse_args()
    args.epochs = int(args.epochs) if args.epochs else 26
    args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '17,23'
    args.lr = float(args.lr) if args.lr else 0.00125
    args.lr_warmup = float(args.lr_warmup) if args.lr_warmup else 8000
    args.wd = float(args.wd) if args.wd else 1e-4
    num_gpus = len(args.gpus.split(','))
    if num_gpus == 1:
        pass
        # args.lr_warmup = -1
    else:
        args.lr *= num_gpus
        args.lr_warmup /= num_gpus
    return args