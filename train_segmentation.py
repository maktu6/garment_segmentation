import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.loss import *
from gluoncv.utils import LRScheduler
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model, get_deeplab_plus_xception_coco
from gluoncv.utils.parallel import *
from gluoncv.data import get_segmentation_dataset
from utils.argument import parse_args_for_segm as parse_args
from utils.logger import build_logger

def replace_conv(block, index, nclass):
    """replace the last conv with a new conv which has `nclass` channels"""
    ctx = list(block[4].params.values())[0].list_ctx()
    in_channels = list(block[4].params.values())[0].shape[1]
    
    new_layer = nn.Conv2D(in_channels=in_channels, 
                      channels=nclass, kernel_size=1)
    new_layer.initialize(ctx=ctx)
    block._children[str(index)] = new_layer

def reset_nclass(model, nclass):
    """reset the number of classes for model"""
    if 'deeplabv3plus' in model.name:
        replace_conv(model.head.block, 8, nclass)
    elif 'deeplabv3' in model.name:
        replace_conv(model.head.block, 4, nclass)
    else:
        raise NotImplementedError("do not support %s"%model.name)
    replace_conv(model.auxlayer.block, 4, nclass)
    model.nclass = nclass

class MixSoftmaxCrossEntropyLossEpsilon(MixSoftmaxCrossEntropyLoss):
    def _aux_forward(self, F, pred1, pred2, label, **kwargs):
        """Compute loss including auxiliary output"""
        loss1 = self.hybrid_forward_epsilon(F, pred1, label, **kwargs)
        loss2 = self.hybrid_forward_epsilon(F, pred2, label, **kwargs)
        return loss1 + self.aux_weight * loss2

    def hybrid_forward_epsilon(self, F, pred, label):
        """Compute loss"""
        softmaxout = F.SoftmaxOutput(
            pred, label.astype(pred.dtype), ignore_label=self._ignore_label,
            multi_output=self._sparse_label,
            use_ignore=True, normalization='valid' if self._size_average else 'null')
        loss = -F.pick(F.log(softmaxout+1e-12), label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class Trainer(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}
        if args.dataset.lower() == 'imaterialist':
            from utils.iMaterialistDataset import iMaterialistSegmentation
            trainset = iMaterialistSegmentation(root='datasets/imaterialist', \
                            split=args.train_split, mode='train', **data_kwargs)
            valset = iMaterialistSegmentation(root='datasets/imaterialist', \
                            split='val', mode='val', **data_kwargs)
        else:
            trainset = get_segmentation_dataset(
                args.dataset, split=args.train_split, mode='train', **data_kwargs)
            valset = get_segmentation_dataset(
                args.dataset, split='val', mode='val', **data_kwargs)
        self.train_data = gluon.data.DataLoader(
            trainset, args.batch_size, shuffle=True, last_batch='rollover',
            num_workers=args.workers)
        self.eval_data = gluon.data.DataLoader(valset, args.test_batch_size,
            last_batch='rollover', num_workers=args.workers)
        # create network
        if args.model_zoo is not None:
            if args.model_zoo == "deeplab_plus_xception_coco":
                model = get_deeplab_plus_xception_coco(pretrained=False)
                self.logger.info("model: %s"%args.model_zoo)
            else:
                model = get_model(args.model_zoo, pretrained=True)
        else:
            model = get_segmentation_model(model=args.model, dataset=args.dataset,
                                           backbone=args.backbone, norm_layer=args.norm_layer,
                                           norm_kwargs=args.norm_kwargs, aux=args.aux,
                                           crop_size=args.crop_size)
        if args.dataset.lower() == 'imaterialist':
            nclass = iMaterialistSegmentation.NUM_CLASS
            reset_nclass(model, nclass)
        model.cast(args.dtype)
        # print(model)
        self.net = DataParallelModel(model, args.ctx, args.syncbn)
        self.evaluator = DataParallelModel(SegEvalModel(model), args.ctx)
        # resume checkpoint if needed
        if args.resume is not None:
            if os.path.isfile(args.resume):
                model.load_parameters(args.resume, ctx=args.ctx)
                self.logger.info("loading checkpoint from %s for resuming training"%args.resume)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'" \
                    .format(args.resume))
        # create criterion
        criterion = MixSoftmaxCrossEntropyLossEpsilon(args.aux, aux_weight=args.aux_weight)
        self.criterion = DataParallelCriterion(criterion, args.ctx, args.syncbn)
        # optimizer and lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr,
                                        nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_data),
                                        power=0.9)
        kv = mx.kv.create(args.kvstore)
        optimizer_params = {'lr_scheduler': self.lr_scheduler,
                            'wd':args.weight_decay,
                            'momentum': args.momentum,
                            'learning_rate': args.lr
                           }
        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True

        if args.no_wd:
            for k, v in self.net.module.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        self.optimizer = gluon.Trainer(self.net.module.collect_params(), 'sgd',
                                       optimizer_params, kvstore = kv)
        # evaluation metrics
        self.metric = gluoncv.utils.metrics.SegmentationMetric(trainset.num_class)
        self.best_mIoU = 0.0

    def training(self, epoch):
        self.is_best = False
        tbar = tqdm(self.train_data)
        train_loss = 0.0
        alpha = 0.2
        for i, (data, target) in enumerate(tbar):
            with autograd.record(True):
                outputs = self.net(data.astype(args.dtype, copy=False))
                losses = self.criterion(outputs, target)
                mx.nd.waitall()
                autograd.backward(losses)
            self.optimizer.step(self.args.batch_size)

            current_loss = 0.0
            for loss in losses:
                current_loss += np.mean(loss.asnumpy()) / len(losses)
            # check whether nan in losses
            if np.isnan(current_loss):
                self.logger.warning("Raise nan,Batch %d, losses: %s"%(i, losses))
            else:
                train_loss += current_loss
            tbar.set_description('Epoch %d, mloss %.3f'%\
                (epoch, train_loss/(i+1)))
            if self.args.log_interval and not (i + 1) % self.args.log_interval:
                self.logger.info('Epoch %d, Batch %d, current loss %.3f, mean loss %.3f'%\
                (epoch, i, current_loss, train_loss/(i+1)))

            mx.nd.waitall()

    def validation(self, epoch):
        #total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        self.metric.reset()
        tbar = tqdm(self.eval_data)
        for i, (data, target) in enumerate(tbar):
            outputs = self.evaluator(data.astype(args.dtype, copy=False))
            outputs = [x[0] for x in outputs]
            targets = mx.gluon.utils.split_and_load(target, args.ctx, even_split=False)
            self.metric.update(targets, outputs)
            pixAcc, mIoU = self.metric.get()
            tbar.set_description('Epoch %d, pixAcc: %.3f, mIoU: %.3f'%\
                (epoch, pixAcc, mIoU))
            mx.nd.waitall()
        if mIoU>self.best_mIoU:
            self.best_mIoU = mIoU
            self.is_best = True # for save checkpoint


def make_save_dir(args):
    directory = "train_logs/%s/%s/%s/" % (args.dataset, args.model, args.checkname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def save_checkpoint(net, directory, is_best=False, epoch=None):
    """Save Checkpoint"""
    if epoch is None:
        filename='checkpoint.params'
    else:
        filename='checkpoint_{}.params'.format(epoch)
    filename = directory + filename
    net.save_parameters(filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.params')


if __name__ == "__main__":
    args = parse_args()
    save_dir = make_save_dir(args)
    logger = build_logger(os.path.join(save_dir, 'train.log'), True) 
    logger.info(args)
    trainer = Trainer(args, logger)
    if args.eval:
        logger.info('Evaluating model: {}'.format(args.resume))
        trainer.validation(args.start_epoch)
    else:
        logger.info('Starting Epoch:{}'.format(args.start_epoch))
        logger.info('Total Epochs: {}'.format(args.epochs))
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val:
                trainer.validation(epoch)
            # save every epoch
            save_checkpoint(trainer.net.module, save_dir, trainer.is_best, epoch)
