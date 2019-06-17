import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.loss import *
from gluoncv.utils import LRScheduler
from gluoncv.model_zoo.segbase import *
from gluoncv.utils.parallel import *
from utils.custom_load import (get_custom_segm_dataset, \
                               get_pretrained_segmentation_model, resume_checkpoint)

class MixSoftmaxCrossEntropyLossStable(MixSoftmaxCrossEntropyLoss):
    def _aux_forward(self, F, pred1, pred2, label, **kwargs):
        """Compute loss including auxiliary output"""
        loss1 = self.hybrid_forward_logsoftmax(F, pred1, label, **kwargs)
        loss2 = self.hybrid_forward_logsoftmax(F, pred2, label, **kwargs)
        return loss1 + self.aux_weight * loss2

    def hybrid_forward_epsilon(self, F, pred, label):
        """Compute loss (adding epsilon)"""
        epsilon = 1e-12
        softmaxout = F.SoftmaxOutput(
            pred, label.astype(pred.dtype), ignore_label=self._ignore_label,
            multi_output=self._sparse_label,
            use_ignore=True, normalization='valid' if self._size_average else 'null')
        if self._sparse_label:
            loss = -F.pick(F.log(softmaxout+epsilon), label, axis=1, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(F.log(softmaxout+epsilon) * label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
    
    def hybrid_forward_logsoftmax(self, F, pred, label):
        """Compute loss (using `F.log_softmax()`)"""
        pred =  F.log_softmax(pred, 1)
        if self._sparse_label:
            loss = -F.pick(pred, label, axis=1, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred*label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)



class Trainer(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        # dataset and dataloader
        trainset, valset = get_custom_segm_dataset("train", args)
        self.train_data = gluon.data.DataLoader(
            trainset, args.batch_size, shuffle=True, last_batch='rollover',
            num_workers=args.workers)
        self.eval_data = gluon.data.DataLoader(valset, args.test_batch_size,
            last_batch='rollover', num_workers=args.workers)
        # create network
        if args.model_zoo is not None:
            model = get_pretrained_segmentation_model(args)
            self.logger.info("model: %s"%args.model_zoo)
        else:
            model = get_segmentation_model(model=args.model, dataset=args.dataset,
                                           backbone=args.backbone, norm_layer=args.norm_layer,
                                           norm_kwargs=args.norm_kwargs, aux=args.aux,
                                           crop_size=args.crop_size)
        model.cast(args.dtype)
        # print(model)
        self.net = DataParallelModel(model, args.ctx, args.syncbn)
        self.evaluator = DataParallelModel(SegEvalModel(model), args.ctx)
        # resume checkpoint if needed
        if args.resume is not None:
            resume_checkpoint(model, args)
            self.logger.info("loading checkpoint from %s for resuming training"%args.resume)
        # create criterion
        criterion = MixSoftmaxCrossEntropyLossStable(args.aux, aux_weight=args.aux_weight)
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


if __name__ == "__main__":
    from utils.argument import parse_args_for_segm as parse_args
    from utils.logger import build_logger
    from utils.custom_load import make_save_dir, save_checkpoint

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
