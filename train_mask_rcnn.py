"""Train Mask RCNN end to end."""
import os

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import logging
import time
import numpy as np
import cv2
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import MaskRCNNDefaultTrainTransform, \
    MaskRCNNDefaultValTransform
from gluoncv.utils.metrics.coco_instance import COCOInstanceMetric

from utils.metric import RCNNAccMetric, RCNNL1LossMetric, RPNAccMetric, RPNL1LossMetric, \
    MaskAccMetric, MaskFGAccMetric
from utils.argument import parse_args

def get_dataset(dataset, args):
    if dataset.lower() == 'coco':
        train_dataset = gdata.COCOInstance(splits='instances_train2017')
        val_dataset = gdata.COCOInstance(splits='instances_val2017', skip_empty=False)
        val_metric = COCOInstanceMetric(val_dataset, args.save_prefix + '_eval', cleanup=True)
    elif dataset.lower() == 'imaterialist':
        from utils.iMaterialistDataset import COCOiMaterialist
        train_dataset = COCOiMaterialist(root='datasets/imaterialist/', 
                                         splits='rle_instances_train')
        val_dataset = COCOiMaterialist(root='datasets/imaterialist/', 
                                         splits='resize_rle_instances_val', skip_empty=False)
        val_metric = COCOInstanceMetric(val_dataset, args.save_prefix + '_eval', cleanup=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, train_transform, val_transform, batch_size,
                   num_workers, multi_stage):
    """Get dataloader."""
    # allow different shapes in same batch
    train_bfn = batchify.Tuple(*[batchify.Append() for _ in range(6)])
    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(train_transform(net.short, net.max_size, net, ashape=net.ashape,
                                                multi_stage=multi_stage)),
        batch_size, True, batchify_fn=train_bfn, last_batch='rollover', num_workers=num_workers)
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(2)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(val_transform(net.short, net.max_size)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric, args):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    if not args.disable_hybridization:
        net.hybridize(static_alloc=args.static_alloc)
    for ib, batch in enumerate(val_data):
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        det_masks = []
        det_infos = []
        for x, im_info in zip(*batch):
            # get prediction results
            ids, scores, bboxes, masks = net(x)
            det_bboxes.append(clipper(bboxes, x))
            det_ids.append(ids)
            det_scores.append(scores)
            det_masks.append(masks)
            det_infos.append(im_info)
        # update metric
        for det_bbox, det_id, det_score, det_mask, det_info in zip(det_bboxes, det_ids, det_scores,
                                                                   det_masks, det_infos):
            for i in range(det_info.shape[0]):
                # numpy everything
                det_bbox = det_bbox[i].asnumpy()
                det_id = det_id[i].asnumpy()
                det_score = det_score[i].asnumpy()
                det_mask = det_mask[i].asnumpy()
                det_info = det_info[i].asnumpy()
                # filter by conf threshold
                im_height, im_width, im_scale = det_info
                valid = np.where(((det_id >= 0) & (det_score >= 0.001)))[0]
                det_id = det_id[valid]
                det_score = det_score[valid]
                det_bbox = det_bbox[valid] / im_scale
                det_mask = det_mask[valid]
                # fill full mask
                im_height, im_width = int(round(im_height / im_scale)), int(
                    round(im_width / im_scale))
                full_masks = []
                for bbox, mask in zip(det_bbox, det_mask):
                    mask = gdata.transforms.mask.fill(mask, bbox, (im_width, im_height))
                    if args.dataset.lower() == 'imaterialist':
                        # compute metric at size (512, 512)
                        mask = cv2.resize(mask, (512, 512), cv2.INTER_NEAREST)
                    full_masks.append(mask)
                full_masks = np.array(full_masks)
                eval_metric.update(det_bbox, det_id, det_score, full_masks)
    return eval_metric.get()


def get_lr_at_iter(alpha):
    return 1. / 3. * (1 - alpha) + alpha


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().setattr('grad_req', 'null')
    net.collect_train_params().setattr('grad_req', 'write')
    trainer = gluon.Trainer(
        net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
        'sgd',
        {'learning_rate': args.lr,
         'wd': args.wd,
         'momentum': args.momentum,
         'clip_gradient': 5})

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    lr_warmup = float(args.lr_warmup)  # avoid int division

    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1 / 9.)  # == smoothl1
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1
    rcnn_mask_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    metrics = [mx.metric.Loss('RPN_Conf'),
               mx.metric.Loss('RPN_SmoothL1'),
               mx.metric.Loss('RCNN_CrossEntropy'),
               mx.metric.Loss('RCNN_SmoothL1'),
               mx.metric.Loss('RCNN_Mask')]

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    rcnn_mask_metric = MaskAccMetric()
    rcnn_fgmask_metric = MaskFGAccMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric,
                rcnn_acc_metric, rcnn_bbox_metric,
                rcnn_mask_metric, rcnn_fgmask_metric]

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    if args.verbose:
        logger.info('Trainable parameters:')
        logger.info(net.collect_train_params().keys())
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        for metric in metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        if not args.disable_hybridization:
            net.hybridize(static_alloc=args.static_alloc)
        base_lr = trainer.learning_rate
        for i, batch in enumerate(train_data):
            if epoch == 0 and i <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup)
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info(
                            '[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
                    trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx)
            batch_size = len(batch[0])
            losses = []
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]
            with autograd.record():
                for data, label, gt_mask, rpn_cls_targets, rpn_box_targets, rpn_box_masks in zip(
                        *batch):
                    gt_label = label[:, :, 4:5]
                    gt_box = label[:, :, :4]
                    cls_pred, box_pred, mask_pred, roi, samples, matches, rpn_score, rpn_box, anchors = net(
                        data, gt_box)
                    # losses of rpn
                    rpn_score = rpn_score.squeeze(axis=-1)
                    num_rpn_pos = (rpn_cls_targets >= 0).sum()
                    rpn_loss1 = rpn_cls_loss(rpn_score, rpn_cls_targets,
                                             rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
                    rpn_loss2 = rpn_box_loss(rpn_box, rpn_box_targets,
                                             rpn_box_masks) * rpn_box.size / num_rpn_pos
                    # rpn overall loss, use sum rather than average
                    rpn_loss = rpn_loss1 + rpn_loss2
                    # generate targets for rcnn
                    cls_targets, box_targets, box_masks = net.target_generator(roi, samples,
                                                                               matches, gt_label,
                                                                               gt_box)
                    # losses of rcnn
                    num_rcnn_pos = (cls_targets >= 0).sum()
                    rcnn_loss1 = rcnn_cls_loss(cls_pred, cls_targets,
                                               cls_targets >= 0) * cls_targets.size / \
                                 cls_targets.shape[0] / num_rcnn_pos
                    rcnn_loss2 = rcnn_box_loss(box_pred, box_targets, box_masks) * box_pred.size / \
                                 box_pred.shape[0] / num_rcnn_pos
                    rcnn_loss = rcnn_loss1 + rcnn_loss2
                    # generate targets for mask
                    mask_targets, mask_masks = net.mask_target(roi, gt_mask, matches, cls_targets)
                    # loss of mask
                    mask_loss = rcnn_mask_loss(mask_pred, mask_targets, mask_masks) * \
                                mask_targets.size / mask_targets.shape[0] / mask_masks.sum()
                    # overall losses
                    losses.append(rpn_loss.sum() + rcnn_loss.sum() + mask_loss.sum())
                    metric_losses[0].append(rpn_loss1.sum())
                    metric_losses[1].append(rpn_loss2.sum())
                    metric_losses[2].append(rcnn_loss1.sum())
                    metric_losses[3].append(rcnn_loss2.sum())
                    metric_losses[4].append(mask_loss.sum())
                    add_losses[0].append([[rpn_cls_targets, rpn_cls_targets >= 0], [rpn_score]])
                    add_losses[1].append([[rpn_box_targets, rpn_box_masks], [rpn_box]])
                    add_losses[2].append([[cls_targets], [cls_pred]])
                    add_losses[3].append([[box_targets, box_masks], [box_pred]])
                    add_losses[4].append([[mask_targets, mask_masks], [mask_pred]])
                    add_losses[5].append([[mask_targets, mask_masks], [mask_pred]])
                autograd.backward(losses)
                for metric, record in zip(metrics, metric_losses):
                    metric.update(0, record)
                for metric, records in zip(metrics2, add_losses):
                    for pred in records:
                        metric.update(pred[0], pred[1])
            trainer.step(batch_size)
            # update metrics
            if args.log_interval and not (i + 1) % args.log_interval:
                msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics + metrics2])
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                    epoch, i, args.log_interval * batch_size / (time.time() - btic), msg))
                btic = time.time()

        msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
        logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
            epoch, (time.time() - tic), msg))
        if not (epoch + 1) % args.val_interval:
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric, args)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, logger, best_map, current_map, epoch, args.save_interval, args.save_prefix)


if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx)  # 1 batch per device

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)

    # network
    module_list = []
    if args.use_fpn:
        module_list.append('fpn')
    if args.dataset.lower() == 'imaterialist':
        # pretrained on coco dataset
        net_name = '_'.join(('mask_rcnn', *module_list, args.network, 'coco'))
        # 'mask_rcnn_%s_coco'%(args.network)
        net = get_model(net_name, pretrained=True)
        # reuse the previously trained weights for specified classes
        # {'tie':'tie', 'umbrella':'umbrella', 'bag, wallet':'handbag', 'glove':'baseball glove'}
        net.reset_class(train_dataset.CLASSES, reuse_weights={16: 27, 26: 25, 24: 26, 17: 35})
    else:
        net_name = '_'.join(('mask_rcnn', *module_list, args.network, args.dataset))
        net = get_model(net_name, pretrained_base=True)
    args.save_prefix += net_name
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    net.collect_params().reset_ctx(ctx)

    # modify train transform to support RLE segmentations
    if args.dataset.lower() == 'imaterialist':
        from utils.rle_transform import MaskRCNNTrainTransformRLE
        MaskRCNNDefaultTrainTransform = MaskRCNNTrainTransformRLE
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, MaskRCNNDefaultTrainTransform, MaskRCNNDefaultValTransform,
        args.batch_size, args.num_workers, args.use_fpn)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)
