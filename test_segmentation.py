import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
from tqdm import tqdm
import numpy as np

import mxnet as mx
from mxnet import gluon

import gluoncv
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.data import ms_batchify_fn
from gluoncv.utils.viz import get_color_pallete
from gluoncv.utils.parallel import DataParallelModel

from utils.argument import parse_args_for_segm as parse_args
from utils.custom_load import (get_custom_segm_dataset, \
                               get_pretrained_segmentation_model, resume_checkpoint)

def test(args):
    # output folder
    outdir = 'train_logs/outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # dataset and dataloader
    testset = get_custom_segm_dataset("test", args)
    test_data = gluon.data.DataLoader(
        testset, args.test_batch_size, shuffle=False, last_batch='keep',
        batchify_fn=ms_batchify_fn if args.tta else None, num_workers=args.workers)
    # create network
    if args.model_zoo is not None:
        model = get_pretrained_segmentation_model(args)
        if args.resume is not None:
            resume_checkpoint(model, args)
            print("loading checkpoint from %s for testing"%args.resume)
    else:
        model = get_segmentation_model(model=args.model, dataset=args.dataset, ctx=args.ctx,
                                       backbone=args.backbone, norm_layer=args.norm_layer,
                                       norm_kwargs=args.norm_kwargs, aux=args.aux,
                                       base_size=args.base_size, crop_size=args.crop_size)
        # load pretrained weight
        assert args.resume is not None, '=> Please provide the checkpoint using --resume'
        resume_checkpoint(model, args)
    # print(model)
    if args.tta:
        evaluator = MultiEvalModel(model, testset.num_class, ctx_list=args.ctx, 
                                scales=[0.75, 1.0, 1.25, 1.5, 1.75])
    else:
        evaluator = DataParallelModel(SegEvalModel(model), args.ctx)
    metric = gluoncv.utils.metrics.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)
    for i, (data, dsts) in enumerate(tbar):
        if args.eval:
            if args.tta:
                predicts = [pred[0].expand_dims(0) for pred in evaluator.parallel_forward(data)]
                targets = [target.as_in_context(predicts[0].context).expand_dims(0) \
                        for target in dsts]
            else:
                data = data.astype(args.dtype, copy=False)
                predicts = evaluator(data)
                predicts = [x[0] for x in predicts]
                if args.test_flip:
                    assert (data.ndim ==4)
                    fdata = data.flip(3)
                    fpredicts = evaluator(fdata)
                    predicts = [(x+y[0].flip(3))/2 for x, y in zip(predicts, fpredicts)]
                targets = mx.gluon.utils.split_and_load(dsts, args.ctx, even_split=False)
            metric.update(targets, predicts)
            pixAcc, mIoU = metric.get()
            tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
            mx.nd.waitall()
        else:
            im_paths = dsts
            predicts = evaluator.parallel_forward(data)
            for predict, impath in zip(predicts, im_paths):
                predict = mx.nd.squeeze(mx.nd.argmax(predict[0], 1)).asnumpy() + \
                    testset.pred_offset
                mask = get_color_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(outdir, outname))

if __name__ == "__main__":
    args = parse_args()
    if args.tta:
        args.test_batch_size = args.ngpus
        # args.crop_size = int(args.crop_size*1.75)
    print('Testing model: ', args.resume)
    test(args)
