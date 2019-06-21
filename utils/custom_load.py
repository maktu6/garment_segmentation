import os
import shutil
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model, get_deeplab_plus_xception_coco
from gluoncv.data import get_segmentation_dataset
from gluoncv.utils.block import freeze_bn

from .iMaterialistDataset import iMaterialistSegmentation

#--------------------------------dataset---------------------------------------
def get_input_transform():
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    return input_transform

def get_custom_segm_dataset(mode, args):
    """ get custom segmentation dataset for training or testing
    Args:
        mode (str): "train" or "test"
        args: configs from argparse
    Returns:
        if mode is "train", return tuple(trainset, valset)
        if mode is "test", return testset
    """
    # image transform
    input_transform = get_input_transform()
    if mode == "train":
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}
        if args.dataset.lower() == 'imaterialist':
            trainset = iMaterialistSegmentation(root='datasets/imaterialist', \
                            split=args.train_split, mode='train', **data_kwargs)
            valset = iMaterialistSegmentation(root='datasets/imaterialist', \
                            split='val', mode='val', **data_kwargs)
        else:
            trainset = get_segmentation_dataset(
                args.dataset, split=args.train_split, mode='train', **data_kwargs)
            valset = get_segmentation_dataset(
                args.dataset, split='val', mode='val', **data_kwargs)
        return trainset, valset
    elif mode == "test":
        if args.eval:
            split_name, mode_name = 'val', 'testval'
        else:
            # TODO: it seems mode='test' for dataset is not implemented
            split_name, mode_name = 'test', 'test'
        if args.dataset.lower() == 'imaterialist':
            testset = iMaterialistSegmentation(root='datasets/imaterialist', \
                            split=split_name, mode=mode_name, transform=input_transform)
        else:
            testset = get_segmentation_dataset(
                args.dataset, split=split_name, mode=mode_name, transform=input_transform)

        return testset
    else:
        raise NotImplementedError("mode = %s is not supported"%mode)

#--------------------------------model---------------------------------------
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

def get_pretrained_segmentation_model(args, ctx=None):
    if ctx is None:
        ctx = args.ctx
    if args.model_zoo == "deeplab_plus_xception_coco":
        # TODO: waiting for the coco pretrained model
        model = get_deeplab_plus_xception_coco(pretrained=False, ctx=ctx)
    else:
        model = get_model(args.model_zoo, pretrained=True, ctx=ctx)
    # reset nclass
    if args.dataset.lower() == 'imaterialist':
        nclass = iMaterialistSegmentation.NUM_CLASS
        reset_nclass(model, nclass)
    if args.freeze_bn:
        freeze_bn(model)
    return model

#--------------------------------checkpoint---------------------------------------
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

def resume_checkpoint(model, args):
    if os.path.isfile(args.resume):
        model.load_parameters(args.resume, ctx=args.ctx)
    else:
        raise RuntimeError("=> no checkpoint found at '{}'" \
            .format(args.resume))
