from random import randint
import pycocotools.mask as cocomask
import cv2

import mxnet as mx
from gluoncv.data.transforms import image as timage
from gluoncv.data.transforms import bbox as tbbox
from gluoncv.data.transforms import mask as tmask
from gluoncv.data.transforms.presets.rcnn import MaskRCNNDefaultTrainTransform, \
    MaskRCNNDefaultValTransform
# import torch

class MaskRCNNTrainTransformRLE(MaskRCNNDefaultTrainTransform):
    """RLE instead of polygon for segmentation"""
    def __call__(self, src, label, segm):
        """Apply transform to training image/label."""
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        if self._random_resize:
            short = randint(self._short[0], self._short[1])
        else:
            short = self._short
        img = timage.resize_short_within(src, short, self._max_size, interp=1)
        bbox = tbbox.resize(label, (w, h), (img.shape[1], img.shape[0]))
        # segm = [tmask.resize(polys, (w, h), (img.shape[1], img.shape[0])) for polys in segm]

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])
        # segm = [tmask.flip(polys, (w, h), flip_x=flips[0]) for polys in segm]

        # gt_masks (n, im_height, im_width) of uint8 -> float32 (cannot take uint8)
        # masks = [mx.nd.array(tmask.to_mask(polys, (w, h))) for polys in segm]
        masks = cocomask.decode(segm) # hxwxn
        mask_list = []
        for i in range(masks.shape[-1]):
            mask = cv2.resize(masks[:,:,i], (img.shape[1],img.shape[0]),
                interpolation=cv2.INTER_NEAREST)
            mask_list.append(mx.nd.array(mask))
       # n * (im_height, im_width) -> (n, im_height, im_width)
        masks = mx.nd.stack(*mask_list, axis=0)
        if flips[0]:
            masks = mx.nd.flip(masks, axis=2)
        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype(img.dtype), masks

        # generate RPN target so cpu workers can help reduce the workload
        # feat_h, feat_w = (img.shape[1] // self._stride, img.shape[2] // self._stride)
        gt_bboxes = mx.nd.array(bbox[:, :4])
        if self._multi_stage:
            oshapes = []
            anchor_targets = []
            for feat_sym in self._feat_sym:
                oshapes.append(feat_sym.infer_shape(data=(1, 3, img.shape[1], img.shape[2]))[1][0])
            for anchor, oshape in zip(self._anchors, oshapes):
                anchor = anchor[:, :, :oshape[2], :oshape[3], :].reshape((-1, 4))
                anchor_targets.append(anchor)
            anchor_targets = mx.nd.concat(*anchor_targets, dim=0)
            cls_target, box_target, box_mask = self._target_generator(
                gt_bboxes, anchor_targets, img.shape[2], img.shape[1])
        else:
            oshape = self._feat_sym.infer_shape(data=(1, 3, img.shape[1], img.shape[2]))[1][0]
            anchor = self._anchors[:, :, :oshape[2], :oshape[3], :].reshape((-1, 4))

            cls_target, box_target, box_mask = self._target_generator(
                gt_bboxes, anchor, img.shape[2], img.shape[1])
        return img, bbox.astype(img.dtype), masks, cls_target, box_target, box_mask

""""
# seek a better way to resize masks
# 1. resize via pytorch
masks = masks.transpose(2,0,1) # 1xnxhxw
masks = torch.from_numpy(masks).unsqueeze(0).to(torch.float) # may raise out of memory
masks = torch.nn.functional.interpolate(masks, (img.shape[0], img.shape[1]), mode='nearest')
masks = mx.nd.array(masks[0].numpy())
# 2. split as 3 channel and then resize as image
num_group = masks.shape[-1]//3
num_left = masks.shape[-1]%3
for i in range(num_group):
    mask = cv2.resize(masks[:,:,i*3:(i+1)*3], (img.shape[1],img.shape[0]),
        interpolation=cv2.INTER_NEAREST)
    mask_list.extend([mx.nd.array(mask[:,:,j]) for j in range(3)])
for k in range(num_left):
    mask = cv2.resize(masks[:,:,num_group*3+k], (img.shape[1],img.shape[0]),
        interpolation=cv2.INTER_NEAREST)
    mask_list.append(mx.nd.array(mask))
"""
