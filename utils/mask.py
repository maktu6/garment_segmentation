# borrowed form https://github.com/dmlc/gluon-cv/blob/master/gluoncv/utils/viz/mask.py
# add `output_shape` to control the result
import numpy as np
import mxnet as mx
import cv2
from gluoncv.data.transforms.mask import fill
import pycocotools.mask as cocomask

def rle_encode(img):
    """Return run-length encoding string from `img`"""
    pixels = np.concatenate([[0], img.ravel(order='F'), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def expand_mask(masks, bboxes, im_shape, scores=None, thresh=0.5, labels=None, output_shape=None):
    """Expand instance segmentation mask to full image size.

    Parameters
    ----------
    masks : numpy.ndarray or mxnet.nd.NDArray
        Binary images with shape `N, M, M`
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes
    im_shape : tuple
        Tuple of length 2: (width, height)
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    output_shape: tuple
        specify the output shape (width, height)
    
    Returns
    -------
    if labels is None, return Binary images with shape `N, height, width` (numpy.ndarray)
    otherwise return a dict {'mask': numpy.ndarray, 'label': list, 'score': list,
                             'bbox': a list of array, 'coco_rle': a list of dict}


    """
    if len(masks) != len(bboxes):
        raise ValueError('The length of bboxes and masks mismatch, {} vs {}'
                         .format(len(bboxes), len(masks)))
    if scores is not None and len(masks) != len(scores):
        raise ValueError('The length of scores and masks mismatch, {} vs {}'
                         .format(len(scores), len(masks)))

    if isinstance(masks, mx.nd.NDArray):
        masks = masks.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    sorted_inds = np.argsort(-areas)

    full_masks = []
    cls_ids = []
    score_list = []
    bbox_list = []
    coco_rle_list = []
    for i in sorted_inds:
        if scores is not None and scores[i] < thresh:
            continue
        else:
            score_list.append(scores[i][0])
        if labels is not None:
            cls_id = int(labels.flat[i])
            assert cls_id != -1
            cls_ids.append(cls_id)
        mask = masks[i]
        bbox = bboxes[i]
        mask = fill(mask, bbox, im_shape)
        # encode to coco_rle
        mask_f = np.asfortranarray(mask.reshape((mask.shape[0], -1, 1), order='F'))
        coco_rle = cocomask.encode(mask_f)[0]
        if output_shape:
            mask = cv2.resize(mask, output_shape, cv2.INTER_NEAREST)
        bbox_list.append(bbox)
        coco_rle_list.append(coco_rle)
        full_masks.append(mask)
    full_masks = np.array(full_masks)
    if labels is None:
        return full_masks
    return {'mask':full_masks, 'label':cls_ids, 
            'score': score_list, 'bbox': bbox_list, 'coco_rle': coco_rle_list}