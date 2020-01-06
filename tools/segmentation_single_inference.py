import os
import numpy as np
from PIL import Image, ImageOps
from skimage import io
import cv2
import mxnet as mx
import gluoncv
import mxnet.ndarray as F
from mxnet.gluon.data.vision import transforms
from gluoncv.utils.viz import get_color_pallete

def pad_img(im):
    w, h = im.size
    long_size = max(w, h)

    padh = long_size - h
    padw = long_size - w

    im_pad = ImageOps.expand(im, border=(0, 0, padw, padh), fill=0)
    return im_pad

def process_img(im, crop_size, img_transform):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    assert len(crop_size) == 2
    
    im = pad_img(im)
    im = im.resize(crop_size, resample=Image.BILINEAR)
    # final transform
    im_tensor = F.array(np.array(im), mx.cpu(0))
    im_tensor = img_transform(im_tensor)
    return im_tensor, im

def compute_nopad_size(ori_size, crop_size):
    w, h = ori_size
    long_size = max(w, h)

    padh = long_size - h
    padw = long_size - w

    crop_h = crop_size[0] - round(padh/long_size*crop_size[0])
    crop_w = crop_size[1] -round(padw/long_size*crop_size[1])
    return (crop_h, crop_w)

def main(args):
    BINARY_SEGMENTATION = True
    weight_path = args.weight #'train_logs/imaterialist/deeplabv3/res101_alterBG_v1/model_best.params'
    im_name = args.img_path # 'datasets/zhong/dress_back.jpg'
    # 'datasets/zhong/results/deeplabv3/'
    result_dir = os.path.join(args.result_dir, os.path.basename(im_name)[:-4])
    th = 0.5 # for binary segmentation
    crop_size = (480,480) 
    ctx = mx.gpu()
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    # load deeplab v3
    model = gluoncv.model_zoo.DeepLabV3(nclass=14, backbone='resnet101', ctx=ctx)
    model.load_parameters(weight_path, ctx=ctx)

    # read image
    ori_img = cv2.imread(im_name)[:,:,::-1]
    ori_img = Image.fromarray(ori_img)
    ori_size = ori_img.size

    img, img_crop = process_img(ori_img, crop_size, transform_fn)
    img = img.expand_dims(0).as_in_context(ctx)

    # predict mask
    output = model.demo(img)
    if BINARY_SEGMENTATION:
        output = mx.nd.softmax(output, axis=1)
        predict = mx.nd.squeeze(output[:,1,:,:]).asnumpy()
    else:
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()



    img_crop = np.array(img_crop)
    if BINARY_SEGMENTATION:
        mask = (predict>=th).astype('uint8')
    else:
        mask = (predict>0).astype('uint8')
        # mask_vis = get_color_pallete(predict)

    crop_h, crop_w = compute_nopad_size(ori_size, crop_size)
    mask_crop = mask[:crop_h, :crop_w]
    img_crop = np.array(img_crop)[:crop_h, :crop_w, :]
    im_vis1 = img_crop*mask_crop[:,:,None]
    im_vis2 = img_crop*(1-mask_crop[:,:,None])


    ori_img = img_crop[:crop_h,:crop_w,:]
    mask = (mask_crop*255).astype('uint8')

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for key in ['ori_img', 'mask', 'im_vis1', 'im_vis2']:
        io.imsave(os.path.join(result_dir, key+'.jpg'), eval(key))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')
    # model and dataset 
    parser.add_argument('--weight', type=str,
                        help='the path of model weight')
    parser.add_argument('--img-path', type=str,
                        help='the path of input image')
    parser.add_argument('--result-dir', type=str,
                        help='the root for save results')

    args = parser.parse_args()
    main(args)