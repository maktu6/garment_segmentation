import argparse
import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import numpy as np
import pandas as pd
from tqdm import tqdm
import mxnet as mx
from gluoncv import model_zoo, data, utils

from utils.mask import expand_mask, rle_encode

CLASSES = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 
           'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 
           'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 
           'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 
           'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 
           'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']

def write_ans(model_name, weight_path, test_root, sample_csv_path, ctx=mx.gpu()):
    """ Write answer for submission
    Args:
        model_name: the pretrained network name
        weight_path: the path of the model weight
        test_root: The dir to save submission file
        sample_csv_path: the path of the sample submission
        ctx: the device for mxnet, default is `mx.gpu()`
    Return:
        the DataFrame of the submission
    """
    # laod net
    net = model_zoo.get_model(model_name, pretrained_base=False)
    net.initialize()
    net.reset_class(CLASSES)
    net.load_parameters(weight_path)
    net.collect_params().reset_ctx(ctx)
    # load sample 
    sample_df = pd.read_csv(sample_csv_path)
    # write answer
    sub_list = []
    for img_name in tqdm(list(sample_df.ImageId)):
        img_path = os.path.join(test_root, img_name)
        x, orig_img = data.transforms.presets.rcnn.load_test(img_path)
        x = x.as_in_context(ctx)
        ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in net(x)]
        width, height = orig_img.shape[1], orig_img.shape[0]
        ans_dict = expand_mask(masks, bboxes, (width, height), 
                                            scores, labels=ids, output_shape=(512,512))
        masks_resize, class_ids = ans_dict['mask'], ans_dict['label']
        score_list, bbox_list = ans_dict['score'], ans_dict['bbox']
        for i, cid in enumerate(class_ids):
            # skip if the mask area is 0
            if np.sum(masks_resize[i,:,:]) == 0:
                continue
            rle = rle_encode(masks_resize[i,:,:])
            bbox = ' '.join([str(x) for x in (np.round(bbox_list[i]).astype('int'))])
            sub_list.append([img_name, rle, cid, score_list[i], bbox])
    # save csv file
    col_names = list(sample_df.columns)
    col_names.extend(['score', 'bbox'])
    submission_df = pd.DataFrame(sub_list, columns=col_names)
    return submission_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write answer for submission")
    parser.add_argument("--model_name", type=str, default='mask_rcnn_resnet50_v1b_coco',
                        help="the pretrained network name")
    parser.add_argument("--weight_path", type=str, default='train_logs/ftCOCO_noWarmUp/maskRCNN_resnet50_0000_0.0000.params',
                        help="the path of the model weight")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="The dir to save submission file (default is the dir of weight_path")
    parser.add_argument("--test_root", type=str, default='datasets/imaterialist/test',
                        help="the root of the test dataset")
    parser.add_argument("--sample_csv_path", type=str, default='datasets/imaterialist/sample_submission.csv',
                        help="the path of the sample submission")
    parser.add_argument('--gpu', type=int, default=0, help='the index of GPU')
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.weight_path)
    ctx = mx.gpu(args.gpu)
    submission_df = write_ans(args.model_name, args.weight_path, 
                              args.test_root, args.sample_csv_path, ctx)
    csv_path = os.path.join(args.save_dir, "submission.csv")
    submission_df.to_csv(csv_path, index=False)


