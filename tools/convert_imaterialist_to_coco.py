# conversion borrowed from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/tools/cityscapes/convert_cityscapes_to_coco.py
import cv2
import numpy as np
import os
from tqdm import tqdm
from pycocotools import mask as maskUtils
import multiprocessing

# !!! Some ImageIds have incorrect image sizes in train.csv. 
# Do transpose and vertical flip on their mask
WRONG_IMG_IDS = ['f4d6e71fbffc3e891e5009fef2c8bf6b.jpg',
                 '2ab8c02ce17612733ddee218b4ce1fd1.jpg']
TO_REMOVE = 0

# borrowed from https://github.com/fastai/fastai/blob/master/fastai/vision/image.py#L416
def rle_decode(mask_rle, shape, class_id=1):
    "Return an image array from run-length encoded string `mask_rle` with `shape`."
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint)
    for low, up in zip(starts, ends): img[low:up] = class_id
    return img.reshape(shape, order='F').astype('uint8').copy()

def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy

def mask2polygons(mask):
    """ convert mask to polygons
    Note: maybe not a good choice since the holes in mask will be ignored,
    RETR_EXTERNAL means extract only the outer contours, 
    CHAIN_APPROX_SIMPLE removes all redundant points
    """
    contour, hier = findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [c.reshape(-1).tolist() for c in contour]
    return polygons

def poly_to_box(poly):
    """Convert a polygon into a tight bounding box."""
    x0 = min(min(p[::2]) for p in poly)
    x1 = max(max(p[::2]) for p in poly)
    y0 = min(min(p[1::2]) for p in poly)
    y1 = max(max(p[1::2]) for p in poly)
    box_from_poly = [x0, y0, x1, y1]

    return box_from_poly

def xyxy_to_xywh(xyxy_box):
    xmin, ymin, xmax, ymax = xyxy_box
    xywh_box = (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE)
    return xywh_box

def extract_image_annotations(i, img_id, seg_df, use_polygon=False):
    """ extract image and annotations for coco dict
    Args:
        i: the index in ImageId list
        img_id: ImageId from "train.csv"
        seg_df: all rows belong to img_id in "train.csv"
        use_polygon: if True, the masks are encoded as polygons, else RLE
        (more details see https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py)
    Return:
        image: a dict of image infomation
        ann_list: a list of annotations
    """
    # image
    image = {}
    image['id'] = i
    image['file_name'] = img_id
    w, h = int(seg_df.iloc[0]["Width"]), int(seg_df.iloc[0]["Height"])
    image['width'] = w
    image['height'] = h
    # annotations
    ann_list = []
    for j in range(len(seg_df)):
        ann = {}
        seg_l = seg_df.iloc[j]
        ann['id'] = int(seg_l._name)
        ann['image_id'] = image['id']
        ann['category_id'] = int(seg_l['ClassId'].split('_')[0])
        # decode RLE
        mask = rle_decode(seg_l['EncodedPixels'], (h, w), 1)
        # since some ImageIds have incorrect sizes in train.csv
        if img_id in WRONG_IMG_IDS:
            mask = mask.transpose()[::-1,:]
        ann['iscrowd'] = 0
        if use_polygon:
            ann['segmentation'] = mask2polygons(mask)
            ann['area'] = int(np.sum(mask))
            # box
            xyxy_box = poly_to_box(ann['segmentation'])
            xywh_box = xyxy_to_xywh(xyxy_box)
            ann['bbox'] = xywh_box
        else:
            # mask need to be (h, w, 1) and Fortran contiguous
            mask = np.asfortranarray(mask.reshape((mask.shape[0], -1, 1), order='F'))
            rle = maskUtils.encode(mask)[0]
            ann['area'] = int(maskUtils.area(rle))
            box = maskUtils.toBbox(rle)
            box[2:] += TO_REMOVE
            ann['bbox'] = [int(x) for x in box]
            # decode rle as str for saving as json file
            rle['counts'] = str(rle['counts'], encoding='utf-8')
            ann['segmentation'] = rle
        ann_list.append(ann)
    return image, ann_list

def extractor(args):
    return extract_image_annotations(*args)

def convert_imaterialist2coco(img_ids, csv_df, label_descriptions, use_polygon=False, num_workers=2, start_id=0):
    """ convert iMaterialis dataset to COCO-style annotations
    Args:
        img_ids: a list of ImageId
        csv_df: load from "train.csv"
        label_descriptions: load from "label_descriptions.json"
        use_polygon: if True, the masks are encoded as polygons, else RLE
        (more details see https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py)
        num_workers: number of workers for multiprocessing
        start_id: for differnt img_ids for train and val
    """
    images = []
    annotations = []
#     for i in tqdm(range(len(img_ids))):
#         img_id, seg_df = img_ids[i], csv_df[csv_df.ImageId==img_id]
#         image, ann_list = extract_image_annotations(i, img_id, seg_df, use_polygon)
#         images.append(image); annotations.extend(ann_list)
    def generate_args():
        for i in tqdm(range(len(img_ids))):
            img_id = img_ids[i]
            seg_df = csv_df[csv_df.ImageId==img_id]
            yield (i+start_id, img_id, seg_df, use_polygon)

    with multiprocessing.Pool(num_workers) as pool:
        imap_unordered_it = pool.imap_unordered(extractor, generate_args())
        for r in imap_unordered_it:
            image, ann_list = r
            images.append(image)
            annotations.extend(ann_list)
    # sorted by image id
    images.sort(key=lambda x:x['id'])
    annotations.sort(key=lambda x:x['image_id'])
    # coco style dict
    ann_dict = {}
    ann_dict['images'] = images
    ann_dict['annotations'] = annotations
    ann_dict['categories'] = label_descriptions['categories']
    ann_dict['info'] = label_descriptions['info']
    return ann_dict

def resize_rles(ann_dict, shape=(512, 512)):
    """ resize RLEs for val set, since it require (512, 512) for computing metric
    Args:
        ann_dict: a coco-style dict
        shape: (width, height)
    """
    for ann_dict in tqdm(ann_dict['annotations']):
        # rle to mask
        rle = ann_dict['segmentation']
        mask = maskUtils.decode(rle) # hxw
        # resize mask
        mask = cv2.resize(mask, shape, interpolation=cv2.INTER_NEAREST)
        mask = np.asfortranarray(mask.reshape((shape[1], shape[0], 1), order='F'))
        # mask to rle
        rle = maskUtils.encode(mask)[0]
        rle['counts'] = str(rle['counts'], encoding='utf-8')
        ann_dict['segmentation'] = rle
    return ann_dict

def main(data_root, val_num, copy_val, use_polygon=False, num_workers=2, resize_val=True):
    # load csv file
    csv_path = os.path.join(data_root, "train.csv")
    train_df = pd.read_csv(csv_path)
    print('-'*40)
    print("- loading %s successfully. "%csv_path)
    # load label_descriptions
    with open(os.path.join(data_root,'label_descriptions.json'), 'r') as f:
        label_descriptions =json.load(f)

    # split train set and val set
    # only select the images with attributes as val set
    ImageId_list = pd.unique(train_df.ImageId)
    attr_df = train_df[train_df.ClassId.str.contains('_')]
    attr_img_ids = pd.unique(attr_df.ImageId)
    val_img_ids = random.sample(list(attr_img_ids), val_num)
    train_img_ids = list(set(ImageId_list)-set(val_img_ids))
    print("- iMaterialis dataset contains %d images which are split %d for train \
        and %d for val. "%(len(attr_img_ids), len(train_img_ids), len(val_img_ids)))

    # val images
    val_dir = os.path.join(data_root, 'val')
    if copy_val:
        # copy val images to "val" folder
        os.makedirs(val_dir, exist_ok=True)
        for img_id in tqdm(val_img_ids):
            src_path = os.path.join(data_root, 'train/'+img_id)
            shutil.copy(src_path, val_dir)
    else:
        # check all val images in "val" folder
        ori_list = os.listdir(val_dir)
        for img_id in val_img_ids:
            assert img_id in ori_list
        print("- all val images in %s. "%val_dir)

    # make and save json files for train and val
    save_dir = os.path.join(data_root, 'annotations')
    os.makedirs(save_dir, exist_ok=True)
    set_names = ["val", "train"] 
    for i, img_ids in enumerate([val_img_ids, train_img_ids]):
        json_path = os.path.join(save_dir, 'rle_instances_%s.json'%set_names[i])
        if os.path.exists(json_path):
            print("- %s are skipped since it existed. ")
            continue
        else:
            print("- Processing %s set ..."%set_names[i])
        # img_ids in val set resume from train set
        start_id = 0 if set_names[i]=="train" else len(train_img_ids)
        ann_dict = convert_imaterialist2coco(img_ids, train_df, label_descriptions, use_polygon, num_workers, start_id=start_id)
        with open(json_path, 'w') as outfile:
            json.dump(ann_dict, outfile)
        # resize RLEs for val set, since it requires (512, 512) for computing metric
        if resize_val and set_names[i]=="val":
            resize_rles(ann_dict)
            json_path = json_path.replace('rle_instances', 'resize_rle_instances')
            with open(json_path, 'w') as outfile:
                json.dump(ann_dict, outfile)

if __name__ == "__main__":
    import pandas as pd 
    import json
    import shutil
    import random
    import argparse

    """ file structure
    imaterialist
    ├── annotations
    │   ├── rle_instances_train.json
    |   └── rle_instances_val.json
    ├── test
    ├── train
    ├── val
    ├── label_descriptions.json
    ├── sample_submission.csv
    └── train.csv
    """
    parser = argparse.ArgumentParser(description='Convert iMaterialis dataset to COCO-style annotations')
    parser.add_argument("--data_root", type=str, required=True, 
                        help="the root of iMaterialis dataset (e.g. datasets/imaterialist)")
    parser.add_argument("--val_num", type=int, default=1000, 
                        help="the size of validation set")
    parser.add_argument("--copy_val", action="store_true", 
                        help="copy val images to 'val' folder")
    parser.add_argument("--use_polygon", action="store_true", 
                        help="the masks are encoded as polygons (Default is RLE)")
    parser.add_argument("--num_workers", type=int, default=2, 
                        help="number of workers for multiprocessing")
    parser.add_argument("--seed", type=int, default=666, 
                        help="the random seed for splitting train and val")
    args = parser.parse_args()

    random.seed(args.seed)
    main(args.data_root, args.val_num, args.copy_val, args.use_polygon, args.num_workers)
