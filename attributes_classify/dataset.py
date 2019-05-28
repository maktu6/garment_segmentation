import numpy as np
import pandas as pd
import cv2
import os
from collections import OrderedDict
from pycocotools import mask as coco_mask
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

from box import rle_decode, get_image_boxes

ATTR_FOR_CLASSIFY = OrderedDict({"length": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                 "opening_type": [10,11,13],
                                 "opening": [12,15,16,17,18],
                                 "sym": [19,20],
                                 "silhouette": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
                                 "fit": [39, 40, 41, 42, 43],
                                 "manufacturing_techniques": [44, 47, 48, 49, 50, 51, 54, 55, 57, 58, 60],
                                 "textile_pattern": [62, 64, 65, 68, 69, 71, 73, 74, 75, 76, 83],
                                 "animal": [77, 78, 79, 80, 81, 82],
                                 "waistline": [85, 86, 87, 88, 89, 90, 91],
                                 "other1": [14, 45,52,56,59, 61,63, 84],
                                 "other2":[53,67,46,66],
                                 "geometric":[70],
                                 "stripe":[72]})

def built_attr2label(attr_for_classify):
    attr2label = {}
    for key in attr_for_classify.keys():
        attr2label[key] = {}
        for i, attr_id in enumerate(attr_for_classify[key]):
            attr2label[key][attr_id] = i+1
        attr2label[key][92] = 0
    return attr2label

# def get_label_dict(train_df, attr_for_classify):
#     label_dict = {}
#     for key in attr_for_classify.keys():
#         label_dict[key] = train_df[key].values.astype(int)
#     label_dict['ImageId'] =  list(train_df['ImageId'].values)
#     label_dict['mask'] =  list(train_df['coco_rle'].values)
#     return label_dict

def crop_target(img, rle, size, del_bg=False, coco_rle=False):
    if coco_rle:
        mask = coco_mask.decode({"size":(img.size[1], img.size[0]), "counts":rle})
    else:
        # polygons to mask
        mask = rle_decode(rle, (512,512))
        mask = cv2.resize(mask, img.size, cv2.INTER_NEAREST)
    DOWN_SIZE = 512 # for saving memroy
    if min(img.size)>DOWN_SIZE:
        mask = transforms.functional.resize(Image.fromarray(mask), DOWN_SIZE, interpolation=Image.NEAREST)
        img = transforms.functional.resize(img, DOWN_SIZE, interpolation=Image.BILINEAR)
    ys, xs = np.where(mask)
    box = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]])
    img_box = get_image_boxes(box, img, size=size, mask=(mask if del_bg else None))[0]
    return img_box

class MyDataset(data.Dataset):
    def __init__(self, root, csv, attr_for_classify, transform=None, del_bg=True, infer=False):
        self.root = os.path.join(root, ("test" if infer else "train"))
        self.classes_dict = attr_for_classify
        self.infer = infer
        if isinstance(csv, str):
            train_df = pd.read_csv(csv)
        else:
            train_df = csv
        self.label_dict = self.get_label_dict(train_df)
        self.attr2label = built_attr2label(attr_for_classify)
        self.transform = transform
        self.del_bg = del_bg

    def __getitem__(self, index):
        labels = []
        img_name = self.label_dict['ImageId'][index]
        rle = self.label_dict['mask'][index]
        if not self.infer:
            for key in self.classes_dict.keys():
                attr_id = self.label_dict[key][index]
                label = self.attr2label[key][attr_id]
                labels.append(label)
        path = os.path.join(self.root, img_name)
        img = Image.open(path)
        sample = crop_target(img, rle, size=299, del_bg=self.del_bg, coco_rle=not self.infer)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, labels
    
    def __len__(self):
        return len(self.label_dict['ImageId'])

    def get_label_dict(self, train_df):
        label_dict = {}
        if not self.infer:
            for key in self.classes_dict.keys():
                label_dict[key] = train_df[key].values.astype(int)
        label_dict['ImageId'] =  list(train_df['ImageId'].values)
        if self.infer:
            rle = train_df['EncodedPixels'].values
        else:
            rle = train_df['coco_rle'].values
        label_dict['mask'] =  list(rle)
        return label_dict

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(16,8))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated