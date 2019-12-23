from gluoncv.data import COCOInstance, COCOSegmentation
from pycocotools.coco import COCO
import numpy as np
from PIL import Image, ImageOps
import os
import pickle
import random
from io import BytesIO


def randomJPEGcompression(image, min_quality=75):
    qf = random.randrange(min_quality, 100)
    outputIoStream = BytesIO()
    image = Image.fromarray(image)
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return np.array(Image.open(outputIoStream))

def random_alter_background(img_np, mask_np, white_prob=0.3):
    if random.random()<white_prob:
        # gray or while
        if random.random()<0.5:
            bg_value = np.random.randint(220, 250, size=(1,1,1), dtype="uint8")
        else:
            bg_value = np.random.randint(250, 256, size=(1,1,1), dtype="uint8")
    else:
        # random color
        bg_value = np.random.randint(0,255,size=(1,1,3), dtype="uint8")
    # replace the background
    bg_mask = mask_np[:,:,None]==0
    bg = bg_value*bg_mask
    img_new_np = img_np*(~bg_mask)+bg
    return img_new_np

class COCOiMaterialist(COCOInstance):
    
    CLASSES=['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 
             'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 
             'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 
             'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 
             'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 
             'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 
             'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']
    
    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []
        segms = []
        for split in self._splits:
            anno = os.path.join(self._root, 'annotations', split) + '.json'
            _coco = COCO(anno)
            self._coco.append(_coco)
            classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
            if not classes == self.classes:
                raise ValueError("Incompatible category names with COCO: ")
            assert classes == self.classes
            json_id_to_contiguous = {
                v: k for k, v in enumerate(_coco.getCatIds())}
            if self.json_id_to_contiguous is None:
                self.json_id_to_contiguous = json_id_to_contiguous
                self.contiguous_id_to_json = {
                    v: k for k, v in self.json_id_to_contiguous.items()}
            else:
                assert self.json_id_to_contiguous == json_id_to_contiguous

            # iterate through the annotations
            image_ids = sorted(_coco.getImgIds())
            for entry in _coco.loadImgs(image_ids):
                filename = entry['file_name']
                dirname = split.split('_')[-1] # "train" or "val"
                abs_path = os.path.join(self._root, dirname, filename)
                if not os.path.exists(abs_path):
                    raise IOError('Image: {} not exists.'.format(abs_path))
                label, segm = self._check_load_bbox(_coco, entry)
                # skip images without objects
                if self._skip_empty and label is None:
                    continue
                items.append(abs_path)
                labels.append(label)
                segms.append(segm)
        return items, labels, segms

    def _check_load_bbox(self, coco, entry):
        """Check and load ground-truth labels"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        valid_segs = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            if obj.get('ignore', 0) == 1:
                continue
            # crowd objs cannot be used for segmentation
            if obj.get('iscrowd', 0) == 1:
                continue
            # need accurate floating point box representation
            x1, y1, w, h = obj['bbox']
            x2, y2 = x1 + np.maximum(0, w), y1 + np.maximum(0, h)
            # clip to image boundary
            x1 = np.minimum(width, np.maximum(0, x1))
            y1 = np.minimum(height, np.maximum(0, y1))
            x2 = np.minimum(width, np.maximum(0, x2))
            y2 = np.minimum(height, np.maximum(0, y2))
            # require non-zero seg area and more than 1x1 box size
            if obj['area'] > self._min_object_area and x2 > x1 and y2 > y1 \
                    and (x2 - x1) * (y2 - y1) >= 4:
                contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
                valid_objs.append([x1, y1, x2, y2, contiguous_cid])

                segs = obj['segmentation'] # polygon or RLE
                assert isinstance(segs, list) or isinstance(segs, dict), '{}'.format(obj.get('iscrowd', 0))
                if isinstance(segs, list):
                    valid_segs.append([np.asarray(p).reshape(-1, 2).astype('float32')
                                    for p in segs if len(p) >= 6])
                else:
                    valid_segs.append(segs)
        # there is no easy way to return a polygon placeholder: None is returned
        # in validation, None cannot be used for batchify -> drop label in transform
        # in training: empty images should be be skipped
        if not valid_objs:
            valid_objs = None
            valid_segs = None
        else:
            valid_objs = np.asarray(valid_objs).astype('float32')
        return valid_objs, valid_segs

class iMaterialistSegmentation(COCOSegmentation):
    """only using categories less than 13 for segmentation"""
    CAT_LIST = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    NUM_CLASS = 14
    def __init__(self, root=os.path.expanduser('datasets/imaterialist'),
                 split='train', mode=None, transform=None, tta=None, alter_bg=False, **kwargs):
        super(COCOSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        from pycocotools import mask
        if split == 'train':
            print('train set')
            ann_file = os.path.join(root, 'annotations/rle_instances_train.json')
            ids_file = os.path.join(root, 'annotations/train_ids.mx')
            self.root = os.path.join(root, 'train')
        else:
            print('val set')
            ann_file = os.path.join(root, 'annotations/rle_instances_val.json')
            ids_file = os.path.join(root, 'annotations/val_ids.mx')
            self.root = os.path.join(root, 'val')
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            with open(ids_file, 'rb') as f:
                self.ids = pickle.load(f)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.transform = transform
        self.alter_bg = alter_bg
        if self.alter_bg:
            self.NUM_CLASS = 2
        if self.mode != "train":
            self.tta = tta

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            m = coco_mask.decode(instance['segmentation'])
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    @property
    def classes(self):
        """Category names."""
        if self.alter_bg:
            return ('background', 'garment')
        else:
            return ('background', 'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 
                    'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 
                    'dress', 'jumpsuit', 'cape')

    def _sync_pad(self, img, mask):
        w, h = img.size
        long_size = max(w, h)

        padh = long_size - h
        padw = long_size - w

        im_pad = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask_pad = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # region for padding (set -1 later)
        ignore_w = round((1-padw/float(long_size))*self.crop_size) if padw != 0 else None
        ignore_h = round((1-padh/float(long_size))*self.crop_size) if padh != 0 else None
        return im_pad, mask_pad, (ignore_w, ignore_h)

    def _resize_short_within(self, img, short, max_size, mult_base=1, interp=Image.BILINEAR):
        """Resizes the original image by setting the shorter edge to size
        and setting the longer edge accordingly. Also this function will ensure
        the new image will not exceed ``max_size`` even at the longer side.

        Parameters
        ----------
        img : PIL.Image
            The original image.
        short : int
            Resize shorter side to ``short``.
        max_size : int
            Make sure the longer side of new image is smaller than ``max_size``.
        mult_base : int, default is 1
            Width and height are rounded to multiples of `mult_base`.
        interp : default is Image.BILINEAR
        Returns
        -------
        PIL.Image
            An 'PIL.Image' containing the resized image.
        """
        w, h = img.size
        im_size_min, im_size_max = (h, w) if w > h else (w, h)
        scale = float(short) / float(im_size_min)
        if np.round(scale * im_size_max / mult_base) * mult_base > max_size:
            # fit in max_size
            scale = float(np.floor(max_size / mult_base) * mult_base) / float(im_size_max)
        new_w, new_h = (int(np.round(w * scale / mult_base) * mult_base),
                        int(np.round(h * scale / mult_base) * mult_base))
        img = img.resize((new_w, new_h), interp)
        return img


    def _testval_sync_transform(self, img, mask, padding=True):
        """ resize image and mask while keeping ratio"""
        if padding:
            # padding and resize
            img, mask, keep_size = self._sync_pad(img, mask)
            img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            mask = mask.resize(img.size, Image.NEAREST)
        else:
            # resize without padding
            short_size = self.crop_size*1.75
            if max(img.size) > short_size:
                img = self._resize_short_within(img, short_size, short_size*2)
                mask = mask.resize(img.size, Image.NEAREST)
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        if padding:
            mask[keep_size[1]:, keep_size[0]:] = -1
        return img, mask

    def _random_alter_background(self, img, mask):
        # alter background and random jpeg quality
        img_np = img.asnumpy().astype('uint8')
        mask_np = mask.asnumpy()
        img_new_np = random_alter_background(img_np, mask_np)
        img_new_np = randomJPEGcompression(img_new_np)
        img_new = self._img_transform(img_new_np)
        return img_new

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width'])
        if self.alter_bg:
            mask = (mask>0).astype('uint8')
        mask = Image.fromarray(mask)
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
            if self.alter_bg and (random.random() < self.alter_bg):
                img = self._random_alter_background(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            # resize without padding for memory reduction when test time augmentation
            img, mask = self._testval_sync_transform(img, mask, not self.tta)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask