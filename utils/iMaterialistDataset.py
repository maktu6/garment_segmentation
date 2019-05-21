import numpy as np
from gluoncv.data import COCOInstance
from pycocotools.coco import COCO
import os

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