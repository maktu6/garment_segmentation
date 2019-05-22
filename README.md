# Garment Segmentation
## Requirements
- [pycocotools](https://github.com/cocodataset/cocoapi)
- [mxnet](http://mxnet.incubator.apache.org/versions/master/install/)
- [gluoncv](https://gluon-cv.mxnet.io/index.html)
> Note: please install nightly build gluoncv for less bugs  
> `pip install gluoncv --pre --upgrade`

## Dataset 
[iMaterialist (Fashion) 2019 at FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/overview)  
- Convert imaterialist to COCO style  
```bash
python tools/convert_imaterialist_to_coco.py --data_root datasets/imaterialist
```
- files structure
```
datasets
└─imaterialist
    ├── annotations
    │   ├── resize_rle_instances_val.json
    │   ├── rle_instances_train.json
    |   └── rle_instances_val.json
    ├── test
    ├── train
    ├── val
    ├── label_descriptions.json
    ├── sample_submission.csv
    └── train.csv
```
## Training
```bash
python train_mask_rcnn.py --save-prefix train_logs/ftCOCO_noWarmUp/ \
			  -j 0 --lr 0.0008 --lr-warmup -1 --val-interval 1
```
## Reference
- [Train Mask RCNN end-to-end on MS COCO](https://gluon-cv.mxnet.io/build/examples_instance/train_mask_rcnn_coco.html)