# Garment Segmentation
## Requirements
- [pycocotools](https://github.com/cocodataset/cocoapi)
- [mxnet](http://mxnet.incubator.apache.org/versions/master/install/)
- [gluoncv](https://gluon-cv.mxnet.io/index.html)  

*Note: please install nightly build gluoncv for less bugs*
```bash
pip install gluoncv --pre --upgrade
```

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
*Note: all using  COCO pretrained models, only the categories less than 13 (main apparels) for segmentation.*

- [Mask R-CNN](https://arxiv.org/abs/1703.06870)
```bash
python train_mask_rcnn.py --save-prefix train_logs/xxx/ --val-interval 1
```
- [DeepLabV3](https://arxiv.org/abs/1706.05587)
```bash
python train_segmentation.py --model deeplabv3 --dataset imaterialist \
                            --model-zoo deeplab_resnet101_coco --aux \
                            --checkname res101 --epochs 30 --lr 0.001 --ngpus 1 \
                            --workers 2 --batch-size 2 --test-batch-size 2
```
- [DeepLabV3+](https://arxiv.org/abs/1802.02611)
```bash
python train_segmentation.py --model deeplabv3plus --dataset imaterialist \
                             --model-zoo deeplab_plus_xception_coco --aux \
                             --base-size 576 --crop-size 512 \
                             --checkname xception --epochs 30 --lr 0.001 --ngpus 1 \
                             --workers 1 --batch-size 1 --test-batch-size 1 
```
## Reference
- [Train Mask RCNN end-to-end on MS COCO](https://gluon-cv.mxnet.io/build/examples_instance/train_mask_rcnn_coco.html)
- [Reproducing SoTA on Pascal VOC Dataset](https://gluon-cv.mxnet.io/build/examples_segmentation/voc_sota.html)
