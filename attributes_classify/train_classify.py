import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from dataset import MyDataset, built_attr2label, imshow, ATTR_FOR_CLASSIFY
from trainer import load_model, save_model, train_model
from CyclicLR import CyclicLR

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(10, resample=Image.BILINEAR, expand=True),
        transforms.Resize((230, 230)),  # 224 for resnet and densenet
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean, std
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

root = '../datasets/imaterialist/'
train_csv = os.path.join(root, 'attribute_label', 'for_classify_train.csv')
val_csv = os.path.join(root, 'attribute_label', 'for_classify_val.csv')


batch_sizes = {"train":48, "val":48}
train_dataset = MyDataset(root, train_csv, ATTR_FOR_CLASSIFY, transform=data_transforms["train"], use_jitter=False)
val_dataset = MyDataset(root, val_csv, ATTR_FOR_CLASSIFY, transform=data_transforms["val"])
image_datasets = {"train":train_dataset, "val":val_dataset}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x],
                                             shuffle=(x=="train"), num_workers=3, drop_last=(x=="train"))
              for x in ["train", "val"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
dataloaders['size'] = dataset_sizes
use_gpu = torch.cuda.is_available()

attr2label = built_attr2label(ATTR_FOR_CLASSIFY)
num_out = 0
label_index = [0]
for key in attr2label.keys():
    num_out += len(attr2label[key])
    label_index.append(num_out)

model = load_model(num_out)

criterions = [nn.CrossEntropyLoss() for _ in attr2label.keys()]

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

step_size = 2 * (dataset_sizes['train'] / batch_sizes['train'])
lr_scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.008, mode='triangular2', step_size=step_size) # 4 epochs 1 cycle
save_dir = '../train_logs/garment_attribute_classify/res50_edge/'
model_ft = train_model(dataloaders, model, criterions, optimizer, lr_scheduler, label_index, 
                       num_epochs=35, save_freq=5, count_loss_weight=0, save_dir=save_dir)