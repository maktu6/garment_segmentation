import time
import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

def load_model(num_out, use_gpu=True, weight_path=None, model_name="resnet50"):
    if model_name=="resnet50":
        model = models.resnet50(pretrained=True)
    else:
        raise NotImplementedError("%s is not support"%model_name)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_out)
    if weight_path:
        model.load_state_dict(torch.load(weight_path))
        print("load model weight from %s"%weight_path)
    if use_gpu:
        model.cuda()
    return model

def save_model(model, epoch, save_dir=None):
    if save_dir is None:
        save_dir = "../train_logs/garment_attribute_classify/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state_dict = model.state_dict()
    pth = os.path.join(save_dir, "%s_cycle_epoch%s.pth" % (model._get_name(), epoch))
    torch.save(state_dict, pth)

def train_model(dataloaders, model, criterions, optimizer, scheduler, label_index, 
                num_epochs=25, save_freq=1, count_loss_weight=1.0, 
                is_inception=False, use_gpu=True, save_dir=None):
    dataset_sizes = dataloaders['size']
    since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_global_corrects = 0
            running_count_loss = 0.0

            # Iterate over data.
#             count  = 0
            for data in tqdm(dataloaders[phase]):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = inputs.to('cuda')
                    labels = [label.to('cuda') for label in labels]
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase=="train"):
                    if is_inception and phase == "train":
                        raise NotImplementedError("TODO: loss caculatation")
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterions(outputs, labels)
                        loss2 = criterions(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                
                preds = []
                for i in range(len(label_index)-1):
                    _, pred = torch.max(outputs[:, label_index[i]:label_index[i+1]], 1)
                    preds.append(pred)
                loss = 0
                i = 0
                count_pred = torch.zeros_like(labels[0]).float()
                for criterion, label in zip(criterions, labels):
                    # extrate (1-the first value after softmax)
                    # regularzation the num of attributes
                    count_pred += 1-torch.nn.functional.softmax(
                        outputs[:, label_index[i]:label_index[i+1]], 1)[:,0]
                    loss += criterion(outputs[:, label_index[i]:label_index[i+1]], label)
                    i += 1
                loss = loss/i
                count_gt = torch.zeros_like(labels[0]).float()
                for label in labels:
                    count_gt += (label>0).float()
                count_loss = torch.nn.functional.l1_loss(count_pred, count_gt)*count_loss_weight
                loss += count_loss

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.batch_step()
                    #print(optimizer.param_groups[0]['lr'])

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_count_loss += count_loss
                # accuracy
                corrects = torch.zeros_like(preds[0], device='cpu')
                for pred, label in zip(preds, labels):
                    corrects += (pred==label).to('cpu').long()
                running_corrects += torch.sum(corrects).item()/len(preds)
                running_global_corrects += torch.sum(corrects==len(preds)).item()
#                 for pred, label in zip(preds, labels):
#                     running_corrects += torch.sum(pred == label)
#                 print(running_corrects)
#                 print(running_corrects.float() / dataset_sizes[phase])
#                 count += 1
#                 if count%100 == 0:
#                     print("batch %d:" % count)
#                     print(phase+"_loss:", loss.data[0])
#                     print(phase+"_acc:", torch.sum(preds == labels.data)/len(labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_count_loss = running_count_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_global_acc = running_global_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f}, Count Loss: {:.4f}, Acc: {:.4f}, Global Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_count_loss, epoch_acc, epoch_global_acc))
            
            if phase == "train" and (epoch%save_freq==0):
                save_model(model, epoch, save_dir=save_dir)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                save_model(model, "best", save_dir=save_dir)
#                 best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc in epoch{}: {:4f}'.format(best_epoch, best_acc))
    return model