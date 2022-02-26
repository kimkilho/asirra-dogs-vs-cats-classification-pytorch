import os
import os.path as osp
import argparse
import copy
import time
import random
SEED = 2022
random.seed(SEED)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from dataset import AsirraDataset, Rescale, RandomFlip


# FIXME:
SRC_DATA_ROOT_DIR = "C:\\Users\\user\\PycharmProjects\\asirra-dogs-vs-cats-classification-pytorch\\asirra"
CHANNEL_MEAN = [0.485, 0.456, 0.406]
CHANNEL_STD = [0.229, 0.224, 0.225]
CLASS_NAMES_DICT = {
    'asirra': ['cat', 'dog'],
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--init_learning_rate', type=float, default=0.0003)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--weight_decay', type=float, default=0.0005)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    dataset_name = 'asirra'

    dst_model_dir = osp.join(os.getcwd(), 'outputs')
    if not osp.exists(dst_model_dir):
        os.makedirs(dst_model_dir)
    dst_model_path = osp.join(dst_model_dir, '{0}_ResNet50.pth'.format(dataset_name))

    src_trainval_img_dir = osp.join(SRC_DATA_ROOT_DIR, 'train-overfit-samples')    # FIXME
    trainval_img_filenames = os.listdir(src_trainval_img_dir)
    random.shuffle(trainval_img_filenames)

    val_ratio = 0.0    # FIXME: 0.2
    if val_ratio > 0:
        val_size = int(len(trainval_img_filenames) * val_ratio)
        val_img_filenames = trainval_img_filenames[:val_size]
        train_img_filenames = trainval_img_filenames[val_size:]
    else:
        val_img_filenames = None
        train_img_filenames = trainval_img_filenames

    class_names = CLASS_NAMES_DICT[dataset_name]
    if dataset_name == 'asirra':
        flip_mode = 'horizontal'
    else:
        flip_mode = 'horizontal_and_vertical'

    trainset = AsirraDataset(img_dir=src_trainval_img_dir,
                             img_filenames=train_img_filenames,
                             class_names=class_names,
                             transform=transforms.Compose([
                                 Rescale((224, 224)),
                                 RandomFlip(mode=flip_mode),
                                 transforms.ToTensor(),
                                 transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD)
                             ]))
    if val_img_filenames:
        valset = AsirraDataset(img_dir=src_trainval_img_dir,
                               img_filenames=val_img_filenames,
                               class_names=class_names,
                               transform=transforms.Compose([
                                   Rescale((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD)
                               ]))
    else:
        valset = None

    if valset:
        image_datasets = {x: ds for x, ds in zip(['train', 'val'], [trainset, valset])}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                      shuffle=shuffle, num_workers=4)
                       for x, shuffle in zip(['train', 'val'], [True, False])}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    else:
        dataloaders = {'train': torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                            shuffle=True, num_workers=4)}
        dataset_sizes = {'train': len(trainset)}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train_model(model, criterion, optimizer, model_path=None, num_epochs=5):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {0}/{1}'.format(epoch, num_epochs-1))
            print('-'*10)

            # Each epoch has a training (and validation phase)
            if valset:
                phases = ['train', 'val']
            else:
                phases = ['train']

            for phase in phases:
                if phase == 'train':
                    model.train()    # Set model in training mode
                else:
                    model.eval()    # Set model to evaluation mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    # track history only if in 'train' phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward pass + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} loss: {:.4f} acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # Deep copy and save the model weights
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if model_path:
                        torch.save(model.state_dict(), model_path)

            print('')

        time_elapsed = time.time() - since
        print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val acc: {:.4f}'.format(best_acc))

        # Load best model weights
        model.load_state_dict(best_model_wts)
        return model

    num_classes = len(class_names)
    model_ft = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)    # Set the size of output to be num_classes

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.init_learning_rate)

    # Train and evaluate
    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           model_path=dst_model_path, num_epochs=args.num_epochs)
    torch.save(model_ft.state_dict(), dst_model_path)

    print('')


if __name__ == '__main__':
    main()
