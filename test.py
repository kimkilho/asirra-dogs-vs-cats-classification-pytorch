import os
import os.path as osp

import numpy as np
from sklearn import metrics
import torch
import torchvision.transforms as transforms

from dataset import AsirraDataset, Rescale


# FIXME:
SRC_DATA_ROOT_DIR = "C:\\Users\\user\\PycharmProjects\\asirra-dogs-vs-cats-classification-pytorch\\asirra"
CHANNEL_MEAN = [0.485, 0.456, 0.406]
CHANNEL_STD = [0.229, 0.224, 0.225]
CLASS_NAMES_DICT = {
    'asirra': ['cat', 'dog'],
}


def main():
    dataset_name = 'asirra'
    dst_model_dir = osp.join(os.getcwd(), 'outputs')
    dst_model_path = osp.join(dst_model_dir, '{0}_ResNet50.pth'.format(dataset_name))

    src_test_img_dir = osp.join(SRC_DATA_ROOT_DIR, 'test')
    test_img_filenames = os.listdir(src_test_img_dir)

    class_names = CLASS_NAMES_DICT[dataset_name]
    testset = AsirraDataset(img_dir=src_test_img_dir,
                            img_filenames=test_img_filenames,
                            class_names=class_names,
                            transform=transforms.Compose([
                                Rescale((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD)
                            ]))
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=32,    # FIXME
                                                  shuffle=False, num_workers=4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = len(class_names)
    model_ft = torch.load(dst_model_path).to(device)

    correct = 0
    total = 0
    preds_np_list = []
    labels_np_list = []
    with torch.no_grad():
        for mb_idx, (inputs, labels) in enumerate(test_dataloader):
            if mb_idx % 100 == 0:
                print('Minibatch #{0}/{1}'.format(mb_idx, len(testset) // 32))
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)

            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            preds_np = preds.cpu().numpy()    # (n,)
            labels_np = labels.cpu().numpy()  # (n,)
            preds_np_list.append(preds_np)
            labels_np_list.append(labels_np)

    preds_np = np.concatenate(preds_np_list, axis=0)    # (N,)
    labels_np = np.concatenate(labels_np_list, axis=0)  # (N,)

    if num_classes > 2:
        c_mat = metrics.confusion_matrix(labels_np, preds_np)
        accuracy = np.sum(labels_np == preds_np) / labels_np.shape[0]

        print('Accuracy of the network on the {} test images: {} %%'.format(len(test_img_filenames), 100 * accuracy))
        print('Confusion matrix (row: true, col: pred): ')
        print(c_mat)
        np.savetxt('_test_confusion_matrix_{}.csv'.format(dataset_name), c_mat, delimiter=',')
        print('')

    else:    # num_classes == 2
        tp = np.sum(np.logical_and(labels_np == 1, preds_np == 1))
        fn = np.sum(np.logical_and(labels_np == 1, preds_np == 0))
        fp = np.sum(np.logical_and(labels_np == 0, preds_np == 1))
        tn = np.sum(np.logical_and(labels_np == 0, preds_np == 0))

        accuracy = (tp+tn) / (tp+fn+fp+tn)

        print('Accuracy of the network on the {} test images: {} %%'.format(len(testset), 100 * accuracy))
        print('TP: {}, FN: {}, FP: {}, TN: {}'.format(tp, fn, fp, tn))
        print('')


if __name__ == '__main__':
    main()
