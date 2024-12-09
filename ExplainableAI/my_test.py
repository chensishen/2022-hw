import os
import sys
import numpy as np
import random
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.wcs.docstrings import imgpix_matrix
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace
from torch.autograd import Variable
from argparse import Namespace
from tqdm.auto import tqdm


# 设置全局的随机种子
def all_seed(seed=6666):
    """
    设置随机种子
    """
    np.random.seed(seed)
    random.seed(seed)
    # CPU
    torch.manual_seed(seed)
    # GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    # python 全局
    os.environ['PYTHONHASHSEED'] = str(seed)
    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(f'Set env random_seed = {seed}')


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        def building_block(indim, outdim):
            return [
                nn.Conv2d(indim, outdim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(outdim),
                nn.ReLU(),
            ]

        def stack_blocks(indim, outdim, block_num):
            layer_list = building_block(indim, outdim)
            for i in range(block_num - 1):
                layer_list += building_block(indim, outdim)
            layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            return layer_list

        cnn_list = []
        cnn_list += stack_blocks(3, 64, 1)
        cnn_list += stack_blocks(64, 128, 1)
        cnn_list += stack_blocks(128, 256, 1)
        cnn_list += stack_blocks(256, 512, 1)
        cnn_list += stack_blocks(512, 512, 1)
        self.cnn = nn.Sequential(*cnn_list)
        dnn_list = [
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11),
        ]
        self.fc = nn.Sequential(*dnn_list)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):
        self.paths = paths
        self.labels = labels
        tr_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        evl_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        self.transform = tr_transform if mode == 'train' else evl_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        X = Image.open(self.paths[idx])
        X = self.transform(X)
        Y = self.labels[idx]
        return X, Y

    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)


def get_paths_labels(path):
    def my_key(name):
        return int(name.replace(".jpg", "").split("_")[1]) + 1000000 * int(name.split("_")[0].split("_")[0])

    imgnames = os.listdir(path)
    imgnames.sort(key=my_key)
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split("_")[0]))
    return imgpaths, labels

# 模型预测函数，LIME 将调用此函数获取模型的分类概率。
# 该函数需要接受图像输入并返回预测概率。



if __name__ == '__main__':
    args = Namespace(
        ckptpath='./checkpoints.ckpt',
        dataset_dir='../food11/training',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        seed=6666,
    )
    all_seed(args.seed)
    model = Classifier().to(args.device)
    checkpoint = torch.load(args.ckptpath,weights_only=False)
    model.load_state_dict(checkpoint)
    # print(model)
    # checkpoint = torch.load(args.ckptpath)
    # print(checkpoint.keys())
    # print({k: v.size() for k, v in checkpoint.items()})

    def predict(ipt):
        # ipt: numpy array, (batches, height, width, channels)
        model.eval()
        input = torch.FloatTensor(ipt).permute(0, 3, 1, 2)
        # pytorch tensor, (batches, channels, height, width)
        output = model(input.cuda())
        return output.data.cpu().numpy()
    # 图像分割函数，用于将图像分割为超像素（superpixels）。LIME 使用超像素来生成解释。
    def segmentation(input):
        return slic(input, n_segments=200, compactness=1, sigma=1, start_label=1)

    train_paths, train_labels = get_paths_labels(args.dataset_dir)
    # print(train_paths,train_labels)
    train_set = FoodDataset(train_paths, train_labels, mode='evl')
    img_indices = [i for i in range(10)]
    images, labels = train_set.getbatch(img_indices)
    fig, axs = plt.subplots(1, len(img_indices), figsize=(15, 8))
    _ = [(i.set_xticks([]), i.set_yticks([])) for i in axs]
    for i, img in enumerate(images):
        axs[i].imshow(img.cpu().permute(1, 2, 0)) #改变张量a的维度顺序，而不会改变数据本身的内容。
    plt.show()
    fig, axs = plt.subplots(1, len(img_indices), figsize=(15, 8))
    _ = [(i.set_xticks([]), i.set_yticks([])) for i in axs]
    for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
        x = image.astype(np.double)
        # numpy array for lime
        explainer = lime_image.LimeImageExplainer() # LIME 的图像解释器，用于生成图像分类模型的局部解释。
        explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=explain_instance#lime.lime_image.LimeImageExplainer.explain_instance
        # 从解释中提取可视化的图像和遮罩。
        # lime_img用颜色标记解释区域的图像。mask二值化的掩码图，指示哪些区域参与了解释。
        lime_img, mask = explaination.get_image_and_mask(
            label=label.item(), # 指定要解释的类别（标签）。
            positive_only=False, # 是否只显示对该类别有正面影响的区域。
            hide_rest=False, # 是否隐藏不相关区域。
            num_features=11,  # 进行解释的最大特征数-图片mask颜色不一样
            min_weight=0.05 # 最小权重，决定显示的区域对预测的最小贡献。
        )
        # turn the result from explainer to the image
        # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=get_image_and_mask#lime.lime_image.ImageExplanation.get_image_and_mask
        axs[idx].imshow(lime_img)
    plt.show()
    plt.close()
