import os
import math
import random
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchviz import make_dot

# 后期查一下*********
from torch.utils.data import ConcatDataset, Dataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder, VisionDataset

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def model_plot(model_class, input_sample):
    clf = model_class()
    y = clf(input_sample)
    clf_view = make_dot(y, params=dict(list(clf.named_parameters()) + [('x', input_sample)]))
    return clf_view

def all_seed(seed=6666):
    random.seed(seed)
    np.random.seed(seed)
    # cpu/gpu
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # python 全局
    os.environ['PYTHONHASHSEED'] = str(seed)
    # cudnn********
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(f'Set env random_seed = {seed}')


test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    # 可以改变*********************************
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
])

def quick_observe(train_dir_root):
    """
    快速观察训练集中的9张照片
    """
    pics_path = [os.path.join(train_dir_root, i) for i in os.listdir(train_dir_root)]
    labels = [i.split('_')[0] for i in os.listdir(train_dir_root)]
    idxs = np.arange(len(labels))
    sample_idx = np.random.choice(idxs, size=9, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    for idx_, i in enumerate(sample_idx):
        row = idx_ // 3
        col = idx_ % 3
        img=Image.open(pics_path[i])
        axes[row, col].imshow(img)
        c = labels[i]
        axes[row, col].set_title(f'class_{c}')

    plt.show()

class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files=None):
        super(Dataset).__init__()
        self.path = path
        self.transform = tfm
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = self.files
        print(f"One {path} sample", self.files[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(fname)
        img = self.transform(img)
        try:
            lable = int(fname.split("\\")[-1].split("_")[0])
        except:
            lable = -1
        return img, lable


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=512 * 4 * 4, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=11),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def trainer(train_loader, valid_loader, model, config, device, rest_net_flag=False):
    # 对于分类任务, 我们常用cross-entropy评估模型表现.
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)  # 交叉熵计算时，label范围为[0, n_classes-1]
    # 初始化优化器 *********************
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # 模型存储位置
    current_time = datetime.now()
    save_path = (config['save_path'] if False else config['resnet_save_path']) + '-' + str(
        current_time.day) + '-' + str(current_time.hour) + '-' + str(current_time.minute) + '.pth'

    writer = SummaryWriter()
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    for epoch in range(n_epochs):
        model.train()
        train_loss_record = []
        train_accs = []
        train_pbar = tqdm(train_loader, position=0, leave=True)  # *************
        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            #稳定训练的技巧 *******************
            if config['clip_flag']:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            step += 1
            acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()
            train_loss_record.append(loss.detach().item())
            train_accs.append(acc.detach().item())
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix(loss=loss.detach().item(), acc=acc.detach().item())

        mean_train_acc = sum(train_accs) / len(train_accs)
        mean_train_loss = sum(train_loss_record) / len(train_loss_record)
        # print('mean_train_acc:',mean_train_acc,'mean_train_loss:', mean_train_loss)
        writer.add_scalar('Train/Loss', mean_train_loss, step)
        writer.add_scalar('Train/Acc', mean_train_acc, step)

        model.eval()
        test_loss_record = []
        test_accs = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
                acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()
            test_loss_record.append(loss.detach().item())
            test_accs.append(acc.detach().item())
        mean_valid_acc = sum(test_accs) / len(test_accs)
        mean_valid_loss = sum(test_loss_record) / len(test_loss_record)
        print(f'epoch [{epoch + 1}/{n_epochs}]: '
              f'Train_loss : {mean_train_loss:.4f},acc:{mean_train_acc:.4f} '
              f'Valid loss: {mean_valid_loss:.4f},acc: {mean_valid_acc:.4f}')
        writer.add_scalar('Valid/Loss', mean_valid_loss, step)
        writer.add_scalar('Valid/Acc', mean_valid_acc, step)
        if mean_valid_acc < best_loss:
            best_loss = mean_valid_acc
            torch.save(model.state_dict(), save_path)
            print('Saving model with loss {:.4f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'seed': 6666,
    'n_epochs': 10,
    'batch_size': 64,
    'learning_rate': 0.0003,
    'early_stop': 300,
    'clip_flag': True,
    'weight_decay': 1e-5,
    'path': "./ml2022spring-hw3b/food11",
    'save_path': f'./models/model',
    'resnet_save_path': './models/resnet_model',
}
print(device)
all_seed(config['seed'])

# train_dir_root = './ml2022spring-hw3b/food11/training'
# quick_observe(train_dir_root)

path = config['path']
train_set = FoodDataset(os.path.join(path, "training"), train_tfm)
# ********************************
train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

valid_set = FoodDataset(os.path.join(path, "validation"), train_tfm)
valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

test_set = FoodDataset(os.path.join(path, "test"), test_tfm)
test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

test_set = FoodDataset(os.path.join(path,"test"), tfm=train_tfm)
test_loader_extra1 = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

test_set = FoodDataset(os.path.join(path,"test"), tfm=train_tfm)
test_loader_extra2 = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

test_set = FoodDataset(os.path.join(path,"test"), tfm=train_tfm)
test_loader_extra3 = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)


if __name__ == "__main__":
    model = Classifier().to(device)
    trainer(train_loader, valid_loader, model, config, device)

    # for x, y in train_loader:
    #     print(x.shape, y)
    # x = torch.zeros(1,3,128,128).requires_grad_(True)
    # print(model_plot(Classifier,x))
    # model = Classifier()
    # x = model(x)
    # print(model)
    # print(x.shape)

