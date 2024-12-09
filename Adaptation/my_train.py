# import matplotlib.pyplot as plt
#
# def no_axis_show(img,title='',cmap=None):
#     fig = plt.imshow(img,interpolation = 'nearest',cmap = cmap)
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)
#     plt.title(title)
#
# titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'tetevision', 'dog', 'dolphin', 'spider']
# plt.figure(figsize=(18,4))
# for i in range(10):
#     plt.subplot(2,10,i+1)
#     fig = no_axis_show(plt.imread(f'real_or_drawing/train_data/{i}/{500*i}.bmp'),title=titles[i])
#
# for i in range(10):
#     plt.subplot(2,10,10+i+1)
#     fig = no_axis_show(plt.imread(f'real_or_drawing/test_data/0/' + str(i).rjust(5, '0') + '.bmp'))
#
# plt.show()
#
# import cv2
# import matplotlib.pyplot as plt
# titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
# plt.figure(figsize=(18,18))
#
# original_img = plt.imread(f'real_or_drawing/train_data/0/0.bmp')
# plt.subplot(1, 5, 1)
# no_axis_show(original_img, title='original')
#
# gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
# plt.subplot(1, 5, 2)
# no_axis_show(gray_img, title='gray scale', cmap='gray')
#
# canny_50100 = cv2.Canny(gray_img, 50, 100)
# plt.subplot(1, 5, 3)
# no_axis_show(canny_50100, title='Canny(50, 100)', cmap='gray')
#
# canny_150200 = cv2.Canny(gray_img, 150, 200)
# plt.subplot(1, 5, 4)
# no_axis_show(canny_150200, title='Canny(150, 200)', cmap='gray')
#
# canny_250300 = cv2.Canny(gray_img, 250, 300)
# plt.subplot(1, 5, 5)
# no_axis_show(canny_250300, title='Canny(250, 300)', cmap='gray')
#
# plt.show()

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch import optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import cv2


src_transform = transforms.Compose([
    # 将 RGB  转换成 greyscale
    transforms.Grayscale(),
    # cv2 不支持 skimage.Image, 需要先装换成 np.array 并采用 cv2.Canny 算法
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    # np.array -> skimag.Image
    transforms.ToPILImage(),
    # 50% 水平翻转(数据增强)
    transforms.RandomHorizontalFlip(),
    # 旋转 +- 15 度,并对旋转后空缺的地方填充0
    transforms.RandomRotation(15,fill=(0,)),
    # 转成tensor 作为模型输入
    transforms.ToTensor(),
])

# 区别是不进行 Canny 识别轮廓
tgt_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15,fill=(0,)),
    transforms.ToTensor(),
])

source_dataset = ImageFolder('real_or_drawing/train_data', transform=src_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=tgt_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.layer(x)
        return x

# 标签预测器（Label Predictor）：对源域数据的标签进行分类。
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.layer(x)
        return x

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

class_criterion = nn.CrossEntropyLoss().cuda()
domain_criterion = nn.BCEWithLogitsLoss().cuda()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())

# 域自适应（Domain Adaptation）**任务
def train_epoch(data_loader, target_dataloader, lamb):
    '''
        Args:
        source_dataloader: source data的dataloader 用于加载源域数据（source domain）。
        target_dataloader: target data的dataloader 用于加载目标域数据（target domain）。
        lamb: 控制域自适应(domain adaptatoin) 和分类的平衡
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    # 分别累计域分类器的损失和特征提取器的损失，用于评估模型训练状态。
    running_D_loss, running_F_loss = 0.0, 0.0
    # 预测正确的数量和总样本数，用于计算分类精度。
    total_hit, total_num = 0.0, 0.0
    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        # 混合 source data 和 target data,否则会误导正在运行的batch_norm的参数(运行源数据和目标数据的 mean/var 不同)
        mixed_data = torch.cat([source_data, target_data], 0)
        # 构造域标签，1表示源域，0表示目标域。
        domain_label = torch.zeros([source_data.shape[0]+target_data.shape[0], 1]).cuda()
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : 训练 domain classifier
        feature = feature_extractor(mixed_data)
        # 在第一部我们不需要训练特征提取器，所以需要对特征进行 .detach() 操作，防止反向传播
        domain_logits = domain_classifier(feature.detach())
        # 域分类器的目标是最小化与 domain_label 的交叉熵损失。
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : 训练 feature extractor 和 label classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        # 这里的特征未使用 .detach()，因此更新 feature_extractor 的参数，使源域和目标域的特征分布更难以区分（即实现对抗）。
        domain_logits = domain_classifier(feature)
        # loss = label分类损失 - lamb * 域分类损失.
        # 让 feature_extractor 的参数方向与 domain_classifier 的优化方向相反。
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        # 对源域数据进行分类预测，计算正确预测的样本数。
        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i,end='\r')

    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num

if __name__=="__main__":
    for epoch in range(200):
        train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=0.1)

        torch.save(feature_extractor.state_dict(), './feature_extractor.bin')
        torch.save(label_predictor.state_dict(), './label_predictor.bin')

        print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))

    result = []
    label_predictor.eval()
    feature_extractor.eval()
    for i, (test_data, _) in enumerate(test_dataloader):
        test_data = test_data.cuda()
        class_logits = label_predictor(feature_extractor(test_data))
        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        result.append(x)

    import pandas as pd
    result = pd.concatenate(result)

    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv('result.csv')










