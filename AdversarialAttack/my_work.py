import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8

# 平均值和标准差是根据cifar10数据集计算的统计数据
cifar_10_mean = (0.491, 0.482, 0.447) # cifar_10 图片数据三个通道的均值
cifar_10_std = (0.202, 0.199, 0.201) # cifar_10 图片数据三个通道的标准差

# 将mean和std转换为三维张量，用于未来的运算
mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

epsilon = 8/255/std

root = './data' # 用于存储`benign images`的目录
# benign images: 不包含对抗性扰动的图像
# adversarial images: 包括对抗性扰动的图像

import os
import glob
import shutil
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_10_mean, cifar_10_std)
])

class AdvDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.images = []
        self.labels = []
        self.names = []
        '''
           data_dir
           ├── class_dir
           │   ├── class1.png
           │   ├── ...
           │   ├── class20.png
        '''
        for i, class_dir in enumerate(sorted(glob.glob(os.path.join(data_dir, '*')))):
            images = sorted(glob.glob(os.path.join(class_dir, '*')))
            self.images += images
            self.labels += [i] * len(images)
            self.names += [os.path.relpath(imgs,data_dir) for imgs in images]
        self.transform = transform
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        label = self.labels[idx]
        return image, label
    def __getname__(self):
        return self.names
    def __len__(self):
        return len(self.images)

adv_set = AdvDataset(root, transform)
adv_names = adv_set.__getname__()
adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)
# print(f'number of images = {adv_set.__len__()}')

# 评估模型在良性图像上的性能
def epoch_benign(model, loader, loss_fn):
    model.eval()
    train_acc, train_loss = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1)==y).sum().item()
        train_loss += loss.item()*x.shape[0]
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)

# 快速梯度符号法（FGSM，Fast Gradient Sign Method），用于生成对抗样本（adversarial examples）
# FGSM 是一种常用的对抗攻击方法，它通过对输入数据施加微小的扰动来欺骗深度学习模型。
def fgsm(model,x,y,loss_fn,epsilon=epsilon):
    """
    :param model:神经网络模型
    :param x:图片
    :param y:真实标签
    :param loss_fn:用于度量模型输出和真实标签之间的误差。
    :param epsilon:扰动强度系数，控制对抗样本的变化幅度。
    :return:返回生成的对抗样本。
    """
    # 使用 .detach() 从计算图中分离出来，避免干扰梯度计算，并通过 .clone() 深拷贝确保原始数据不被修改。
    x_adv = x.detach().clone() # 用良性图片初始化 x_adv
    x_adv.requires_grad = True # 需要获取 x_adv 的梯度
    loss = loss_fn(model(x_adv), y) # 计算损失
    loss.backward()
    # fgsm: 在x_adv上使用梯度上升来最大化损失
    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon*grad.sign()
    return x_adv

# 迭代快速梯度符号法（I-FGSM，Iterative Fast Gradient Sign Method），是 FGSM 的扩展版本。
# 通过多次迭代，每次施加一个小的扰动，I-FGSM 通常能够生成更强的对抗样本。
# 在“全局设置”部分中将alpha设置为步长
# alpha和num_iter可以自己决定设定成何值
alpha = 0.8 / 255 / std
def ifgsm(model,x,y,loss_fn,epsilon=epsilon,alpha=alpha,num_iter=20):
    x_adv = x
    # num_iter 次迭代
    for i in range(num_iter):
        x_adv = fgsm(model,x_adv,y,loss_fn,alpha) # 用（ε=α）调用fgsm以获得新的x_adv
        x_adv = torch.max(torch.min(x_adv, x+epsilon),x-epsilon) # x_adv 裁剪到 [x-epsilon, x+epsilon]范围
        # 限制对抗样本的变化不会超过原始输入的约束范围，防止过大扰动导致明显的人工痕迹。
    return x_adv

# 带动量的迭代快速梯度符号法（MI-FGSM, Momentum Iterative Fast Gradient Sign Method）。
# MI-FGSM 在 I-FGSM 的基础上引入了动量机制（momentum），通过累积梯度信息增强对抗样本生成的方向性，
# 使其更难被防御。
def mifgsm(model,x,y,loss_fn,epsilon=epsilon,alpha=alpha,num_iter=20,decay=0.7):
    # decay: 动量衰减系数，用于控制动量的更新幅度。
    x_adv = x
    # 初始化 momentum tensor
    momentum = torch.zeros_like(x).detach().to(device)
    # num_iter 次迭代
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        # TODO: Momentum calculation
        grad = x_adv.grad.detach() + (1-decay)*momentum
        momentum = grad
        x_adv = x_adv + alpha*grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+epsilon),x-epsilon)  # x_adv 裁剪到 [x-epsilon, x+epsilon]范围
    return x_adv

# 执行对抗性攻击 并 生成对抗性示例
def gen_adv_examples(model, loader, attack, loss_fn):
    model.eval()
    adv_names = []
    train_acc, train_loss = 0.0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y,loss_fn)  # 获得对抗性示例
        yp = model(x_adv)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1)==y).sum().item()
        train_loss += loss.item()*x.shape[0]
        # 保存对抗性示例
        """
        将对抗样本 x_adv 反标准化和反归一化：
        然后将值裁剪到有效范围：
        第一轮裁剪:像素值范围 [0, 1]
        第二轮裁剪:像素值范围 [0, 255]
        """
        adv_ex = (x_adv*std +mean).clamp(min=0., max=1.)
        adv_ex = (adv_ex*255).clamp(min=0, max=255)
        # 转换为 NumPy 数组，并对像素值取整。
        adv_ex = adv_ex.detach().cpu().data.numpy().round()
        adv_ex = adv_ex.transpose((0, 2, 3, 1))
        # 如果是第一批，直接初始化 adv_examples。
        # 否则，将当前批次的对抗样本拼接到 adv_examples。
        adv_examples = adv_ex if i ==0 else np.r_[adv_examples, adv_ex]
    return adv_examples, train_acc / len(loader.dataset), train_loss / len(loader.dataset)

# 定义函数 create_dir，实现对抗样本的存储功能。
def create_dir(data_dir, adv_dir, adv_examples, adv_names):
    """
    :param data_dir:原始数据集目录，用于参考或复制结构。
    :param adv_dir:对抗样本存放的目标目录。
    :param adv_examples: 对抗样本数组，包含多个对抗图像。
    :param adv_names:对应对抗样本的文件名列表。
    """
    if os.path.exists(adv_dir) is not True:
        _ = shutil.copytree(data_dir, adv_dir)
    for example, name in zip(adv_examples, adv_names):
        # Image.fromarray(): 使用 Pillow 创建图像对象。
        im = Image.fromarray(example.astype(np.uint8))
        im.save(os.path.join(adv_dir, name))

from pytorchcv.model_provider import  get_model as ptcv_get_model

model = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device)
loss_fn = nn.CrossEntropyLoss()

def train1():
    benign_acc, benign_loss = epoch_benign(model, adv_loader, loss_fn)
    print(f'[ Base(未Attack图片评估) ] benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')

    adv_examples, fgsm_acc, fgsm_loss = gen_adv_examples(model, adv_loader, fgsm, loss_fn)
    print(f'[ Attack(FGSM Attack图片评估) ] fgsm_acc = {fgsm_acc:.5f}, fgsm_loss = {fgsm_loss:.5f}')

    create_dir(root, 'fgsm', adv_examples, adv_names)

    adv_examples, ifgsm_acc, ifgsm_loss = gen_adv_examples(model, adv_loader, ifgsm, loss_fn)
    print(f'[ Attack(I-FGSM Attack图片评估) ] ifgsm_acc = {ifgsm_acc:.5f}, ifgsm_loss = {ifgsm_loss:.5f}')

    create_dir(root, 'ifgsm', adv_examples, adv_names)

    adv_examples, mifgsm_acc, mifgsm_loss = gen_adv_examples(model, adv_loader, mifgsm, loss_fn)
    print(f'[ Attack(MI-FGSM Attack图片评估) ] mifgsm_acc = {mifgsm_acc:.5f}, mifgsm_loss = {mifgsm_loss:.5f}')

    create_dir(root, 'mifgsm_decay=0.7', adv_examples, adv_names)

class ensembleNet(nn.Module):
    def __init__(self, model_names):
        super().__init__()
        self.model = nn.ModuleList([ptcv_get_model(name,pretrained=True) for name in model_names])
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        for i ,m in enumerate(self.model):
            # TODO: sum up logits from multiple models
            if i == 0:
                res = m(x)
                continue
            res += m(x)
        # 用于对最终的输出 logits 进行归一化，将它们转化为概率分布。
        # 参数 dim=1 表示沿类别维度执行 Softmax 操作（适用于分类任务）。
        return self.softmax(res)

import matplotlib.pyplot as plt

model_names = [
    'nin_cifar10',
    'resnet20_cifar10',
    'preresnet20_cifar10'
]
ensemble_model = ensembleNet(model_names).to(device)
ensemble_model.eval()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def test1():
    plt.figure(figsize=(10, 20))
    cnt = 0
    for i, cls_name in enumerate(classes):
        path =  f'{cls_name}/{cls_name}1.png'
        # 未Attack图片（benign image）
        cnt += 1
        # 10,4,cnt
        plt.subplot(len(classes), 4, cnt)
        im = Image.open(f'./data/{path}')
        logit = model(transform(im).unsqueeze(0).to(device))[0]
        predict = logit.argmax(-1).item()
        prob = logit.softmax(-1)[predict].item()
        plt.title(f'benign: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
        plt.axis('off')
        plt.imshow(np.array(im))
        # Attack后图片（adversarial image）
        cnt += 1
        plt.subplot(len(classes), 4, cnt)
        im = Image.open(f'./ifgsm/{path}')
        logit = model(transform(im).unsqueeze(0).to(device))[0]
        predict = logit.argmax(-1).item()
        prob = logit.softmax(-1)[predict].item()
        plt.title(f'adversarial: {cls_name}1.png\n{classes[predict]}: {prob:.2%}')
        plt.axis('off')
        plt.imshow(np.array(im))
    plt.tight_layout()
    plt.show()

# test1()

def test2():
    # original image
    path = f'dog/dog2.png'
    im = Image.open(f'./data/{path}')
    logit = model(transform(im).unsqueeze(0).to(device))[0]
    predict = logit.argmax(-1).item()
    prob = logit.softmax(-1)[predict].item()
    plt.title(f'benign: dog2.png\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')
    plt.imshow(np.array(im))
    plt.tight_layout()
    plt.show()

    # adversarial image
    im = Image.open(f'./ifgsm/{path}')
    logit = model(transform(im).unsqueeze(0).to(device))[0]
    predict = logit.argmax(-1).item()
    prob = logit.softmax(-1)[predict].item()
    plt.title(f'adversarial: dog2.png\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')
    plt.imshow(np.array(im))
    plt.tight_layout()
    plt.show()

# test2()

def test3():
    import imgaug.augmenters as iaa
    # 预处理image
    path = f'dog/dog11.png'
    im = Image.open(f'./fgsm/{path}')
    x = transforms.ToTensor()(im) * 255
    x = x.permute(1, 2, 0).numpy()
    compressed_x = x.astype(np.uint8)

    logit = model(transform(compressed_x).unsqueeze(0).to(device))[0]
    predict = logit.argmax(-1).item()
    prob = logit.softmax(-1)[predict].item()
    plt.title(f'JPEG adversarial: dog2.png\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')
    plt.imshow(compressed_x)
    plt.tight_layout()
    plt.show()

    # TODO: use "imgaug" package to perform JPEG compression (compression rate = 70)
    # 创建一个压缩强度为 70 的 JPEG 压缩模型。
    cmp_model = iaa.arithmetic.JpegCompression(compression=90)
    compressed_x = cmp_model(images=compressed_x)

    logit = model(transform(compressed_x).unsqueeze(0).to(device))[0]
    predict = logit.argmax(-1).item()
    prob = logit.softmax(-1)[predict].item()
    plt.title(f'JPEG assive Defense: dog2.png\n{classes[predict]}: {prob:.2%}')
    plt.axis('off')
    plt.imshow(compressed_x)
    plt.tight_layout()
    plt.show()

test3()















