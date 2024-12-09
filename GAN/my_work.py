import os
import glob
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm

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


class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, index):
        fname = self.fnames[index]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

def get_dataset(root):
    fname = glob.glob(os.path.join(root, '*.jpg'))
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fname, transform)
    return dataset

# temp_dataset = get_dataset(os.path.join(workspace_dir, 'faces'))
# images = [temp_dataset[i] for i in range(8,12)]
# grid_img = torchvision.utils.make_grid(images,nrow=4)
# plt.figure(figsize=(10,10))
# plt.imshow(grid_img.permute(1,2,0))
# plt.show()

# 网络权重初始化 提升深度学习模型的训练稳定性和收敛速度。
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02) # 卷积层权重初始化为正态分布，均值为0，标准差为0.02
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) # 批归一化层的权重初始化为正态分布，均值为1，标准差为0.02
        m.bias.data.fill_(0)

# 生成器
class Generator(nn.Module):
    """
    Input shape: (batch, in_dim)
    Output shape: (batch, 3, 64, 64)
    """
    def __init__(self,in_dim,feature_dim=64):
        super(Generator, self).__init__()
        # input: 输入随机一维向量 (batch, 100) 随机生成噪点数据 -> (batch, 64 * 8 * 4 * 4)
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU()
        )

        # y.view(y.size(0), -1, 4, 4) -> 转成 (batch, feature_dim * 8, 4, 4)

        # 上采样并提取特征：逐步将channel中的特征信息转到 height and width 维度
        self.l2 = nn.Sequential(
            self.dconv_bn_relu(feature_dim * 8, feature_dim * 4),  # out_put -> (batch, feature_dim * 4, 8, 8)
            self.dconv_bn_relu(feature_dim * 4, feature_dim * 2),  # out_put -> (batch, feature_dim * 2, 16, 16)
            self.dconv_bn_relu(feature_dim * 2, feature_dim),  # out_put -> (batch, feature_dim, 32, 32)
        )

        # out_put -> (batch, 3, 64, 64) channel dim=1
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            #将最后一层的特征（形状 (batch, feature_dim, 32, 32)）转为目标图像（形状 (batch, 3, 64, 64)）。
            nn.Tanh() # Tanh 激活函数将输出值限制在 [-1, 1] 范围内，适配通常对图像归一化的范围。
        )
        self.apply(weights_init)

    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            # 双倍 height and width 放大图像尺寸
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4) # (batch_size, feature_dim * 8, 4, 4)
        y = self.l2(y)
        y = self.l3(y)
        return y

# Discriminator: 判别器  生成img 和 真实img
class Discriminator(nn.Module):
    """
    输入: (batch, 3, 64, 64)
    输出: (batch)
    """

    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator, self).__init__()

        # input: (batch, 3, 64, 64)
        """
        设置Discriminator的注意事项:
        在WGAN中需要移除最后一层 sigmoid
        """
        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1),  # output -> (batch, 64, 32, 32)
            nn.LeakyReLU(0.2), #线性整流。
            # 提取更高层次的特征，并逐步降低图像的空间分辨率
            self.conv_bn_lrelu(feature_dim, feature_dim * 2),  # output -> (batch, 128, 32, 32)
            self.conv_bn_lrelu(feature_dim * 2, feature_dim * 4),  # output -> (batch, 256, 32, 32)
            self.conv_bn_lrelu(feature_dim * 4, feature_dim * 8),  # output -> (batch, 512, 32, 32)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),  # output -> (batch, 1, 1, 1)
            # nn.Sigmoid()
        )
        self.apply(weights_init)

    def conv_bn_lrelu(self, in_dim, out_dim):
        """
        设置Discriminator的注意事项:
        在WGAN-GP中不能使用 nn.Batchnorm， 需要使用 nn.InstanceNorm2d 替代
        """
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            # nn.BatchNorm2d(out_dim), #因为 WGAN-GP 不需要在批次之间共享归一化信息。
            nn.BatchNorm2d(out_dim),
            #用于归一化特征分布，稳定训练。
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(-1)
        return y


class TrainerGAN():
    def __init__(self,config):
        self.config = config
        self.G = Generator(100)
        self.D = Discriminator(3)
        self.loss = nn.BCELoss()
        """
           优化器设置注意：
           GAN: 使用 Adam optimizer
           WGAN: 使用 RMSprop optimizer
           WGAN-GP: 使用 Adam optimizer 
           参数 betas=(0.5, 0.999) 是 GAN 中常用的配置，帮助稳定训练。
        """
        # self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        self.opt_D = torch.optim.RMSprop(self.D.parameters(),lr=self.config["lr"])
        # self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        self.opt_G = torch.optim.RMSprop(self.G.parameters(),lr=self.config["lr"])
        self.dataloader = None

        self.log_dir = os.path.join(self.config["workspace_dir"], 'logs')
        self.ckpt_dir = os.path.join(self.config["workspace_dir"], 'checkpoints')

        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO,
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')

        self.steps = 0
        #是训练中固定的一批随机噪声，用于生成样本，便于观察生成器在不同训练阶段的表现。
        self.z_samples = Variable(torch.randn(100, self.config["z_dim"])).cuda()

    def prepare_environment(self):
        """
        训练前环境、数据与模型准备
        """
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 基于时间更新日志和ckpt文件名
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.log_dir, time + f'_{self.config["model_type"]}')
        self.ckpt_dir = os.path.join(self.ckpt_dir, time + f'_{self.config["model_type"]}')
        os.makedirs(self.log_dir)
        os.makedirs(self.ckpt_dir)

        dataset = get_dataset(os.path.join(self.config["workspace_dir"], 'faces'))
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)

        # 模型准备
        self.G = self.G.cuda()
        self.D = self.D.cuda()
        self.G.train()
        self.D.train()

        def gp(self):
            """
            实现梯度惩罚功能
            实现 gp 函数需要以下步骤：
            在真实数据和生成数据之间采样一个插值样本。
            计算插值样本的梯度。
            将梯度的范数约束到 1。
            """
            pass

    def train(self):
        """
        训练 generator 和 discriminator
        """
        self.prepare_environment()

        for e,epoch in enumerate(range(self.config["n_epoch"])):
            progress_bar = tqdm(self.dataloader, desc=f'Epoch {e+1}')
            for i,data in enumerate(progress_bar):
                imgs = data.cuda()
                bs = imgs.size(0)
                # *********************
                # *    Train D-判别器  *
                # *********************
                z = Variable(torch.randn(bs,self.config["z_dim"])).cuda()
                r_imgs = Variable(imgs).cuda()
                # 生成器生成假照片
                f_imgs = self.G(z)

                r_label = torch.ones((bs)).cuda()
                f_label = torch.zeros((bs)).cuda()

                # Discriminator前向传播 # 判别器的预测
                r_logit = self.D(r_imgs)
                f_logit = self.D(f_imgs)

                """
                DISCRIMINATOR损失设置注意:
                GAN:  loss_D = (r_loss + f_loss)/2
                WGAN: loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
                WGAN-GP: gradient_penalty = self.gp(r_imgs, f_imgs)
                loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
                """

                # discriminator的损失: (评估 是否能区分真实图片和生成图片)
                #    生成fake->判别logit VS 0    &  real->判别logit VS 1

                # r_loss = self.loss(r_logit, r_label)
                # f_loss = self.loss(f_logit, f_label)
                # loss_D = (r_loss + f_loss) / 2

                loss_D = -torch.mean(r_logit) + torch.mean(f_logit)

                # Discriminator 反向传播
                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                """
                设置 WEIGHT CLIP 注意:
                WGAN: 使用以下code
                """
                for p in self.D.parameters():
                    p.data.clamp_(-self.config["clip_value"], self.config["clip_value"])

                # *********************
                # *    Train G-生成器  *
                # *********************
                if self.steps % self.config["n_critic"] == 0:
                    # 生成一些假照片
                    z = Variable(torch.randn(bs, self.config["z_dim"])).cuda()
                    f_imgs = self.G(z)

                    # Generator前向传播
                    f_logit = self.D(f_imgs)

                    """
                    生成器损失函数设置注意：
                        GAN: loss_G = self.loss(f_logit, r_label)
                        WGAN: loss_G = -torch.mean(self.D(f_imgs))
                        WGAN-GP: loss_G = -torch.mean(self.D(f_imgs))
                    """
                    # 生成器损失(评估 生成图片和真实 是否很接近): 生成->判别logit VS 1
                    # loss_G = self.loss(f_logit, r_label)
                    loss_G = -torch.mean(self.D(f_imgs))

                    # Generator反向传播
                    self.G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()

                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
                self.steps += 1
            self.G.eval()
            f_imgs_sample = (self.G(self.z_samples).data + 1) / 2.0
            filename = os.path.join(self.log_dir, f'Epoch_{epoch + 1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            logging.info(f'Save some samples to {filename}.')

            # 在训练过程中显示图片
            grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

            self.G.train()

            if (e + 1) % 5 == 0 or e == 0:
                # 保存checkpoints.
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{e}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.ckpt_dir, f'D_{e}.pth'))

    logging.info('Finish training')

    # def inference(self, G_path, n_generate=1000, n_output=30, show=False):
    #     """
    #     1. G_path： 生成器ckpt路径
    #     2. 可以使用此函数生成最终答案
    #     """
    #
    #     self.G.load_state_dict(torch.load(G_path))
    #     self.G.cuda()
    #     self.G.eval()
    #     z = Variable(torch.randn(n_generate, self.config["z_dim"])).cuda()
    # 生成的图片值通常是 [-1, 1] 范围内的，需要将其转换到 [0, 1] 范围内，以便保存为图片。
    #     imgs = (self.G(z).data + 1) / 2.0
    #
    #     os.makedirs('output', exist_ok=True)
    #     for i in range(n_generate):
    #         torchvision.utils.save_image(imgs[i], f'output/{i + 1}.jpg')
    #
    #     if show:
    #         row, col = n_output // 10 + 1, 10
    #         grid_img = torchvision.utils.make_grid(imgs[:n_output].cpu(), nrow=row)
    #         plt.figure(figsize=(row, col))
    #         plt.imshow(grid_img.permute(1, 2, 0))
    #         plt.show()

if __name__ == '__main__':
    workspace_dir = './faces'

    config = {
        "model_type": "GAN",
        "batch_size": 64,
        "lr": 1e-4,
        "n_epoch": 5,
        "n_critic": 1,
        "z_dim": 100,
        "workspace_dir": workspace_dir,
        "clip_value":0.01,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_seed(2022)
    trainer = TrainerGAN(config)
    trainer.train()