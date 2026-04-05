from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# ─────────────────────────────────────────────
# 1. 命令行参数解析
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='数据加载子进程数（Windows 必须设为 0）', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='每批训练样本数')
parser.add_argument('--imageSize', type=int, default=64, help='输入图像尺寸（需为 2 的幂次）')
parser.add_argument('--nz', type=int, default=100, help='潜在向量 z 的维度')
parser.add_argument('--ngf', type=int, default=64, help='Generator 卷积核数量基数')
parser.add_argument('--ndf', type=int, default=64, help='Discriminator 卷积核数量基数')
parser.add_argument('--niter', type=int, default=25, help='训练轮数（epoch 数）')
parser.add_argument('--lr', type=float, default=0.0002, help='Adam 学习率')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam 的 beta1 参数')
parser.add_argument('--dry-run', action='store_true', help='只跑一个 iteration，验证流程是否正常')
parser.add_argument('--ngpu', type=int, default=1, help='使用的 GPU 数量')
parser.add_argument('--netG', default='', help='Generator 权重路径（用于继续训练）')
parser.add_argument('--netD', default='', help='Discriminator 权重路径（用于继续训练）')
parser.add_argument('--outf', default='.', help='输出目录（图片和模型保存位置）')
parser.add_argument('--manualSeed', type=int, help='固定随机种子，便于复现')
parser.add_argument('--classes', default='bedroom', help='LSUN 数据集的类别（逗号分隔）')
parser.add_argument('--accel', action='store_true', default=False, help='启用 GPU 加速（CUDA / Intel XPU）')

opt = parser.parse_args()
print(opt)

# 创建输出目录
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# ─────────────────────────────────────────────
# 2. 随机种子设置（保证实验可复现）
# ─────────────────────────────────────────────
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# 开启 cuDNN 自动寻找最优卷积算法，提升训练速度
cudnn.benchmark = True

# ─────────────────────────────────────────────
# 3. 设备选择：GPU（CUDA/XPU）或 CPU
# ─────────────────────────────────────────────
if opt.accel and torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
    raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)

# ─────────────────────────────────────────────
# 4. 数据集加载
#    - 所有图像统一 resize 到 imageSize x imageSize
#    - 像素值归一化到 [-1, 1]（对应 Generator 的 Tanh 输出）
#    - nc：图像通道数（RGB=3，灰度=1）
# ─────────────────────────────────────────────
if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # 自定义文件夹数据集，按子目录分类
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    # LSUN 大规模场景数据集，需提前下载
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    # CIFAR-10：10 类彩色图，自动下载
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
    # MNIST：手写数字灰度图，自动下载
    # 注意：原始尺寸 28x28，建议 --imageSize 32 避免 D 过强
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(opt.imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),  # 单通道归一化
                       ]))
    nc=1

elif opt.dataset == 'fake':
    # 随机生成假数据，用于快速验证流程（无需真实数据集）
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

# 从 opt 中提取超参数为局部变量，方便后续使用
ngpu = int(opt.ngpu)
nz = int(opt.nz)    # 潜在向量维度
ngf = int(opt.ngf)  # Generator 滤波器基数
ndf = int(opt.ndf)  # Discriminator 滤波器基数


# ─────────────────────────────────────────────
# 5. 权重初始化
#    论文建议：Conv 层权重 ~ N(0, 0.02)
#              BatchNorm 权重 ~ N(1, 0.02)，偏置 = 0
# ─────────────────────────────────────────────
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


# ─────────────────────────────────────────────
# 6. Generator（生成器）
#    输入：随机噪声向量 z，形状 (batch, nz, 1, 1)
#    输出：生成图像，形状 (batch, nc, imageSize, imageSize)
#    结构：一系列转置卷积（上采样）逐步放大特征图
#          4x4 → 8x8 → 16x16 → 32x32 → 64x64
#    激活：中间层用 ReLU，输出层用 Tanh（值域 [-1,1]）
# ─────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入 z: (nz) x 1 x 1
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 特征图: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 特征图: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 特征图: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 特征图: (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出图像: (nc) x 64 x 64
        )

    def forward(self, input):
        # 多 GPU 并行（单 GPU 或 CPU 时走 else 分支）
        if (input.is_cuda or input.is_xpu) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


# 实例化 Generator，应用权重初始化，可选加载已有权重
netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


# ─────────────────────────────────────────────
# 7. Discriminator（判别器）
#    输入：图像，形状 (batch, nc, imageSize, imageSize)
#    输出：每张图为真实图像的概率，形状 (batch,)
#    结构：一系列步长为 2 的卷积（下采样）逐步压缩特征图
#          64x64 → 32x32 → 16x16 → 8x8 → 4x4 → 1x1
#    激活：中间层用 LeakyReLU（斜率 0.2），输出层用 Sigmoid
# ─────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入图像: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征图: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征图: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征图: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征图: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # 输出概率值 [0, 1]
        )

    def forward(self, input):
        if (input.is_cuda or input.is_xpu) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)  # 展平为 (batch,)


# 实例化 Discriminator，应用权重初始化，可选加载已有权重
netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# ─────────────────────────────────────────────
# 8. 损失函数与优化器
# ─────────────────────────────────────────────
# BCELoss：二元交叉熵，用于衡量 D 的判断与真实标签的差距
criterion = nn.BCELoss()

# fixed_noise：固定噪声，每个 epoch 用同一批噪声生成图片，便于观察训练进展
fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)

real_label = 1  # 真实图像的标签
fake_label = 0  # 生成图像的标签

# Adam 优化器，论文推荐 lr=0.0002, beta1=0.5
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if opt.dry_run:
    opt.niter = 1  # dry-run 模式只跑 1 个 epoch

# ─────────────────────────────────────────────
# 9. 训练循环
#    每个 iteration 分两步：
#    步骤一：训练 D，目标是区分真假图像
#    步骤二：训练 G，目标是骗过 D
# ─────────────────────────────────────────────
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):

        ############################
        # 步骤一：更新 Discriminator
        # 目标：最大化 log(D(x)) + log(1 - D(G(z)))
        # 即：D 对真图输出接近 1，对假图输出接近 0
        ###########################

        netD.zero_grad()

        # --- 用真实 real 图像训练 D ---
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        # 真实图像的标签全为 1
        label = torch.full((batch_size,), real_label,
                           dtype=real_cpu.dtype, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)  # D 对真图的损失
        errD_real.backward()
        D_x = output.mean().item()  # D(x)：D 对真图的平均判断值，理想值接近 1

        # --- 用生成 fake 图像训练 D ---
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)          # G 生成假图
        label.fill_(fake_label)     # 假图标签全为 0
        output = netD(fake.detach())  # detach：不让梯度流回 G
        errD_fake = criterion(output, label)  # D 对假图的损失
        errD_fake.backward()
        D_G_z1 = output.mean().item()  # D(G(z))第一次：D 对假图的判断，理想值接近 0

        errD = errD_real + errD_fake  # D 的总损失
        optimizerD.step()

        ############################
        # 步骤二：更新 Generator
        # 目标：最大化 log(D(G(z)))
        # 即：让 D 把假图误判为真图（输出接近 1）
        ###########################

        netG.zero_grad()
        label.fill_(real_label)   # G 希望 D 把假图判为真（标签设为 1）
        output = netD(fake)       # 用更新后的 D 重新判断假图
        errG = criterion(output, label)  # G 的损失
        errG.backward()
        D_G_z2 = output.mean().item()  # D(G(z))第二次：G 更新后 D 对假图的判断，理想值接近 0.5
        optimizerG.step()

        # 打印训练进度
        # D(x) 理想值：0.5~0.8；D(G(z)) 理想值：前期接近 0，后期接近 0.5
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # 每 100 个 iteration 保存一次样本图
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)  # 用固定噪声生成，便于对比不同 epoch 的进展
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

        if opt.dry_run:
            break

    # 每个 epoch 结束保存模型权重（可用 --netG / --netD 加载继续训练）
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
