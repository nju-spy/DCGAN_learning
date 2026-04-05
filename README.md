# 深度卷积生成对抗网络（DCGAN）

本项目实现了论文 [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)。

每训练 100 个 iteration，会将真实样本图和生成样本图保存到磁盘：
- `real_samples.png` — 来自数据集的真实图像
- `fake_samples_epoch_XXX.png` — Generator 在当前 epoch 的生成图像

每个 epoch 结束后，模型权重保存为：
- `netG_epoch_%d.pth` — Generator 权重
- `netD_epoch_%d.pth` — Discriminator 权重

## 工作原理

DCGAN 由两个网络组成，互相对抗训练：

- **Generator（G）**：接收随机噪声向量 z，通过转置卷积逐步上采样，生成与真实图像尺寸相同的假图像。目标是让 D 无法区分真假。
- **Discriminator（D）**：接收图像（真实或生成），输出一个 [0,1] 的概率值，判断图像是否为真实图像。目标是准确区分真假。

两者交替训练，G 不断提升生成质量，D 不断提升判别能力，最终达到博弈均衡。

---

### Generator 网络结构与尺度变化

Generator 的核心任务是把一个 `(nz, 1, 1)` 的随机噪声向量"画"成一张 `(nc, 64, 64)` 的图像。
每一层都是**转置卷积（ConvTranspose2d）+ BatchNorm + ReLU**，空间尺寸每层翻倍，通道数每层减半，最终一层用 Tanh 把像素值压到 `[-1, 1]`。

> 默认超参数：`nz=100`，`ngf=64`，`nc=3`（RGB）

| 层序 | 操作 | 参数（in_ch, out_ch, kernel, stride, pad） | 输出形状（C × H × W） | 激活 | 设计意图 |
|------|------|------------------------------------------|----------------------|------|---------|
| 输入 | — | — | `nz × 1 × 1` | — | 随机噪声 z，作为"种子" |
| 1 | ConvTranspose2d + BN | `nz → ngf×8`，4, 1, 0 | `512 × 4 × 4` | ReLU | stride=1, pad=0 把 1×1 直接展开为 4×4；通道最多，抽象语义最丰富 |
| 2 | ConvTranspose2d + BN | `ngf×8 → ngf×4`，4, 2, 1 | `256 × 8 × 8` | ReLU | stride=2 每次将边长翻倍；通道减半，逐渐从"语义"过渡到"结构" |
| 3 | ConvTranspose2d + BN | `ngf×4 → ngf×2`，4, 2, 1 | `128 × 16 × 16` | ReLU | 同上，细节层次继续丰富 |
| 4 | ConvTranspose2d + BN | `ngf×2 → ngf×1`，4, 2, 1 | `64 × 32 × 32` | ReLU | 接近真实分辨率，学习局部纹理 |
| 5 | ConvTranspose2d | `ngf×1 → nc`，4, 2, 1 | `3 × 64 × 64` | Tanh | 输出 RGB 图像；Tanh 把值域限制在 [-1,1]，与数据归一化对齐 |

相应的，Discriminator 下采样把 `(nc, 64, 64)` 的图像，映射到 [0,1] 概率值。

**尺度变化一览（H/W）：**

```
1×1  →[层1]→  4×4  →[层2]→  8×8  →[层3]→  16×16  →[层4]→  32×32  →[层5]→  64×64
```

**各设计选择的原因：**

| 设计 | 原因 |
|------|------|
| 转置卷积（而非插值上采样） | 让网络自己学习上采样方式，可学习参数更灵活，论文实验证明效果更好 |
| 通道数 `ngf×8 → ngf×4 → … → nc` 逐层减半 | 深层通道多→表示复杂语义；浅层通道少→专注于像素级细节，是 CNN 的通用设计范式 |
| BatchNorm（除输出层外） | 稳定梯度流、防止训练崩溃；输出层去掉 BN 是为了让 Tanh 输出的值域不被强制收缩 |
| 中间层 ReLU，输出层 Tanh | ReLU 保持稀疏激活、训练稳定；Tanh 输出 [-1,1] 与数据预处理（`Normalize(0.5, 0.5)`）对齐 |
| bias=False | 有 BatchNorm 时 bias 冗余（BN 自带可学习的平移参数 β），去掉可减少参数量 |
| kernel=4, stride=2, pad=1 | 经典的"×2 上采样"公式：`H_out = (H_in-1)×stride - 2×pad + kernel = 2H_in`，整数倍放大无棋盘格伪影 |

---

### 损失函数设计

两个网络都使用**二元交叉熵（BCELoss）**，但优化方向相反。

**Discriminator 的损失：**

```
Loss_D = -[log(D(x)) + log(1 - D(G(z)))]
```

- `D(x)`：D 对真实图像的判断，标签为 1，希望输出接近 1
- `D(G(z))`：D 对生成图像的判断，标签为 0，希望输出接近 0
- 两项相加，D 同时学习"认出真图"和"识破假图"

1. 真实图像 x 送入 D，计算 loss: BCELoss(D(x), 1)
2. 噪声 z 生成假图 G(z)，计算 loss: BCELoss(D(G(z)), 0)
3. 合并两个 loss，反向传播，更新 D

**Generator 的损失：**

```
Loss_G = -log(D(G(z)))
```

- G 把生成图送入 D，但标签设为 1（假装是真图）
- D 输出越接近 1，Loss_G 越小，说明 G 成功骗过了 D
- 注意：G 的参数更新时，D 的权重是冻结的

**训练健康的参考指标：**

| 指标 | 理想范围 | 含义 |
|------|---------|------|
| `Loss_D` | 0.3 ~ 0.8 | D 有一定难度，但没被完全骗过 |
| `Loss_G` | 1.0 ~ 3.0 | G 在努力骗 D，但还没完全成功 |
| `D(x)` | 0.6 ~ 0.85 | D 能识别大部分真图，但不过分自信 |
| `D(G(z))` | 0.3 ~ 0.5 | G 生成的图有一定迷惑性 |

---

### 防止 D 和 G 能力差距过大

GAN 训练最常见的问题是 D 碾压 G（`D(x)≈1, D(G(z))≈0`），导致 G 梯度消失、无法学习。

**调参方向：**

| 问题 | 现象 | 解决方法 |
|------|------|---------|
| D 太强 | `Loss_D < 0.1`，`D(G(z)) < 0.05` | 降低 `--ndf`，或提高 `--lr`（只对 G）|
| G 太强 | `Loss_G < 0.5`，`D(x) < 0.5` | 降低 `--ngf`，或增加 D 的训练次数 |
| 图像尺寸不匹配 | MNIST 用默认 64 时 D 过强 | 加 `--imageSize 32` |
| 训练不稳定 | Loss 剧烈震荡 | 降低学习率，如 `--lr 0.0001` |

**实用技巧：**

- MNIST 使用 `--imageSize 32`，避免 28×28 图像被放大后纹理过于简单
- `--beta1 0.5` 是 GAN 训练的经验值，比默认的 0.9 更稳定
- 如果 D 已经过强，可以用 `--netG` 单独加载 G 的权重，重新初始化 D 再训练

---

## 下载数据集

LSUN 数据集可以克隆 [此仓库](https://github.com/fyu/lsun) 后运行：

```bash
python download.py -c bedroom
```

MNIST、CIFAR-10 会在训练时通过 `torchvision` 自动下载。

## 使用方法

```
python main.py --dataset DATASET [选项]
```

**必填参数：**

| 参数 | 可选值 |
|------|--------|
| `--dataset` | cifar10 \| lsun \| mnist \| imagenet \| folder \| lfw \| fake |

**常用可选参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataroot` | — | 数据集路径（fake 数据集可不填）|
| `--outf` | `.` | 输出目录（图片和模型保存位置）|
| `--niter` | 25 | 训练轮数（epoch 数）|
| `--batchSize` | 64 | 每批训练样本数 |
| `--imageSize` | 64 | 输入图像尺寸（需为 2 的幂次，MNIST 建议用 32）|
| `--nz` | 100 | 潜在向量 z 的维度 |
| `--ngf` | 64 | Generator 卷积核数量基数 |
| `--ndf` | 64 | Discriminator 卷积核数量基数 |
| `--lr` | 0.0002 | Adam 学习率 |
| `--beta1` | 0.5 | Adam 的 β₁ 参数 |
| `--ngpu` | 1 | 使用的 GPU 数量 |
| `--workers` | 2 | 数据加载子进程数（Windows 上必须设为 0）|
| `--accel` | False | 启用 GPU 加速（CUDA / Intel XPU）|
| `--netG` | — | 加载已有 Generator 权重继续训练 |
| `--netD` | — | 加载已有 Discriminator 权重继续训练 |
| `--manualSeed` | 随机 | 固定随机种子，便于复现 |
| `--classes` | bedroom | LSUN 数据集的类别（逗号分隔）|
| `--dry-run` | — | 仅跑一个 iteration，验证环境是否正常 |

## 示例

**MNIST（Windows 推荐）：**
```bash
python main.py --dataset mnist --dataroot ./data --outf ./output --ngpu 1 --niter 25 --workers 0 --accel --imageSize 32
```

**从 checkpoint 继续训练：**
```bash
python main.py --dataset mnist --dataroot ./data --outf ./output --workers 0 --accel --netG ./output/netG_epoch_24.pth --netD ./output/netD_epoch_24.pth --niter 10
```

**快速验证环境：**
```bash
python main.py --dataset fake --dry-run
```
