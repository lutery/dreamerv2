import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn


class ObsEncoder(nn.Module):
    def __init__(self, input_shape, embedding_size, info):
        """
        :param input_shape: tuple containing shape of input，输入观察的形状
        :param embedding_size: Supposed length of encoded vector，希望的编码向量的维度
        :param info: dict containing num of hidden layers, size of hidden layers, activation function, output distribution etc.
        """
        super(ObsEncoder, self).__init__()
        self.shape = input_shape
        activation = info['activation']
        d = info['depth']
        k  = info['kernel']
        self.k = k
        self.d = d
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_shape[0], d, k),
            activation(),
            nn.Conv2d(d, 2*d, k),
            activation(),
            nn.Conv2d(2*d, 4*d, k),
            activation(),
        )

        # 这一层是确保输出的维度是embedding_size
        if embedding_size == self.embed_size:
            self.fc_1 = nn.Identity()
        else:
            self.fc_1 = nn.Linear(self.embed_size, embedding_size)

    def forward(self, obs):
        # batch_shape应该是 l, n
        batch_shape = obs.shape[:-3]
        # img_shape应该是 c, h, w
        img_shape = obs.shape[-3:]
        # 将l,n维度展品送入卷积层，顺序应该整体还是序列时间顺序
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        # 完成后再reshape回来，此时shape应该是 l, n, k * w * h
        embed = torch.reshape(embed, (*batch_shape, -1))
        embed = self.fc_1(embed)
        return embed

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.shape[1:], 0, self.k, 1)
        conv2_shape = conv_out_shape(conv1_shape, 0, self.k, 1)
        conv3_shape = conv_out_shape(conv2_shape, 0, self.k, 1)
        embed_size = int(4*self.d*np.prod(conv3_shape).item())
        return embed_size

class ObsDecoder(nn.Module):
    def __init__(self, output_shape, embed_size, info):
        """
        :param output_shape: tuple containing shape of output obs
        :param embed_size: the size of input vector, for dreamerv2 : modelstate 
        """
        super(ObsDecoder, self).__init__()
        c, h, w = output_shape
        activation = info['activation']
        d = info['depth']
        k  = info['kernel']
        # 计算第一层的卷积输出尺寸，这里应该是计算每一层的反卷积的输出尺寸
        # 得到最终输入的状态需要是什么尺寸
        conv1_shape = conv_out_shape(output_shape[1:], 0, k, 1)
        # 计算第二层的卷积输出尺寸
        conv2_shape = conv_out_shape(conv1_shape, 0, k, 1)
        # 计算第三层的卷积输出尺寸
        conv3_shape = conv_out_shape(conv2_shape, 0, k, 1)
        self.conv_shape = (4*d, *conv3_shape)
        self.output_shape = output_shape
        # 这层是确保输出的维度是符合np.prod(self.conv_shape)
        if embed_size == np.prod(self.conv_shape).item():
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(embed_size, np.prod(self.conv_shape).item())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4*d, 2*d, k, 1),
            activation(),
            nn.ConvTranspose2d(2*d, d, k, 1),
            activation(),
            nn.ConvTranspose2d(d, c, k, 1),
        )

    def forward(self, x):
        batch_shape = x.shape[:-1] # 这里应该是l, n
        embed_size = x.shape[-1] # 潜入尺寸 todo 实际运行的时这里输入的维度的意义
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        # 将维度转变为 (squeezed_size, *self.conv_shape)
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        # 反卷积输出模拟的环境观察
        x = self.decoder(x)
        # 转变回 (l, n, *self.output_shape)也就是 (l, n, c, h, w)
        # 这里输出的是每个维度的均值
        mean = torch.reshape(x, (*batch_shape, *self.output_shape))
        # 根据每个维度的均值和方差构建一个正态分布
        # td.Independent 的主要作用是将多维分布的各个维度视为独立的，从而在计算对数概率时可以将各个维度的对数概率相加。这对于处理高维观测数据非常有用，因为它简化了对数概率的计算
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.output_shape))
        return obs_dist
    
def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))
