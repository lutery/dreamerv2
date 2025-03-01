from collections import namedtuple
import torch.distributions as td
import torch
import torch.nn.functional as F
from typing import Union

# logit: 根据t-1动作和t-1的离散状态的logits
# stoch: 根据logit菜样得到的随机状态
# deter: rnn得到的确定性状态（根据前一时刻的动作和先验的状态）
RSSMDiscState = namedtuple('RSSMDiscState', ['logit', 'stoch', 'deter'])
# mean: 根据t-1动作和t-1的连续状态的均值
# std: 根据t-1动作和t-1的连续状态的标准差
# stoch: 根据mean和std采样得到的随机状态
# deter: rnn得到的确定性状态（根据前一时刻的动作和先验的状态）
RSSMContState = namedtuple('RSSMContState', ['mean', 'std', 'stoch', 'deter'])  

RSSMState = Union[RSSMDiscState, RSSMContState]

class RSSMUtils(object):
    '''utility functions for dealing with rssm states'''
    def __init__(self, rssm_type, info):
        self.rssm_type = rssm_type
        # 这边根据状态是连续的还是离散的选择不同的type
        if rssm_type == 'continuous':
            self.deter_size = info['deter_size']
            self.stoch_size = info['stoch_size']
            self.min_std = info['min_std']
        elif rssm_type == 'discrete':
            self.deter_size = info['deter_size']
            self.class_size = info['class_size']
            self.category_size = info['category_size']
            self.stoch_size  = self.class_size*self.category_size
        else:
            raise NotImplementedError

    def rssm_seq_to_batch(self, rssm_state, batch_size, seq_len):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                seq_to_batch(rssm_state.logit[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.stoch[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.deter[:seq_len], batch_size, seq_len)
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                seq_to_batch(rssm_state.mean[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.std[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.stoch[:seq_len], batch_size, seq_len),
                seq_to_batch(rssm_state.deter[:seq_len], batch_size, seq_len)
            )
        
    def rssm_batch_to_seq(self, rssm_state, batch_size, seq_len):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                batch_to_seq(rssm_state.logit, batch_size, seq_len),
                batch_to_seq(rssm_state.stoch, batch_size, seq_len),
                batch_to_seq(rssm_state.deter, batch_size, seq_len)
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                batch_to_seq(rssm_state.mean, batch_size, seq_len),
                batch_to_seq(rssm_state.std, batch_size, seq_len),
                batch_to_seq(rssm_state.stoch, batch_size, seq_len),
                batch_to_seq(rssm_state.deter, batch_size, seq_len)
            )
        
    def get_dist(self, rssm_state):
        '''
        根据传入的RSSM状态，构建返回一个概率分布
        '''
        if self.rssm_type == 'discrete':
            shape = rssm_state.logit.shape
            logit = torch.reshape(rssm_state.logit, shape = (*shape[:-1], self.category_size, self.class_size))
            return td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1)
        elif self.rssm_type == 'continuous':
            return td.independent.Independent(td.Normal(rssm_state.mean, rssm_state.std), 1)

    def get_stoch_state(self, stats):
        '''
        这个函数好像是根据计算得到的概率分布，然后采样得到随机状态
        '''
        if self.rssm_type == 'discrete':
            logit = stats['logit']
            shape = logit.shape # (batch_size, stoch_size)
            # 将logit转换为(batch_size, category_size, class_size)，那么category_size*class_size就要stoch_size
            # todo 结合后续学习class_size和category_size分别代表什么意思？有什么意义
            logit = torch.reshape(logit, shape = (*shape[:-1], self.category_size, self.class_size))
            # 然后根据OneHotCategorical创建一个概率分布
            dist = torch.distributions.OneHotCategorical(logits=logit)        
            # stoch shape = (batch_size, category_size, class_size)
            stoch = dist.sample()
            '''
            这行代码的目的是通过一种称为 "Straight-Through Estimator" 的技术来实现梯度传播。具体来说，它在离散采样过程中保持梯度信息，以便在反向传播时能够更新模型参数。
具体解释
dist.sample()：首先，从 OneHotCategorical 分布中采样得到 stoch，这是一个 one-hot 编码的张量，表示离散的随机变量。

dist.probs：这是 OneHotCategorical 分布的概率，表示每个类别的概率分布。

dist.probs.detach()：使用 detach() 方法从计算图中分离出 dist.probs，使其在反向传播时不会计算梯度。

dist.probs - dist.probs.detach()：这部分计算结果是一个零梯度的张量，因为 dist.probs 和 dist.probs.detach() 的值是相同的，但 dist.probs.detach() 没有梯度。

stoch += dist.probs - dist.probs.detach()：这一步将采样得到的 stoch 加上零梯度的 dist.probs，从而在前向传播时保持 stoch 的值不变，但在反向传播时，梯度将通过 dist.probs 传播。

作用
这种技术的作用是：

保持离散采样的值：在前向传播时，stoch 保持为离散采样的值。
实现梯度传播：在反向传播时，梯度通过 dist.probs 传播，从而实现对模型参数的更新。
这种方法在训练包含离散变量的神经网络时非常有用，因为离散采样本身是不可微的，而这种技术允许我们在不改变采样值的情况下实现梯度传播。
            '''
            stoch += dist.probs - dist.probs.detach()
            # 返回的stoch shape = (batch_size, stoch_size)
            return torch.flatten(stoch, start_dim=-2, end_dim=-1)

        elif self.rssm_type == 'continuous':
            # 如果是连续的随机状态，直接返回均值和标准差
            mean = stats['mean']
            std = stats['std']
            # min_std标准差的最小值，超参数，避免标准差过小，导致数值不稳定
            # F.softplus(std)函数将标准差 std 转换为正值
            # 因为过小的标准差可能导致数值不稳定和梯度爆炸。
            std = F.softplus(std) + self.min_std
            '''
            torch.randn_like(mean)：生成一个与 mean 张量形状相同的标准正态分布（均值为 0，标准差为 1）的随机张量。
std * torch.randn_like(mean)：将生成的标准正态分布随机张量乘以标准差 std，得到一个标准差为 std 的正态分布随机张量。
mean + std * torch.randn_like(mean)：将上述结果加上均值 mean，得到一个均值为 mean，标准差为 std 的正态分布随机张量。
            '''
            return mean + std*torch.randn_like(mean), std

    def rssm_stack_states(self, rssm_states, dim):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                torch.stack([state.logit for state in rssm_states], dim=dim),
                torch.stack([state.stoch for state in rssm_states], dim=dim),
                torch.stack([state.deter for state in rssm_states], dim=dim),
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
            torch.stack([state.mean for state in rssm_states], dim=dim),
            torch.stack([state.std for state in rssm_states], dim=dim),
            torch.stack([state.stoch for state in rssm_states], dim=dim),
            torch.stack([state.deter for state in rssm_states], dim=dim),
        )

    def get_model_state(self, rssm_state):
        # 结合确定性状态和随机状态得到模型状态
        if self.rssm_type == 'discrete':
            return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)
        elif self.rssm_type == 'continuous':
            return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)

    def rssm_detach(self, rssm_state):
        if self.rssm_type == 'discrete':
            return RSSMDiscState(
                rssm_state.logit.detach(),  
                rssm_state.stoch.detach(),
                rssm_state.deter.detach(),
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                rssm_state.mean.detach(),
                rssm_state.std.detach(),  
                rssm_state.stoch.detach(),
                rssm_state.deter.detach()
            )

    def _init_rssm_state(self, batch_size, **kwargs):
        '''
        方法的作用是初始化 RSSM（Recurrent State-Space Model，递归状态空间模型）的状态。在 DreamerV2 算法中，RSSM 状态包括确定性状态（deterministic state）和随机状态（stochastic state）。初始化这些状态是模型开始运行时的必要步骤
        对比dreamerv1算法，也是存在类似的初始化，一般来说都是初始化为0

        始化确定性状态 deter_state 为全零张量，形状为 (batch_size, deter_size)。
        初始化随机状态 stoch_state 为全零张量，形状为 (batch_size, stoch_size)。
        返回初始化的确定性状态和随机状态
        这个应该是一个工具类，想要初始化状态时随时返回
        '''
        if self.rssm_type  == 'discrete':
            return RSSMDiscState(
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.deter_size, **kwargs).to(self.device),
            )
        elif self.rssm_type == 'continuous':
            return RSSMContState(
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, self.deter_size, **kwargs).to(self.device),
            )
            
def seq_to_batch(sequence_data, batch_size, seq_len):
    """
    converts a sequence of length L and batch_size B to a single batch of size L*B
    """
    shp = tuple(sequence_data.shape)
    batch_data = torch.reshape(sequence_data, [shp[0]*shp[1], *shp[2:]])
    return batch_data

def batch_to_seq(batch_data, batch_size, seq_len):
    """
    converts a single batch of size L*B to a sequence of length L and batch_size B
    """
    shp = tuple(batch_data.shape)
    seq_data = torch.reshape(batch_data, [seq_len, batch_size, *shp[1:]])
    return seq_data

