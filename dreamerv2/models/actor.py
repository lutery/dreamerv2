import torch 
import torch.nn as nn
import numpy as np

class DiscreteActionModel(nn.Module):
    '''
    离散动作模型
    '''
    def __init__(
        self,
        action_size,
        deter_size,
        stoch_size,
        embedding_size,
        actor_info,
        expl_info
    ):
        super().__init__()
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.embedding_size = embedding_size
        self.layers = actor_info['layers']
        self.node_size = actor_info['node_size']
        self.act_fn = actor_info['activation']
        self.dist = actor_info['dist']
        self.act_fn = actor_info['activation']
        self.train_noise = expl_info['train_noise']
        self.eval_noise = expl_info['eval_noise']
        self.expl_min = expl_info['expl_min']
        self.expl_decay = expl_info['expl_decay']
        self.expl_type = expl_info['expl_type']
        self.model = self._build_model()

    def _build_model(self):
        '''
        输入维度：

        self.deter_size：确定性状态的维度，表示通过 RNN 计算得到的隐状态。
        self.stoch_size：随机状态的维度，表示通过先验或后验分布生成的隐状态。
        self.deter_size + self.stoch_size：输入层的总维度，是确定性状态和随机状态的拼接。

        self.node_size：隐藏层的节点数量，表示每个隐藏层的输出维度。
        self.layers：隐藏层的数量，表示网络中隐藏层的层数。

        todo 搞清楚在实际运行时，每个输入的谁是从哪里来的，然后到哪里去。
        '''
        model = [nn.Linear(self.deter_size + self.stoch_size, self.node_size)]
        model += [self.act_fn()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.node_size, self.node_size)]
            model += [self.act_fn()]

        # 输出动作分布，仅实现了 one-hot 动作分布，也就是离散动作分户。
        if self.dist == 'one_hot':
            model += [nn.Linear(self.node_size, self.action_size)]
        else:
            raise NotImplementedError
        return nn.Sequential(*model) 

    def forward(self, model_state):
        # 获取预测的动作概率分布
        action_dist = self.get_action_dist(model_state)
        # 从动作概率分布中采样一个动作
        action = action_dist.sample()
        '''
        选中的代码行：

        这行代码是实现离散动作的直通估计器(Straight-Through Estimator)技术，它允许梯度通过离散采样操作传播
        在强化学习中，从动作分布中采样离散动作是常见操作
        但采样操作是不可微的，这会阻止梯度从损失函数流回到生成动作分布的网络
        '''
        action = action + action_dist.probs - action_dist.probs.detach()
        return action, action_dist

    def get_action_dist(self, modelstate):
        # 根据后验状态的确定性状态和随机状态得到 动作logits
        logits = self.model(modelstate)
        if self.dist == 'one_hot':
            return torch.distributions.OneHotCategorical(logits=logits)         
        else:
            raise NotImplementedError
            
    def add_exploration(self, action: torch.Tensor, itr: int, mode='train'):
        '''
        给预测的动作增加噪音，实现探索
        '''
        if mode == 'train':
            expl_amount = self.train_noise
            expl_amount = expl_amount - itr/self.expl_decay
            expl_amount = max(self.expl_min, expl_amount)
        elif mode == 'eval':
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError
            
        if self.expl_type == 'epsilon_greedy':
            if np.random.uniform(0, 1) < expl_amount:
                index = torch.randint(0, self.action_size, action.shape[:-1], device=action.device)
                action = torch.zeros_like(action)
                action[:, index] = 1
            return action

        raise NotImplementedError