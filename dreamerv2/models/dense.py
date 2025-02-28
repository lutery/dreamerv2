import numpy as np
import torch.nn as nn
import torch.distributions as td

class DenseModel(nn.Module):
    '''
    动作生成（actor）
    值函数估计（critic）
    '''
    def __init__(
            self, 
            output_shape,
            input_size, 
            info,
        ):
        """
        :param output_shape: tuple containing shape of expected output
        :param input_size: size of input features
        :param info: dict containing num of hidden layers, size of hidden layers, activation function, output distribution etc.
        """
        super().__init__()
        self._output_shape = output_shape # 如果是用于reward和value，那么输出的output_shape是1维的
        self._input_size = input_size # stoch_size + deter_size：
        self._layers = info['layers']
        self._node_size = info['node_size']
        self.activation = info['activation']
        self.dist = info['dist']
        self.model = self.build_model()

    def build_model(self):
        # 全连接神经网络
        # todo 搞清楚self.input_size是输入什么东西
        # 
        model = [nn.Linear(self._input_size, self._node_size)]
        model += [self.activation()]
        for i in range(self._layers-1):
            model += [nn.Linear(self._node_size, self._node_size)]
            model += [self.activation()]
        model += [nn.Linear(self._node_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, input):
        dist_inputs = self.model(input)
        if self.dist == 'normal':
            return td.independent.Independent(td.Normal(dist_inputs, 1), len(self._output_shape))
        if self.dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=dist_inputs), len(self._output_shape))
        if self.dist == None:
            return dist_inputs

        raise NotImplementedError(self._dist)