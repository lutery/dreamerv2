import torch
import torch.nn as nn
from dreamerv2.utils.rssm import RSSMUtils, RSSMContState, RSSMDiscState

class RSSM(nn.Module, RSSMUtils):
    '''
    在 DreamerV2 算法中，RSSM（Recurrent State-Space Model，递归状态空间模型）的作用是用于在隐空间中建模环境的动态变化。它是 DreamerV2 的核心组件之一，负责将环境的观察转换为隐状态表示，并根据隐状态和动作预测未来的状态。

具体来说，RSSM 的主要作用包括：

状态表示：将环境的观察（observations）转换为隐状态（latent states），这些隐状态捕捉了环境的动态特征。
状态预测：根据当前的隐状态和动作，预测下一个隐状态。这使得模型能够在隐空间中进行模拟和规划。
重构观察：从隐状态重构出原始观察，这有助于训练过程中进行对比学习（contrastive learning）。
奖励预测：从隐状态预测奖励，这有助于训练代理的策略和价值函数。
    '''
    def __init__(
        self,
        action_size,
        rssm_node_size,
        embedding_size,
        device,
        rssm_type,
        info,
        act_fn=nn.ELU,  
    ):
        nn.Module.__init__(self)
        RSSMUtils.__init__(self, rssm_type=rssm_type, info=info)
        self.device = device
        self.action_size = action_size
        self.node_size = rssm_node_size
        self.embedding_size = embedding_size
        self.act_fn = act_fn
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size) # 有一个RNN层
        self.fc_embed_state_action = self._build_embed_state_action()
        # 先验分布预测
        self.fc_prior = self._build_temporal_prior()
        # 后验分布预测
        self.fc_posterior = self._build_temporal_posterior()

        '''
        fc_embed_state_action->self.rnn->self.fc_prior
        fc_embed_state_action->self.rnn->self.fc_posterior
        '''
    
    def _build_embed_state_action(self):
        """
        model is supposed to take in previous stochastic state and previous action 
        and embed it to deter size for rnn input
        """
        # 将随机状态stochastic state 和前一个动作嵌入到一个确定的deter_size 维度
        # 也就是提取了特征后输入到RNN中
        # todo 这是模型在上一个时间步预测的随机状态，表示环境的隐状态
        # todo 这是代理在上一个时间步执行的动作
        fc_embed_state_action = [nn.Linear(self.stoch_size + self.action_size, self.deter_size)]
        fc_embed_state_action += [self.act_fn()]
        return nn.Sequential(*fc_embed_state_action)
    
    def _build_temporal_prior(self):
        """
        model is supposed to take in latest deterministic state 
        and output prior over stochastic state
        选中的注释解释了模型的另一个关键功能：如何从最新的确定性状态（deterministic state）生成随机状态（stochastic state）的先验分布（prior）
        """
        # deter_size 是确定性状态的维度
        # node_size 是 RSSM 的节点大小
        # stoch_size 是随机状态的大小
        temporal_prior = [nn.Linear(self.deter_size, self.node_size)]
        temporal_prior += [self.act_fn()]
        if self.rssm_type == 'discrete':
            temporal_prior += [nn.Linear(self.node_size, self.stoch_size)]
        elif self.rssm_type == 'continuous':
             temporal_prior += [nn.Linear(self.node_size, 2 * self.stoch_size)]
        return nn.Sequential(*temporal_prior)

    def _build_temporal_posterior(self):
        """
        model is supposed to take in latest embedded observation and deterministic state 
        and output posterior over stochastic states
        选中的注释解释了模型的另一个关键功能：如何从最新的嵌入观察（embedded observation）和确定性状态（deterministic state）生成随机状态（stochastic state）的后验分布（posterior）
        最新的嵌入观察（latest embedded observation）：
        这是模型在当前时间步从环境中获取的观察数据，经过编码器处理后得到的嵌入表示。

        确定性状态（deterministic state）：
        这是模型在当前时间步通过 RNN 计算得到的确定性状态，表示环境的隐状态。

        生成随机状态的后验分布（output posterior over stochastic states）：
        模型使用最新的嵌入观察和确定性状态来生成一个后验分布，这个分布描述了随机状态的可能值。
        后验分布通常用参数化的概率分布（如高斯分布）来表示，参数包括均值和方差。
        """
        temporal_posterior = [nn.Linear(self.deter_size + self.embedding_size, self.node_size)]
        temporal_posterior += [self.act_fn()]
        if self.rssm_type == 'discrete':
            temporal_posterior += [nn.Linear(self.node_size, self.stoch_size)]
        elif self.rssm_type == 'continuous':
            temporal_posterior += [nn.Linear(self.node_size, 2 * self.stoch_size)]
        return nn.Sequential(*temporal_posterior)
    
    def rssm_imagine(self, prev_action, prev_rssm_state, nonterms=True):
        '''
        模拟t时刻动作,rssm状态，t时刻的非终止状态

        返回根据t时刻的动作和rssm状态，预测的t+1时刻的rssm状态，也就是先验状态
        '''
        # 随机状态和动作嵌入结合(stoch_size+action_size)抽取得到随机状态动作嵌入（shape=deter_size）
        # 如果遇到中止状态那么prev_action就是0，prev_rssm_state.stoch*nonterms就是0
        # state_action_embed中提取的对应特征也是0
        state_action_embed = self.fc_embed_state_action(torch.cat([prev_rssm_state.stoch*nonterms, prev_action],dim=-1))
        # 将其提取的动作特征输入到RNN中，得到deter_state确定性状态
        deter_state = self.rnn(state_action_embed, prev_rssm_state.deter*nonterms)
        # 根据观察时离散还是连续，得到不同的随机状态
        if self.rssm_type == 'discrete':
            # 如果是离散的随机状态，需要将输出的结果作为 logits，然后得到随机分布状态
            prior_logit = self.fc_prior(deter_state)
            stats = {'logit':prior_logit}
            # 根据logits得到随机状态
            prior_stoch_state = self.get_stoch_state(stats)
            prior_rssm_state = RSSMDiscState(prior_logit, prior_stoch_state, deter_state)

        elif self.rssm_type == 'continuous':
            # 对于连续的随机状态，需要将输出的结果分成两部分，一部分是均值，一部分是标准差
            prior_mean, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
            stats = {'mean':prior_mean, 'std':prior_std}
            # 根据logits得到随机状态
            prior_stoch_state, std = self.get_stoch_state(stats)
            prior_rssm_state = RSSMContState(prior_mean, std, prior_stoch_state, deter_state)
        return prior_rssm_state

    def rollout_imagination(self, horizon:int, actor:nn.Module, prev_rssm_state):
        rssm_state = prev_rssm_state
        next_rssm_states = []
        action_entropy = []
        imag_log_probs = []
        for t in range(horizon):
            action, action_dist = actor((self.get_model_state(rssm_state)).detach())
            rssm_state = self.rssm_imagine(action, rssm_state)
            next_rssm_states.append(rssm_state)
            action_entropy.append(action_dist.entropy())
            imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))

        next_rssm_states = self.rssm_stack_states(next_rssm_states, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)
        return next_rssm_states, imag_log_probs, action_entropy

    def rssm_observe(self, obs_embed, prev_action, prev_nonterm, prev_rssm_state):
        '''
        obs_embed: t+1时刻的观察嵌入特征
        prev_action: t时刻的动作,如果是终止状态则是0
        prev_nonterm: t时刻的非终止状态
        prev_rssm_state: t时刻的RSSM状态

        返回预测得到的先验rssm状态和后验rssm状态
        '''
        prior_rssm_state = self.rssm_imagine(prev_action, prev_rssm_state, prev_nonterm)
        deter_state = prior_rssm_state.deter
        # 结合确定性状态（动作和RSSM状态）和t+1时刻的观察嵌入特征
        # 后验状态是结合了实际的观察
        x = torch.cat([deter_state, obs_embed], dim=-1)
        if self.rssm_type == 'discrete':
            # 得到后验状态的logits
            posterior_logit = self.fc_posterior(x)
            stats = {'logit':posterior_logit}
            posterior_stoch_state = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMDiscState(posterior_logit, posterior_stoch_state, deter_state)
        
        elif self.rssm_type == 'continuous':
            # 得到后验状态的均值和方差
            posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1)
            stats = {'mean':posterior_mean, 'std':posterior_std}
            posterior_stoch_state, std = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMContState(posterior_mean, std, posterior_stoch_state, deter_state)
        return prior_rssm_state, posterior_rssm_state

    def rollout_observation(self, seq_len:int, obs_embed: torch.Tensor, action: torch.Tensor, nonterms: torch.Tensor, prev_rssm_state):
        '''
        seq_len: 序列长度
        obs_embed: 观察嵌入特征
        action: 动作
        nonterms: 非终止状态
        prev_rssm_state: 前一个RSSM状态,在t=0时刻是初始化的状态,全0
        '''
        
        priors = [] # 存储t时刻的先验状态
        posteriors = [] # 存储t时刻的后验状态
        # 遍历序列长度
        for t in range(seq_len):
            # t时刻的动作 乘以 t时刻的非终止状态
            # 如果不终止，那么prev_action 就是 t时刻的动作
            # 如果终止，那么prev_action 就是 t时刻的动作乘以0,最终结果也是0
            prev_action = action[t]*nonterms[t]
            # 根据t+1时刻的观察��入特征，t时刻的动作，t时刻的非终止状态，t1时刻的RSSM状态
            # 得到t时刻的先验状态和后验状态
            prior_rssm_state, posterior_rssm_state = self.rssm_observe(obs_embed[t], prev_action, nonterms[t], prev_rssm_state)
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            prev_rssm_state = posterior_rssm_state
        # 将收集到的1-nt时刻的先验和后验状态��接成一个batch
        prior = self.rssm_stack_states(priors, dim=0)
        post = self.rssm_stack_states(posteriors, dim=0)
        return prior, post
        