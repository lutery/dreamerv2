import numpy as np
import torch 
import torch.optim as optim
import os 

from dreamerv2.utils.module import get_parameters, FreezeParameters
from dreamerv2.utils.algorithm import compute_return

from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.utils.buffer import TransitionBuffer

class Trainer(object):
    def __init__(
        self, 
        config,
        device,
    ):
        self.device = device
        self.config = config
        self.action_size = config.action_size
        self.pixel = config.pixel
        self.kl_info = config.kl
        self.seq_len = config.seq_len
        self.batch_size = config.batch_size
        self.collect_intervals = config.collect_intervals
        self.seed_steps = config.seed_steps # 初始化种子数据集的长度，用于预热缓冲区
        self.discount = config.discount_
        self.lambda_ = config.lambda_
        self.horizon = config.horizon
        self.loss_scale = config.loss_scale
        self.actor_entropy_scale = config.actor_entropy_scale
        self.grad_clip_norm = config.grad_clip

        self._model_initialize(config)
        self._optim_initialize(config)

    def collect_seed_episodes(self, env):
        '''
        相当于预热缓冲区
        方法的作用通常是收集初始的种子数据集（seed episodes），这些数据集用于初始化强化学习算法的经验回放缓冲区。在强化学习中，种子数据集可以通过随机策略在环境中采样得到，以便为后续的训练提供初始经验
        '''
        s, done  = env.reset(), False 
        for i in range(self.seed_steps):
            a = env.action_space.sample()
            ns, r, done, _ = env.step(a)
            if done:
                self.buffer.add(s,a,r,done)
                s, done  = env.reset(), False 
            else:
                self.buffer.add(s,a,r,done)
                s = ns    

    def train_batch(self, train_metrics):
        """ 
        trains the world model and imagination actor and critic for collect_interval times using sequence-batch data from buffer
        """
        actor_l = []
        value_l = []
        obs_l = []
        model_l = []
        reward_l = []
        prior_ent_l = []
        post_ent_l = []
        kl_l = []
        pcont_l = []
        mean_targ = []
        min_targ = []
        max_targ = []
        std_targ = []

        for i in range(self.collect_intervals):
            obs, actions, rewards, terms = self.buffer.sample()
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)                         #t, t+seq_len 
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)                 #t-1, t+seq_len-1
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)   #t-1 to t+seq_len-1
            nonterms = torch.tensor(1-terms, dtype=torch.float32).to(self.device).unsqueeze(-1)  #t-1 to t+seq_len-1

            # 计算损失
            model_loss, kl_loss, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior = self.representation_loss(obs, actions, rewards, nonterms)
            
            self.model_optimizer.zero_grad()
            model_loss.backward()
            grad_norm_model = torch.nn.utils.clip_grad_norm_(get_parameters(self.world_list), self.grad_clip_norm)
            self.model_optimizer.step()

            # 根据后验状态计算actor和critic的损失
            actor_loss, value_loss, target_info = self.actorcritc_loss(posterior)

            self.actor_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            actor_loss.backward()
            value_loss.backward()

            grad_norm_actor = torch.nn.utils.clip_grad_norm_(get_parameters(self.actor_list), self.grad_clip_norm)
            grad_norm_value = torch.nn.utils.clip_grad_norm_(get_parameters(self.value_list), self.grad_clip_norm)

            self.actor_optimizer.step()
            self.value_optimizer.step()
            # 到这里训练结束，后续是记录训练过程中的统计值

            with torch.no_grad():
                prior_ent = torch.mean(prior_dist.entropy())
                post_ent = torch.mean(post_dist.entropy())

            prior_ent_l.append(prior_ent.item())
            post_ent_l.append(post_ent.item())
            actor_l.append(actor_loss.item())
            value_l.append(value_loss.item())
            obs_l.append(obs_loss.item())
            model_l.append(model_loss.item())
            reward_l.append(reward_loss.item())
            kl_l.append(kl_loss.item())
            pcont_l.append(pcont_loss.item())
            mean_targ.append(target_info['mean_targ'])
            min_targ.append(target_info['min_targ'])
            max_targ.append(target_info['max_targ'])
            std_targ.append(target_info['std_targ'])

        train_metrics['model_loss'] = np.mean(model_l)
        train_metrics['kl_loss']=np.mean(kl_l)
        train_metrics['reward_loss']=np.mean(reward_l)
        train_metrics['obs_loss']=np.mean(obs_l)
        train_metrics['value_loss']=np.mean(value_l)
        train_metrics['actor_loss']=np.mean(actor_l)
        train_metrics['prior_entropy']=np.mean(prior_ent_l)
        train_metrics['posterior_entropy']=np.mean(post_ent_l)
        train_metrics['pcont_loss']=np.mean(pcont_l)
        train_metrics['mean_targ']=np.mean(mean_targ)
        train_metrics['min_targ']=np.mean(min_targ)
        train_metrics['max_targ']=np.mean(max_targ)
        train_metrics['std_targ']=np.mean(std_targ)

        return train_metrics

    def actorcritc_loss(self, posterior):
        '''
        param posterior: 后验状态
        '''
        with torch.no_grad():
            # self.RSSM.rssm_seq_to_batch 将RSSM状态转换为batch形式也就是将l n合并为一个维度
            # 然后再将后验状态分离出梯度计算
            batched_posterior = self.RSSM.rssm_detach(self.RSSM.rssm_seq_to_batch(posterior, self.batch_size, self.seq_len-1))
        
        # 冻结非actor和value的参数
        with FreezeParameters(self.world_list):
            imag_rssm_states, imag_log_prob, policy_entropy = self.RSSM.rollout_imagination(self.horizon, self.ActionModel, batched_posterior)
        
        # 结合预测的想象状态中的确定性状态和随机状态，得到模型状态
        imag_modelstates = self.RSSM.get_model_state(imag_rssm_states)
        # 冻结除动作模型以外的所有模型
        with FreezeParameters(self.world_list+self.value_list+[self.TargetValueModel]+[self.DiscountModel]):
            # 根据预测的状态计算奖励分布，有点像dqn的c51
            imag_reward_dist = self.RewardDecoder(imag_modelstates)
            imag_reward = imag_reward_dist.mean
            # 预测价值分布
            imag_value_dist = self.TargetValueModel(imag_modelstates)
            imag_value = imag_value_dist.mean
            # 预测终止状态分布
            discount_dist = self.DiscountModel(imag_modelstates)
            discount_arr = self.discount*torch.round(discount_dist.base_dist.probs)              #mean = prob(disc==1)

        actor_loss, discount, lambda_returns = self._actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy)
        value_loss = self._value_loss(imag_modelstates, discount, lambda_returns)     
        
        # 记录训练过程中的统计值
        mean_target = torch.mean(lambda_returns, dim=1)
        max_targ = torch.max(mean_target).item()
        min_targ = torch.min(mean_target).item() 
        std_targ = torch.std(mean_target).item()
        mean_targ = torch.mean(mean_target).item()
        target_info = {
            'min_targ':min_targ,
            'max_targ':max_targ,
            'std_targ':std_targ,
            'mean_targ':mean_targ,
        }
    
        # 返回动作损失，价值损失，统计信息
        return actor_loss, value_loss, target_info

    def representation_loss(self, obs, actions, rewards, nonterms):
        '''
        param obs: 观察
        param actions: 动作
        param rewards: 奖励
        param nonterms: 非终止状态

        return model_loss, kl_loss, obs_loss, reward_loss, pcont_loss, prior, posterior
        '''

        # 得到观察的嵌入特征
        embed = self.ObsEncoder(obs)                                         #t to t+seq_len   
        # 获取rssm初始化特征，每次训练计算损失时都需要初始化
        prev_rssm_state = self.RSSM._init_rssm_state(self.batch_size)   
        # 得到先验和后验状态时间序列
        prior, posterior = self.RSSM.rollout_observation(self.seq_len, embed, actions, nonterms, prev_rssm_state)
        # 根据后验状态的确定性状态和随机状态得到模型状态
        # todo 后验是有根据实际的状态得到的，所以在实际使用时是如何使用的呢？
        post_modelstate = self.RSSM.get_model_state(posterior)               #t to t+seq_len   
        # 输出一个观察分布，一个奖励分布，一个折扣因子分布
        # todo 为什么要输出分布而不是直接输出值
        # 以下预测输入的是时刻与计算损失的时刻不包含最后一个时刻
        obs_dist = self.ObsDecoder(post_modelstate[:-1])                     #t to t+seq_len-1  
        reward_dist = self.RewardDecoder(post_modelstate[:-1])               #t to t+seq_len-1  
        # todo 这个不是折扣模型吗？为什么和nonterms计算损失
        pcont_dist = self.DiscountModel(post_modelstate[:-1])                #t to t+seq_len-1   
        
        # todo 为啥计算观察损失时不包含最后一个时刻，和post_modelstate[:-1对应
        obs_loss = self._obs_loss(obs_dist, obs[:-1])
        # todo 而计算奖励损失时不包含第一个时刻，上面偏差一个时刻，相当于和预测t+1时刻的奖励计算损失
        reward_loss = self._reward_loss(reward_dist, rewards[1:])
        pcont_loss = self._pcont_loss(pcont_dist, nonterms[1:])
        # 是的，在 DreamerV2 算法中，_kl_loss 损失用于衡量先验状态（prior state）和后验状态（posterior state）之间的差异。具体来说，_kl_loss 计算的是先验分布和后验分布之间的 Kullback-Leibler (KL) 散度。通过最小化这个损失，算法可以使先验状态和后验状态尽可能接近
        prior_dist, post_dist, div = self._kl_loss(prior, posterior)
    
        # 汇总模型损失
        model_loss = self.loss_scale['kl'] * div + reward_loss + obs_loss + self.loss_scale['discount']*pcont_loss
        return model_loss, div, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior

    def _actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):
        '''
        param imag_reward: dreamerv2预测的奖励
        param imag_value: dreamerv2预测的价值
        param discount_arr: discount因子
        param imag_log_prob: dreamerv2的预测动作的对数概率 ，通过预测一个动作的对数概率，然后通过这个对数概率知道该往哪个方向走
        param policy_entropy: dreamerv2的预测动作的熵

        return 动作损失，折扣，lambda_returns（ppo的长期回报序列）
        '''

        # 计算预测的动作回报
        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1], lambda_=self.lambda_)
        
        if self.config.actor_grad == 'reinforce':
            # 基于策略梯度：传统的REINFORCE算法风格
            advantage = (lambda_returns-imag_value[:-1]).detach() # 使用优势函数：计算返回值与基线（价值估计）之间的差值作为优势
            objective = imag_log_prob[1:].unsqueeze(-1) * advantage # 通过动作的对数概率乘以优势函数来更新策略
            '''
            更高的方差
            更慢的收敛
            但在某些情况下可能探索性更好
            '''

        elif self.config.actor_grad == 'dynamics':
            '''
            基于路径导数：直接通过世界模型的动态梯度进行反向传播
            使用返回值：直接最大化预期返回值
            梯度更新方式：梯度通过想象的状态-动作轨迹直接反向传播
            '''
            objective = lambda_returns

            '''
            在实践中，dynamics 策略是 DreamerV2 的默认选择，因为它通常提供更好的性能和更快的收敛。然而，提供 reinforce 选项可能是为了在特定环境或情况下提供替代方案。
            '''
        else:
            raise NotImplementedError

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]]) # 首先，将第一个折扣因子强制设为 1.0，确保从当前时间步开始的奖励权重是 100%
        discount = torch.cumprod(discount_arr[:-1], 0) # 然后，计算累积折扣乘积，表示未来奖励在当前决策中的重要性逐渐降低
        policy_entropy = policy_entropy[1:].unsqueeze(-1) # 选取策略熵（不包括第一个时间步
        '''
        策略熵是衡量行动"随机性"的指标。想象一个掷骰子的游戏：

        低熵：总是倾向于选择某个数字
        高熵：完全随机地选择任何数字
        '''

        # objective:主要目标（返回值或优势函数）
        # self.actor_entropy_scale * policy_entropy 加上策略熵（乘以一个系数来调整探索程度）
        # discount 加权平均,用折扣因子对每个时间步的目标进行加权,近期回报权重高，远期回报权重低
        # torch.mean 先对批次维度求均值 l n中的n
        # torch.sum 再对时间维度求和l n中的l
        actor_loss = -torch.sum(torch.mean(discount * (objective + self.actor_entropy_scale * policy_entropy), dim=1)) 
        return actor_loss, discount, lambda_returns

    def _value_loss(self, imag_modelstates, discount, lambda_returns):
        with torch.no_grad():
            value_modelstates = imag_modelstates[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        # 用动作模型预测的模型状态（后验的随机状态和确定状态）预测价值分布
        value_dist = self.ValueModel(value_modelstates) 
        # 为什么要乘以：value_discount
        '''
        为什么在 _value_loss 中要乘以 value_discount
        在 DreamerV2 的 _value_loss 方法中，乘以 value_discount 是一个重要的设计，其目的是按时间步长的重要性对值函数的损失进行加权。让我们详细解释原因：

        价值损失函数的计算
        为什么要乘以 value_discount
        时间步长加权：

        value_discount 是累积折扣因子，表示每个时间步长的相对重要性
        较近的未来时间步长有更高的折扣值（接近1），较远的未来时间步长有较低的折扣值
        通过乘以这个折扣，算法对近期预测的准确性给予更高的权重
        一致性与策略损失：

        在 _actor_loss 中，也使用了相同的折扣因子来加权目标
        这种一致性确保策略和值函数的训练是协调的，都更关注近期回报
        不确定性处理：

        预测越远的未来，不确定性就越大
        通过折扣，算法减少了对高不确定性远期预测的依赖
        终止状态的考虑：

        折扣因子考虑了可能的终止状态
        如果某条路径可能很快终止，那么其后续步骤的权重会相应降低
        实际效果
        这种设计的实际效果是：

        稳定训练：减少远期预测误差对训练的影响
        更好的近期预测：值函数更准确地预测近期回报
        更高效的学习：资源集中在更重要和更确定的预测上
        这种按折扣加权的值函数损失是 DreamerV2 能有效学习长期预测同时保持稳定性的关键技术之一。

        也就是说越远的价值预测越不准确，方差越高，所以要降低权重，越近的价值预测，方差越低，所以权重越高
        所以这里的价值预测关键优化点就在alue_dist.log_prob(value_target)，如果与value_target相差大，那么就会将对应的价值概率降低，如果与value_target相差小，那么就会将对应的价值概率提高
        因为这里预测的是价值分布，所以要通过价值分布的对数概率来计算价值损失
        '''
        value_loss = -torch.mean(value_discount*value_dist.log_prob(value_target).unsqueeze(-1))
        return value_loss
            
    def _obs_loss(self, obs_dist, obs):
        # 利用实际的观察，获取对应的观察分布的概率的log值
        # 如果想要obs_loss最小，那么obs_dist.log_prob(obs)应该最大
        # 如果想要obs_dist.log_prob(obs)应该最大，obs_dist在obs处的概率应该最大
        # 从而达到优化的目的
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss
    
    def _kl_loss(self, prior, posterior):
        '''
        返回先验分布，后验分布，KL散度损失
        '''

        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        if self.kl_info['use_kl_balance']:
            '''
            它控制是否使用 KL 平衡技术来计算 KL 损失
            当 use_kl_balance 设置为 True 时，算法会采用 KL 平衡技术，即计算两个方向的 KL 散度并加权平均：

kl_lhs：计算后验分布到先验分布的 KL 散度
kl_rhs：计算先验分布到后验分布的 KL 散度
然后通过以下公式计算平衡的 KL 损失：

其中 alpha 是平衡系数（kl_balance_scale），控制两个方向 KL 散度的权重。

为什么需要 KL 平衡？
KL 散度是非对称的，即 KL(P||Q) 不等于 KL(Q||P)。这两个方向的散度有不同的优化特性：

KL(posterior||prior)：倾向于使后验分布更加"模式覆盖"（mode-covering）
KL(prior||posterior)：倾向于使后验分布更加"模式寻找"（mode-seeking）
通过平衡两个方向的 KL 散度，可以综合两者的优点，使模型既能覆盖多种可能的模式，又能准确地定位特定模式。
            '''
            alpha = self.kl_info['kl_balance_scale']
            # 计算两个分布之间的KL散度
            # 因为要尽可能接近，所以要计算两个分布之间的KL散度两次（颠倒顺序计算一次），通过计算这两个值，可以更全面地衡量先验和后验分布之间的差异
            # todo 这里使用detach是不是在计算损失时，不能存在两个互相可以计算梯度的张量？从而为了在计算 KL 散度时，防止梯度通过这些分布传播，从而避免影响模型的其他部分
            kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
            if self.kl_info['use_free_nats']:
                '''
                在 free nats 技术中，其目的并不是限制 KL 损失的上限，而是设置一个下界，防止 KL 损失变得太小，从而使得模型忽略一定程度的正则化。也就是说，使用 torch.max 的目的是：

如果计算得到的 KL 值低于 free nats 阈值，则用 free nats 替换（即不允许 KL 损失低于这个值），从而保证有一定的 KL 惩罚存在；
如果默认 free_nats 为 0，则 torch.max(kl, 0) 没有实际影响，相当于没有启用该功能。
因此，如果目标是防止 KL 损失过低（而不是过高），那么使用 torch.max 是正确的做法。使用 torch.min 则会限制 KL 的上限，这并不是 free nats 的目的。
                '''
                free_nats = self.kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
            # 通过 alpha 权重平衡两个 KL 散度的值，从而得到最终的 KL 损失
            kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs

        else: 
            # 这边仅计算一个KL散度损失
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss
    
    def _reward_loss(self, reward_dist, rewards):
        # 原理同_obs_loss
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss
    
    def _pcont_loss(self, pcont_dist, nonterms):
        # 将booean转换为float：布尔值可以转换为浮点数。布尔值 True 会被转换为 1.0，而布尔值 False 会被转换为 0.0
        pcont_target = nonterms.float()
        # 接下来原理同_obs_loss
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss

    def update_target(self):
        '''
        同步更新目标值网络
        '''
        mix = self.config.slow_target_fraction if self.config.use_slow_target else 1
        for param, target_param in zip(self.ValueModel.parameters(), self.TargetValueModel.parameters()):
            target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)

    def save_model(self, iter):
        save_dict = self.get_save_dict()
        model_dir = self.config.model_dir
        save_path = os.path.join(model_dir, 'models_%d.pth' % iter)
        torch.save(save_dict, save_path)

    def get_save_dict(self):
        return {
            "RSSM": self.RSSM.state_dict(),
            "ObsEncoder": self.ObsEncoder.state_dict(),
            "ObsDecoder": self.ObsDecoder.state_dict(),
            "RewardDecoder": self.RewardDecoder.state_dict(),
            "ActionModel": self.ActionModel.state_dict(),
            "ValueModel": self.ValueModel.state_dict(),
            "DiscountModel": self.DiscountModel.state_dict(),
        }
    
    def load_save_dict(self, saved_dict):
        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.RewardDecoder.load_state_dict(saved_dict["RewardDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])
        self.ValueModel.load_state_dict(saved_dict["ValueModel"])
        self.DiscountModel.load_state_dict(saved_dict['DiscountModel'])
            
    def _model_initialize(self, config):
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size = config.rssm_info['deter_size']
        '''
        判断 RSSM（递归状态空间模型）的类型是“连续”还是“离散”。

        具体来说，代码根据 config.rssm_type 的值来配置不同类型的 RSSM 模型：

        如果 config.rssm_type 是 'continuous'，则使用 stoch_size（随机状态的大小）来配置连续类型的 RSSM。
        如果 config.rssm_type 是 'discrete'，则使用 category_size 和 class_size 来配置离散类型的 RSSM。
        '''
        if config.rssm_type == 'continuous':
            stoch_size = config.rssm_info['stoch_size']
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']
            class_size = config.rssm_info['class_size']
            stoch_size = category_size*class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        # 别代表随机状态（stochastic state）和确定性状态（deterministic state）的维度。这两个参数是递归状态空间模型（RSSM）的核心组成部分，用于表示和预测环境的隐状态
        modelstate_size = stoch_size + deter_size 
    
        self.buffer = TransitionBuffer(config.capacity, obs_shape, action_size, config.seq_len, config.batch_size, config.obs_dtype, config.action_dtype)
        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, self.device, config.rssm_type, config.rssm_info).to(self.device)
        self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size, embedding_size, config.actor, config.expl).to(self.device)
        self.RewardDecoder = DenseModel((1,), modelstate_size, config.reward).to(self.device)
        self.ValueModel = DenseModel((1,), modelstate_size, config.critic).to(self.device)
        self.TargetValueModel = DenseModel((1,), modelstate_size, config.critic).to(self.device)
        self.TargetValueModel.load_state_dict(self.ValueModel.state_dict())
        
        # 判断是否加入折扣因子模型，用于预测长期的回报
        if config.discount['use']:
            '''
            在 DreamerV2 中的作用是预测环境中的折扣因子（discount factor）或者说是终止概率。它实际上是一个终止预测器（termination predictor），用于学习预测某个状态是否为终止状态或者离终止状态有多远。
            '''
            self.DiscountModel = DenseModel((1,), modelstate_size, config.discount).to(self.device)
        
        # 构建环境观察编码器和解码器（分像素空间和线性空间）
        if config.pixel:
            self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device)
            self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(self.device)
        else:
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(self.device)
            self.ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(self.device)

    def _optim_initialize(self, config):
        model_lr = config.lr['model']
        actor_lr = config.lr['actor']
        value_lr = config.lr['critic']
        # 包含环境编码器、RSSM、奖励解码器、观察解码器和折扣模型
        self.world_list = [self.ObsEncoder, self.RSSM, self.RewardDecoder, self.ObsDecoder, self.DiscountModel]
        self.actor_list = [self.ActionModel]
        self.value_list = [self.ValueModel]
        # 包含策略和价值网络
        self.actorcritic_list = [self.ActionModel, self.ValueModel]
        self.model_optimizer = optim.Adam(get_parameters(self.world_list), model_lr)
        self.actor_optimizer = optim.Adam(get_parameters(self.actor_list), actor_lr)
        self.value_optimizer = optim.Adam(get_parameters(self.value_list), value_lr)

    def _print_summary(self):
        print('\n Obs encoder: \n', self.ObsEncoder)
        print('\n RSSM model: \n', self.RSSM)
        print('\n Reward decoder: \n', self.RewardDecoder)
        print('\n Obs decoder: \n', self.ObsDecoder)
        if self.config.discount['use']:
            print('\n Discount decoder: \n', self.DiscountModel)
        print('\n Actor: \n', self.ActionModel)
        print('\n Critic: \n', self.ValueModel)