import torch
import torch.distributions as td

# 定义 logits
logits = torch.tensor([[[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]])
print("logints shape:", logits.shape)

# 创建 OneHotCategorical 分布
dist = td.OneHotCategorical(logits=logits)

# 从分布中采样
samples = dist.sample()
print("Samples:", samples)
print("Samples shape:", samples.shape)

# 计算样本的对数概率
log_probs = dist.log_prob(samples)
print("Log probabilities:", log_probs)
print("Log probabilities shape:", log_probs.shape)

# 获取归一化的概率
probs = dist.probs
print("Probabilities:", probs)
print("Probabilities shape:", probs.shape)


import torch
import torch.nn.functional as F

action_logits = torch.rand(5)
print("Action logits shape:", action_logits.shape)
action_probs = F.softmax(action_logits, dim=-1)

dist = torch.distributions.Categorical(action_probs)
action = dist.sample()
print("Sampled action:", action)
print(dist.log_prob(action), torch.log(action_probs[action]))