import wandb
import argparse
import os
import torch
import numpy as np
import gym
from dreamerv2.utils.wrapper import GymMinAtar, OneHotAction, breakoutPOMDP, space_invadersPOMDP, seaquestPOMDP, asterixPOMDP, freewayPOMDP
from dreamerv2.training.config import MinAtarConfig
from dreamerv2.training.trainer import Trainer
from dreamerv2.training.evaluator import Evaluator

pomdp_wrappers = {
    'breakout':breakoutPOMDP,
    'seaquest':seaquestPOMDP,
    'space_invaders':space_invadersPOMDP,
    'asterix':asterixPOMDP,
    'freeway':freewayPOMDP,
}

def main(args):
    wandb.login()
    env_name = args.env
    exp_id = args.id + '_pomdp'

    '''make dir for saving results'''
    result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
    model_dir = os.path.join(result_dir, 'models')                                                  #dir to save learnt models
    os.makedirs(model_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device:
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    print('using :', device)  

    PomdpWrapper = pomdp_wrappers[env_name]
    env = PomdpWrapper(OneHotAction(GymMinAtar(env_name)))
    obs_shape = env.observation_space.shape
    action_size = env.action_space.shape[0]
    obs_dtype = bool
    action_dtype = np.float32
    batch_size = args.batch_size
    seq_len = args.seq_len # 采集的序列长度

    # atari 游戏配置
    config = MinAtarConfig(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype = obs_dtype,
        action_dtype = action_dtype,
        seq_len = seq_len,
        batch_size = batch_size,
        model_dir=model_dir, 
    )

    config_dict = config.__dict__
    # 训练器
    trainer = Trainer(config, device)
    # 验证器
    evaluator = Evaluator(config, device)
    
    # wandb可视化训练配置
    with wandb.init(project='mastering MinAtar with world models', config=config_dict):
        """training loop"""
        print('...training...')
        train_metrics = {} # 这个应该只是保存的可视化训练指标
        # 预测缓冲区
        trainer.collect_seed_episodes(env)
        obs, score = env.reset(), 0
        done = False
        # 获取初始化的RSSM状态，包含确定性状态和随机状态
        # 在dreamerv2的算法中，动作和观察偏移一个动作是正确的，因为实际开始游戏时，动作时没有的，游戏观察时有的，所以动作和游戏观察时存在一个时间差
        prev_rssmstate = trainer.RSSM._init_rssm_state(1)
        # 获取初始化的Action，初始为0 tensor
        prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
        episode_actor_ent = [] # 保存每次episode的actor熵，仅作为可视化指标
        scores = []
        best_mean_score = 0
        train_episodes = 0
        best_save_path = os.path.join(model_dir, 'models_best.pth')
        for iter in range(1, trainer.config.train_steps):  
            if iter%trainer.config.train_every == 0:
                train_metrics = trainer.train_batch(train_metrics)
            if iter%trainer.config.slow_target_update == 0:
                trainer.update_target()                
            if iter%trainer.config.save_every == 0:
                trainer.save_model(iter)
            with torch.no_grad():
                embed = trainer.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(trainer.device))  
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
                action, action_dist = trainer.ActionModel(model_state)
                action = trainer.ActionModel.add_exploration(action, iter).detach() # 预测的动作增加噪音，进行探索
                action_ent = torch.mean(action_dist.entropy()).item() # 计算熵
                episode_actor_ent.append(action_ent)

            # 环境执行动作，所以iter的维度是step？
            next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
            score += rew

            if done:
                # 如果游戏结束的处理方式
                train_episodes += 1
                # 记录探索的轨迹
                trainer.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                # 记录指标
                train_metrics['train_rewards'] = score
                train_metrics['action_ent'] =  np.mean(episode_actor_ent)
                train_metrics['train_steps'] = iter
                wandb.log(train_metrics, step=train_episodes)
                scores.append(score)
                # 判断是否保存最好的模型
                if len(scores)>100:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    if current_average>best_mean_score:
                        best_mean_score = current_average 
                        print('saving best model with mean score : ', best_mean_score)
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)
                
                # 重制模型
                obs, score = env.reset(), 0
                done = False
                # 重制RSSM状态
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                # 重制动作
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
                # 重制收集的actor熵
                episode_actor_ent = []
            else:
                # 和done一致
                trainer.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done)
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action

    '''evaluating probably best model'''
    # 完成训练，验证模型
    evaluator.eval_saved_agent(env, best_save_path)

if __name__ == "__main__":

    """there are tonnes of HPs, if you want to do an ablation over any particular one, please add if here"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help='mini atari env name')
    parser.add_argument("--id", type=str, default='0', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence Length (chunk length)')
    args = parser.parse_args()
    main(args)