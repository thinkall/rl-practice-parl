# Modified from https://github.com/PaddlePaddle/PARL/blob/develop/examples/tutorials/homework/lesson5/ddpg_quadrotor/train.py

# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import gym

import parl
from parl import layers
from paddle import fluid
from parl.utils import logger
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围内
from parl.utils import ReplayMemory # 经验回放
from parl.algorithms import DDPG
from parl.utils.scheduler import LinearDecayScheduler, PiecewiseScheduler

from model import BipedalWalkerModel
from agent import BipedalWalkerAgent

import matplotlib.pyplot as plt


def run_episode(env, agent, rpm):
    obs = env.reset()
    # print(obs)
    total_reward, steps = 0, 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)

        # 给输出动作增加探索扰动，输出限制在 [-1.0, 1.0] 范围内
        action = np.clip(np.random.normal(action, 1.0), -1.0, 1.0)
        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)
        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
            batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.squeeze(action)
            action = np.clip(action, -1.0, 1.0)  # the action should be in range [-1.0, 1.0]
            action = action_mapping(action, env.action_space.low[0],
                                    env.action_space.high[0])

            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward
            steps += 1

            if render:
                env.render()
            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


def draw_results(train_score_list, test_score_list, title='', path='./fig_dir/'):
    # 画出训练过程reward历史曲线
    plt.figure(figsize=[14, 6])
    plt.subplot(121)
    plt.plot(train_score_list, color='green', label='train')
    plt.title('Train History {}'.format(title))
    plt.legend()
    plt.subplot(122)
    plt.plot(test_score_list, color='red', label='test')
    plt.title('Test History {}'.format(title))
    plt.legend()
    if path != '' and not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + title + '.png')
    plt.show()


def main(ACTOR_LR=0.0002, CRITIC_LR=0.001, model_tag=1, load_model=False, go_steps=1, f_best=''):
    # 创建环境
    env = gym.make('BipedalWalker-v3')
    env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 根据parl框架构建agent
    model = BipedalWalkerModel(act_dim, model_tag=model_tag)
    algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = BipedalWalkerAgent(algorithm, obs_dim, act_dim)

    # parl库也为DDPG算法内置了ReplayMemory，可直接从 parl.utils 引入使用
    rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)

    # 启动训练
    logger.info(
        'Params: ACTOR_LR={}, CRITIC_LR={}, model_tag={}, Pid {}'.format(ACTOR_LR, CRITIC_LR, model_tag, os.getpid()))
    test_flag = 0
    total_steps = 0
    early_stop = 0
    last_reward = -1e9
    best_reward = -1e9
    # 与下文结合，使得学习率在训练到90%时保持初始学习率的1%水平继续训练
    actor_lr_scheduler = [(0, ACTOR_LR), (int(TRAIN_TOTAL_STEPS / 5), ACTOR_LR / 5),
                          (int(TRAIN_TOTAL_STEPS / 2), ACTOR_LR / 10)]
    critic_lr_scheduler = [(0, CRITIC_LR), (int(TRAIN_TOTAL_STEPS / 5), CRITIC_LR / 5),
                           (int(TRAIN_TOTAL_STEPS / 2), CRITIC_LR / 10)]
    actor_lr_scheduler = PiecewiseScheduler(actor_lr_scheduler)
    critic_lr_scheduler = PiecewiseScheduler(critic_lr_scheduler)
    train_score_list = []
    test_score_list = []

    # load best results and continue training
    if load_model == True and os.path.exists(f_best + '.ckpt'):
        agent.restore(f_best + '.ckpt')
        rpm.load('rpm_{}'.format(model_tag))
        actor_lr_scheduler.step(step_num=go_steps)
        critic_lr_scheduler.step(step_num=go_steps)
        logger.info('load model success. Pid {}'.format(os.getpid()))

    while total_steps < TRAIN_TOTAL_STEPS:
        train_reward, steps = run_episode(env, agent, rpm)
        total_steps += steps
        # logger.info('Steps: {} Reward: {} Pid: {}'.format(total_steps, train_reward, os.getpid())) # 打印训练reward
        train_score_list.append(train_reward)

        # 可以在这里修改学习率, 可以用 parl.utils.scheduler 中的 LinearDecayScheduler 进行修改，也可以自行修改
        agent.alg.actor_lr = max(actor_lr_scheduler.step(step_num=steps), ACTOR_LR / 100)
        agent.alg.critic_lr = max(critic_lr_scheduler.step(step_num=steps), CRITIC_LR / 100)

        if total_steps // TEST_EVERY_STEPS >= test_flag:  # 每隔一定step数，评估一次模型
            while total_steps // TEST_EVERY_STEPS >= test_flag:
                test_flag += 1

            evaluate_reward = evaluate(env, agent)
            test_score_list.append(evaluate_reward)
            logger.info(
                'Steps {}, Test reward: {}, Pid {}'.format(total_steps, evaluate_reward, os.getpid()))  # 打印评估的reward

            with open('train_{}.pickle'.format(model_tag), 'wb') as f:
                pickle.dump([train_score_list, test_score_list], f)
                print('history saved')

            # 每评估一次，优于最优模型就保存一次模型和记忆回放，以训练的step数命名，DEBUG时则一直保存模型和图片
            if evaluate_reward > best_reward or DEBUG:  # velocity control task reward will always be negative
                ckpt = 'model_dir/steps_{}_evaluate_reward_{}_ACTOR_LR_{}_CRITIC_LR_{}_model_tag_{}'.format(total_steps,
                                                                                                            int(
                                                                                                                evaluate_reward),
                                                                                                            ACTOR_LR,
                                                                                                            CRITIC_LR,
                                                                                                            model_tag)
                agent.save(ckpt + '.ckpt')
                rpm.save('rpm_{}'.format(model_tag))
                logger.info('Current actor_lr: {}  critic_lr: {}  Pid {} ckpt {}'.format(agent.alg.actor_lr,
                                                                                         agent.alg.critic_lr,
                                                                                         os.getpid(), ckpt))

                # 每次保存模型时画出当前reward趋势图
                draw_results(train_score_list, test_score_list,
                             '_'.join([str(ACTOR_LR), str(CRITIC_LR), str(model_tag), str(total_steps)]))

                # update best reward
                best_reward = evaluate_reward

            # early_stop, 超过20%训练进度且连续5次测评reward下降则提前终止
            if evaluate_reward > last_reward:
                early_stop = 0
            else:
                early_stop += 1
            last_reward = evaluate_reward
            if total_steps > TRAIN_TOTAL_STEPS / 5 and early_stop >= 5:
                logger.info(
                    'No good results, stop training. Params: ACTOR_LR={}, CRITIC_LR={}, model_tag={}, Pid {}'.format(
                        ACTOR_LR, CRITIC_LR, model_tag, os.getpid()))
                break
    # 训练结束，画出reward趋势图，并保存最终模型
    draw_results(train_score_list, test_score_list,
                 '_'.join([str(ACTOR_LR), str(CRITIC_LR), str(model_tag), str(total_steps)]))
    ckpt = 'model_dir/steps_{}_evaluate_reward_{}_ACTOR_LR_{}_CRITIC_LR_{}_model_tag_{}'.format(total_steps,
                                                                                                int(evaluate_reward),
                                                                                                ACTOR_LR, CRITIC_LR,
                                                                                                model_tag)
    agent.save(ckpt + '.ckpt')
    rpm.save('rpm_{}'.format(model_tag))
    with open('train_{}.pickle'.format(model_tag), 'wb') as f:
        pickle.dump([train_score_list, test_score_list], f)
        print('history saved')
    logger.info('Current actor_lr: {}  critic_lr: {}  Pid {} ckpt {}'.format(agent.alg.actor_lr, agent.alg.critic_lr,
                                                                             os.getpid(), ckpt))


def parallel(num_cores=2, num_gpus=0):
    assert isinstance(num_cores, int)
    assert num_cores > 0
    from multiprocessing import Pool

    # 多进程不能使用GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    print('Parent process %s.' % os.getpid())

    p = Pool(num_cores)
    used_gpus = 0
    for ACTOR_LR in [0.0001, 0.0002, 0.0005, 0.001, 0.002]:  # 0.0002
        for CRITIC_LR in [0.001, 0.005, 0.01]:  # 0.001
            for model_tag in [2]:
                p.apply_async(main, args=(ACTOR_LR, CRITIC_LR, model_tag))

    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


def one(ACTOR_LR=0.0002, CRITIC_LR=0.001, model_tag=2, load_model=False, go_steps=1, f_best='', gpu=''):
    # 默认不使用GPU，在我的服务器出现以下错误：
    # ExternalError:  Cublas error, CUBLAS_STATUS_NOT_INITIALIZED  at (/paddle/paddle/fluid/platform/cuda_helper.h:32)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    main(ACTOR_LR, CRITIC_LR, model_tag, load_model, go_steps, f_best)


def find_best():
    best = -1000000
    best_f = ''
    for _, _, files in os.walk('./model_dir'):
        for f in files:
            if f[-4:] == 'ckpt':
                reward = int(f[f.find('reward') + 7:f.find('ACTOR') - 1])
                if reward >= best:
                    best = reward
                    best_f = f
    return 'model_dir/' + best_f[:-5]


def fine_tune(ACTOR_LR=0.0002, CRITIC_LR=0.005, episodes=10, model_tag=2, load_model=True, go_steps=1):
    for i in range(episodes):
        f_best = find_best()
        logger.info('Current best: {}, finetune it...'.format(f_best))
        # todo: extract go_steps from f_best
        one(ACTOR_LR=ACTOR_LR, CRITIC_LR=CRITIC_LR, model_tag=model_tag, load_model=load_model, go_steps=go_steps, f_best=f_best)


def test(total_steps, evaluate_reward, ACTOR_LR=0.0002, CRITIC_LR=0.001, model_tag=2, render=False):
    # 请设置ckpt为你训练中效果最好的一次评估保存的模型文件名称
    ckpt = 'model_dir/steps_{}_evaluate_reward_{}_ACTOR_LR_{}_CRITIC_LR_{}_model_tag_{}.ckpt'.format(total_steps, int(evaluate_reward), ACTOR_LR, CRITIC_LR, model_tag)
    env = gym.make('BipedalWalker-v3')
    env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model = BipedalWalkerModel(act_dim, model_tag=model_tag)
    algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = BipedalWalkerAgent(algorithm, obs_dim, act_dim)
    # 加载模型
    if os.path.exists(ckpt):
        agent.restore(ckpt)
        logger.info('Test Model file {}'.format(ckpt))
    else:
        logger.info('No Model file {}'.format(ckpt))
        return -1
    evaluate_reward = evaluate(env, agent, render=render)
    logger.info('Evaluate reward: {}'.format(evaluate_reward)) # 打印评估的reward


def test_best(render=False):
    best = -1000000
    best_f = ''
    for _, _, files in os.walk('./model_dir'):
        for f in files:
            if f[-4:] == 'ckpt':
                reward = int(f[f.find('reward')+7:f.find('ACTOR')-1])
                if reward >= best:
                    best = reward
                    best_f = f
    res = best_f.split('_')
    test(res[1], res[4], float(res[7]), float(res[10]), int(res[-1].split('.')[0]), render=render)


if __name__ == '__main__':
    ACTOR_LR = 0.0001  # Actor网络更新的 learning rate
    CRITIC_LR = 0.001  # Critic网络更新的 learning rate

    GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
    TAU = 0.001  # target_model 跟 model 同步参数 的 软更新参数
    MEMORY_SIZE = 1e6  # replay memory的大小，越大越占用内存
    MEMORY_WARMUP_SIZE = 1e3  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
    REWARD_SCALE = 0.01  # reward 的缩放因子
    BATCH_SIZE = 512  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
    TRAIN_TOTAL_STEPS = 1e6  # 总训练步数
    TEST_EVERY_STEPS = 1e3  # 每个N步评估一下算法效果，每次评估5个episode求平均reward

    DEBUG = False
    # parallel(15)  # train from beginning
    test_best(render=True)  # test current best model
    # fine_tune(ACTOR_LR=0.0001, CRITIC_LR=0.01, episodes=10, go_steps=1)  # finetune existing model
    # one(ACTOR_LR=0.0001, CRITIC_LR=0.001, model_tag=1, load_model=False, go_steps=1, f_best='', gpu='')
