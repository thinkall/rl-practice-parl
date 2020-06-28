# Modified from https://github.com/PaddlePaddle/PARL/blob/develop/examples/tutorials/homework/lesson3/dqn_mountaincar/train.py

#-*- coding: utf-8 -*-

import os
import gym
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import parl
from parl.utils import ReplayMemory  # 经验回放
from parl.algorithms import DQN  # DQN
from parl.utils import logger
from parl.utils.scheduler import LinearDecayScheduler, PiecewiseScheduler
from ple import PLE
from ple.games.flappybird import FlappyBird

from agent import FlappyBirdAgent
from model import FlappyBirdModel


SHOW_SCORE = 0  # 是否显示当前得分，大于0显示
RECORD = 0  # 是否需要录屏，大于0录


def action_mapping(action):
    # one hot action in model will return 0 or 1,
    # need to transfer to 119 or None for real action in ple.
    return 119 if action == 0 else None


# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0 
    step = 0
    env.reset_game()
    if IMAGE_MODE:
        obs = env.getScreenRGB()
        # preprocess image if in image mode
        obs = preprocess(obs)  # from shape (288, 512, 3) to (9216,)
    else:
        obs = env.getGameState()
        obs = [obs[k] for _, k in enumerate(obs)]
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        reward = env.act(action_mapping(action))
        if IMAGE_MODE:
            next_obs = env.getScreenRGB()
            # preprocess image if in image mode
            next_obs = preprocess(obs)  # from shape (288, 512, 3) to (9216,)
        else:
            next_obs = env.getGameState()
            next_obs = [next_obs[k] for _, k in enumerate(next_obs)]
        done = env.game_over()

        rpm.append(obs=obs, act=action, reward=reward, next_obs=next_obs, terminal=done)

        # print(obs, action, reward, next_obs, done, action_mapping(action))

        # train model
        if (rpm.size() > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample_batch(BATCH_SIZE)
            # PARL自带ReplayMemory的输出需要修改一下再输入learn函数
            batch_action = batch_action[:, 0]
            batch_done = np.array(batch_done).astype('float32')
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 增加显示当前得分的功能
if SHOW_SCORE > 0:
    fig = plt.figure(figsize=[2.9, 0.5])
    plt.ion()


def draw_score_now(episode_reward):
    plt.clf()  # 清除之前画的图
    plt.xlim(0, 1)
    plt.ylim(0, 0.1)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.text(0, 0, 'Score: {}'.format(episode_reward), fontsize=24, color='Red')
    plt.show(block=False)


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False, rounds=5):
    assert rounds > 0
    global RECORD, SHOW_SCORE
    eval_reward = []
    env.init()
    for i in range(rounds):
        episode_reward = 0
        env.reset_game()
        if IMAGE_MODE:
            obs = env.getScreenRGB()
            # preprocess image if in image mode
            obs = preprocess(obs)  # from shape (288, 512, 3) to (9216,)
        else:
            obs = env.getGameState()
            obs = [obs[k] for _, k in enumerate(obs)]
        while True:
            action = agent.predict(obs)
            reward = env.act(action_mapping(action))
            if IMAGE_MODE:
                obs = env.getScreenRGB()
                # preprocess image if in image mode
                obs = preprocess(obs)  # from shape (288, 512, 3) to (9216,)
            else:
                obs = env.getGameState()
                obs = [obs[k] for _, k in enumerate(obs)]
            isOver = env.game_over()

            episode_reward += reward
            if render:
                env.display_screen = True
            else:
                env.display_screen = False

            # 显示得分
            if SHOW_SCORE > 0:
                draw_score_now(episode_reward)
            if RECORD > 0:
                RECORD += 1
                if 5 <= RECORD <= 10:
                    # 前几帧画面空白，等出内容了再暂停
                    # 留时间将得分和游戏画面移动到合适的位置，可以为录屏做准备
                    time.sleep(5)

            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def preprocess(image):
    # if use IMAGE as input, model should use cnn, not only fc
    from PIL import Image
    """ 预处理 (288, 512, 3) uint8 frame into (72, 128) float 2 维 float matrix """
    image = image[::4, ::4, 0]  # 下采样，缩放4倍
    image = Image.fromarray(image)
    image = image.convert('L')  # 转为灰度图像
    image = np.array(image)
    return image.astype(np.float)  # .ravel()  # with ravel to flatten


def main(lr_scheduler, max_episode=10000, load_model=False, go_steps=1, f_pretrain=''):
    # Initiate game
    game = FlappyBird()  # (height=512, width=288)
    env = PLE(game, fps=30, display_screen=False)
    env1 = PLE(game, fps=30, display_screen=True)
    env.init()
    env1.init()
    if IMAGE_MODE:
        obs_dim = preprocess(env.getScreenRGB()).shape
    else:
        obs_dim = len(env.getGameState())
    act_dim = len(env.getActionSet())
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, act_dim)  # DQN的经验回放池
    # 根据parl框架构建agent
    model = FlappyBirdModel(act_dim=act_dim)
    algorithm = DQN(model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = FlappyBirdAgent(algorithm, obs_dim=obs_dim, act_dim=act_dim,
                            e_greed=0.1,  # 有一定概率随机选取动作，探索
                            e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

    # Use Shceduler to adjust learning rate during training process
    _lr_scheduler = PiecewiseScheduler(lr_scheduler)

    # load best results and continue training
    if load_model and os.path.exists(f_pretrain):
        agent.restore(f_pretrain)
        rpm.load('./dqn_model.rpm.npz')
        _lr_scheduler.step(step_num=go_steps)
        logger.info('load model success. Pid {}'.format(os.getpid()))

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while rpm.size() < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)

    episode = 0
    last_reward = -1e9
    train_history = []
    test_history = []
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        for i in range(0, 50):
            total_reward = run_episode(env, agent, rpm)
            episode += 1
            train_history.append(total_reward)
            agent.alg.lr = _lr_scheduler.step(step_num=1)

        # test part
        eval_reward = evaluate(env1, agent, render=True)  # render=True 查看显示效果
        test_history.append(eval_reward)
        with open('train1.pickle', 'wb') as f:
            pickle.dump([train_history, test_history], f)
            print('data saved')
        logger.info('episode:{}, e_greed:{}, lr:{}, test_reward:{}, train_reward:{}'.format(
            episode, agent.e_greed, agent.alg.lr, eval_reward, total_reward))

        # 训练得到改善，保存模型
        if eval_reward > last_reward:
            save_path = './model_dir/dqn_model_{}_{}'.format(episode, eval_reward)
            agent.save(save_path + '.ckpt')
            rpm.save('./dqn_model.rpm')

        last_reward = eval_reward

    # 训练结束，保存模型
    save_path = './model_dir/dqn_model_final.ckpt'
    agent.save(save_path)
    rpm.save('./dqn_model.rpm')


def test(f_model='', rounds=5):
    # Initiate game
    game = FlappyBird()  # (height=512, width=288)
    env = PLE(game, fps=30, display_screen=True)
    env.init()
    if IMAGE_MODE:
        obs_dim = preprocess(env.getScreenRGB()).shape
    else:
        obs_dim = len(env.getGameState())
    act_dim = len(env.getActionSet())
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
    model = FlappyBirdModel(act_dim=act_dim)
    algorithm = DQN(model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = FlappyBirdAgent(algorithm, obs_dim=obs_dim, act_dim=act_dim,
                            e_greed=0.1,  # 有一定概率随机选取动作，探索
                            e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

    # load best results and continue training
    if os.path.exists(f_model):
        agent.restore(f_model)
        logger.info('load model success. Pid {}'.format(os.getpid()))
        eval_reward = evaluate(env, agent, render=True)  # render=True 查看显示效果

    logger.info('test_reward:{}'.format(eval_reward))
    return eval_reward


if __name__ == '__main__':
    HEADLESS = False  # if no display like a server, set to True
    if HEADLESS:
        # Headless Usage
        os.putenv('SDL_VIDEODRIVER', 'fbcon')
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    IMAGE_MODE = False  # todo: try using RGB image as input
    LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
    MEMORY_SIZE = 100000  # replay memory的大小，越大越占用内存
    MEMORY_WARMUP_SIZE = 1000  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn, 要大于300
    BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
    LEARNING_RATE = 0.0005  # 学习率
    # learning rate adjustment schedule: (train_step, learning_rate)
    lr_scheduler = [(0, LEARNING_RATE), (2000, LEARNING_RATE/2), (5000, LEARNING_RATE/5)]
    GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
    ## Finetune an existing model
    # main(lr_scheduler, max_episode=100000, load_model=True, go_steps=1, f_pretrain='./model_dir/dqn_model_1600_2274.5.ckpt')

    ## Train a new model
    # main(lr_scheduler, max_episode=20000, load_model=False, go_steps=1, f_pretrain='')

    ## Test a pre-trained model
    test(f_model='./model_dir/dqn_model_1600_2274.5.ckpt', rounds=1)
