# Explore Quadrotor environment

import sys
import time
import numpy as np
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围内
from rlschool import make_env  # 使用 RLSchool 创建飞行器环境

if len(sys.argv) == 1:
    task = 'velocity_control'
else:
    task = sys.argv[1]

# for velocity_control: Yellow arrow is the expected velocity vector; orange arrow is the real velocity vector.
env = make_env("Quadrotor", task=task)  # velocity_control, hovering_control, no_collision
env.reset()
env.render()
reset = False
step = 1
total_reward = 0
ts = time.time()
print(env.action_space.low[0], env.action_space.high[0])
while not reset:
    # action = np.array([5., 5., 5., 5.], dtype=np.float32)
    action = np.random.random(4)*2-1
    action = action_mapping(action, env.action_space.low[0],
                            env.action_space.high[0])
    state, reward, reset, info = env.step(action)
    total_reward += reward
    env.render()
    # print('---------- step %s ----------' % step)
    # print('state:', state)
    # print('info:', info)
    # print('action:', action)
    # print('reward:', reward)
    step += 1
env.close()
print('total reward: ', total_reward)
te = time.time()
print('time cost: ', te - ts)