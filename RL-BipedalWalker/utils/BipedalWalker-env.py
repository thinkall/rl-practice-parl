# Explore CarRacing Environment

import numpy as np
import gym
from pyglet.window import key


a = np.array([0.0, 0.0, 0.0])


def key_press(k, mod):
    global restart, isquit
    if k == key.ENTER: restart = True
    if k == key.ESCAPE: isquit = True
    if k == key.LEFT:  a[0] = -1.0
    if k == key.RIGHT: a[0] = +1.0
    if k == key.UP:    a[1] = +1.0
    if k == key.DOWN:  a[2] = +0.8  # set 1.0 for wheels to block to zero rotation


def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0: a[0] = 0
    if k == key.RIGHT and a[0] == +1.0: a[0] = 0
    if k == key.UP:    a[1] = 0
    if k == key.DOWN:  a[2] = 0


# env = gym.make('CarRacing-v0')
env = gym.make('BipedalWalker-v3')  # To solve the game you need to get 300 points in 1600 time steps.
# env = gym.make('LunarLanderContinuous-v2')
env.render()
env.reset()
print(env.action_space)
print(env.action_space.sample())
print(env.action_space.low[0], env.action_space.high[0])
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release
record_video = False
if record_video:
    from gym.wrappers.monitor import Monitor
    env = Monitor(env, './video-test', force=True)
isopen = True
isquit = False
while isopen:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
        # print(env.action_space.sample())
        s, r, done, info = env.step(a)  # env.step(env.action_space.sample()) # take a random action
        total_reward += r
        if steps % 200 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        isopen = env.render()
        if done or restart or isopen == False:
            break
    print(isquit)
    if isquit:
        break
env.close()
print('total_reward: {}'.format(total_reward))