# Explore FlappyBird environment

import os
import numpy as np
from random import choice
from pyglet.window import key
from ple.games.flappybird import FlappyBird
from ple.games.snake import Snake
from ple.games.pixelcopter import Pixelcopter
from ple.games.monsterkong import MonsterKong
from ple.games.raycastmaze import RaycastMaze
from ple import PLE

## Headless Usage
# os.putenv('SDL_VIDEODRIVER', 'fbcon')
# os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initiate game
game = FlappyBird()  # (height=512, width=288)
# game = Pixelcopter()
# game = MonsterKong()
# game = RaycastMaze(height=480, width=480)
# game = Snake(height=480, width=480)
# game = FlappyBird()

env = PLE(game, fps=30, display_screen=True)
env.init()
reward = 0.0
for i in range(1000):
   if env.game_over():
       print(step_reward)
       print(env.game_over())
       break
       # env.reset_game()


   observation = env.getScreenRGB()  # (288, 512, 3)
   action = 119 if np.random.random() > 0.9 else None  # choice(env.getActionSet())
   step_reward = env.act(action)

   if i > 5:
       env.display_screen = False
       print("I'm still running")

   reward += step_reward
   if i == 0:
       print(type(observation), observation.shape)
       print(env.getActionSet())
       print(len(env.getActionSet()))
       print(action)
       step_state = env.getGameState()
       print(type(step_state), step_state)
       print(len(step_state))
       print([step_state[k] for _, k in enumerate(step_state)])
       print(env.fps)
       print(env.game_over())
   # print(action)

print('total_reward: {}'.format(reward))


