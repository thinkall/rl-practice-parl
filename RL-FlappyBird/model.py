# -*- coding: utf-8 -*-
import parl
from parl import layers


class FlappyBirdModel(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 100
        hid2_size = 100
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q
