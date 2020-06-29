# Modified from https://github.com/PaddlePaddle/PARL/blob/develop/examples/tutorials/homework/lesson5/ddpg_quadrotor/quadrotor_model.py

# -*- coding: utf-8 -*-

import paddle.fluid as fluid
import parl
from parl import layers


class ActorModel(parl.Model):
    def __init__(self, act_dim, model_tag):
        self.model_tag = model_tag
        if self.model_tag == 1:
            # simple model
            hid1_size = 100
            hid2_size = 100
            self.fc1 = layers.fc(size=hid1_size, act='tanh', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
            self.fc2 = layers.fc(size=hid2_size, act='tanh', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
            self.fc3 = layers.fc(size=act_dim, act='tanh', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        else:
            hid1_size = 128
            hid2_size = 256
            hid3_size = 128
            self.fc1 = layers.fc(size=hid1_size, act='relu', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
            self.fc2 = layers.fc(size=hid2_size, act='relu', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
            self.fc3 = layers.fc(size=hid3_size, act='relu', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
            self.fc4 = layers.fc(size=act_dim, act='tanh', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))

    def policy(self, obs):
        if self.model_tag == 1:
            hid = self.fc1(obs)
            hid = self.fc2(hid)
            logits = self.fc3(hid)
        else:
            hid1 = self.fc1(obs)
            hid2 = self.fc2(hid1)
            hid3 = self.fc3(hid2)
            logits = self.fc4(hid3)
        return logits


class CriticModel(parl.Model):
    def __init__(self, model_tag):
        self.model_tag = model_tag
        if self.model_tag == 1:
            hid_size = 100

            self.fc1 = layers.fc(size=hid_size, act='relu', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
            self.fc2 = layers.fc(size=1, act=None)
        else:
            hid1_size = 100
            hid2_size = 100

            self.fc1 = layers.fc(size=hid1_size, act='relu', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
            self.fc2 = layers.fc(size=hid2_size, act='relu', param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
            self.fc3 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        # 输入 state, action, 输出对应的Q(s,a)
        if self.model_tag == 1:
            concat = layers.concat([obs, act], axis=1)
            hid = self.fc1(concat)
            Q = self.fc2(hid)
            Q = layers.squeeze(Q, axes=[1])
        else:
            hid1 = self.fc1(obs)
            concat = layers.concat([hid1, act], axis=1)
            hid2 = self.fc2(concat)
            Q = self.fc3(hid2)
            Q = layers.squeeze(Q, axes=[1])
        return Q


class BipedalWalkerModel(parl.Model):
    def __init__(self, act_dim, model_tag):
        self.model_tag = model_tag
        self.actor_model = ActorModel(act_dim, self.model_tag)
        self.critic_model = CriticModel(self.model_tag)

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()