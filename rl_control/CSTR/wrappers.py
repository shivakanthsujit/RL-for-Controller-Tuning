import gym
import numpy as np


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, amount=1):
        super().__init__(env)
        self.amount = amount

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.amount and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class EarlyStopping(gym.Wrapper):
    def __init__(self, env, check_samples=[45, 46, 47, 48, 75, 76, 77, 78, 115, 116, 117, 118], tol=5e-4):
        super().__init__(env)
        self.check_samples = check_samples
        self.tol = tol

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        t_step = self.env.system.k - 1
        if t_step in self.check_samples:
            if np.abs(self.env.system.E[t_step]) > self.tol:
                done = True
                reward += -1.0
        return obs, reward, done, info
