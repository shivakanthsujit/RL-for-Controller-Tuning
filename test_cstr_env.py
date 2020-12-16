import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_checker import check_env

from cstr_control_env import GymCSTRPID


def test_env(env):
    check_env(env, warn=True)
    obs = env.reset()
    done = False
    tot_r = 0.0
    action = np.ones(env.action_space.shape[0]) * -1.0
    while not done:
        obs, reward, done, info = env.step(action)
        tot_r += reward
    print("Total Reward for [-1, -1, -1]=", tot_r)

    obs = env.reset()
    done = False
    tot_r = 0.0
    while not done:
        obs, reward, done, info = env.step(action * -1.0)
        tot_r += reward
    print("Total Reward for [1, 1, 1]=", tot_r)

    obss = []
    obs = env.reset()
    obss.append(obs)
    done = False
    actions = []
    tot_r = 0.0
    while not done:
        aa = env.action_space.sample()
        actions.append(aa)
        obs, reward, done, info = env.step(aa)
        obss.append(obs)
        tot_r += reward
    print("Total Reward for Random Actions=", tot_r)

    env.render()
    env.system.plot_model_regions()
    labels = env.system.inputs
    n_inputs = len(labels)
    plt.figure(figsize=(16, 16))
    for i in range(n_inputs):
        plt.subplot(n_inputs // 2 + 1, 2, i + 1)
        plt.plot(np.arange(len(np.array(obss)[:, i])), np.array(obss)[:, i], label=labels[i])
        plt.legend()
        plt.grid()


if __name__ == "__main__":
    env = GymCSTRPID()
    test_env(env)
