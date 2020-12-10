from stable_baselines3.common.env_checker import check_env
from pid_control_env import GymNonLinearTankPID
import matplotlib.pyplot as plt
import numpy as np

env = GymNonLinearTankPID()
check_env(env, warn=True)

obs = env.reset()
print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())
actions = []
done = False
tot_r = 0.0
while not done:
    aa = env.action_space.sample()
    actions.append(aa)
    obs, reward, done, info = env.step([-1, -1, -1])
    tot_r += reward

print("Total Reward=", tot_r)
env.render()
plt.show()
plt.figure(figsize=(16, 9))
plt.plot(np.arange(len(np.array(actions)[:, 0])), np.array(actions)[:, 0], label="Kp")
plt.plot(np.arange(len(np.array(actions)[:, 1])), np.array(actions)[:, 1], label="Ki")
plt.plot(np.arange(len(np.array(actions)[:, 2])), np.array(actions)[:, 2], label="Kd")
plt.legend()