from stable_baselines3.common.env_checker import check_env
from cstr_control_env import GymCSTRPID
import matplotlib.pyplot as plt

env = GymCSTRPID()
check_env(env, warn=True)

obs = env.reset()
done = False
tot_r = 0.0
while not done:
    obs, reward, done, info = env.step([-1, -1, -1])
    tot_r += reward
print("Total Reward=", tot_r)

obs = env.reset()
done = False
tot_r = 0.0
while not done:
    obs, reward, done, info = env.step([1, 1, 1])
    tot_r += reward
print("Total Reward=", tot_r)

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
print("Total Reward=", tot_r)
env.render()
plt.show()
env.system.plot_model_regions()
labels = ["Qc", "Y", "T", "R"]
# print(len(obss))
plt.figure(figsize=(16, 16))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(
        np.arange(len(np.array(obss)[:, i])), np.array(obss)[:, i], label=labels[i]
    )
    plt.legend()
    plt.grid()