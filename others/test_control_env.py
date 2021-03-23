import os

import matplotlib.pyplot as plt
import numpy as np
from control_env import GymSystem
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan, VecNormalize

tag_name = "PID_Control_PPO"
log_dir = os.path.join("./logs", tag_name)
save_path = None

env = GymSystem()
env = make_vec_env(lambda: env, n_envs=4, monitor_dir=log_dir)
try:
    env = VecNormalize.load(os.path.join(log_dir, "vec_normalize.pkl"), env)
except RuntimeError:
    raise RuntimeError("VecNormalize stats not found.")

env = VecCheckNan(env, raise_exception=True)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
if save_path is not None:
    model.load(save_path)
else:
    raise RuntimeError("Saved Model not found.")

test_env = GymSystem()
obs = test_env.reset()
obss = [obs]
norm_obss = [model.env.normalize_obs(obs)]
done = False
tot_r = 0.0
while not done:
    norm_obs = model.env.normalize_obs(obs)
    action, _ = model.predict(norm_obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    obss.append(obs)
    norm_obss.append(norm_obs)
    tot_r += reward
test_env.render()
obss = np.array(obss)
norm_obss = np.array(norm_obss)
plt.figure(figsize=(16, 6))
plt.plot(np.arange(len(obss[:, 0])), obss)
plt.grid()
plt.xlabel("time")
plt.ylabel("Value")
plt.figure(figsize=(16, 6))

plt.plot(np.arange(len(norm_obss[:, 0])), norm_obss)
plt.grid()
plt.xlabel("time")
plt.ylabel("Value")
