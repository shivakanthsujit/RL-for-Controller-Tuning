import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize

from cstr_control_env import GymCSTRPID

torch.autograd.set_detect_anomaly(True)


tag_name = "baseline"
model_dir = "./models"
logs_dir = "./logs"

eval_env = GymCSTRPID()
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecCheckNan(eval_env, raise_exception=True)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(logs_dir, tag_name),
    n_eval_episodes=20,
    log_path=os.path.join(logs_dir, tag_name),
    eval_freq=5000,
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(10000, "./models", tag_name)
callback = CallbackList([checkpoint_callback, eval_callback])
env = GymCSTRPID()
env = make_vec_env(lambda: env, n_envs=10)
env = VecCheckNan(env, raise_exception=True)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

model.learn(500000, reset_num_timesteps=False, callback=callback, tb_log_name=tag_name)

test_env = GymCSTRPID()
obss = []
obs = test_env.reset()
obss.append(obs)
done = False
agent_actions = []
tot_r = 0.0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    agent_actions.append(action.copy())
    obs, reward, done, info = test_env.step(action)
    obss.append(obs)
    tot_r += reward

test_env.render("plot")
plt.figure(figsize=(16, 9))
test_env.system.plot_model_regions()
labels = env.system.inputs
n_inputs = len(labels)
plt.figure(figsize=(16, 16))
for i in range(n_inputs):
    plt.subplot(n_inputs // 2 + 1, 2, i + 1)
    plt.plot(np.arange(len(np.array(obss)[:, i])), np.array(obss)[:, i], label=labels[i])
    plt.legend()
    plt.grid()
