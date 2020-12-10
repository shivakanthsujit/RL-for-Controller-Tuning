import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from pid_control_env import GymNonLinearTankPID
import matplotlib.pyplot as plt
import numpy as np

torch.autograd.set_detect_anomaly(True)
tag_name = "baseline"
checkpoint_callback = CheckpointCallback(10000, "./models", tag_name)
eval_env = GymNonLinearTankPID(disturbance=True)
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecCheckNan(eval_env, raise_exception=True)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/" + tag_name + "/",
    n_eval_episodes=10,
    log_path="./logs/" + tag_name + "/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

callback = CallbackList([checkpoint_callback, eval_callback])
env = GymNonLinearTankPID(disturbance=True)
env = make_vec_env(lambda: env, n_envs=5)
env = VecCheckNan(env, raise_exception=True)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

model.learn(1000000, reset_num_timesteps=False, callback=callback, tb_log_name=tag_name)

test_env = GymNonLinearTankPID(deterministic=True, disturbance=True)
obs = test_env.reset()
actions = []
agent_actions = []
done = False
tot_r = 0.0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    actions.append(action)
    agent_actions.append(action.copy())
    obs, reward, done, info = test_env.step(action)
    tot_r += reward
    test_env.render("plot")

plt.figure(figsize=(16, 9))
plt.plot(np.arange(len(np.array(actions)[:, 0])), np.array(actions)[:, 0], label="Kp")
plt.plot(np.arange(len(np.array(actions)[:, 1])), np.array(actions)[:, 1], label="Ki")
plt.plot(np.arange(len(np.array(actions)[:, 2])), np.array(actions)[:, 2], label="Kd")
plt.legend()
plt.xlabel("time")
plt.ylabel("Value")
plt.title("PID Parameters")
plt.grid()

plt.figure(figsize=(16, 9))
plt.plot(
    np.arange(len(np.array(agent_actions)[:, 0])),
    np.array(agent_actions)[:, 0],
    label="Kp",
)
plt.plot(
    np.arange(len(np.array(agent_actions)[:, 1])),
    np.array(agent_actions)[:, 1],
    label="Ki",
)
plt.plot(
    np.arange(len(np.array(agent_actions)[:, 2])),
    np.array(agent_actions)[:, 2],
    label="Kd",
)
plt.legend()
plt.xlabel("time")
plt.ylabel("Value")
plt.title("Agent Actions")
plt.grid()