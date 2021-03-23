import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan, VecNormalize

from cstr_control_env import CSTRFuzzyPID2, CSTRFuzzyPID3, GymCSTR, lamda
from wrappers import ActionRepeat, EarlyStopping

torch.autograd.set_detect_anomaly(True)
print("CUDA Available: ", torch.cuda.is_available())

early_stopping = True
print("Using Early Stopping: ", early_stopping)

action_repeat = True
action_repeat_value = 2
print("Using Action Repeat: ", action_repeat, action_repeat_value)

use_sde = False
print("Using gSDE: ", use_sde)

algo = "PPO"
print("Algorithm: ", algo)
tag_name = f"CSTRFuzzyPID2_{algo}_AR_{action_repeat}_{action_repeat_value}_lamda_{lamda}_use_sde_{use_sde}_ES_{early_stopping}"
print("Run Name: ", tag_name)

model_dir = "./models"
log_dir = os.path.join("./logs", tag_name)

# env = GymCSTR(system=CSTRFuzzyPID)
env = GymCSTR(system=CSTRFuzzyPID2)
if early_stopping:
    env = EarlyStopping(env)
if action_repeat:
    env = ActionRepeat(env, action_repeat_value)
env = make_vec_env(lambda: env, n_envs=1, monitor_dir=log_dir)
if os.path.exists(os.path.join(log_dir, "vec_normalize.pkl")):
    print("Found VecNormalize Stats. Using stats")
    env = VecNormalize.load(os.path.join(log_dir, "vec_normalize.pkl"), env)
else:
    print("No previous stats found. Using new VecNormalize instance.")
    env = VecNormalize(env)
env = VecCheckNan(env, raise_exception=True)

if algo == "PPO":
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/", use_sde=use_sde)
elif algo == "SAC":
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/", use_sde=use_sde)
else:
    raise NotImplementedError()

best_model_path = os.path.join(log_dir, "best_model.zip")
if os.path.exists(best_model_path):
    print("Found previous checkpoint. Loading from checkpoint.")
    if algo == "PPO":
        model = PPO.load(best_model_path, env)
    elif algo == "SAC":
        model = SAC.load(best_model_path, env)
    else:
        raise NotImplementedError()
print(model)

test_env = GymCSTR(
    #     system=CSTRFuzzyPID,
    system=CSTRFuzzyPID2,
    disturbance=False,
    deterministic=True,
)

if action_repeat:
    test_env = ActionRepeat(test_env, action_repeat_value)

obs = test_env.reset()
obss = [obs]
norm_obss = [model.env.normalize_obs(obs)]
actions = []
done = False
tot_r = 0.0
while not done:
    action, _ = model.predict(model.env.normalize_obs(obs), deterministic=True)
    obs, reward, done, info = test_env.step(action)
    obss.append(obs)
    norm_obss.append(model.env.normalize_obs(obs))
    actions.append(action)
    tot_r += reward
print("Evaluation Reward: ", tot_r)
test_env.render()
plt.show()

axis, axis_name = test_env.system.get_axis()
labels = test_env.system.state_names.copy()
obss = np.array(obss)
norm_obss = np.array(norm_obss)

plt.figure(figsize=(16, 4))
lineObj = plt.plot(axis[test_env.system.ksp :: action_repeat_value], actions)
plt.legend(lineObj, ["Kp", "taui", "taud"])
plt.ylabel("Value")
plt.xlabel(axis_name)
plt.xlim(axis[0], axis[-1])
plt.title("Agent Outputs")
plt.show()

plt.figure(figsize=(16, 9))
plt.subplot(2, 1, 1)
lineObj = plt.plot(axis[test_env.system.ksp :: action_repeat_value], obss[:-1])
plt.legend(lineObj, labels)
plt.ylabel("Value")
plt.xlabel(axis_name)
plt.xlim(axis[0], axis[-1])
plt.grid()
plt.title("Unnormalized States")

plt.subplot(2, 1, 2)
lineObj = plt.plot(axis[test_env.system.ksp :: action_repeat_value], norm_obss[:-1])
plt.legend(lineObj, labels)
plt.ylabel("Value")
plt.xlabel(axis_name)
plt.xlim(axis[0], axis[-1])
plt.grid()
plt.title("Normalized States")


values = np.arange(0, test_env.n_states)
n_values = np.max(values) + 1
regions = np.eye(n_values)[values]
regions = regions - 0.5

for i, region in enumerate(regions):
    action, _ = model.predict(model.env.normalize_obs(region), deterministic=True)
    action = test_env.convert_action(action)
    print(test_env.system.state_names[i])
    print("Kp: {:.3f}, TauI: {:.3f}, TauD: {:.3f}".format(action[0], action[1], action[2]))
