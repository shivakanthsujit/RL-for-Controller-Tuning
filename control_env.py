import gym
import matplotlib.pyplot as plt
import numpy as np
from control.matlab import c2d, ssdata, tf, tf2ss
from gym import spaces
from simple_pid import PID

from utils import fig2data

uinit = 0.0
umin = -40.0
umax = 40.0
yinit = 4.0
delt = 0.1
slew_rate = 20.0
num = [1]
dem = [3, 1]


class System:
    def __init__(
        self,
        uinit=uinit,
        yinit=yinit,
        num=num,
        dem=dem,
        delt=delt,
        ttfinal=None,
        disturbance=True,
        deterministic=False,
    ):
        # Simulation settings
        self.delt = delt  # sample time
        self.ttfinal = ttfinal  # final simulation time
        self.ksp = 10
        self.slew_rate = slew_rate
        self.umin = umin
        self.umax = umax
        self.input_low = np.array([self.umin])
        self.input_high = np.array([self.umax])
        self.reset_init(uinit, yinit, num, dem, ttfinal, disturbance, deterministic)

    @property
    def state_names(self):
        return ["Setpoint(k)", "Output(k)", "Output(k-1)"]

    @property
    def input_names(self):
        return ["Input(k)"]

    @property
    def n_states(self):
        return len(self.get_state())

    @property
    def n_actions(self):
        return self.input_low.shape[0]

    def reset_init(
        self,
        uinit=uinit,
        yinit=yinit,
        num=num,
        dem=dem,
        ttfinal=None,
        disturbance=True,
        deterministic=False,
    ):
        #  plant initial conditions
        self.uinit = uinit
        self.yinit = yinit
        self.num = num
        self.dem = dem
        self.disturbance = disturbance
        if deterministic:
            self.r = np.concatenate(
                (
                    np.ones((self.ksp, 1)) * self.yinit,
                    np.ones((40, 1)) * 5.0,
                    np.ones((30, 1)) * 3.0,
                    np.ones((40, 1)) * 6,
                )
            )
        else:
            self.r = np.concatenate(
                (
                    np.ones((self.ksp, 1)) * self.yinit,
                    np.ones((40, 1)) * np.random.uniform(6.0, 8.0),
                    np.ones((30, 1)) * np.random.uniform(3.0, 6.0),
                    np.ones((40, 1)) * np.random.uniform(3.0, 4.0),
                )
            )

        sim_time = len(self.r) * self.delt
        self.ttfinal = ttfinal if ttfinal is not None and ttfinal < sim_time else sim_time
        self.tt = np.arange(0, self.ttfinal, self.delt)  # time vector
        self.kfinal = len(self.tt)  # number of time intervals

        self.linear_tf = tf(self.num, self.dem)
        self.linear_ss = tf2ss(self.linear_tf)
        #  discretize the plant with a sample time, delt
        self.sysd_plant = c2d(self.linear_ss, self.delt)
        [self.phi, self.gamma, self.cd, self.dd] = ssdata(self.sysd_plant)
        self.xinit = self.yinit / self.sysd_plant.C
        return self.reset()

    def reset_env(self):
        #  Control vector
        self.u = np.zeros((self.kfinal + 1, 1))
        self.u[: self.ksp] = np.ones((self.ksp, 1)) * self.uinit
        self.du = np.zeros((self.kfinal, 1))

        # Environment State vector
        self.x = np.zeros((self.kfinal + 1, self.sysd_plant.A.shape[0], 1))
        self.x[: self.ksp] = np.repeat(np.array(self.xinit), self.ksp).reshape(self.ksp, self.sysd_plant.A.shape[0], 1)

        # Output vector
        self.y = np.zeros((self.kfinal + 1, 1))
        self.y[: self.ksp] = np.ones((self.ksp, 1)) * self.yinit

        # Error vector
        self.E = np.zeros((self.kfinal, 1))
        self.k = self.ksp - 1
        return self.r[self.k][0], self.y[self.k][0]

    def reset(self):
        # Input vector
        self.input = np.zeros((self.kfinal + 1, self.n_actions))
        self.input[: self.ksp] = np.ones((self.ksp, 1)) * self.uinit
        return self.reset_env()

    def step_env(self, u):
        # Slew rate
        if self.slew_rate:
            u = np.clip(u, self.u[self.k - 1] - self.slew_rate, self.u[self.k - 1] + self.slew_rate)
        # Disturbance
        d = 0.05 * np.random.randn() if self.disturbance else 0.0
        self.u[self.k] = u + d
        # Control Constraints
        self.u[self.k] = np.clip(self.u[self.k], self.umin, self.umax)
        #  plant equations
        self.x[self.k + 1] = np.matmul(self.phi, self.x[self.k]) + np.matmul(self.gamma, self.u[self.k]).T
        self.y[self.k + 1] = np.matmul(self.cd, self.x[self.k + 1])

        self.k = self.k + 1
        return self.r[self.k][0], self.y[self.k][0]

    def step(self, u):
        self.input[self.k] = np.array(u)
        return self.step_env(u)

    def get_state(self):
        return self.r[self.k][0], self.y[self.k][0], self.y[self.k - 1][0]

    def plot(self, save=False):
        plt.figure(figsize=(16, 16))
        plt.subplot(3, 1, 1)
        plt.step(
            self.tt[: self.k],
            self.r[: self.k],
            linestyle="dashed",
            label="Setpoint",
        )
        plt.plot(self.tt[: self.k], self.y[: self.k], label="Plant Output")
        plt.ylabel("y")
        plt.xlabel("time")
        plt.title("Plant Output")
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.step(self.tt[: self.k], self.u[: self.k], label="Control Input")
        plt.ylabel("u")
        plt.xlabel("time")
        plt.title("Control Action")
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 3)
        for i in range(self.n_actions):
            plt.plot(self.tt[: self.k], self.input[: self.k, i], label=self.input_names[i])
        plt.ylabel("Value")
        plt.xlabel("time")
        plt.title("Inputs")
        plt.grid()
        plt.legend()
        if save:
            plt.tight_layout()
            return fig2data(plt.gcf())

    def ise(self):
        return ((self.r[: self.k] - self.y[: self.k]) ** 2).sum()


min_gains = [0.0, 0.0, 0.0]
max_gains = [20.0, 20.0, 20.0]


class SystemSimplePID(System):
    @property
    def input_names(self):
        return ["Kp(k)", "Ki(k)", "Kd(k)"]

    def reset(self, *args, **kwargs):
        self.Gc = PID(
            5, 1, 0, setpoint=self.yinit, sample_time=self.delt, output_limits=(self.umin, self.umax), auto_mode=False
        )
        self.Gc.set_auto_mode(True, last_output=self.uinit)
        self.slew_rate = None
        self.input_low = np.array(min_gains)
        self.input_high = np.array(max_gains)
        # Input vector
        self.input = np.zeros((self.kfinal + 1, self.n_actions))
        self.input[: self.ksp] = np.ones((self.ksp, self.n_actions)) * np.array([5, 1, 0])
        self.gains = []
        return self.reset_env(*args, **kwargs)

    def step(self, Kp, Ki, Kd):
        self.E[self.k] = self.r[self.k][0] - self.y[self.k][0]
        self.Gc.setpoint = self.r[self.k][0]
        self.Gc.tunings = (Kp, Ki, Kd)
        u = self.Gc(self.y[self.k][0], self.delt)

        self.input[self.k] = np.array([Kp, Ki, Kd])
        self.gains.append(self.Gc.components)
        return self.step_env(u)

    def plot_gain_components(self):
        plt.plot(self.tt[: len(self.gains)], np.array(self.gains)[:, 0], label="Proportional")
        plt.show()
        plt.plot(self.tt[: len(self.gains)], np.array(self.gains)[:, 1], label="Integral")
        plt.show()
        plt.plot(self.tt[: len(self.gains)], np.array(self.gains)[:, 2], label="Derivative")
        plt.show()


class GymSystem(gym.Env):
    def __init__(
        self,
        uinit=uinit,
        yinit=yinit,
        num=num,
        dem=dem,
        system=SystemSimplePID,
        disturbance=True,
        deterministic=False,
    ):
        super().__init__()

        self.uinit = uinit
        self.yinit = yinit
        self.num = num
        self.dem = dem
        self.disturbance = disturbance
        self.deterministic = deterministic
        self.system = system(
            uinit=self.uinit,
            yinit=self.yinit,
            num=self.num,
            dem=self.dem,
            disturbance=self.disturbance,
            deterministic=self.deterministic,
        )

        self.n_actions = self.system.n_actions
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_actions,))
        self.n_states = self.system.n_states
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(self.n_states,), dtype=np.float32)

    def convert_state(self):
        obs = self.system.get_state()
        obs = np.array(obs).astype(np.float32)
        return obs

    def convert_action(self, action):
        actions = (action + 1) * (self.system.input_high - self.system.input_low) * 0.5 + self.system.input_low
        return actions

    def unconvert_action(self, action):
        actions = (2.0 * action - (self.system.input_high + self.system.input_low)) / (
            self.system.input_high - self.system.input_low
        )
        return actions

    def reset(self):
        _ = self.system.reset()
        obs = self.convert_state()
        return obs

    def step(self, action, debug=False):
        if debug:
            print("Original: ", action)
        action = self.convert_action(action)
        if debug:
            print("Converted: ", action)
        obs = self.system.step(*action)
        # Calculate error and reward
        e = obs[0] - obs[1]
        scale = 0.01
        e_squared = scale * np.abs(e) ** 2
        e_squared = np.minimum(e_squared, 10.0)
        tol = (0.1 - np.abs(e)) if np.abs(e) <= 0.05 else 0.0
        reward = -e_squared + tol
        done = bool(self.system.k == self.system.kfinal - 1)
        info = {}
        obs = self.convert_state()
        return obs, reward, done, info

    def render(self, mode="plot"):
        if mode == "plot":
            print("ISE: ", self.system.ise())
            self.system.plot()
        elif mode == "rgb_array":
            return self.system.plot(save=True)

    def close(self):
        pass


env_name = "SystemPID-v0"
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]
gym.envs.register(
    id=env_name,
    entry_point="control_env:GymSystem",
)

if __name__ == "__main__":
    env = GymSystem(disturbance=False, deterministic=False)
    obss = []
    obs = env.reset()
    obss.append(obs)
    done = False
    tot_r = 0.0
    while not done:
        # obs, r, done, _ = env.step(env.unconvert_action(np.array([0.0, 0.0, 0.0])), True)
        obs, r, done, _ = env.step(np.array([-1, -1, -1]), True)
        # obs, r, done, _ = env.step(
        #     env.action_space.sample(),
        # )
        tot_r += r
        obss.append(obs)
    print(tot_r)
    env.render()
    # plt.show()
    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(np.array(obss)[:, 0])), np.array(obss))
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("Value")
    plt.show()
