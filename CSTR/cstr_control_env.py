import gym
import matplotlib.pyplot as plt
import numpy as np
from control.matlab import *
from gym import spaces
from scipy.integrate import solve_ivp
from simple_pid import PID

from utils import fig2data

uinit = 103.0
umin = 95.0
umax = 112.0
Tinit = 438.7763
yinit = 0.1
delt = 0.083
slew_rate = 20.0


class CSTR:
    def __init__(
        self,
        uinit=uinit,
        Tinit=Tinit,
        yinit=yinit,
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
        self.reset_init(uinit, Tinit, yinit, ttfinal, disturbance, deterministic)

    @property
    def state_names(self):
        names = ["Setpoint(k)", "Output(k)", "Output(k-1)", "Model Region(k)"]
        assert len(names) == self.n_states
        return names

    @property
    def input_names(self):
        names = ["Input(k)"]
        assert len(names) == self.n_actions
        return names

    @property
    def n_states(self):
        return len(self.get_state())

    @property
    def n_actions(self):
        return self.input_low.shape[0]

    def reset_init(
        self,
        uinit=uinit,
        Tinit=Tinit,
        yinit=yinit,
        ttfinal=None,
        disturbance=True,
        deterministic=False,
    ):
        #  plant initial conditions
        self.uinit = uinit
        self.Tinit = Tinit
        self.yinit = yinit

        self.disturbance = disturbance
        if deterministic:
            self.r = np.concatenate(
                (
                    np.ones((self.ksp, 1)) * self.yinit,
                    np.ones((40, 1)) * 0.11,
                    np.ones((30, 1)) * 0.09,
                    np.ones((40, 1)) * 0.1,
                )
            )
        else:
            self.r = np.concatenate(
                (
                    np.ones((self.ksp, 1)) * self.yinit,
                    np.ones((40, 1)) * np.random.uniform(0.11, 0.14),
                    np.ones((30, 1)) * np.random.uniform(0.07, 0.10),
                    np.ones((40, 1)) * np.random.uniform(0.08, 0.11),
                )
            )
        sim_time = len(self.r) * self.delt
        self.ttfinal = ttfinal if ttfinal is not None and ttfinal < sim_time else sim_time
        self.tt = np.arange(0, self.ttfinal, self.delt)  # time vector
        self.kfinal = len(self.tt)  # number of time intervals

        return self.reset()

    def reset_env(self):
        # Reset sim time
        self.tinitial = 0
        self.tfinal = self.tinitial + self.delt
        self.k = self.ksp - 1
        # Input vector
        self.u = np.zeros((self.kfinal + 1, 1))
        self.u[: self.ksp] = np.ones((self.ksp, 1)) * self.uinit
        self.du = np.zeros((self.kfinal, 1))
        # Output vector
        self.y = np.zeros((self.kfinal + 1, 1))
        self.y[: self.ksp] = np.ones((self.ksp, 1)) * self.yinit
        # Temperature Vector
        self.T = np.zeros((self.kfinal + 1, 1))
        self.T[: self.ksp] = np.ones((self.ksp, 1)) * self.Tinit
        # Error vector
        self.E = np.zeros((self.kfinal, 1))
        # State Vector
        self.x = [self.yinit, self.Tinit]
        # Model Vector
        self.m = np.zeros((self.kfinal + 1, 1))
        self.m[: self.ksp] = np.ones((self.ksp, 1)) * self.get_model_region()
        return self.r[self.k][0], self.y[self.k][0]

    def reset(self):
        # Input vector
        self.input = np.zeros((self.kfinal + 1, self.n_actions))
        self.input[: self.ksp] = np.ones((self.ksp, 1)) * self.uinit
        return self.reset_env()

    def dec_ode(self, u):
        def ode_eqn(t, z):
            qc = u
            q = 100.0
            Cao = 1.0
            To = 350.0
            Tco = 350.0
            V = 100.0
            hA = 7e5
            ko = 7.2e10
            AE = 1e4
            delH = 2e5
            rho = 1e3
            rhoc = 1e3
            Cp = 1.0
            Cpc = 1.0
            Ca = z[0]
            T = z[1]
            Ca = np.maximum(1e-6, Ca)
            T = np.maximum(10, T)
            qc = np.maximum(1e-4, qc)
            try:
                f1 = (q / V * (Cao - Ca)) - (ko * Ca * np.exp(-AE / T))
                f2 = (
                    (q / V * (To - T))
                    - ((((-delH) * ko * Ca) / (rho * Cp)) * np.exp(-AE / T))
                    + (((rhoc * Cpc) / (rho * Cp * V)) * qc * (1 - np.exp(-hA / (qc * rho * Cp))) * (Tco - T))
                )
            except RuntimeWarning:
                print("qc: ", qc)
                print("Ca: ", Ca)
                print("T: ", T)
                raise RuntimeError
            return [f1, f2]

        return ode_eqn

    def step_env(self, u):
        # Slew rate
        if self.slew_rate:
            u = np.clip(u, self.u[self.k - 1] - self.slew_rate, self.u[self.k - 1] + self.slew_rate)
        # Disturbance
        d = 0.05 * np.random.randn() if self.disturbance else 0.0
        self.u[self.k] = u + d
        # Control Constraints
        self.u[self.k] = np.clip(self.u[self.k], self.umin, self.umax)
        # Solve ODE and update state vector and output vector
        sol = solve_ivp(self.dec_ode(self.u[self.k]), [self.tinitial, self.tfinal], self.x)
        self.x[0] = sol.y[0][-1]
        self.x[1] = sol.y[1][-1]
        self.y[self.k + 1] = self.x[0]
        # Current model region
        self.m[self.k] = self.get_model_region()
        # Update time and sampling instant
        self.tinitial = self.tfinal
        self.tfinal += self.delt
        self.k = self.k + 1
        return self.r[self.k][0], self.y[self.k][0]

    def step(self, u):
        self.input[self.k] = np.array(u)
        return self.step_env(u)

    def get_state(self):
        return self.r[self.k][0], self.y[self.k][0], self.y[self.k - 1][0], self.m[self.k - 1][0]

    def plot(self, save=False):
        plt.figure(figsize=(16, 16))
        plt.subplot(3, 1, 1)
        plt.step(
            self.tt[: self.k],
            self.r[: self.k],
            linestyle="dashed",
            label="Setpoint",
        )
        ise = f"{self.ise():.3e}"
        piecewise_ise = "{:.3e}, {:.3e}, {:.3e}".format(*self.get_piecewise_ise())
        title = f"ISE: {ise}, Piecewise ISE: {piecewise_ise}"
        plt.plot(self.tt[: self.k], self.y[: self.k], label="Plant Output")
        plt.ylabel("y")
        plt.xlabel("time")
        plt.title(title)
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
            img = fig2data(plt.gcf())
            plt.close()
            return img

    def get_model_region(self):
        m_values = np.linspace(-0.2, 0.2, 6)
        m = m_values[0]
        if self.u[self.k] >= 97.0 and self.u[self.k] <= 100.0:
            m = m_values[1]
        elif self.u[self.k] > 100.0 and self.u[self.k] <= 103.0:
            m = m_values[2]
        elif self.u[self.k] > 103.0 and self.u[self.k] <= 106.0:
            m = m_values[3]
        elif self.u[self.k] > 106.0 and self.u[self.k] <= 109.0:
            m = m_values[4]
        elif self.u[self.k] > 109.0:
            m = m_values[5]
        return m

    def plot_model_regions(self, use_sample_instant=True):
        axis = self.tt[: self.k].copy()
        if use_sample_instant:
            axis = np.arange(self.k)
        plt.figure(figsize=(12, 6))
        plt.plot(axis, self.m[: self.k])
        plt.title("Model Region")
        plt.xlabel("Time")
        plt.ylabel("Linear Region Number")
        plt.grid()
        plt.show()

    def ise(self):
        return ((self.r[: self.k] - self.y[: self.k]) ** 2).sum()

    def get_piecewise_ise(self):
        return (
            ((self.r[10:50] - self.y[10:50]) ** 2).sum(),
            ((self.r[50:80] - self.y[50:80]) ** 2).sum(),
            ((self.r[80:120] - self.y[80:120]) ** 2).sum(),
        )


min_gains = [0.0, 0.0, 0.0]
max_gains = [140.0, 140.0, 140.0]


class CSTRManualPID(CSTR):
    @property
    def input_names(self):
        return ["Kp(k)", "Ki(k)", "Kd(k)"]

    def reset(self):
        self.input_low = np.array(min_gains)
        self.input_high = np.array(max_gains)
        # Input vector
        self.input = np.zeros((self.kfinal + 1, self.n_actions))
        self.input[: self.ksp] = np.ones((self.ksp, self.n_actions)) * np.array([5, 1, 0])
        self.gains = []
        return self.reset_env()

    def step(self, Kp, Ki, Kd):
        self.E[self.k] = self.r[self.k][0] - self.y[self.k][0]
        proportional = Kp * (self.E[self.k] - self.E[self.k - 1])
        integral = 0.5 * Ki * self.delt * (self.E[self.k] + self.E[self.k - 1])
        derivative = (Kd / self.delt) * (self.E[self.k] - 2 * self.E[self.k - 1] + self.E[self.k - 2])
        du = proportional + integral + derivative
        u = self.u[self.k - 1] + du

        self.input[self.k] = np.array([Kp, Ki, Kd])
        self.gains.append([proportional[0], integral[0], derivative[0]])
        return self.step_env(u)

    def plot_gain_components(self):
        plt.plot(self.tt[: len(self.gains)], np.array(self.gains)[:, 0], label="Proportional")
        plt.show()
        plt.plot(self.tt[: len(self.gains)], np.array(self.gains)[:, 1], label="Integral")
        plt.show()
        plt.plot(self.tt[: len(self.gains)], np.array(self.gains)[:, 2], label="Derivative")
        plt.show()


min_gains = [0.0, 0.0, 0.0]
max_gains = [140.0, 2.0, 2.0]


class CSTRPID(CSTR):
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
        self.gain_components = []
        return self.reset_env(*args, **kwargs)

    def step(self, Kp, taui, taud):
        Ki = Kp / (taui + 1e-2)
        Kd = Kp * taud
        self.E[self.k] = self.r[self.k][0] - self.y[self.k][0]
        self.Gc.setpoint = self.r[self.k][0]
        self.Gc.tunings = (Kp, Ki, Kd)
        u = self.Gc(self.y[self.k][0], self.delt)

        self.input[self.k] = np.array([Kp, Ki, Kd])
        self.gains.append([Kp, taui, taud])
        self.gain_components.append(self.Gc.components)
        return self.step_env(u)

    def plot_gains(self):
        plt.figure(figsize=(16, 9))
        plt.subplot(3, 1, 1)
        plt.plot(self.tt[: len(self.gains)], np.array(self.gains)[:, 0], label="Kp")
        plt.ylabel("Value")
        plt.xlabel("time")
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.tt[: len(self.gains)], np.array(self.gains)[:, 1], label="taui")
        plt.ylabel("Value")
        plt.xlabel("time")
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.tt[: len(self.gains)], np.array(self.gains)[:, 2], label="taud")
        plt.ylabel("Value")
        plt.xlabel("time")
        plt.grid()
        plt.legend()

    def plot_gain_components(self):
        plt.figure(figsize=(16, 9))
        plt.subplot(3, 1, 1)
        plt.plot(self.tt[: len(self.gain_components)], np.array(self.gain_components)[:, 0], label="Proportional")
        plt.ylabel("Value")
        plt.xlabel("time")
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.tt[: len(self.gain_components)], np.array(self.gain_components)[:, 1], label="Integral")
        plt.ylabel("Value")
        plt.xlabel("time")
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.tt[: len(self.gain_components)], np.array(self.gain_components)[:, 2], label="Derivative")
        plt.ylabel("Value")
        plt.xlabel("time")
        plt.grid()
        plt.legend()


class GymCSTR(gym.Env):
    def __init__(
        self,
        uinit=uinit,
        Tinit=Tinit,
        yinit=yinit,
        system=CSTRPID,
        disturbance=True,
        deterministic=True,
    ):
        super().__init__()
        self.uinit = uinit
        self.Tinit = Tinit
        self.yinit = yinit
        self.disturbance = disturbance
        self.deterministic = deterministic

        self.system = system(
            uinit=self.uinit,
            Tinit=self.Tinit,
            yinit=self.yinit,
            disturbance=self.disturbance,
            deterministic=self.deterministic,
        )

        n_actions = self.system.n_actions
        self.action_space = spaces.Box(-1.0, 1.0, (n_actions,))
        self.n_states = self.system.n_states
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(self.n_states,), dtype=np.float32)

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
        scale = 15.0
        e_squared = scale * np.abs(e) ** 2
        e_squared = np.minimum(e_squared, 10.0)
        tol = (0.1 - np.abs(e)) if np.abs(e) <= 1e-4 else 0.0
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


env_name = "CSTRPID-v0"
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]
gym.envs.register(
    id=env_name,
    entry_point="cstr_control_env:GymCSTR",
)

if __name__ == "__main__":
    Kp = 120.0
    taui = 0.3367
    taud = 0.19
    Ki = Kp / taui
    Kd = Kp * taud

    args = {"disturbance": False, "deterministic": True}
    env = GymCSTR(**args)
    obss = []
    obs = env.reset()
    obss.append(obs)
    done = False
    tot_r = 0.0
    while not done:
        obs, r, done, _ = env.step(env.unconvert_action(np.array([Kp, taui, taud])))
        tot_r += r
        obss.append(obs)
    print(tot_r)
    env.render()
    plt.show()
    # env.system.plot_gains()
    # plt.show()
    # env.system.plot_gain_components()
    # plt.show()
