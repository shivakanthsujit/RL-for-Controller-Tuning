import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import piecewise
from scipy.integrate import solve_ivp

from utils import fig2data
import gym
import matplotlib.pyplot as plt
import numpy as np
from control.matlab import *
from gym import spaces
from scipy.integrate import solve_ivp
from simple_pid import PID


disturbance = True
deterministic = False
regulatory = False
regulatory_points = [300, 350]
regulatory_disturbance = 150
disturbance_value = 20.0
stable_region=False
parameter_uncertainity=False

def dec_ode(Tj):
    def ode_eqn(t, z):
        Ca0 = 10
        T0 = 298
        f = 1
        E = (11843 * 1.0) if not parameter_uncertainity else (11843 * 1.05)
        ko = 9703 * 3600
        H = 5960

        UA = (150 * 1.0) if not parameter_uncertainity else (150 * 0.8)
        rowCp = 500
        R = 1.987
        V = 1

        dz1 = (f / V) * (Ca0 - z[0]) - ko * z[0] * np.exp(-E / (R * z[1]))
        dz2 = (
            (f / V) * (T0 - z[1]) + (H / (rowCp)) * ko * z[0] * np.exp(-E / (R * z[1])) - (UA * (1 / rowCp)) * (z[1] - Tj)
        )
        return [dz1, dz2]

    return ode_eqn

def get_init_states(u):
        Ca = 5.518
        ybar = 339.1
        t_span = np.linspace(0, 10, 100)
        sol = solve_ivp(dec_ode(u), t_span=[t_span[0], t_span[-1]], y0=[Ca, ybar])
        return sol.y[0][-1], sol.y[1][-1]

uinit = 298.0
umin = -1000.0
umax = 1200.0
Tbar = 5.443
ybar = 339.849
Tinit, yinit = get_init_states(uinit)
delt = 0.01
slew_rate = None
ksp = 100

Kp = 125.0
taui = 0.3367
taud = 0.19
Ki = Kp / taui
Kd = Kp * taud

class CSTR2:
    def __init__(
        self,
        uinit=uinit,
        Tinit=Tinit,
        yinit=yinit,
        delt=delt,
        ttfinal=None,
        disturbance=disturbance,
        deterministic=deterministic,
        regulatory=regulatory,
        regulatory_points = regulatory_points,
        regulatory_disturbance = regulatory_disturbance,
        disturbance_value=disturbance_value,
        stable_region=stable_region,
        parameter_uncertainity=parameter_uncertainity,
    ):
        # Simulation settings
        self.delt = delt  # sample time
        self.ttfinal = ttfinal  # final simulation time
        self.ksp = ksp
        self.slew_rate = slew_rate
        self.umin = umin
        self.umax = umax
        self.input_low = np.array([self.umin])
        self.input_high = np.array([self.umax])

        #  plant initial conditions
        self.uinit = uinit
        self.Tinit = Tinit
        self.yinit = yinit

        self.disturbance = disturbance
        self.deterministic = deterministic
        self.regulatory = regulatory
        self.regulatory_points = regulatory_points
        self.regulatory_disturbance = regulatory_disturbance
        self.disturbance_value = disturbance_value
        self.stable_region=stable_region
        self.parameter_uncertainity = parameter_uncertainity
        self.reset()

    @property
    def state_names(self):
        names = names = ["Setpoint(k)", "Output(k)", "Output(k-1)", "Temp(k)", "Temp(k-1)"]
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

    def reset(self):
        # self.del_indices = [500] # For random figure
        self.del_indices = [200, 200, 200, 200] # For random figure 2
        # self.del_indices = [100, 300, 300, 400] # Normal setpoint change
        self.indices = [self.ksp]
        for del_index in self.del_indices:
            self.indices.append(self.indices[-1] + del_index)

        if self.deterministic:
            if self.stable_region:
                self.uinit = 295.0
                self.Tinit, self.yinit = self.get_init_states(self.uinit)
                # self.setpoints = [self.yinit, 340] # For random figure
                self.setpoints = [self.yinit, 330.0, 350.0, 340.0, 310.0] # For random figure 2
                # self.setpoints = [self.yinit, 320.0, 370.0, 350.0, 310.0] # Normal setpoint change
                # self.setpoints = [self.yinit, 320.0, 385.0, 400.0, 390.0] # For generalization
            else:
                # self.setpoints = [self.yinit, 350.0, 370.0, 335.0, 310.0]
                # self.setpoints = [self.yinit, 340.0] # For random figure
                self.setpoints = [self.yinit, 345.0, 370.0, 350.0, 310.0] # For random figure 2
                # self.setpoints = [self.yinit, 345.0, 370.0, 350.0, 310.0] # Normal setpoint change
                # self.setpoints = [self.yinit, 355.0, 385.0, 400.0, 390.0] # For generalization
        else:
            self.uinit = np.random.randint(290.0, 301.0)
            self.Tinit, self.yinit = self.get_init_states(self.uinit)
            # self.setpoints = [self.yinit, 340.0] # For random figure
            self.setpoints = [self.yinit, 335.0, 365.0, 350.0, 315.0]
            self.setpoints += np.random.uniform(-5.0, 5.0, len(self.setpoints))

        self.setpoint_indices = [self.ksp] + self.del_indices

        assert len(self.setpoints) == len(self.setpoint_indices)

        self.r = np.concatenate([np.ones((self.setpoint_indices[i], 1)) * self.setpoints[i] for i in range(len(self.setpoint_indices))])

        # if self.deterministic:
        #     self.r = np.concatenate(
        #         (
        #             np.ones((self.ksp, 1)) * self.yinit,
        #             np.ones((100, 1)) * 345.0,
        #             np.ones((300, 1)) * 370.0,
        #             np.ones((300, 1)) * 350.0,
        #             np.ones((400, 1)) * 310.0,
        #         )
        #     )
        #     if self.regulatory:
        #         self.r = np.concatenate(
        #             (
        #                 np.ones((self.ksp, 1)) * self.yinit,
        #                 np.ones((400, 1)) * 345.0,
        #             )
        #         )
        # else:
        #     self.uinit = np.random.randint(290.0, 301.0)
        #     self.Tinit, self.yinit = CSTR2.get_init_states(self.uinit)
        #     self.r = np.concatenate(
        #         (
        #             np.ones((self.ksp, 1)) * self.yinit,
        #             np.ones((100, 1)) * (335.0 + np.random.uniform(-5.0, 5.0)),
        #             np.ones((300, 1)) * (365.0 + np.random.uniform(-5.0, 5.0)),
        #             np.ones((300, 1)) * (350.0 + np.random.uniform(-5.0, 5.0)),
        #             np.ones((400, 1)) * (315.0 + np.random.uniform(-5.0, 5.0)),
        #         )
        #     )
        sim_time = len(self.r) * self.delt
        self.ttfinal = self.ttfinal if self.ttfinal is not None and self.ttfinal < sim_time else sim_time
        self.tt = np.arange(0, self.ttfinal, self.delt)  # time vector
        self.kfinal = len(self.tt)  # number of time intervals
        return self.reset_input()

    def reset_input(self):
        # Input vector
        self.input = np.zeros((self.kfinal + 1, self.n_actions))
        self.input[: self.ksp] = np.ones((self.ksp, 1)) * self.uinit
        return self.reset_env()

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
        self.x = [self.Tinit, self.yinit]
        return self.r[self.k][0], self.y[self.k][0]

    def dec_ode(self, Tj, T0=298):
        def ode_eqn(t, z):
            Ca0 = 10

            f = 1
            E = (11843 * 1.0) if not self.parameter_uncertainity else (11843 * 1.05)
            ko = 9703 * 3600
            H = 5960

            UA = (150 * 1.0) if not self.parameter_uncertainity else (150 * 0.8)
            rowCp = 500
            R = 1.987
            V = 1

            dz1 = (f / V) * (Ca0 - z[0]) - ko * z[0] * np.exp(-E / (R * z[1]))
            dz2 = (
                (f / V) * (T0 - z[1]) + (H / (rowCp)) * ko * z[0] * np.exp(-E / (R * z[1])) - (UA * (1 / rowCp)) * (z[1] - Tj)
            )
            return [dz1, dz2]

        return ode_eqn

    def get_init_states(self, u):
        Ca = 5.518
        ybar = 339.1
        t_span = np.linspace(0, 10, 100)
        sol = solve_ivp(self.dec_ode(u), t_span=[t_span[0], t_span[-1]], y0=[Ca, ybar])
        return sol.y[0][-1], sol.y[1][-1]

    def step_env(self, u):
        self.E[self.k] = self.r[self.k][0] - self.y[self.k][0]
        # Slew rate
        if self.slew_rate:
            u = np.clip(u, self.u[self.k - 1] - self.slew_rate, self.u[self.k - 1] + self.slew_rate)
        # Disturbance
        d = self.disturbance_value * np.random.randn() if self.disturbance else 0.0
        self.u[self.k] = u + d
        # Control Constraints
        self.u[self.k] = np.clip(self.u[self.k], self.umin, self.umax)
        # Solve ODE and update state vector and output vector
        if self.regulatory and self.regulatory_points[0] < self.k < self.regulatory_points[1]:
            sol = solve_ivp(self.dec_ode(self.u[self.k], T0=298+self.regulatory_disturbance), [self.tinitial, self.tfinal], self.x)
        else:
            sol = solve_ivp(self.dec_ode(self.u[self.k]), [self.tinitial, self.tfinal], self.x)
        self.x[0] = sol.y[0][-1]
        self.x[1] = sol.y[1][-1]
        self.T[self.k + 1] = self.x[0]
        self.y[self.k + 1] = self.x[1]
        # Update time and sampling instant
        self.tinitial = self.tfinal
        self.tfinal += self.delt
        self.k = self.k + 1
        return self.r[self.k][0], self.y[self.k][0]

    def step(self, u):
        self.input[self.k] = np.array(u)
        return self.step_env(u)

    def get_state(self):
        return (
            self.r[self.k][0] - ybar,
            self.y[self.k][0] - ybar,
            self.y[self.k - 1][0] - ybar,
            self.T[self.k][0] - Tbar,
            self.T[self.k - 1][0] - Tbar,
        )

    def get_axis(self, use_sample_instant=True):
        axis = self.tt[: self.k].copy()
        axis_name = "Time (min)"
        if use_sample_instant:
            axis = np.arange(self.k)
            axis_name = "Sampling Instants"
        return axis, axis_name

    def plot(self, save=False, use_sample_instant=True):
        axis, axis_name = self.get_axis(use_sample_instant)
        plt.figure(figsize=(16, 20))
        plt.subplot(4, 1, 1)
        plt.step(axis, self.r[: self.k], linestyle="dashed", label="Setpoint", where="post")
        plt.plot(axis, self.y[: self.k], label="Plant Output")
        plt.ylabel("Reactor Temperature Tr (K)")
        plt.xlabel(axis_name)
        ise = f"{self.ise():.3e}"
        title = f"ISE: {ise}"
        plt.title(title)
        plt.xlim(axis[0], axis[-1])
        plt.grid()
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.step(axis, self.u[: self.k], label="Control Input", where="post")
        plt.ylabel("Jacket Temperature Tj (K)")
        plt.xlabel(axis_name)
        plt.title("Control Action")
        plt.xlim(axis[0], axis[-1])
        plt.grid()
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.step(axis, self.T[: self.k], label="Concentration", where="post")
        plt.ylabel("Concentration Ca (mol/l)")
        plt.xlabel(axis_name)
        plt.title("Concentration")
        plt.xlim(axis[0], axis[-1])
        plt.grid()
        plt.legend()

        plt.subplot(4, 1, 4)
        for i in range(self.n_actions):
            plt.plot(axis[self.ksp :], self.input[self.ksp : self.k, i], label=self.input_names[i])
        plt.ylabel("Value")
        plt.xlabel(axis_name)
        plt.title("Inputs")
        plt.xlim(axis[0], axis[-1])
        plt.grid()
        plt.legend()
        if save:
            plt.tight_layout()
            img = fig2data(plt.gcf())
            plt.close()
            return img

    def ise(self):
        return ((self.r[: self.k] - self.y[: self.k]) ** 2).sum()

    def get_piecewise_ise(self):
        indices = [x_ for x_ in self.indices if x_ <= self.k] + [self.k]
        piecewise = [((self.r[indices[i]:indices[i+1]] - self.y[indices[i]:indices[i+1]]) ** 2).sum() for i in range(len(indices)-1)]
        return piecewise

    def iae(self):
        return abs(self.r[: self.k] - self.y[: self.k]).sum()

    def get_piecewise_iae(self):
        indices = [x_ for x_ in self.indices if x_ <= self.k] + [self.k]
        piecewise = [abs(self.r[indices[i]:indices[i+1]] - self.y[indices[i]:indices[i+1]]).sum() for i in range(len(indices)-1)]
        return piecewise


lamda = 1.0
min_gains = [0.0, 0.0, 0.0]
max_gains = [100.0 / lamda, 2.0, 2.0]


class CSTR2PID(CSTR2):
    @property
    def input_names(self):
        return ["Kp(k)", "Ki(k)", "Kd(k)"]

    def reset_input(self, *args, **kwargs):
        auto = False
        self.Gc = PID(
            Kp, Ki, Kd, setpoint=self.yinit, sample_time=self.delt, output_limits=(self.umin, self.umax), auto_mode=auto
        )
        self.Gc.set_auto_mode(not auto, last_output=self.uinit)
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
        Ki = Kp / (taui + 1e-1)
        Kd = Kp * taud
        self.Gc.setpoint = self.r[self.k][0]
        self.Gc.tunings = (Kp, Ki, Kd)
        u = self.Gc(self.y[self.k][0], self.delt)

        self.input[self.k] = np.array([Kp, Ki, Kd])
        self.gains.append([Kp, taui, taud])
        self.gain_components.append(self.Gc.components)
        return self.step_env(u)

    def plot_gains(self, save=False, use_sample_instant=True):
        axis, axis_name = self.get_axis(use_sample_instant)
        plt.figure(figsize=(16, 9))
        labels = ["Proportional", "Integral", "Derivative"]
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(axis[self.ksp : self.ksp + len(self.gains)], np.array(self.gains)[:-1, i], label=labels[i])
            plt.ylabel("Value")
            plt.xlabel(axis_name)
            plt.xlim(axis[0], axis[-1])
            plt.grid()
            plt.legend()
        if save:
            plt.tight_layout()
            img = fig2data(plt.gcf())
            plt.close()
            return img

    def plot_gain_components(self, use_sample_instant=True):
        axis, axis_name = self.get_axis(use_sample_instant)
        plt.figure(figsize=(16, 9))
        labels = ["Proportional", "Integral", "Derivative"]
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(
                axis[self.ksp : self.ksp + len(self.gain_components)], np.array(self.gain_components)[:-1, i], label=labels[i]
            )
            plt.ylabel("Value")
            plt.xlabel(axis_name)
            plt.xlim(axis[0], axis[-1])
            plt.grid()
            plt.legend()


class GymCSTR2(gym.Env):
    def __init__(
        self,
        uinit=uinit,
        Tinit=Tinit,
        yinit=yinit,
        system=CSTR2PID,
        disturbance=disturbance,
        deterministic=deterministic,
        regulatory=regulatory,
        disturbance_value=disturbance_value,
        stable_region=stable_region,
        parameter_uncertainity=parameter_uncertainity
    ):
        super().__init__()
        self.uinit = uinit
        self.Tinit = Tinit
        self.yinit = yinit
        self.disturbance = disturbance
        self.deterministic = deterministic
        self.regulatory=regulatory
        self.disturbance_value=disturbance_value
        self.stable_region=stable_region
        self.parameter_uncertainity = parameter_uncertainity

        self.system = system(
            uinit=self.uinit,
            Tinit=self.Tinit,
            yinit=self.yinit,
            disturbance=self.disturbance,
            deterministic=self.deterministic,
            regulatory=self.regulatory,
            disturbance_value=self.disturbance_value,
            stable_region=self.stable_region,
            parameter_uncertainity=self.parameter_uncertainity,
        )

        self.n_actions = self.system.n_actions
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_actions,))
        self.n_states = self.system.n_states
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(self.n_states,), dtype=np.float32)

    def convert_state(self):
        obs = self.system.get_state()
        obs = np.array(obs).astype(np.float32)
        return obs

    def convert_action(self, action):
        actions = (action + 1) * (self.system.input_high - self.system.input_low) * 0.5 + self.system.input_low
        actions = np.clip(actions, self.system.input_low, self.system.input_high)
        return actions

    def unconvert_action(self, action):
        actions = (2.0 * action - (self.system.input_high + self.system.input_low)) / (
            self.system.input_high - self.system.input_low
        )
        actions = np.clip(actions, -1.0, 1.0)
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
        scale = 0.001
        e_squared = scale * np.abs(e) ** 2
        e_squared = np.minimum(e_squared, 5.0)
        tol = (2.0 - np.abs(e)) if np.abs(e) <= 1e-2 else 0.0
        reward = -e_squared + tol
        done = bool(self.system.k == self.system.kfinal - 1)
        info = {}
        obs = self.convert_state()
        return obs, reward, done, info

    def render(self, mode="human"):
        if mode == "human":
            print("ISE: ", self.system.ise())
            self.system.plot()
        elif mode == "rgb_array":
            return self.system.plot(save=True)

    def close(self):
        pass
