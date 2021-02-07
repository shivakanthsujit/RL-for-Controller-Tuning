import io

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from control.matlab import *
from gym import spaces
from scipy.integrate import solve_ivp
from simple_pid import PID


def fig2data(fig, dpi=72):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @param dpi DPI of saved image
    @return a numpy 3D array of RGBA values
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def dec_ode(u):
    def ode_eqn(t, z):
        qc = u
        q = 100
        Cao = 1
        To = 350
        Tco = 350
        V = 100
        hA = 7e5
        ko = 7.2e10
        AE = 1e4
        delH = 2e5
        rho = 1e3
        rhoc = 1e3
        Cp = 1
        Cpc = 1
        Ca = z[0]
        T = z[1]
        f1 = (q / V * (Cao - Ca)) - (ko * Ca * np.exp(-AE / T))
        f2 = (
            (q / V * (To - T))
            - ((((-delH) * ko * Ca) / (rho * Cp)) * np.exp(-AE / T))
            + (((rhoc * Cpc) / (rho * Cp * V)) * qc * (1 - np.exp(-hA / (qc * rho * Cp))) * (Tco - T))
        )
        return [f1, f2]

    return ode_eqn


def get_init_states(u):
    Tbar = 438.86881957
    ybar = 0.09849321
    t_span = np.linspace(0, 10, 100)
    sol = solve_ivp(dec_ode(u), t_span=[t_span[0], t_span[-1]], y0=[ybar, Tbar])
    return sol.y[0][-1], sol.y[1][-1]


uinit = 103.0
umin = 95.0
umax = 112.0
Tbar = 438.86881957
ybar = 0.09849321
yinit, Tinit = get_init_states(uinit)
delt = 0.083
slew_rate = 20.0

Kp = 125.0
taui = 0.3367
taud = 0.19
Ki = Kp / taui
Kd = Kp * taud

disturbance = True
deterministic = False


class CSTR:
    def __init__(
        self,
        uinit=uinit,
        Tinit=Tinit,
        yinit=yinit,
        delt=delt,
        ttfinal=None,
        disturbance=disturbance,
        deterministic=deterministic,
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

        #  plant initial conditions
        self.uinit = uinit
        self.Tinit = Tinit
        self.yinit = yinit

        self.disturbance = disturbance
        self.deterministic = deterministic
        self.reset()

    @property
    def state_names(self):
        names = names = ["Setpoint(k)", "Output(k)", "Output(k-1)", "Temp(k)", "Temp(k-1)", "Model Region(k)"]
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
        if self.deterministic:
            self.r = np.concatenate(
                (
                    np.ones((self.ksp, 1)) * self.yinit,
                    np.ones((40, 1)) * 0.11038855,
                    np.ones((30, 1)) * 0.08823159,
                    np.ones((40, 1)) * 0.09849321,
                )
            )
        else:
            self.uinit = np.random.randint(100.0, 106.0)
            self.yinit, self.Tinit = get_init_states(self.uinit)
            self.r = np.concatenate(
                (
                    np.ones((self.ksp, 1)) * self.yinit,
                    np.ones((40, 1)) * np.random.uniform(0.07, 0.13),
                    np.ones((30, 1)) * np.random.uniform(0.07, 0.13),
                    np.ones((40, 1)) * np.random.uniform(0.07, 0.13),
                )
            )
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
        self.x = [self.yinit, self.Tinit]
        # Model Vector
        self.m = np.zeros((self.kfinal + 1, 1))
        self.m[: self.ksp] = np.ones((self.ksp, 1)) * self.get_model_region()
        return self.r[self.k][0], self.y[self.k][0]

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
        self.E[self.k] = self.r[self.k][0] - self.y[self.k][0]
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
        self.T[self.k + 1] = self.x[1]
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
        return (
            self.r[self.k][0] - ybar,
            self.y[self.k][0] - ybar,
            self.y[self.k - 1][0] - ybar,
            self.T[self.k][0] - Tbar,
            self.T[self.k - 1][0] - Tbar,
            self.m[self.k - 1][0],
        )

    def get_axis(self, use_sample_instant=True):
        axis = self.tt[: self.k].copy()
        axis_name = "Time (min)"
        if use_sample_instant:
            axis = np.arange(self.k)
            axis_name = "Sampling Instants (x0.083min)"
        return axis, axis_name

    def plot(self, save=False, use_sample_instant=True):
        axis, axis_name = self.get_axis(use_sample_instant)
        plt.figure(figsize=(16, 20))
        plt.subplot(4, 1, 1)
        plt.step(axis, self.r[: self.k], linestyle="dashed", label="Setpoint", where="post")
        plt.plot(axis, self.y[: self.k], label="Plant Output")
        plt.ylabel("Concentration Ca (mol/l)")
        plt.xlabel(axis_name)
        ise = f"{self.ise():.3e}"
        piecewise_ise = "{:.3e}, {:.3e}, {:.3e}".format(*self.get_piecewise_ise())
        title = f"ISE: {ise}, Piecewise ISE: {piecewise_ise}"
        plt.title(title)
        plt.xlim(axis[0], axis[-1])
        plt.grid()
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.step(axis, self.u[: self.k], label="Control Input", where="post")
        plt.ylabel("Coolant Flowrate qc (l/min)")
        plt.xlabel(axis_name)
        plt.title("Control Action")
        plt.xlim(axis[0], axis[-1])
        plt.grid()
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.step(axis, self.T[: self.k], label="Temperature", where="post")
        plt.ylabel("Temperature T (K)")
        plt.xlabel(axis_name)
        plt.title("Temperature")
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
        axis, axis_name = self.get_axis(use_sample_instant)
        plt.figure(figsize=(12, 6))
        plt.plot(axis, self.m[: self.k])
        plt.title("Model Region")
        plt.xlabel(axis_name)
        plt.ylabel("Linear Region Number")
        plt.xlim(axis[0], axis[-1])
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


lamda = 0.5
min_gains = [0.0, 0.0, 0.0]
max_gains = [140.0 / lamda, 2.0, 2.0]


class CSTRPID(CSTR):
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


class CSTRFuzzyPID(CSTRPID):
    @property
    def state_names(self):
        names = names = ["Very Low(k)", "Low(k)", "Medium(k)", "High(k)", "Very High(k)"]
        assert len(names) == self.n_states
        return names

    @property
    def n_states(self):
        return len(self.get_state())

    def reset_input(self):
        self.q_range = np.arange(97, 109, 0.001)
        # Generate fuzzy membership functions
        very_low = fuzz.trimf(self.q_range, [97, 97, 100])
        low = fuzz.trimf(self.q_range, [97, 100, 103])
        medium = fuzz.trimf(self.q_range, [100, 103, 106])
        high = fuzz.trimf(self.q_range, [103, 106, 109])
        very_high = fuzz.trimf(self.q_range, [106, 109, 109])
        self.membership_fns = [very_low, low, medium, high, very_high]
        return super().reset_input()

    def fuzzify(self, u):
        u = np.clip(u, self.q_range[0], self.q_range[-1])
        fuzzy_values = np.array([fuzz.interp_membership(self.q_range, self.membership_fns[i], u) for i in range(5)])
        return fuzzy_values

    def get_state(self):
        u = self.u[self.k - 1][0]
        state = self.fuzzify(u)
        return state


class CSTRFuzzyPID2(CSTRFuzzyPID):
    def get_state(self):
        state = super().get_state()
        state -= 0.5
        return state


class CSTRFuzzyPID3(CSTRPID):
    @property
    def state_names(self):
        names = names = ["M1(k)", "M2(k)", "M3(k)", "M4(k)"]
        assert len(names) == self.n_states
        return names

    @property
    def n_states(self):
        return len(self.get_state())

    def reset_input(self):
        self.q_range = np.arange(95, 113, 0.001)
        # Generate fuzzy membership functions
        m1 = fuzz.trapmf(self.q_range, [95, 95, 106.2, 107.4])
        m2 = fuzz.trimf(self.q_range, [106.2, 107.4, 108])
        m3 = fuzz.trimf(self.q_range, [107.4, 108, 109.2])
        m4 = fuzz.trapmf(self.q_range, [108, 109.2, 113, 113])
        self.membership_fns = [m1, m2, m3, m4]
        return super().reset_input()

    def fuzzify(self, u):
        u = np.clip(u, self.q_range[0], self.q_range[-1])
        fuzzy_values = np.array(
            [fuzz.interp_membership(self.q_range, self.membership_fns[i], u) for i in range(len(self.membership_fns))]
        )
        return fuzzy_values

    def get_state(self):
        u = self.u[self.k - 1][0]
        state = self.fuzzify(u)
        state -= 0.5
        return state


class GymCSTR(gym.Env):
    def __init__(
        self,
        uinit=uinit,
        Tinit=Tinit,
        yinit=yinit,
        system=CSTRPID,
        disturbance=disturbance,
        deterministic=deterministic,
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
        scale = 15.0
        e_squared = scale * np.abs(e) ** 2
        e_squared = np.minimum(e_squared, 10.0)
        tol = (0.1 - np.abs(e)) if np.abs(e) <= 5e-4 else 0.0
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


# env_name = "CSTRPID-v0"
# if env_name in gym.envs.registry.env_specs:
#     del gym.envs.registry.env_specs[env_name]
# gym.envs.register(
#     id=env_name,
#     entry_point="cstr_control_env:GymCSTR",
# )
