import control
import gym
import matplotlib.pyplot as plt
import numpy as np
from control.matlab import *
from gym import spaces
from scipy.integrate import solve_ivp

uinit = 105
T = 438.7763
yinit = 0.1
delt = 0.083
slew_rate = 20


class CSTRPIDBase:
    def __init__(
        self,
        uinit=uinit,
        T=T,
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
        self.inputs = ["Setpoint", "Output", "Model Region"]
        self.reset_init(uinit, T, yinit, ttfinal, disturbance, deterministic)

    def reset_init(
        self,
        uinit=uinit,
        T=T,
        yinit=yinit,
        ttfinal=None,
        disturbance=True,
        deterministic=False,
    ):
        #  plant initial conditions
        self.uinit = uinit
        self.yinit = yinit
        self.T = T
        self.slew_rate = slew_rate

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
                    np.ones((self.ksp, 1)) * 0.1,
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

    def reset(self):
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
        # Error vector
        self.E = np.zeros((self.kfinal, 1))
        # State Vector
        self.x = [self.yinit, self.T]
        # Model Vector
        self.m = np.zeros((self.kfinal + 1, 1))
        self.m[: self.ksp] = np.ones((self.ksp, 1)) * self.get_model_region()
        # PID Gains Vector
        self.gains = np.zeros((self.kfinal + 1, 3))
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

    def step(self, Kc, tau_i, tau_d):
        if self.k == self.kfinal - 1:
            print("Environment is done. Call `reset()` to start again.")
            return
        # Add to gains vector
        self.gains[self.k] = np.array([Kc, tau_i, tau_d])
        # Disturbance
        d = 0.05 * np.random.randn() + 0.05 if self.disturbance else 0.0

        # Set point error
        self.E[self.k] = self.r[self.k][0] - self.y[self.k][0]

        # Calculate control action using Discrete PID Implementation
        self.du[self.k] = Kc * (
            self.E[self.k]
            - self.E[self.k - 1]
            + (self.delt / tau_i) * self.E[self.k]
            + (tau_d / self.delt) * (self.E[self.k] - 2 * self.E[self.k - 1] + self.E[self.k - 2])
        )
        # Slew rate control to prevent controller output from exploding
        self.du[self.k] = np.clip(
            self.du[self.k],
            self.du[self.k - 1] - self.slew_rate,
            self.du[self.k - 1] + self.slew_rate,
        )
        # Update input vector
        self.u[self.k] = self.u[self.k - 1] + self.du[self.k] + d
        self.u[self.k] = np.maximum(1e-4, self.u[self.k])

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

    def plot(self, use_sample_instant=True):
        if self.k == 0:
            print("Simulation not started.")
        else:
            axis = self.tt[: self.k].copy()
            if use_sample_instant:
                axis = np.arange(self.k)

            plt.figure(figsize=(12, 12))
            plt.subplot(2, 1, 1)
            plt.step(axis, self.r[: self.k], linestyle="dashed", label="Setpoint")
            plt.plot(axis, self.y[: self.k], label="Plant Output")
            plt.ylabel("y")
            plt.xlabel("time")
            plt.title("Plant Output")
            plt.grid()
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.step(axis, self.u[: self.k], label="Control Input")
            plt.ylabel("u")
            plt.xlabel("time")
            plt.title("Control Action")
            plt.grid()
            plt.legend()
            plt.show()

            labels = ["Kp", "Ki", "Kd"]
            plt.figure(figsize=(12, 12))
            plt.suptitle("PID Gains")
            for i in range(3):
                plt.subplot(3, 1, i + 1)
                plt.plot(np.arange(len(self.gains[: self.k, i])), self.gains[: self.k, i], label=labels[i])
                plt.legend()
                plt.grid()
                plt.xlabel("time")
                plt.ylabel("Value")
            plt.show()

    def get_state(self):
        return self.r[self.k][0], self.y[self.k][0], self.m[self.k - 1]

    def get_model_region(self):
        m_values = np.linspace(-1, 1, 6)
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


qcbar = 106.0
tbar = 438.0


class GymCSTRPID(gym.Env):
    def __init__(
        self,
        system=CSTRPIDBase,
        uinit=uinit,
        T=T,
        yinit=yinit,
        disturbance=True,
        deterministic=True,
    ):
        super().__init__()
        self.uinit = uinit
        self.T = T
        self.yinit = yinit
        self.disturbance = disturbance
        self.deterministic = deterministic

        self.system = system(
            uinit=self.uinit,
            T=self.T,
            yinit=self.yinit,
            disturbance=self.disturbance,
            deterministic=self.deterministic,
        )

        n_actions = 3
        obs_dim = len(self.system.get_state())
        self.action_space = spaces.Box(-1.0, 1.0, (n_actions,))
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32)

    def convert_state(self):
        obs = self.system.get_state()
        obs = np.array(obs).astype(np.float32)
        return obs

    def reset(self):
        _ = self.system.reset()
        obs = self.convert_state()
        return obs

    def convert_action(self, action):
        scaling = [[70.0, 70.0], [0.5, 0.50001], [0.5, 0.5]]
        actions = [0, 0, 0]
        for i in range(len(action)):
            actions[i] = action[i] * scaling[i][0] + scaling[i][1]
        return actions

    def step(self, action, debug=False):
        # Rescale to appropriate range
        action = self.convert_action(action)
        if debug:
            print(action)
        # Execute one timestep in sim
        obs = self.system.step(*action)
        # Calculate error and reward
        e = obs[0] - obs[1]
        e_squared = np.abs(e) ** 2
        offset = 0.002 if np.abs(e) > 1e-4 else 0.0
        reward = -(e_squared + offset)
        # Check if sim is over and get RL State Vector
        done = bool(self.system.k == (self.system.kfinal - 1))
        obs = self.convert_state()
        # Ignore
        info = {}
        return obs, reward, done, info

    def render(self, mode="plot"):
        if mode != "plot":
            raise NotImplementedError()
        print("ISE: ", self.system.ise())
        self.system.plot()

    def close(self):
        pass


if __name__ == "__main__":
    sim = CSTRPIDBase()
    sim.reset()
    for i in range(sim.kfinal - sim.ksp):
        _, _ = sim.step(119, 0.3367, 0.1926)
    print("ISE: ", sim.ise())
    sim.plot()
