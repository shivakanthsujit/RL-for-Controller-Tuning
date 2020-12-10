import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import spaces

import control
from control.matlab import *

from scipy.integrate import solve_ivp

ubar = 105
T = 438.7763
ybar = 0.1
ysp = 0.11
delt = 0.083


class CSTRPID:
    def __init__(
        self,
        ubar=ubar,
        T=T,
        ybar=ybar,
        ysp=ysp,
        timesp=1,
        delt=delt,
        ttfinal=20,
        disturbance=True,
    ):
        self.reset_init(ubar, T, ybar, ysp, timesp, delt, ttfinal, disturbance)

    def reset_init(
        self,
        ubar=ubar,
        T=T,
        ybar=ybar,
        ysp=ysp,
        timesp=1,
        delt=delt,
        ttfinal=20,
        disturbance=True,
    ):
        self.ubar = ubar
        self.ybar = 0.1
        self.T = T
        self.ysp = ysp  # setpoint change (from 0)
        self.timesp = timesp  # time of setpoint change
        self.delt = delt  # sample time
        self.ttfinal = ttfinal  # final simulation time

        self.ksp = 10
        #       self.r = np.concatenate((np.ones((self.ksp, 1))*0.1, np.ones((40, 1))*0.11, np.ones((30, 1))*0.09, np.ones((40, 1))*0.1))
        self.r = np.concatenate(
            (
                np.ones((self.ksp, 1)) * 0.1,
                np.ones((40, 1)) * np.random.uniform(0.11, 0.14),
                np.ones((30, 1)) * np.random.uniform(0.07, 0.10),
                np.ones((40, 1)) * np.random.uniform(0.08, 0.11),
            )
        )
        self.ttfinal = len(self.r) * self.delt
        self.tt = np.arange(0, self.ttfinal, self.delt)  # time vector
        self.kfinal = len(self.tt)  # number of time intervals

        #  plant initial conditions
        self.uinit = self.ubar
        self.yinit = self.ybar
        self.disturbance = disturbance

        return self.reset()

    def reset(self):
        #  initialize input vector
        self.u = np.zeros((self.kfinal + 1, 1))
        self.u[: self.ksp] = np.ones((self.ksp, 1)) * self.uinit
        self.U = np.zeros((self.kfinal + 1, 1))
        self.dU = np.zeros((self.kfinal, 1))

        self.y = np.zeros((self.kfinal + 1, 1))
        self.y[: self.ksp] = np.ones((self.ksp, 1)) * self.yinit
        self.Y = np.zeros((self.kfinal + 1, 1))

        self.E = np.zeros((self.kfinal, 1))
        self.ss_s1 = [self.ybar, self.T]

        self.m = np.zeros((self.kfinal + 1, 1))
        self.tinitial = 0
        self.tfinal = self.tinitial + self.delt
        self.k = self.ksp - 1
        self.m[: self.ksp] = np.ones((self.ksp, 1)) * self.get_model_region()
        return self.r[self.k][0], self.y[self.k][0]

    def dec_ode(self, u):
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
            Ca = np.maximum(1e-6, Ca)
            T = np.maximum(10, T)
            qc = np.maximum(1e-4, qc)
            try:
                f1 = (q / V * (Cao - Ca)) - (ko * Ca * np.exp(-AE / T))
                f2 = (
                    (q / V * (To - T))
                    - ((((-delH) * ko * Ca) / (rho * Cp)) * np.exp(-AE / T))
                    + (
                        ((rhoc * Cpc) / (rho * Cp * V))
                        * qc
                        * (1 - np.exp(-hA / (qc * rho * Cp)))
                        * (Tco - T)
                    )
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
        R = self.r[self.k][0]
        self.E[self.k] = R - self.y[self.k]
        self.dU[self.k] = Kc * (
            self.E[self.k]
            - self.E[self.k - 1]
            + (self.delt / tau_i) * self.E[self.k]
            + (tau_d / self.delt)
            * (self.E[self.k] - 2 * self.E[self.k - 1] + self.E[self.k - 2])
        )
        self.dU[self.k] = np.clip(
            self.dU[self.k], self.dU[self.k - 1] - 5, self.dU[self.k - 1] + 5
        )
        self.U[self.k] = self.U[self.k - 1] + self.dU[self.k]
        w = 0.05 * np.random.randn() if self.disturbance else 0.0
        d = 0.05 if self.k >= 80 else 0.0
        self.u[self.k] = self.U[self.k] + self.ubar + w + d
        self.u[self.k] = np.maximum(1e-4, self.u[self.k])
        sol = solve_ivp(
            self.dec_ode(self.u[self.k]), [self.tinitial, self.tfinal], self.ss_s1
        )
        self.ss_s1[0] = sol.y[0][-1]
        self.ss_s1[1] = sol.y[1][-1]
        self.m[self.k] = self.get_model_region()
        self.y[self.k + 1] = self.ss_s1[0]
        self.Y[self.k + 1] = self.y[self.k + 1] - self.ybar
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
            plt.figure(figsize=(16, 16))
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

    def get_state(self):
        return self.u[self.k - 1], self.m[self.k - 1], self.ss_s1[1], self.r[self.k][0]

    def get_model_region(self):
        m = -3
        if self.u[self.k] >= 97.0 and self.u[self.k] <= 100.0:
            m = -2
        elif self.u[self.k] > 100.0 and self.u[self.k] <= 103.0:
            m = -1
        elif self.u[self.k] > 103.0 and self.u[self.k] <= 106.0:
            m = 0
        elif self.u[self.k] > 106.0 and self.u[self.k] <= 109.0:
            m = 1
        elif self.u[self.k] > 109.0:
            m = 2
        return m

    def plot_model_regions(self, use_sample_instant=True):
        axis = self.tt[: self.k].copy()
        if use_sample_instant:
            axis = np.arange(self.k)
        plt.figure(figsize=(16, 9))
        plt.plot(axis, self.m[: self.k])
        plt.title("Model Region")
        plt.xlabel("Time")
        plt.ylabel("Linear Region Number")
        plt.grid()
        plt.show()

    def ise(self):
        return ((self.r[: self.k] - self.y[: self.k]) ** 2).sum()


qcbar = 106.0
tbar = 438.0


class GymCSTRPID(gym.Env):
    def __init__(
        self,
        deterministic=True,
        ubar=ubar,
        T=T,
        ybar=ybar,
        ysp=ysp,
        timesp=1,
        disturbance=True,
    ):
        super(GymCSTRPID, self).__init__()
        n_actions = 3
        self.action_space = spaces.Box(-1.0, 1.0, (n_actions,))
        self.observation_space = spaces.Box(
            low=-1000.0, high=1000.0, shape=(4,), dtype=np.float32
        )
        self.deterministic = deterministic
        self.ubar = ubar
        self.ybar = ybar
        self.T = T
        self.ysp = ysp
        self.timesp = timesp
        self.disturbance = disturbance
        self.system = CSTRPID(
            ubar=self.ubar,
            ybar=self.ybar,
            T=self.T,
            ysp=self.ysp,
            timesp=self.timesp,
            disturbance=self.disturbance,
        )

    def convert_state(self):
        obs = self.system.get_state()
        # obs = obs[1]
        obs = np.array(obs).astype(np.float32)
        obs[0] = (obs[0] - qcbar) * 5.0 / 12.0
        obs[2] = (obs[2] - tbar) / 10.0
        return obs

    def reset(self):
        if not self.deterministic:
            self.ubar = np.random.randint(100, 110)
            self.ybar = np.random.uniform(0.09, 0.13)
            self.ysp = np.random.uniform(0.1, 0.14)
            self.timesp = np.random.randint(1, 4)
            _ = self.system.reset_init(
                ubar=self.ubar, ybar=self.ybar, ysp=self.ysp, timesp=self.timesp
            )
        else:
            _ = self.system.reset()
        obs = self.convert_state()
        return obs

    def step(self, action, debug=False):
        reward = 0.0
        scaling = 0.5
        if (
            action[0] <= 1.0
            and action[0] >= -1.0
            and action[1] <= 1.0
            and action[1] >= -1.0
            and action[2] <= 1.0
            and action[2] >= -1.0
        ):
            action[0] = action[0] * 70.0 + 70.0
            action[1] = action[1] * scaling + 0.5 + 0.1
            action[2] = action[2] * scaling + 0.5
        else:
            reward += -5.0
        if debug:
            print(action)
        obs = self.system.step(*action)
        e = obs[0] - obs[1]
        e_squared = np.abs(e)
        reward += -e_squared
        info = {}
        done = bool(self.system.k == (self.system.kfinal - 1))
        obs = self.convert_state()
        return obs, reward, done, info

    def render(self, mode="plot"):
        if mode != "plot":
            raise NotImplementedError()
        print("ISE: ", self.system.ise())
        self.system.plot()

    def close(self):
        pass


if __name__ == "__main__":
    sim = CSTRPID()
    sim.reset()
    for i in range(sim.kfinal - sim.ksp):
        _, _ = sim.step(119, 0.3367, 0.1926)
    print("ISE: ", sim.ise())
    sim.plot()